#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

def try_read_csv(filename):
    """
    Safely attempts to read a CSV file.
    Returns a pandas DataFrame if successful, or None if the file does not exist or fails to load.
    """
    if os.path.isfile(filename):
        try:
            return pd.read_csv(filename)
        except Exception as e:
            print(f"Warning: Could not read '{filename}'. Error: {e}")
            return None
    else:
        print(f"Info: '{filename}' not found. Skipping.")
        return None

def try_read_text(filename):
    """
    Safely attempts to read a text file.
    Returns the text content if successful, or None if file missing or unreadable.
    """
    if os.path.isfile(filename):
        try:
            with open(filename, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not read '{filename}'. Error: {e}")
            return None
    else:
        return None

def analyze_posix_data(posix_df):
    """
    Given a POSIX DataFrame, compute relevant aggregations and summaries.
    """
    # Compute open/close durations from timestamps if they exist.
    if all(col in posix_df.columns for col in [
            'POSIX_F_OPEN_END_TIMESTAMP','POSIX_F_OPEN_START_TIMESTAMP',
            'POSIX_F_CLOSE_END_TIMESTAMP','POSIX_F_CLOSE_START_TIMESTAMP']):
        posix_df['POSIX_F_OPEN_TIME'] = (
            posix_df['POSIX_F_OPEN_END_TIMESTAMP']
            - posix_df['POSIX_F_OPEN_START_TIMESTAMP']
        )
        posix_df['POSIX_F_CLOSE_TIME'] = (
            posix_df['POSIX_F_CLOSE_END_TIMESTAMP']
            - posix_df['POSIX_F_CLOSE_START_TIMESTAMP']
        )
    else:
        # If the time-stamp columns do not exist, fill with zeros
        posix_df['POSIX_F_OPEN_TIME'] = 0.0
        posix_df['POSIX_F_CLOSE_TIME'] = 0.0

    # Create 'directory' column to facilitate grouping
    posix_df['directory'] = posix_df['file_name'].apply(os.path.dirname)

    # Identify which columns exist before constructing our aggregation dictionary
    agg_columns = {
        'POSIX_OPENS': 'sum',
        'POSIX_READS': 'sum',
        'POSIX_WRITES': 'sum',
        'POSIX_BYTES_READ': 'sum',
        'POSIX_BYTES_WRITTEN': 'sum',
        'POSIX_F_READ_TIME': 'sum',
        'POSIX_F_WRITE_TIME': 'sum',
        'POSIX_F_META_TIME': 'sum',
        'POSIX_F_OPEN_TIME': 'sum',
        'POSIX_F_CLOSE_TIME': 'sum'
    }
    # Filter out any columns not in the dataframe
    valid_agg = {k:v for k,v in agg_columns.items() if k in posix_df.columns}

    # Group by directory and apply the valid aggregations
    posix_grouped = posix_df.groupby('directory')
    posix_summary = posix_grouped.agg(valid_agg).reset_index(drop=False)

    # Compute derived metrics if the needed columns exist
    if 'POSIX_F_READ_TIME' in posix_summary.columns and 'POSIX_F_WRITE_TIME' in posix_summary.columns:
        posix_summary['TOTAL_DATA_TIME'] = (
            posix_summary['POSIX_F_READ_TIME'] + posix_summary['POSIX_F_WRITE_TIME']
        )
    else:
        posix_summary['TOTAL_DATA_TIME'] = 0.0

    time_cols = [
        'TOTAL_DATA_TIME',
        'POSIX_F_META_TIME' if 'POSIX_F_META_TIME' in posix_summary.columns else None,
        'POSIX_F_OPEN_TIME' if 'POSIX_F_OPEN_TIME' in posix_summary.columns else None,
        'POSIX_F_CLOSE_TIME' if 'POSIX_F_CLOSE_TIME' in posix_summary.columns else None
    ]
    # Filter out any None entries
    time_cols = [c for c in time_cols if c is not None]

    if time_cols:
        posix_summary['TOTAL_IO_TIME'] = posix_summary[time_cols].sum(axis=1)
    else:
        posix_summary['TOTAL_IO_TIME'] = 0.0

    # Average read/write sizes
    if 'POSIX_READS' in posix_summary.columns and 'POSIX_BYTES_READ' in posix_summary.columns:
        posix_summary['AVG_READ_SIZE'] = posix_summary.apply(
            lambda x: x['POSIX_BYTES_READ'] / x['POSIX_READS'] if x['POSIX_READS'] > 0 else 0.0,
            axis=1
        )
    else:
        posix_summary['AVG_READ_SIZE'] = 0.0

    if 'POSIX_WRITES' in posix_summary.columns and 'POSIX_BYTES_WRITTEN' in posix_summary.columns:
        posix_summary['AVG_WRITE_SIZE'] = posix_summary.apply(
            lambda x: x['POSIX_BYTES_WRITTEN'] / x['POSIX_WRITES'] if x['POSIX_WRITES'] > 0 else 0.0,
            axis=1
        )
    else:
        posix_summary['AVG_WRITE_SIZE'] = 0.0

    # Count how many unique files per directory
    file_counts = posix_df.groupby('directory')['file_name'].nunique()
    posix_summary = posix_summary.merge(
        file_counts.reset_index(name='FILE_COUNT'),
        on='directory',
        how='left'
    )

    return posix_summary

def analyze_lustre_data(lustre_df):
    """
    Given a LUSTRE DataFrame, compute relevant aggregations and summaries.
    """
    # Add directory column
    lustre_df['directory'] = lustre_df['file_name'].apply(os.path.dirname)
    lustre_grouped = lustre_df.groupby('directory')

    # Set up an aggregation dict for columns that exist
    agg_dict_lustre = {}
    for col in ['LUSTRE_STRIPE_WIDTH', 'LUSTRE_STRIPE_SIZE', 'LUSTRE_STRIPE_OFFSET']:
        if col in lustre_df.columns:
            agg_dict_lustre[col] = 'max'

    lustre_summary = lustre_grouped.agg(agg_dict_lustre).reset_index()

    # Count how many unique files per directory
    file_counts = lustre_df.groupby('directory')['file_name'].nunique()
    lustre_summary = lustre_summary.merge(
        file_counts.reset_index(name='FILE_COUNT'),
        on='directory',
        how='left'
    )

    return lustre_summary

def print_posix_overall_stats(posix_summary):
    """
    Compute overall sums and distribution from the POSIX summary DataFrame
    and print them out.
    """
    # Make sure columns exist before summing
    safe_sum = lambda c: posix_summary[c].sum() if c in posix_summary.columns else 0.0

    summary_dict = {
        'TOTAL_OPENS': safe_sum('POSIX_OPENS'),
        'TOTAL_READS': safe_sum('POSIX_READS'),
        'TOTAL_WRITES': safe_sum('POSIX_WRITES'),
        'TOTAL_BYTES_READ': safe_sum('POSIX_BYTES_READ'),
        'TOTAL_BYTES_WRITTEN': safe_sum('POSIX_BYTES_WRITTEN'),
        'AGGREGATE_READ_TIME': safe_sum('POSIX_F_READ_TIME'),
        'AGGREGATE_WRITE_TIME': safe_sum('POSIX_F_WRITE_TIME'),
        'AGGREGATE_META_TIME': safe_sum('POSIX_F_META_TIME'),
        'AGGREGATE_OPEN_TIME': safe_sum('POSIX_F_OPEN_TIME'),
        'AGGREGATE_CLOSE_TIME': safe_sum('POSIX_F_CLOSE_TIME'),
        'AGGREGATE_DATA_TIME': safe_sum('TOTAL_DATA_TIME'),
        'AGGREGATE_IO_TIME': safe_sum('TOTAL_IO_TIME')
    }

    # Fraction of metadata time
    total_io = summary_dict['AGGREGATE_IO_TIME']
    meta_time = summary_dict['AGGREGATE_META_TIME']
    if total_io > 0.0:
        summary_dict['META_TIME_FRACTION'] = meta_time / total_io
    else:
        summary_dict['META_TIME_FRACTION'] = 0.0

    print("=== Overall POSIX Aggregated Stats ===")
    for k, v in summary_dict.items():
        print(f"{k}: {v}")

    # Print distributions
    if 'AVG_READ_SIZE' in posix_summary.columns:
        print("\n=== Distribution of per-directory average read sizes (bytes) ===")
        print(posix_summary['AVG_READ_SIZE'].describe())

    if 'AVG_WRITE_SIZE' in posix_summary.columns:
        print("\n=== Distribution of per-directory average write sizes (bytes) ===")
        print(posix_summary['AVG_WRITE_SIZE'].describe())

    if 'POSIX_F_META_TIME' in posix_summary.columns:
        print("\n=== Distribution of per-directory metadata time (seconds) ===")
        print(posix_summary['POSIX_F_META_TIME'].describe())

        # Top 5 directories by total metadata time
        top5_meta = posix_summary.nlargest(
            5, 'POSIX_F_META_TIME'
        )[['directory', 'POSIX_F_META_TIME', 'FILE_COUNT']]
        print("\n=== Top 5 directories by total metadata time ===")
        print(top5_meta)

def print_lustre_stripe_stats(lustre_summary):
    """
    Check uniqueness of Lustre stripe settings across directories.
    """
    # Only proceed if the columns exist
    stripe_size_col = 'LUSTRE_STRIPE_SIZE'
    stripe_width_col = 'LUSTRE_STRIPE_WIDTH'
    stripe_offset_col = 'LUSTRE_STRIPE_OFFSET'

    print("\n=== LUSTRE Stripe Settings (Unique Values) ===")
    if stripe_size_col in lustre_summary.columns:
        unique_sizes = lustre_summary[stripe_size_col].unique()
        print(f"Stripe Sizes: {unique_sizes}")
    else:
        print("No 'LUSTRE_STRIPE_SIZE' column present.")

    if stripe_width_col in lustre_summary.columns:
        unique_widths = lustre_summary[stripe_width_col].unique()
        print(f"Stripe Widths: {unique_widths}")
    else:
        print("No 'LUSTRE_STRIPE_WIDTH' column present.")

    if stripe_offset_col in lustre_summary.columns:
        unique_offsets = lustre_summary[stripe_offset_col].unique()
        print(f"Stripe Offsets: {unique_offsets}")
    else:
        print("No 'LUSTRE_STRIPE_OFFSET' column present.")


def main():
    # ----------------------------------------------------------------
    # 0. Attempt to read data from each module if available
    # ----------------------------------------------------------------
    header_text = try_read_text('header.txt')

    # The CSV data for each module (or None if missing)
    HEATMAP_data = try_read_csv('HEATMAP.csv')
    HEATMAP_description = try_read_text('HEATMAP_description.txt')

    STDIO_data = try_read_csv('STDIO.csv')
    STDIO_description = try_read_text('STDIO_description.txt')

    POSIX_data = try_read_csv('POSIX.csv')
    POSIX_description = try_read_text('POSIX_description.txt')

    LUSTRE_data = try_read_csv('LUSTRE.csv')
    LUSTRE_description = try_read_text('LUSTRE_description.txt')

    # ----------------------------------------------------------------
    # 1. If POSIX data is present, perform analysis
    # ----------------------------------------------------------------
    if POSIX_data is not None and not POSIX_data.empty:
        print("\n=== Analyzing POSIX data... ===")
        posix_summary = analyze_posix_data(POSIX_data)
        print("POSIX Summary (first few rows):")
        print(posix_summary.head())
        print_posix_overall_stats(posix_summary)
    else:
        print("\nNo POSIX data to analyze (POSIX.csv missing or empty).")

    # ----------------------------------------------------------------
    # 2. If LUSTRE data is present, perform analysis
    # ----------------------------------------------------------------
    if LUSTRE_data is not None and not LUSTRE_data.empty:
        print("\n=== Analyzing LUSTRE data... ===")
        lustre_summary = analyze_lustre_data(LUSTRE_data)
        print("LUSTRE Summary (first few rows):")
        print(lustre_summary.head())
        print_lustre_stripe_stats(lustre_summary)
    else:
        print("\nNo LUSTRE data to analyze (LUSTRE.csv missing or empty).")

    # ----------------------------------------------------------------
    # 3. Additional modules (STDIO, HEATMAP) can be handled similarly
    #    if desired. For now, we simply announce if they exist.
    # ----------------------------------------------------------------
    if STDIO_data is not None:
        print("\nSTDIO data loaded; no specialized analysis implemented.")
    if HEATMAP_data is not None:
        print("\nHEATMAP data loaded; no specialized analysis implemented.")

    print("\n=== Darshan I/O Analysis Completed ===")

if __name__ == "__main__":
    main()