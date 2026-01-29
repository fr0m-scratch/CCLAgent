import json
import os
from litellm import completion
import time
from typing import Dict, Any, Optional


model = "anthropic/claude-3-5-sonnet-latest"
path_root = "/Users/chris/GitHub/PFSAgent/Agent/FSConfigs/Docs"

def get_param_info(param_name: str, description: str) -> Optional[Dict[str, Any]]:
    """
    Query the language model to get detailed parameter information.
    
    Args:
        param_name: The name of the Lustre parameter
        description: The basic description from the original JSON
    
    Returns:
        Dictionary containing detailed parameter information or None if not tunable
    """
    
    prompt = f"""
    Analyze this Lustre filesystem parameter and provide detailed information about it:
    
    Parameter name: {param_name}
    Basic description: {description}
    
    Provide the information in this format if it's a tunable parameter:
    {{
        "definition": "Brief technical definition",
        "effect": "Detailed description of performance/behavior effects",
        "type": "One of: boolean, size, integer, enum, or string",
        "range_description": "Human readable description of valid values",
        "range": {{
            Appropriate range specification:
            For boolean: "values": [0, 1]
            For size: "min": X, "max": Y
            For integer: "min": X, "max": Y
            For enum: "values": [possible, values, list]
            For string: "pattern": "regex_pattern"
        }}
    }}
    
    If this is not a tunable parameter, respond with exactly: "not a tunable parameter"
    """

    try:
        response = completion(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        
        if result == "not a tunable parameter":
            return None
            
        return json.loads(result)
        
    except Exception as e:
        print(f"Error processing parameter {param_name}: {str(e)}")
        return None

def main():
    # Read the original parameter list
    with open(f"{path_root}/lustre.json", "r") as f:
        params = json.load(f)
    
    if "/" in model:
        model_name = model.split("/")[1]
    else:
        model_name = model
    
    output_path = f"{path_root}/lustre_param_descriptions_{model_name}.json"
    
    # Load existing enhanced parameters if file exists
    enhanced_params = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            enhanced_params = json.load(f)
    
    # Process each parameter
    processed_count = 0
    for param in params:
        name = param["name"]
        description = param["description"]
        
        # Skip if already processed
        if name in enhanced_params:
            print(f"Skipping already processed parameter: {name}")
            continue
            
        print(f"Processing parameter: {name}")
        
        # Get enhanced information from the language model
        param_info = get_param_info(name, description)
        
        if param_info:
            enhanced_params[name] = param_info
            processed_count += 1
            
            # Save after each successful parameter processing
            with open(output_path, "w") as f:
                json.dump(enhanced_params, f, indent=4)
            
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    print(f"\nEnhanced parameter descriptions saved to {output_path}")
    print(f"Processed {processed_count} new tunable parameters")
    print(f"Total tunable parameters: {len(enhanced_params)}")

if __name__ == "__main__":
    main() 