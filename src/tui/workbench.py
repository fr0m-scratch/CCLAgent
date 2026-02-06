from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rich.pretty import Pretty
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical
    from textual.timer import Timer
    from textual.widgets import (
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        RichLog,
        Static,
        TabbedContent,
        TabPane,
    )
except ImportError as exc:
    raise RuntimeError("textual and rich are required for the TUI: pip install textual rich") from exc

from ..utils import load_env_file


def _fmt_ts(ts: Optional[float]) -> str:
    if not isinstance(ts, (int, float)):
        return "--:--:--"
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _safe_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        return None
    except Exception:
        return None


def _truncate_text(text: Any, max_len: int = 5000) -> str:
    if text is None:
        return ""
    value = str(text)
    if len(value) <= max_len:
        return value
    return value[:max_len] + f"\n... [truncated {len(value) - max_len} chars]"


def _compact(value: Any, *, depth: int = 0, max_depth: int = 4, max_items: int = 20) -> Any:
    if depth >= max_depth:
        return "<trimmed>"
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                out["..."] = f"{len(value) - max_items} more keys"
                break
            out[str(key)] = _compact(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)
        return out
    if isinstance(value, list):
        if len(value) > max_items:
            return [
                _compact(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)
                for item in value[:max_items]
            ] + [f"... {len(value) - max_items} more items"]
        return [_compact(item, depth=depth + 1, max_depth=max_depth, max_items=max_items) for item in value]
    if isinstance(value, str):
        return _truncate_text(value, max_len=1200)
    return value


def _json_load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _maybe_parse_json_text(text: Any) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except Exception:
        return None


def _extract_index_from_name(name: str, *, prefix: str, suffix: str) -> int:
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return -1
    raw = name[len(prefix) : len(name) - len(suffix)]
    return int(raw) if raw.isdigit() else -1


@dataclass
class AsyncAdviceStatus:
    step: int
    requested_ts: Optional[float] = None
    ready_ts: Optional[float] = None
    call_id: Optional[str] = None
    used_in_decision: Optional[bool] = None
    state: str = "pending"


@dataclass
class WorkbenchState:
    run_dir: Optional[Path] = None
    run_id: str = ""
    phase: str = "starting"
    running: bool = True

    events: List[Dict[str, Any]] = field(default_factory=list)
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    base_steps: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    selected_event_idx: Optional[int] = None
    selected_step: Optional[int] = None
    selected_llm_idx: Optional[int] = None
    selected_async_step: Optional[int] = None

    async_advice: Dict[int, AsyncAdviceStatus] = field(default_factory=dict)
    latest_chat_context: Optional[Dict[str, Any]] = None

    iteration_times_ms: List[float] = field(default_factory=list)
    best_iteration_ms: Optional[float] = None
    native_baseline_ms: Optional[float] = None


class AgentWorkbench(App):
    """Unified modern monitor for both inspect and live modes."""

    EVENT_TABLE_MAX_ROWS = 450
    DIR_STALE_SCAN_S = 1.5

    CSS = """
    Screen {
        layout: vertical;
        background: #060b14;
        color: #e5efff;
    }

    #root {
        height: 1fr;
    }

    #hero {
        height: 7;
        padding: 0 1;
        border-bottom: solid #1b2b44;
        background: #08101d;
    }

    #hero-grid {
        height: 1fr;
    }

    .hero-card {
        width: 1fr;
        margin: 0 1 0 0;
        border: solid #1e3657;
        background: #0d1727;
        color: #dbe9ff;
        padding: 0 1;
        content-align: left middle;
    }

    .hero-title {
        color: #7dc7ff;
        text-style: bold;
    }

    #workspace {
        height: 1fr;
    }

    #left-pane {
        width: 38%;
        min-width: 52;
        border-right: solid #1b2b44;
        background: #0a1423;
    }

    #right-pane {
        width: 62%;
        min-width: 70;
        background: #08101d;
    }

    #left-tabs,
    #right-tabs {
        height: 1fr;
    }

    DataTable {
        height: 1fr;
        background: #0a1525;
        color: #e3edff;
    }

    DataTable > .datatable--header {
        background: #14263d;
        color: #97c7ff;
        text-style: bold;
    }

    TabbedContent {
        background: #08101d;
    }

    Tabs {
        background: #0f1d31;
    }

    Tab {
        color: #8aa8cc;
    }

    Tab.-active {
        color: #61c6ff;
        text-style: bold;
    }

    RichLog {
        height: 1fr;
        background: #07101d;
        color: #dbe9ff;
        padding: 1;
    }

    #chat-pane {
        height: 11;
        border-top: heavy #1f7bd6;
        background: #07101a;
        padding: 0 1;
    }

    #chat-log {
        height: 1fr;
        background: #0a1423;
    }

    #chat-input {
        height: 3;
        margin-top: 1;
        background: #12213a;
        border: solid #2a5f96;
        color: #e5efff;
    }

    .dim {
        color: #87a1c6;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_now", "Refresh"),
        Binding("1", "focus_left('events')", "Events"),
        Binding("2", "focus_left('steps')", "Steps"),
        Binding("3", "focus_left('llm')", "LLM"),
        Binding("4", "focus_left('async')", "Async"),
        Binding("5", "focus_right('overview')", "Overview"),
        Binding("6", "focus_right('event')", "Event Detail"),
        Binding("7", "focus_right('llm')", "LLM Detail"),
        Binding("8", "focus_right('reasoning')", "Reasoning"),
        Binding("9", "focus_right('context')", "Context"),
        Binding("0", "focus_right('pruning')", "Pruning"),
        Binding("p", "focus_right('llmreason')", "LLM Reasoning"),
        Binding("ctrl+l", "focus_chat", "Chat"),
    ]

    def __init__(
        self,
        *,
        bridge: Any = None,
        artifacts_root: str = "artifacts",
        run_dir: Optional[str] = None,
        poll_interval: float = 0.5,
        env_file: str = ".env.local",
        live_mode: bool = False,
    ) -> None:
        super().__init__()
        self.bridge = bridge
        self.artifacts_root = Path(artifacts_root)
        self.run_dir_override = Path(run_dir) if run_dir else None
        self.poll_interval = poll_interval
        self.live_mode = live_mode or bridge is not None

        load_env_file(env_file)

        self.state = WorkbenchState()
        self._poll_timer: Optional[Timer] = None
        self._quitting = False

        self._json_cache: Dict[str, tuple[float, Any]] = {}
        self._trace_offset = 0
        self._trace_path: Optional[Path] = None
        self._trace_mtime = 0.0

        self._seen_event_keys: set[str] = set()
        self._render_signatures: Dict[str, Any] = {}
        self._render_timestamps: Dict[str, float] = {}

        # Table key -> state index mapping.
        self._event_row_to_idx: Dict[str, int] = {}
        self._llm_row_to_idx: Dict[str, int] = {}
        self._step_row_to_step: Dict[str, int] = {}
        self._async_row_to_step: Dict[str, int] = {}
        self._left_scroll_y: Dict[str, float] = {}

        # Performance caches.
        self._refresh_cycle_active = False
        self._run_dir_scan_at = 0.0
        self._cached_latest_run_dir: Optional[Path] = None
        self._poll_marks: Dict[str, float] = {}
        self._dir_scan_cache: Dict[str, tuple[int, float]] = {}
        self._base_steps_signature: tuple = ()
        self._llm_calls_signature: tuple = ()
        self._step_bundle_cache: Dict[int, tuple[tuple, Dict[str, Any]]] = {}
        self._offline_bundle_signature: tuple = ()
        self._offline_bundle_cache: Dict[str, Any] = {}
        self._jsonl_cache: Dict[str, tuple[float, int, List[Dict[str, Any]]]] = {}
        self._advice_file_mtimes: Dict[str, float] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="root"):
            with Container(id="hero"):
                with Horizontal(id="hero-grid"):
                    yield Static("", id="card-run", classes="hero-card")
                    yield Static("", id="card-phase", classes="hero-card")
                    yield Static("", id="card-perf", classes="hero-card")
                    yield Static("", id="card-async", classes="hero-card")
                    yield Static("", id="card-goal", classes="hero-card")

            with Horizontal(id="workspace"):
                with Vertical(id="left-pane"):
                    with TabbedContent(id="left-tabs"):
                        with TabPane("Events", id="left-events"):
                            yield DataTable(id="events-table", cursor_type="row")
                        with TabPane("Steps", id="left-steps"):
                            yield DataTable(id="steps-table", cursor_type="row")
                        with TabPane("LLM Calls", id="left-llm"):
                            yield DataTable(id="llm-table", cursor_type="row")
                        with TabPane("LLM Async", id="left-async"):
                            yield DataTable(id="async-table", cursor_type="row")

                with Vertical(id="right-pane"):
                    with TabbedContent(id="right-tabs"):
                        with TabPane("Overview", id="right-overview"):
                            yield RichLog(id="overview-log", auto_scroll=True, wrap=True)
                        with TabPane("Event Detail", id="right-event"):
                            yield RichLog(id="event-detail-log", auto_scroll=True, wrap=True)
                        with TabPane("LLM Detail", id="right-llm"):
                            yield RichLog(id="llm-detail-log", auto_scroll=True, wrap=True)
                        with TabPane("Online Reasoning", id="right-reasoning"):
                            yield RichLog(id="reasoning-log", auto_scroll=True, wrap=True)
                        with TabPane("Context Engineering", id="right-context"):
                            yield RichLog(id="context-log", auto_scroll=True, wrap=True)
                        with TabPane("Pruning Lens", id="right-pruning"):
                            yield RichLog(id="pruning-log", auto_scroll=True, wrap=True)
                        with TabPane("LLM Reasoning", id="right-llmreason"):
                            yield RichLog(id="llm-reasoning-log", auto_scroll=True, wrap=True)

            with Container(id="chat-pane"):
                yield RichLog(id="chat-log", auto_scroll=True, wrap=True)
                yield Input(
                    id="chat-input",
                    placeholder="Live commands: /set, /setcfg, /setplan, /state, /best, /context, /ctxeng, or ask questions",
                )

        yield Footer()

    def on_mount(self) -> None:
        self._init_tables()
        self._emit_system_chat(
            "Control tower ready. Select an event/step/LLM call on the left. "
            "Bottom chat works in live mode (runtime control) and inspect mode (artifact-backed Q&A)."
        )
        self._poll_timer = self.set_interval(self.poll_interval, self._refresh_now)
        self._refresh_now()

    async def on_unmount(self) -> None:
        if self.bridge is not None:
            self.bridge.stopped.set()

    def action_refresh_now(self) -> None:
        self._refresh_now()

    def action_focus_left(self, tab: str) -> None:
        self.query_one("#left-tabs", TabbedContent).active = f"left-{tab}"
        self._render_active_left_table()

    def action_focus_right(self, tab: str) -> None:
        self.query_one("#right-tabs", TabbedContent).active = f"right-{tab}"
        self._render_active_right_pane()

    def action_focus_chat(self) -> None:
        self.query_one("#chat-input", Input).focus()

    def action_quit(self) -> None:
        self._quitting = True
        if self._poll_timer:
            self._poll_timer.stop()
        if self.bridge is not None:
            try:
                from ..runner import AgentCommand

                self.bridge.send_command(AgentCommand(action="stop", payload={"silent": True}))
            except Exception:
                pass
            self.bridge.stopped.set()
            self.bridge.paused.set()
        self.exit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "chat-input":
            return
        message = event.value.strip()
        if not message:
            return
        event.input.value = ""

        self._emit_chat("You", message, style="bold #63d8ff")

        if self.bridge is None:
            self._handle_inspect_chat(message)
            return

        try:
            from ..runner import AgentCommand

            self.bridge.send_command(AgentCommand(action="chat", payload={"message": message}))
            self._emit_chat("System", "Command sent to agent.", style="dim")
        except Exception as exc:
            self._emit_chat("System", f"Failed to send command: {exc}", style="bold red")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table_id = event.data_table.id or ""
        row_key = self._row_key_to_str(event.row_key)

        if table_id == "events-table":
            idx = self._event_row_to_idx.get(row_key)
            if idx is not None:
                self.state.selected_event_idx = idx
                self.action_focus_right("event")
                self._render_event_detail()
        elif table_id == "steps-table":
            step = self._step_row_to_step.get(row_key)
            if step is not None:
                self.state.selected_step = step
                self.action_focus_right("reasoning")
                self._render_reasoning_detail()
                self._render_context_detail()
        elif table_id == "llm-table":
            idx = self._llm_row_to_idx.get(row_key)
            if idx is not None:
                self.state.selected_llm_idx = idx
                self.action_focus_right("llm")
                self._render_llm_detail()
                self._render_context_detail()
        elif table_id == "async-table":
            step = self._async_row_to_step.get(row_key)
            if step is not None:
                self.state.selected_async_step = step
                self.state.selected_step = step
                self.action_focus_right("reasoning")
                self._render_reasoning_detail()

    def _init_tables(self) -> None:
        events = self.query_one("#events-table", DataTable)
        events.add_columns("time", "phase", "step", "actor", "type", "status", "ms")

        steps = self.query_one("#steps-table", DataTable)
        steps.add_columns("step", "action", "iter_ms", "delta%", "ok", "bottleneck")

        llm = self.query_one("#llm-table", DataTable)
        llm.add_columns("step", "phase", "model", "ctx_tok", "dur_ms", "call")

        async_tbl = self.query_one("#async-table", DataTable)
        async_tbl.add_columns("step", "state", "requested", "ready", "lat_ms", "used", "call")

    # ---------------------------------------------------------------------
    # Refresh Pipeline
    # ---------------------------------------------------------------------

    def _refresh_now(self) -> None:
        if self._quitting:
            return

        self._refresh_cycle_active = True
        try:
            if self.bridge is not None:
                self._poll_bridge_events()

            run_dir = self._resolve_run_dir()
            if run_dir is None:
                self._render_no_run()
                return

            if self.state.run_dir != run_dir:
                self._switch_run(run_dir)

            self._poll_trace_disk()
            step_changed = self._poll_step_files()

            llm_interval = 0.35 if self._llm_artifact_hot_path() else 1.0
            if self._should_poll("llm_calls", llm_interval):
                self._poll_llm_calls()
            if self._should_poll("chat_context", 0.8):
                self._poll_chat_context_snapshots()
            if self._should_poll("llm_advice_files", 0.8):
                self._poll_late_llm_advice_files()

            if step_changed:
                self._derive_iteration_series()
            self._apply_defaults()

            self._render_cards()
            self._render_active_left_table()
            self._render_active_right_pane()
        finally:
            self._refresh_cycle_active = False

    def _render_active_left_table(self) -> None:
        active = self.query_one("#left-tabs", TabbedContent).active
        if active == "left-events":
            self._render_events_table()
        elif active == "left-steps":
            self._render_steps_table()
        elif active == "left-llm":
            self._render_llm_table()
        elif active == "left-async":
            self._render_async_table()
        else:
            self._render_events_table()

    def _render_active_right_pane(self) -> None:
        active = self.query_one("#right-tabs", TabbedContent).active
        if active == "right-overview":
            self._render_overview()
        elif active == "right-event":
            self._render_event_detail()
        elif active == "right-llm":
            self._render_llm_detail()
        elif active == "right-reasoning":
            self._render_reasoning_detail()
        elif active == "right-context":
            self._render_context_detail()
        elif active == "right-pruning":
            self._render_pruning_lens()
        elif active == "right-llmreason":
            self._render_llm_reasoning_timeline()
        else:
            self._render_overview()

    def _render_no_run(self) -> None:
        self.query_one("#card-run", Static).update("Run\nNo artifacts found")
        self.query_one("#card-phase", Static).update("Phase\nwaiting")
        self.query_one("#card-perf", Static).update("Performance\n--")
        self.query_one("#card-async", Static).update("LLM Async\n--")
        self.query_one("#card-goal", Static).update(
            "Goal\nMinimize end-to-end training iteration time"
        )

    def _llm_artifact_hot_path(self) -> bool:
        left_active = self.query_one("#left-tabs", TabbedContent).active
        right_active = self.query_one("#right-tabs", TabbedContent).active
        if left_active == "left-llm":
            return True
        return right_active in ("right-llm", "right-context", "right-llmreason", "right-event")

    def _should_poll(self, key: str, interval_s: float) -> bool:
        now = time.monotonic()
        last = self._poll_marks.get(key, 0.0)
        if now - last >= interval_s:
            self._poll_marks[key] = now
            return True
        return False

    def _should_scan_dir(self, key: str, path: Path, *, stale_s: float) -> bool:
        if not path.exists():
            return False
        try:
            dir_mtime = path.stat().st_mtime_ns
        except Exception:
            return False
        full_key = f"{key}:{path}"
        now = time.monotonic()
        cached = self._dir_scan_cache.get(full_key)
        if cached:
            prev_mtime, last_scan = cached
            if prev_mtime == dir_mtime and (now - last_scan) < stale_s:
                return False
        self._dir_scan_cache[full_key] = (dir_mtime, now)
        return True

    def _resolve_run_dir(self) -> Optional[Path]:
        if self.run_dir_override and self.run_dir_override.exists():
            return self.run_dir_override

        if self.state.run_dir and self.state.run_dir.exists():
            return self.state.run_dir

        now = time.monotonic()
        if (now - self._run_dir_scan_at) < 2.0:
            if self._cached_latest_run_dir and self._cached_latest_run_dir.exists():
                return self._cached_latest_run_dir
            if self._cached_latest_run_dir is None:
                return None

        if not self.artifacts_root.exists():
            self._run_dir_scan_at = now
            self._cached_latest_run_dir = None
            return None

        runs = [p for p in self.artifacts_root.iterdir() if p.is_dir()]
        if not runs:
            self._run_dir_scan_at = now
            self._cached_latest_run_dir = None
            return None
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        self._run_dir_scan_at = now
        self._cached_latest_run_dir = runs[0]
        return runs[0]

    def _switch_run(self, run_dir: Path) -> None:
        self.state = WorkbenchState(run_dir=run_dir, run_id=run_dir.name)
        self._trace_offset = 0
        self._trace_mtime = 0.0
        self._trace_path = run_dir / "trace" / "events.jsonl"
        self._json_cache.clear()
        self._jsonl_cache.clear()
        self._seen_event_keys.clear()
        self._render_signatures.clear()
        self._render_timestamps.clear()
        self._event_row_to_idx.clear()
        self._llm_row_to_idx.clear()
        self._step_row_to_step.clear()
        self._async_row_to_step.clear()
        self._left_scroll_y.clear()
        self._poll_marks.clear()
        self._dir_scan_cache.clear()
        self._base_steps_signature = ()
        self._llm_calls_signature = ()
        self._step_bundle_cache.clear()
        self._offline_bundle_signature = ()
        self._offline_bundle_cache = {}
        self._advice_file_mtimes.clear()
        self._emit_system_chat(f"Switched to run {run_dir.name}")

    def _poll_bridge_events(self) -> None:
        events = self.bridge.poll_events(timeout=0.01, max_events=256)
        for bridge_evt in events:
            payload = bridge_evt.payload if isinstance(bridge_evt.payload, dict) else {}

            if bridge_evt.event_type == "run_started":
                run_id = payload.get("run_id")
                artifacts_dir = payload.get("artifacts_dir")
                if isinstance(run_id, str):
                    self.state.run_id = run_id
                if isinstance(artifacts_dir, str):
                    run_dir = Path(artifacts_dir)
                    if run_dir.exists() or run_dir.parent.exists():
                        self.state.run_dir = run_dir
                self.state.phase = "online"
                self.state.running = True
                self._emit_system_chat(f"Run started: {self.state.run_id or 'unknown'}")
                continue

            if bridge_evt.event_type == "run_completed":
                self.state.phase = "completed"
                self.state.running = False
                best = payload.get("best_time_ms")
                best_text = f" | best={best:.2f}ms" if isinstance(best, (int, float)) else ""
                self._emit_system_chat(f"Run completed{best_text}")
                continue

            if bridge_evt.event_type == "run_error":
                self.state.phase = "error"
                self.state.running = False
                self._emit_chat("System", f"Run error: {payload.get('error', 'unknown error')}", style="bold red")
                continue

            normalized = {
                "ts": payload.get("trace_ts") if isinstance(payload.get("trace_ts"), (int, float)) else time.time(),
                "phase": payload.get("trace_phase", self.state.phase),
                "step": bridge_evt.step,
                "actor": payload.get("trace_actor", "agent"),
                "type": bridge_evt.event_type,
                "payload": payload,
                "refs": payload.get("refs", []),
                "status": payload.get("trace_status") or ("error" if payload.get("error") else "ok"),
                "duration_ms": payload.get("duration_ms"),
                "error": payload.get("error"),
                "source": "bridge",
            }
            self._ingest_event(normalized)

            if bridge_evt.event_type == "chat_response":
                response = payload.get("response", "")
                ctx_tokens = payload.get("context_tokens")
                tokens = f" ({ctx_tokens} ctx tokens)" if isinstance(ctx_tokens, (int, float)) else ""
                self._emit_chat("Agent", f"{response}{tokens}", style="bold #7ddf8d")
                ctx_path = payload.get("chat_context_path")
                if isinstance(ctx_path, str):
                    snap = _json_load(Path(ctx_path))
                    if isinstance(snap, dict):
                        self.state.latest_chat_context = dict(snap, _path=ctx_path)

    def _poll_trace_disk(self) -> None:
        if self.state.run_dir is None:
            return
        path = self.state.run_dir / "trace" / "events.jsonl"
        self._trace_path = path
        if not path.exists():
            return

        try:
            stat = path.stat()
        except Exception:
            return

        size = stat.st_size
        mtime = stat.st_mtime

        # Handle truncation / rotate.
        if self._trace_offset > size:
            self._trace_offset = 0

        if self._trace_offset == 0 and size > 0:
            max_bytes = 2_500_000
            start = max(0, size - max_bytes)
            with open(path, "rb") as handle:
                handle.seek(start)
                raw = handle.read().decode("utf-8", errors="ignore")
            lines = raw.splitlines()
            if start > 0 and lines:
                lines = lines[1:]
            self._trace_offset = size
            self._trace_mtime = mtime
            for line in lines:
                event = self._parse_trace_line(line)
                if event:
                    self._ingest_event(event)
            return

        if mtime <= self._trace_mtime and size == self._trace_offset:
            return

        with open(path, "rb") as handle:
            handle.seek(self._trace_offset)
            raw = handle.read().decode("utf-8", errors="ignore")

        self._trace_offset = size
        self._trace_mtime = mtime

        for line in raw.splitlines():
            event = self._parse_trace_line(line)
            if event:
                self._ingest_event(event)

    def _parse_trace_line(self, line: str) -> Optional[Dict[str, Any]]:
        line = line.strip()
        if not line:
            return None
        try:
            raw = json.loads(line)
        except Exception:
            return None

        payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
        return {
            "ts": raw.get("ts", time.time()),
            "phase": raw.get("phase", "unknown"),
            "step": raw.get("step"),
            "actor": raw.get("actor", "agent"),
            "type": raw.get("type", "unknown"),
            "payload": payload,
            "refs": raw.get("refs", []),
            "status": raw.get("status", "ok"),
            "duration_ms": raw.get("duration_ms"),
            "error": raw.get("error"),
            "source": "trace",
        }

    def _ingest_event(self, event: Dict[str, Any]) -> None:
        key = self._event_key(event)
        if key in self._seen_event_keys:
            return
        self._seen_event_keys.add(key)

        self.state.events.append(event)
        if len(self.state.events) > 5000:
            self.state.events = self.state.events[-5000:]

        phase = event.get("phase")
        if isinstance(phase, str):
            self.state.phase = phase

        evt_type = str(event.get("type", ""))
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if evt_type == "chat_response" and event.get("source") == "trace":
            response = payload.get("response")
            if isinstance(response, str) and response.strip():
                self._emit_chat("Agent", response, style="bold #7ddf8d")
            ctx_path = payload.get("chat_context_path")
            if isinstance(ctx_path, str):
                snap = _json_load(Path(ctx_path))
                if isinstance(snap, dict):
                    self.state.latest_chat_context = dict(snap, _path=ctx_path)
        elif evt_type == "chat.context":
            ctx_path = payload.get("chat_context_path")
            if isinstance(ctx_path, str):
                snap = _json_load(Path(ctx_path))
                if isinstance(snap, dict):
                    self.state.latest_chat_context = dict(snap, _path=ctx_path)

        self._update_async_from_event(event)

    def _event_key(self, event: Dict[str, Any]) -> str:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        call_id = payload.get("call_id") or payload.get("tool_call_id") or ""
        ts = event.get("ts")
        ts_str = f"{float(ts):.6f}" if isinstance(ts, (int, float)) else "0"
        return "|".join(
            [
                ts_str,
                str(event.get("type", "")),
                str(event.get("step", "")),
                str(event.get("actor", "")),
                str(call_id),
            ]
        )

    def _update_async_from_event(self, event: Dict[str, Any]) -> None:
        evt_type = str(event.get("type", ""))
        step = event.get("step")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}

        if not isinstance(step, int):
            return

        status = self.state.async_advice.get(step)
        if status is None:
            status = AsyncAdviceStatus(step=step)
            self.state.async_advice[step] = status

        if evt_type == "llm.advice.async_dispatched":
            status.requested_ts = event.get("ts")
            status.state = "pending"
            return

        if evt_type == "llm.advice.async_ready":
            status.ready_ts = event.get("ts")
            status.call_id = payload.get("call_id") or status.call_id
            status.state = "ready"
            return

        if evt_type == "llm.advice":
            status.ready_ts = event.get("ts")
            status.call_id = payload.get("call_id") or status.call_id
            used = payload.get("used")
            if isinstance(used, bool):
                status.used_in_decision = used
            status.state = "used" if status.used_in_decision else "ready"

    def _poll_step_files(self) -> bool:
        if self.state.run_dir is None:
            return False
        steps_dir = self.state.run_dir / "steps"
        if not steps_dir.exists():
            if self.state.base_steps:
                self.state.base_steps = {}
                self._base_steps_signature = ()
                return True
            return False

        if not self._should_scan_dir("steps", steps_dir, stale_s=self.DIR_STALE_SCAN_S):
            return False

        base_steps: Dict[int, Dict[str, Any]] = {}
        signature_rows: List[tuple[int, int, int]] = []

        for entry in os.scandir(steps_dir):
            if not entry.is_file():
                continue
            name = entry.name
            if not (name.startswith("step_") and name.endswith(".json")):
                continue
            suffix = name[len("step_") : -len(".json")]
            if not suffix.isdigit():
                continue
            step_idx = int(suffix)
            path = Path(entry.path)
            data = self._read_json_cached(path)
            if isinstance(data, dict):
                base_steps[step_idx] = data
                try:
                    st = entry.stat()
                    signature_rows.append((step_idx, st.st_mtime_ns, st.st_size))
                except Exception:
                    pass

        signature_rows.sort(key=lambda item: item[0])
        new_sig = tuple(signature_rows)
        if new_sig == self._base_steps_signature:
            return False

        self.state.base_steps = base_steps
        self._base_steps_signature = new_sig
        return True

    def _poll_llm_calls(self) -> bool:
        if self.state.run_dir is None:
            return False
        llm_dir = self.state.run_dir / "llm"
        if not llm_dir.exists():
            if self.state.llm_calls:
                self.state.llm_calls = []
                self._llm_calls_signature = ()
                return True
            return False

        stale = 0.6 if self._llm_artifact_hot_path() else 2.0
        if not self._should_scan_dir("llm", llm_dir, stale_s=stale):
            return False

        calls: List[Dict[str, Any]] = []
        files = sorted(llm_dir.glob("call_*.json"), key=lambda p: p.name)
        file_sig: List[tuple[str, int, int]] = []
        for path in files:
            try:
                stat = path.stat()
            except Exception:
                continue
            file_sig.append((path.name, stat.st_mtime_ns, stat.st_size))

        new_sig = tuple(file_sig)
        if new_sig == self._llm_calls_signature:
            return False

        for path in files:
            data = self._read_json_cached(path)
            if isinstance(data, dict):
                rec = dict(data)
                rec["_path"] = str(path)
                calls.append(rec)

        self.state.llm_calls = calls
        self._llm_calls_signature = new_sig
        return True

    def _poll_chat_context_snapshots(self) -> None:
        if self.state.run_dir is None:
            return
        online_dir = self.state.run_dir / "online"
        if not online_dir.exists():
            return

        snapshots = list(online_dir.glob("chat_context_step_*.json"))
        if not snapshots:
            return

        latest_path = max(
            snapshots,
            key=lambda p: _extract_index_from_name(
                p.name,
                prefix="chat_context_step_",
                suffix=".json",
            ),
        )
        latest = self._read_json_cached(latest_path)
        if isinstance(latest, dict):
            self.state.latest_chat_context = dict(latest, _path=str(latest_path))

    def _poll_late_llm_advice_files(self) -> None:
        if self.state.run_dir is None:
            return
        online_dir = self.state.run_dir / "online"
        if not online_dir.exists():
            return

        for path in online_dir.glob("llm_advice_step_*.json"):
            key = str(path)
            try:
                mtime = path.stat().st_mtime
            except Exception:
                continue
            if self._advice_file_mtimes.get(key) == mtime:
                continue
            data = self._read_json_cached(path)
            if not isinstance(data, dict):
                continue
            self._advice_file_mtimes[key] = mtime
            step = data.get("step")
            if not isinstance(step, int):
                continue
            status = self.state.async_advice.get(step)
            if status is None:
                status = AsyncAdviceStatus(step=step)
                self.state.async_advice[step] = status
            status.call_id = data.get("call_id") or status.call_id
            if status.requested_ts is None:
                status.state = "ready"

    def _read_json_cached(self, path: Path) -> Optional[Any]:
        try:
            mtime = path.stat().st_mtime
        except Exception:
            return None

        key = str(path)
        cached = self._json_cache.get(key)
        if cached and cached[0] >= mtime:
            return cached[1]

        data = _json_load(path)
        self._json_cache[key] = (mtime, data)
        return data

    def _derive_iteration_series(self) -> None:
        times: List[float] = []
        native_baselines: List[float] = []
        for step in sorted(self.state.base_steps.keys()):
            rec = self.state.base_steps[step]
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics"), dict) else {}
            val = _safe_float(metrics.get("iteration_time_ms"))
            if val is not None:
                times.append(val)
            raw = metrics.get("raw", {}) if isinstance(metrics.get("raw"), dict) else {}
            native = _safe_float(raw.get("native_nccl_tuner_ms"))
            if native is not None:
                native_baselines.append(native)
        self.state.iteration_times_ms = times
        self.state.best_iteration_ms = min(times) if times else None
        self.state.native_baseline_ms = native_baselines[-1] if native_baselines else None

    def _apply_defaults(self) -> None:
        if self.state.selected_event_idx is None and self.state.events:
            self.state.selected_event_idx = max(0, len(self.state.events) - 1)

        if self.state.selected_step is None and self.state.base_steps:
            self.state.selected_step = max(self.state.base_steps.keys())

        if self.state.selected_llm_idx is None and self.state.llm_calls:
            self.state.selected_llm_idx = max(0, len(self.state.llm_calls) - 1)

        if self.state.selected_async_step is None and self.state.async_advice:
            self.state.selected_async_step = max(self.state.async_advice.keys())

    def _should_render(self, slot: str, signature: Any, *, min_interval_s: float = 0.0) -> bool:
        previous = self._render_signatures.get(slot)
        if previous == signature:
            return False
        if self._refresh_cycle_active and min_interval_s > 0.0:
            now = time.monotonic()
            last = self._render_timestamps.get(slot, 0.0)
            if now - last < min_interval_s:
                return False
        self._render_signatures[slot] = signature
        self._render_timestamps[slot] = time.monotonic()
        return True

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------

    def _render_cards(self) -> None:
        run_name = self.state.run_id or (self.state.run_dir.name if self.state.run_dir else "waiting")
        phase = self.state.phase.upper() if self.state.phase else "UNKNOWN"
        mode = "LIVE" if self.live_mode else "INSPECT"
        state_text = "RUNNING" if self.state.running else "STOPPED"

        if self.state.iteration_times_ms:
            baseline = self.state.iteration_times_ms[0]
            latest = self.state.iteration_times_ms[-1]
            best = self.state.best_iteration_ms or latest
            improve = (baseline - best) / baseline * 100.0 if baseline > 0 else 0.0
            native = self.state.native_baseline_ms
            native_gain_text = ""
            if isinstance(native, (int, float)) and native > 0:
                native_gain = (native - best) / native * 100.0
                native_gain_text = f" | vs native {native_gain:.2f}%"
            perf_text = (
                "Performance\n"
                f"latest {latest:.2f} ms | best {best:.2f} ms | gain {improve:.2f}%{native_gain_text}"
            )
        else:
            perf_text = "Performance\nno step metrics yet"

        pending = sum(1 for s in self.state.async_advice.values() if s.state == "pending")
        ready = sum(1 for s in self.state.async_advice.values() if s.state in ("ready", "used"))
        card_sig = (run_name, mode, phase, state_text, perf_text, pending, ready)
        if not self._should_render("cards", card_sig):
            return

        self.query_one("#card-run", Static).update(f"Run\n{run_name}")
        self.query_one("#card-phase", Static).update(f"Mode / Phase\n{mode} | {phase} | {state_text}")
        self.query_one("#card-perf", Static).update(perf_text)
        self.query_one("#card-async", Static).update(f"LLM Async\npending {pending} | ready {ready}")

        self.query_one("#card-goal", Static).update(
            "Objective\n"
            "minimize end-to-end distributed LLM training iteration_time_ms via safe NCCL tuning"
        )

    def _render_events_table(self) -> None:
        table = self.query_one("#events-table", DataTable)
        events = self.state.events[-self.EVENT_TABLE_MAX_ROWS :]
        last_key = self._event_key(events[-1]) if events else ""
        active_tab = self.query_one("#left-tabs", TabbedContent).active
        sig = (len(events), last_key, self.state.selected_event_idx, active_tab)
        if not self._should_render("table_events", sig, min_interval_s=0.15):
            return
        self._capture_left_table_state(table)
        table.clear(columns=False)
        self._event_row_to_idx.clear()

        base_idx = max(0, len(self.state.events) - len(events))
        for offset, event in enumerate(events):
            idx = base_idx + offset
            row_key = str(idx)
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            status = event.get("status", "ok")
            err = payload.get("error") or event.get("error")
            status_cell = "ERR" if err else str(status).upper()
            duration = event.get("duration_ms")
            dur_cell = f"{duration:.1f}" if isinstance(duration, (int, float)) else "-"
            table.add_row(
                _fmt_ts(event.get("ts")),
                str(event.get("phase", "?"))[:8],
                str(event.get("step", "-")),
                str(event.get("actor", "?"))[:10],
                str(event.get("type", "?"))[:48],
                status_cell,
                dur_cell,
                key=row_key,
            )
            self._event_row_to_idx[row_key] = idx
        self._restore_left_table_state(table)

    def _render_steps_table(self) -> None:
        table = self.query_one("#steps-table", DataTable)
        steps = sorted(self.state.base_steps.keys())
        step_sig = []
        for step in steps:
            rec = self.state.base_steps.get(step, {})
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics"), dict) else {}
            action = rec.get("action", {}) if isinstance(rec.get("action"), dict) else {}
            step_sig.append((step, metrics.get("iteration_time_ms"), metrics.get("success"), action.get("kind")))
        active_tab = self.query_one("#left-tabs", TabbedContent).active
        sig = (tuple(step_sig), self.state.selected_step, active_tab)
        if not self._should_render("table_steps", sig, min_interval_s=0.25):
            return
        self._capture_left_table_state(table)
        table.clear(columns=False)
        self._step_row_to_step.clear()

        if not steps:
            self._restore_left_table_state(table)
            return

        baseline = None
        first = self.state.base_steps.get(steps[0], {})
        first_metrics = first.get("metrics", {}) if isinstance(first.get("metrics"), dict) else {}
        baseline = _safe_float(first_metrics.get("iteration_time_ms"))

        for step in steps:
            rec = self.state.base_steps.get(step, {})
            action = rec.get("action", {}) if isinstance(rec.get("action"), dict) else {}
            action_kind = action.get("kind", "?")
            if self.state.run_dir is not None:
                decision = self._read_json_cached(
                    self.state.run_dir / "steps" / f"step_{step}_decision.json"
                )
                if isinstance(decision, dict) and decision.get("action"):
                    action_kind = decision.get("action")
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics"), dict) else {}
            iter_ms = _safe_float(metrics.get("iteration_time_ms"))
            iter_cell = f"{iter_ms:.2f}" if iter_ms is not None else "-"
            delta_cell = "-"
            if baseline and iter_ms is not None:
                delta_cell = f"{(baseline - iter_ms) / baseline * 100.0:+.2f}"
            ok_cell = "yes" if metrics.get("success") else "no"

            bottleneck = self._read_json_cached(self.state.run_dir / "steps" / f"step_{step}_bottleneck.json")
            bottleneck_cls = (
                bottleneck.get("class", "-") if isinstance(bottleneck, dict) else "-"
            )

            row_key = str(step)
            table.add_row(
                str(step),
                str(action_kind),
                iter_cell,
                delta_cell,
                ok_cell,
                str(bottleneck_cls),
                key=row_key,
            )
            self._step_row_to_step[row_key] = step
        self._restore_left_table_state(table)

    def _render_llm_table(self) -> None:
        table = self.query_one("#llm-table", DataTable)
        calls = self.state.llm_calls[-400:]
        last_call_id = calls[-1].get("call_id") if calls else ""
        last_duration = calls[-1].get("duration_ms") if calls else None
        active_tab = self.query_one("#left-tabs", TabbedContent).active
        sig = (len(calls), last_call_id, last_duration, self.state.selected_llm_idx, active_tab)
        if not self._should_render("table_llm", sig, min_interval_s=0.25):
            return
        self._capture_left_table_state(table)
        table.clear(columns=False)
        self._llm_row_to_idx.clear()

        for idx, call in enumerate(calls):
            global_idx = len(self.state.llm_calls) - len(calls) + idx
            row_key = str(global_idx)

            trace = call.get("trace", {}) if isinstance(call.get("trace"), dict) else {}
            step = trace.get("step", "-")
            phase = trace.get("phase", "?")
            model = call.get("model", "?")
            ctx = call.get("context_window", {}) if isinstance(call.get("context_window"), dict) else {}
            ctx_tok = ctx.get("total_tokens", "-")
            dur = call.get("duration_ms")
            dur_cell = f"{dur:.1f}" if isinstance(dur, (int, float)) else "-"
            call_short = str(call.get("call_id", "?"))[:10]

            table.add_row(
                str(step),
                str(phase),
                str(model)[:20],
                str(ctx_tok),
                dur_cell,
                call_short,
                key=row_key,
            )
            self._llm_row_to_idx[row_key] = global_idx
        self._restore_left_table_state(table)

    def _render_async_table(self) -> None:
        table = self.query_one("#async-table", DataTable)
        async_sig = tuple(
            (
                step,
                st.state,
                st.requested_ts,
                st.ready_ts,
                st.used_in_decision,
                st.call_id,
            )
            for step, st in sorted(self.state.async_advice.items())
        )
        active_tab = self.query_one("#left-tabs", TabbedContent).active
        sig = (async_sig, self.state.selected_async_step, active_tab)
        if not self._should_render("table_async", sig, min_interval_s=0.25):
            return
        self._capture_left_table_state(table)
        table.clear(columns=False)
        self._async_row_to_step.clear()

        for step in sorted(self.state.async_advice.keys()):
            st = self.state.async_advice[step]
            req = _fmt_ts(st.requested_ts)
            ready = _fmt_ts(st.ready_ts)
            lat = "-"
            if isinstance(st.requested_ts, (int, float)) and isinstance(st.ready_ts, (int, float)):
                lat = f"{(st.ready_ts - st.requested_ts) * 1000.0:.1f}"
            used = "yes" if st.used_in_decision else ("no" if st.used_in_decision is False else "?")
            call_short = (st.call_id or "")[:10] if st.call_id else "-"
            row_key = str(step)
            table.add_row(str(step), st.state, req, ready, lat, used, call_short, key=row_key)
            self._async_row_to_step[row_key] = step
        self._restore_left_table_state(table)

    def _render_overview(self) -> None:
        events_tail = self.state.events[-500:]
        last_event = self._event_key(events_tail[-1]) if events_tail else ""
        sig = (
            self.state.run_id,
            self.state.phase,
            len(self.state.base_steps),
            len(self.state.llm_calls),
            len(self.state.async_advice),
            self.state.best_iteration_ms,
            last_event,
        )
        if not self._should_render("right_overview", sig, min_interval_s=0.7):
            return

        log = self.query_one("#overview-log", RichLog)
        log.clear()

        log.write(Text("Command Center Overview", style="bold #63d8ff"))
        log.write(
            "Step = choose NCCL config -> run workload -> evaluate metrics -> update strategy."
        )
        log.write(
            "Iteration = inner training loop measurements within each step execution."
        )
        log.write("")

        summary = {
            "run_id": self.state.run_id,
            "phase": self.state.phase,
            "events": len(self.state.events),
            "steps": len(self.state.base_steps),
            "llm_calls": len(self.state.llm_calls),
            "async_entries": len(self.state.async_advice),
            "best_iteration_ms": self.state.best_iteration_ms,
        }
        log.write(Text("Run Summary", style="bold #7ddf8d"))
        log.write(Pretty(summary))

        if self.state.iteration_times_ms:
            baseline = self.state.iteration_times_ms[0]
            best = self.state.best_iteration_ms or baseline
            latest = self.state.iteration_times_ms[-1]
            log.write("")
            log.write(Text("Optimization Progress", style="bold #ffc66d"))
            log.write(
                f"Baseline: {baseline:.2f} ms | Best: {best:.2f} ms | Latest: {latest:.2f} ms | "
                f"Gain: {(baseline - best) / baseline * 100.0:.2f}%"
            )
            native = self.state.native_baseline_ms
            if isinstance(native, (int, float)) and native > 0:
                gain_native = (native - best) / native * 100.0
                status = "met" if gain_native >= 30.0 else "not met"
                log.write(
                    f"Vs native NCCL tuner baseline: {gain_native:.2f}% ({status}, target >= 30.00%)"
                )

        event_counts: Dict[str, int] = {}
        for event in events_tail:
            evt_type = str(event.get("type", "unknown"))
            event_counts[evt_type] = event_counts.get(evt_type, 0) + 1

        if event_counts:
            table = Table(show_header=True, header_style="bold #63d8ff")
            table.add_column("Event Type")
            table.add_column("Count", justify="right")
            for evt_type, count in sorted(event_counts.items(), key=lambda item: item[1], reverse=True)[:15]:
                table.add_row(evt_type, str(count))
            log.write("")
            log.write(Text("Recent Event Mix", style="bold #63d8ff"))
            log.write(table)

        llm_summary = self._build_llm_impact_summary()
        if llm_summary:
            log.write("")
            log.write(Text("LLM Impact Snapshot", style="bold #ffc66d"))
            for line in llm_summary:
                log.write(f"- {line}")

    def _render_event_detail(self) -> None:
        event = self._selected_event()
        sig = (self.state.selected_event_idx, self._event_key(event) if event else "none")
        if not self._should_render("right_event", sig, min_interval_s=0.25):
            return

        log = self.query_one("#event-detail-log", RichLog)
        log.clear()
        if not event:
            log.write("No event selected.")
            return

        payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}
        log.write(Text("Event Inspector", style="bold #63d8ff"))
        log.write(
            Pretty(
                {
                    "time": _fmt_ts(event.get("ts")),
                    "phase": event.get("phase"),
                    "step": event.get("step"),
                    "actor": event.get("actor"),
                    "type": event.get("type"),
                    "status": event.get("status"),
                    "duration_ms": event.get("duration_ms"),
                    "refs": event.get("refs", []),
                }
            )
        )

        evt_type = str(event.get("type", ""))
        log.write("")
        log.write(Text("Highlights", style="bold #7ddf8d"))
        if evt_type == "tool.call":
            log.write(
                f"Tool invoked: `{payload.get('tool', '?')}.{payload.get('method', '?')}` | "
                f"call_id={payload.get('call_id', '-')}"
            )
        elif evt_type == "tool.result":
            result = payload.get("result", {})
            log.write(
                f"Tool result for `{payload.get('tool', '?')}.{payload.get('method', '?')}` | "
                f"status={'error' if payload.get('error') else 'ok'}"
            )
            if isinstance(result, dict) and "iteration_time_ms" in result:
                log.write(f"Returned iteration_time_ms={result.get('iteration_time_ms')}")
        elif evt_type == "llm.call":
            log.write(
                f"LLM call: model={payload.get('model')} | call_id={payload.get('call_id')} | "
                f"path={payload.get('call_path')}"
            )
        elif evt_type in ("decision.select_action", "stop.decision"):
            log.write(f"Decision event: {evt_type}")
            if payload:
                log.write(_truncate_text(payload, max_len=800))
        elif evt_type.startswith("chat"):
            log.write("Chat interaction captured with context-engineering trace.")
        else:
            log.write(f"Event type `{evt_type}` captured in trace.")

        if evt_type.startswith("tool."):
            self._render_tool_event_detail(log, payload, evt_type)

        if evt_type == "llm.call":
            self._render_llm_event_link(log, payload)

        if evt_type.startswith("decision") or "decision" in evt_type:
            log.write("")
            log.write(Text("Decision Payload", style="bold #ffc66d"))
            log.write(Pretty(_compact(payload)))

        if evt_type == "chat_response":
            log.write("")
            log.write(Text("Chat Response", style="bold #7ddf8d"))
            log.write(_truncate_text(payload.get("response", ""), max_len=3000))
            if payload.get("chat_context_path"):
                log.write(f"context snapshot: {payload.get('chat_context_path')}")

        step = event.get("step")
        if isinstance(step, int):
            bundle = self._load_step_bundle(step)
            context_pack = bundle.get("context_pack")
            if isinstance(context_pack, dict):
                log.write("")
                log.write(Text("Step Context Snapshot", style="bold #9ab6ff"))
                retrieval = context_pack.get("retrieval", {}) if isinstance(context_pack.get("retrieval"), dict) else {}
                obs = context_pack.get("observations", {}) if isinstance(context_pack.get("observations"), dict) else {}
                compact_view = {
                    "step": context_pack.get("step"),
                    "phase": context_pack.get("phase"),
                    "observation_keys": list(obs.keys()),
                    "memory_rules": len(retrieval.get("memory_rules", []))
                    if isinstance(retrieval.get("memory_rules"), list)
                    else 0,
                    "rag_chunks": len(retrieval.get("rag_chunks", []))
                    if isinstance(retrieval.get("rag_chunks"), list)
                    else 0,
                }
                log.write(Pretty(compact_view))

    def _render_tool_event_detail(self, log: RichLog, payload: Dict[str, Any], evt_type: str) -> None:
        call_id = payload.get("call_id")
        log.write("")
        log.write(Text("Tool Call / Result", style="bold #7ddf8d"))

        detail = {
            "event_type": evt_type,
            "tool": payload.get("tool"),
            "method": payload.get("method"),
            "call_id": call_id,
            "args": _compact(payload.get("args")),
            "kwargs": _compact(payload.get("kwargs")),
            "result": _compact(payload.get("result")),
            "error": payload.get("error"),
        }
        log.write(Pretty(detail))

        if call_id:
            paired = self._find_paired_tool_event(call_id=call_id, current_type=evt_type)
            if paired:
                log.write("")
                log.write(Text("Paired Tool Event", style="bold #ffc66d"))
                log.write(
                    Pretty(
                        {
                            "type": paired.get("type"),
                            "time": _fmt_ts(paired.get("ts")),
                            "payload": _compact(paired.get("payload")),
                        }
                    )
                )

    def _render_llm_event_link(self, log: RichLog, payload: Dict[str, Any]) -> None:
        call_id = payload.get("call_id")
        log.write("")
        log.write(Text("LLM Artifact Link", style="bold #7ddf8d"))
        log.write(Pretty({"call_id": call_id, "model": payload.get("model"), "call_path": payload.get("call_path")}))

        if not call_id:
            return

        call = self._find_llm_call(call_id)
        if not call:
            return

        log.write("")
        log.write(Text("LLM Call Snapshot", style="bold #63d8ff"))
        response = call.get("response", {}) if isinstance(call.get("response"), dict) else {}
        ctx = call.get("context_window", {}) if isinstance(call.get("context_window"), dict) else {}
        log.write(
            Pretty(
                {
                    "trace": call.get("trace"),
                    "token_estimates": call.get("token_estimates"),
                    "context_window": _compact(ctx),
                    "response_preview": _truncate_text(response.get("content", ""), max_len=700),
                }
            )
        )

    def _render_llm_detail(self) -> None:
        call = self._selected_llm_call()
        call_sig = (
            self.state.selected_llm_idx,
            call.get("call_id") if isinstance(call, dict) else None,
            call.get("duration_ms") if isinstance(call, dict) else None,
        )
        if not self._should_render("right_llm", call_sig, min_interval_s=0.5):
            return

        log = self.query_one("#llm-detail-log", RichLog)
        log.clear()
        if not call:
            log.write("No LLM call selected.")
            return

        trace = call.get("trace", {}) if isinstance(call.get("trace"), dict) else {}
        response = call.get("response", {}) if isinstance(call.get("response"), dict) else {}
        messages = call.get("messages", []) if isinstance(call.get("messages"), list) else []
        system_msg = next(
            (
                msg.get("content", "")
                for msg in messages
                if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "system"
            ),
            "",
        )
        user_msg = next(
            (
                msg.get("content", "")
                for msg in messages
                if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user"
            ),
            "",
        )

        log.write(Text("LLM Intelligence Panel", style="bold #63d8ff"))
        summary = Table(show_header=True, header_style="bold #63d8ff")
        summary.add_column("Field", style="bold #9ab6ff")
        summary.add_column("Value")
        summary.add_row("Call", str(call.get("call_id", "-")))
        summary.add_row("Model", str(call.get("model", "-")))
        summary.add_row("Phase/Step", f"{trace.get('phase', '-')} / {trace.get('step', '-')}")
        summary.add_row("Prompt Version", str(call.get("system_prompt_version", "-")))
        summary.add_row("Duration", f"{call.get('duration_ms', '-')}")
        tok = call.get("token_estimates", {}) if isinstance(call.get("token_estimates"), dict) else {}
        summary.add_row(
            "Token Est.",
            f"prompt={tok.get('prompt_tokens_est', '-')} | response={tok.get('response_tokens_est', '-')}",
        )
        log.write(summary)

        ctx = call.get("context_window", {}) if isinstance(call.get("context_window"), dict) else {}
        sections = ctx.get("sections", []) if isinstance(ctx.get("sections"), list) else []
        if sections:
            log.write("")
            log.write(Text("Context Budget Allocation", style="bold #7ddf8d"))
            sec_tbl = Table(show_header=True, header_style="bold #7ddf8d")
            sec_tbl.add_column("Section")
            sec_tbl.add_column("Tokens", justify="right")
            sec_tbl.add_column("Truncated")
            for item in sections[:14]:
                if not isinstance(item, dict):
                    continue
                sec_tbl.add_row(
                    str(item.get("name", "?")),
                    str(item.get("tokens_after", "-")),
                    "yes" if item.get("truncated") else "no",
                )
            log.write(sec_tbl)

        parsed = _maybe_parse_json_text(response.get("content", "") if isinstance(response, dict) else "")
        insights = self._summarize_llm_output(parsed) if isinstance(parsed, dict) else []
        if insights:
            log.write("")
            log.write(Text("Response Highlights", style="bold #ffc66d"))
            for line in insights:
                log.write(f"- {line}")

        log.write("")
        log.write(Text("Prompt Intent", style="bold #9ab6ff"))
        intent_line = _truncate_text(system_msg.splitlines()[0] if isinstance(system_msg, str) and system_msg else "", max_len=300)
        if intent_line:
            log.write(intent_line)
        else:
            log.write("No system prompt captured.")

        log.write("")
        log.write(Text("Prompt Preview", style="bold #7ddf8d"))
        if system_msg:
            log.write(Text("System", style="bold #9ab6ff"))
            log.write(Syntax(_truncate_text(system_msg, max_len=1800), "markdown", word_wrap=True))
        if user_msg:
            log.write("")
            log.write(Text("User", style="bold #9ab6ff"))
            log.write(Syntax(_truncate_text(user_msg, max_len=2400), "markdown", word_wrap=True))

        log.write("")
        log.write(Text("Response Preview", style="bold #ffc66d"))
        content = response.get("content", "") if isinstance(response, dict) else str(response)
        if isinstance(parsed, dict):
            log.write(Syntax(_truncate_text(json.dumps(_compact(parsed, max_depth=5, max_items=40), indent=2), max_len=3500), "json", word_wrap=True))
        else:
            log.write(Syntax(_truncate_text(content, max_len=3500), "markdown", word_wrap=True))

    def _render_reasoning_detail(self) -> None:
        step = self.state.selected_step
        sig = (step, self._step_artifact_signature(step) if isinstance(step, int) else None)
        if not self._should_render("right_reasoning", sig, min_interval_s=0.5):
            return

        log = self.query_one("#reasoning-log", RichLog)
        log.clear()
        if step is None:
            log.write("No step selected.")
            return

        bundle = self._load_step_bundle(step)
        base = self.state.base_steps.get(step, {})

        log.write(Text(f"Online Strategy - Step {step}", style="bold #63d8ff"))
        log.write(
            "Definition alignment: one step is one online decision/evaluation loop; each run contains many training iterations."
        )

        metrics = base.get("metrics", {}) if isinstance(base.get("metrics"), dict) else {}
        action = base.get("action", {}) if isinstance(base.get("action"), dict) else {}
        baseline = self.state.iteration_times_ms[0] if self.state.iteration_times_ms else None
        iter_ms = _safe_float(metrics.get("iteration_time_ms"))
        delta = None
        if isinstance(baseline, (int, float)) and baseline > 0 and iter_ms is not None:
            delta = (baseline - iter_ms) / baseline * 100.0

        log.write("")
        log.write(Text("Step Snapshot", style="bold #7ddf8d"))
        snap = Table(show_header=True, header_style="bold #7ddf8d")
        snap.add_column("Metric")
        snap.add_column("Value")
        snap.add_row("Action Kind", str(action.get("kind", "-")))
        snap.add_row("Rationale", _truncate_text(action.get("rationale", "-"), max_len=180))
        snap.add_row("Iteration (ms)", f"{iter_ms:.2f}" if iter_ms is not None else "-")
        snap.add_row("Delta vs Baseline", f"{delta:+.2f}%" if delta is not None else "-")
        snap.add_row("Success", "yes" if metrics.get("success") else "no")
        if metrics.get("failure_reason"):
            snap.add_row("Failure Reason", _truncate_text(metrics.get("failure_reason"), max_len=160))
        log.write(snap)

        log.write("")
        log.write(Text("Hypothesis Plan", style="bold #63d8ff"))
        hyp = bundle.get("hypothesis")
        hyp_series = bundle.get("hypothesis_series")
        if isinstance(hyp, dict):
            log.write(f"- hypothesis id: {hyp.get('id', '-')}")
            log.write(f"- summary: {_truncate_text(hyp.get('summary', ''), max_len=220)}")
            patch = hyp.get("patch", {}) if isinstance(hyp.get("patch"), dict) else {}
            if patch:
                patch_table = Table(show_header=True, header_style="bold #9ab6ff")
                patch_table.add_column("Param")
                patch_table.add_column("Value")
                for key, value in list(patch.items())[:8]:
                    patch_table.add_row(str(key), str(value))
                log.write(patch_table)
        if isinstance(hyp_series, dict):
            chosen_id = hyp_series.get("chosen_id")
            series = hyp_series.get("series", []) if isinstance(hyp_series.get("series"), list) else []
            if chosen_id:
                log.write(f"- chosen from hypothesis series: {chosen_id}")
            if series:
                log.write(f"- hypothesis series candidates: {len(series)}")
                series_tbl = Table(show_header=True, header_style="bold #7ddf8d")
                series_tbl.add_column("Rank", justify="right")
                series_tbl.add_column("ID")
                series_tbl.add_column("Pred(ms)", justify="right")
                series_tbl.add_column("Summary")
                for item in series[:5]:
                    if not isinstance(item, dict):
                        continue
                    pred = item.get("predicted_time_ms")
                    pred_text = f"{pred:.2f}" if isinstance(pred, (int, float)) else "-"
                    series_tbl.add_row(
                        str(item.get("rank", "-")),
                        str(item.get("id", "-")),
                        pred_text,
                        _truncate_text(item.get("summary", "-"), max_len=120),
                    )
                log.write(series_tbl)
        hyp_ranked = bundle.get("hypothesis_ranked")
        if isinstance(hyp_ranked, list):
            log.write(f"- hypotheses ranked: {len(hyp_ranked)}")

        log.write("")
        log.write(Text("Numeric Candidate Pruning", style="bold #ffc66d"))
        candidates_trace = bundle.get("candidates_trace")
        pruning_summary = bundle.get("pruning_summary")
        offline = self._load_offline_bundle()
        offline_pruning = offline.get("pruning_guidance")
        if isinstance(offline_pruning, list) and offline_pruning:
            log.write(f"- offline pruning guidance rows: {len(offline_pruning)}")
        if candidates_trace is not None:
            if isinstance(candidates_trace, dict):
                entries = candidates_trace.get("candidates", [])
                count = len(entries) if isinstance(entries, list) else "?"
            else:
                count = len(candidates_trace) if isinstance(candidates_trace, list) else "?"
            log.write(f"- candidates_trace entries: {count}")
        if pruning_summary is not None:
            if isinstance(pruning_summary, dict):
                dropped = pruning_summary.get("dropped", {}) if isinstance(pruning_summary.get("dropped"), dict) else {}
                if dropped:
                    top = sorted(dropped.items(), key=lambda item: item[1], reverse=True)[:4]
                    reason_text = ", ".join(f"{k}:{v}" for k, v in top)
                    log.write(f"- dropped candidates by reason: {reason_text}")
                else:
                    log.write("- pruning summary present, but no dropped candidates recorded")
            else:
                log.write(f"- pruning summary available ({type(pruning_summary).__name__})")

        llm_support = bundle.get("llm_decision_support")
        llm_conv = bundle.get("llm_convergence")
        log.write("")
        log.write(Text("LLM Decision Support", style="bold #9ab6ff"))
        if isinstance(llm_support, dict):
            output = llm_support.get("output", {}) if isinstance(llm_support.get("output"), dict) else {}
            conv = output.get("convergence", {}) if isinstance(output.get("convergence"), dict) else {}
            log.write(
                f"- advice used: {llm_support.get('used_in_decision')} | call: {str(llm_support.get('call_id', '-'))[:12]}"
            )
            if conv:
                log.write(
                    f"- convergence: {conv.get('decision', '?')} "
                    f"(confidence={conv.get('confidence', '-')})"
                )
            pref = output.get("action_preference")
            if pref:
                log.write(f"- action preference: {pref}")
            hyps = output.get("hypotheses")
            if isinstance(hyps, list):
                log.write(f"- hypotheses proposed by llm: {len(hyps)}")
            tool_req = output.get("tool_request", {}) if isinstance(output.get("tool_request"), dict) else {}
            if tool_req:
                log.write(f"- tool request: {tool_req.get('name', 'none')}")
        if isinstance(llm_conv, dict):
            output = llm_conv.get("output", {}) if isinstance(llm_conv.get("output"), dict) else {}
            conv = output.get("convergence", {}) if isinstance(output.get("convergence"), dict) else {}
            if conv:
                log.write(
                    f"- explicit convergence call: {conv.get('decision', '?')} "
                    f"(confidence={conv.get('confidence', '-')}) | used={llm_conv.get('used_in_decision')}"
                )

        log.write("")
        log.write(Text("Decision Rationale", style="bold #7ddf8d"))
        decision = bundle.get("decision", {})
        decision_bundle = bundle.get("decision_bundle", {})
        decision_record = decision_bundle if isinstance(decision_bundle, dict) and decision_bundle else bundle.get("decision_record", {})
        if isinstance(decision, dict):
            log.write(f"- action selected: {decision.get('action', action.get('kind', '?'))}")
            if decision.get("reason"):
                log.write(f"- reason: {_truncate_text(decision.get('reason'), max_len=200)}")
        if isinstance(decision_record, dict):
            chosen_action = decision_record.get("chosen_action", {}) if isinstance(decision_record.get("chosen_action"), dict) else {}
            if chosen_action:
                log.write(f"- chosen kind: {chosen_action.get('kind', '-')}")
                if chosen_action.get("rationale"):
                    log.write(f"- chosen rationale: {_truncate_text(chosen_action.get('rationale'), max_len=220)}")
            why = decision_record.get("why_selected", [])
            if isinstance(why, list) and why:
                first = why[0].get("claim") if isinstance(why[0], dict) else str(why[0])
                log.write(f"- why selected: {_truncate_text(first, max_len=220)}")
            why_not = decision_record.get("why_rejected", [])
            if isinstance(why_not, list) and why_not:
                first_not = why_not[0].get("claim") if isinstance(why_not[0], dict) else str(why_not[0])
                log.write(f"- why not others: {_truncate_text(first_not, max_len=220)}")
            if isinstance(decision_bundle, dict):
                counterfactuals = decision_bundle.get("counterfactuals", [])
                if isinstance(counterfactuals, list) and counterfactuals:
                    log.write(f"- counterfactuals tracked: {len(counterfactuals)}")
                quality_flags = decision_bundle.get("quality_flags", [])
                if isinstance(quality_flags, list) and quality_flags:
                    log.write(f"- quality flags: {', '.join(str(x) for x in quality_flags[:4])}")
        stop_decision = bundle.get("stop_decision")
        rollback = bundle.get("rollback_decision")
        if isinstance(stop_decision, dict):
            log.write(f"- stop decision: {stop_decision.get('reason', '-')}")
        if isinstance(rollback, dict):
            log.write(f"- rollback decision: {rollback.get('reason', '-')}")

        log.write("")
        log.write(Text("Risk / Bottleneck / Derived Metrics", style="bold #63d8ff"))
        risk = bundle.get("risk_report")
        bottle = bundle.get("bottleneck")
        derived = bundle.get("metrics_derived")
        if isinstance(risk, dict):
            log.write(f"- risk score: {risk.get('risk_score', '-')}")
        if isinstance(bottle, dict):
            log.write(f"- bottleneck: {bottle.get('class', '-')} (conf={bottle.get('confidence', '-')})")
        if isinstance(derived, dict):
            vals = derived.get("derived", {}) if isinstance(derived.get("derived"), dict) else derived
            top_keys = list(vals.keys())[:8] if isinstance(vals, dict) else []
            if top_keys:
                log.write(f"- derived metrics keys: {', '.join(top_keys)}")

    def _render_context_detail(self) -> None:
        step = self.state.selected_step
        selected_call = self._selected_llm_call()
        latest_ctx = self.state.latest_chat_context if isinstance(self.state.latest_chat_context, dict) else {}
        sig = (
            step,
            self._step_artifact_signature(step) if isinstance(step, int) else None,
            selected_call.get("call_id") if isinstance(selected_call, dict) else None,
            latest_ctx.get("_path"),
            latest_ctx.get("step"),
        )
        if not self._should_render("right_context", sig, min_interval_s=0.6):
            return

        log = self.query_one("#context-log", RichLog)
        log.clear()

        log.write(Text("Context Engineering Inspector", style="bold #63d8ff"))
        log.write(
            "Shows exactly what context is packed for online decisions and chat responses, including token budgets and section truncation."
        )

        # Selected step context pack.
        if isinstance(step, int):
            bundle = self._load_step_bundle(step)
            context_pack = bundle.get("context_pack")
            if context_pack is not None:
                log.write("")
                log.write(Text(f"Step {step} Context Pack", style="bold #7ddf8d"))
                log.write(Pretty(_compact(context_pack, max_depth=6, max_items=50)))

        # Selected LLM call context window + prompt structure.
        call = selected_call
        if call is not None:
            log.write("")
            log.write(Text("Selected LLM Call Context Window", style="bold #ffc66d"))
            log.write(Pretty(_compact(call.get("context_window", {}), max_depth=6, max_items=50)))

        # Latest chat context snapshot.
        if self.state.latest_chat_context is not None:
            snap = self.state.latest_chat_context
            log.write("")
            log.write(Text("Latest Chat Context Snapshot", style="bold #7ddf8d"))
            snapshot_view = {
                "path": snap.get("_path"),
                "step": snap.get("step"),
                "question": snap.get("question"),
                "context_window": _compact(snap.get("context_window", {}), max_depth=6, max_items=50),
                "sections": _compact(snap.get("sections", []), max_depth=5, max_items=50),
                "step_vs_iteration": snap.get("step_vs_iteration"),
            }
            log.write(Pretty(snapshot_view))

            system_prompt = snap.get("system_prompt")
            assembled = snap.get("assembled_user_prompt")
            if isinstance(system_prompt, str):
                log.write("")
                log.write(Text("Chat System Prompt", style="bold #9ab6ff"))
                log.write(Syntax(_truncate_text(system_prompt, max_len=6000), "markdown", word_wrap=True))
            if isinstance(assembled, str):
                log.write("")
                log.write(Text("Chat Assembled User Prompt", style="bold #9ab6ff"))
                log.write(Syntax(_truncate_text(assembled, max_len=10000), "markdown", word_wrap=True))
        else:
            log.write("")
            log.write("No chat context snapshot yet. Ask a question in live mode to generate one.")

    def _render_pruning_lens(self) -> None:
        step = self.state.selected_step
        sig = (
            step,
            self._step_artifact_signature(step) if isinstance(step, int) else None,
            self._base_steps_signature,
        )
        if not self._should_render("right_pruning", sig, min_interval_s=0.8):
            return

        log = self.query_one("#pruning-log", RichLog)
        log.clear()
        log.write(Text("Pruning Lens (Offline + Online)", style="bold #63d8ff"))
        log.write(
            "Shows pruning rationale and dropped-candidate reasons across the full loop. "
            "All rows are rendered as reasoning summaries, not raw JSON."
        )

        offline = self._load_offline_bundle()
        search_pruning = offline.get("search_space_pruning")
        pruning_guidance = offline.get("pruning_guidance")

        log.write("")
        log.write(Text("Offline Pruning", style="bold #7ddf8d"))
        offline_rows: List[tuple[str, str, str, str]] = []
        if isinstance(search_pruning, list):
            for item in search_pruning[:80]:
                if not isinstance(item, dict):
                    continue
                offline_rows.append(
                    (
                        str(item.get("param", "-")),
                        "offline_reasoner",
                        str(item.get("action", "fix_default")),
                        _truncate_text(item.get("reason", "low_importance"), max_len=90),
                    )
                )
        if isinstance(pruning_guidance, list):
            for item in pruning_guidance[:80]:
                if not isinstance(item, dict):
                    continue
                offline_rows.append(
                    (
                        str(item.get("param", "-")),
                        "llm_guidance",
                        str(item.get("action", "freeze_default")),
                        _truncate_text(item.get("reason", "not provided"), max_len=90),
                    )
                )
        if offline_rows:
            tbl = Table(show_header=True, header_style="bold #7ddf8d")
            tbl.add_column("Param")
            tbl.add_column("Source")
            tbl.add_column("Action")
            tbl.add_column("Reason")
            for row in offline_rows[:24]:
                tbl.add_row(*row)
            log.write(tbl)
            log.write(f"offline pruning rows: {len(offline_rows)}")
        else:
            log.write("No offline pruning artifacts found yet.")

        if isinstance(step, int):
            bundle = self._load_step_bundle(step)
            trace_obj = bundle.get("candidates_trace")
            trace_rows = []
            if isinstance(trace_obj, dict):
                trace_rows = trace_obj.get("candidates", []) if isinstance(trace_obj.get("candidates"), list) else []
            elif isinstance(trace_obj, list):
                trace_rows = trace_obj

            stage_drops: Dict[str, int] = {}
            reason_drops: Dict[str, int] = {}
            selected_ids: List[str] = []
            for cand in trace_rows:
                if not isinstance(cand, dict):
                    continue
                cid = str(cand.get("candidate_id", "?"))
                stages = cand.get("stages", {}) if isinstance(cand.get("stages"), dict) else {}
                for stage_name, info in stages.items():
                    if not isinstance(info, dict):
                        continue
                    if info.get("status") == "kept" and stage_name == "selected":
                        selected_ids.append(cid)
                    if info.get("status") == "dropped":
                        stage_drops[stage_name] = stage_drops.get(stage_name, 0) + 1
                        reason = str(info.get("reason", stage_name))
                        reason_drops[reason] = reason_drops.get(reason, 0) + 1

            log.write("")
            log.write(Text(f"Online Pruning (Step {step})", style="bold #ffc66d"))
            if trace_rows:
                log.write(f"candidate lifecycle entries: {len(trace_rows)}")
                if selected_ids:
                    log.write(f"selected candidate ids: {', '.join(selected_ids[:3])}")
                stage_tbl = Table(show_header=True, header_style="bold #ffc66d")
                stage_tbl.add_column("Stage")
                stage_tbl.add_column("Dropped", justify="right")
                for stage_name, count in sorted(stage_drops.items(), key=lambda item: item[1], reverse=True):
                    stage_tbl.add_row(stage_name, str(count))
                if stage_drops:
                    log.write(stage_tbl)

                reason_tbl = Table(show_header=True, header_style="bold #ffc66d")
                reason_tbl.add_column("Drop Reason")
                reason_tbl.add_column("Count", justify="right")
                for reason, count in sorted(reason_drops.items(), key=lambda item: item[1], reverse=True)[:8]:
                    reason_tbl.add_row(reason, str(count))
                if reason_drops:
                    log.write(reason_tbl)
            else:
                log.write("No online candidate trace for this step.")

            candidates = bundle.get("candidates")
            if isinstance(candidates, list) and candidates:
                log.write("")
                log.write(Text("Top Numeric Candidates", style="bold #9ab6ff"))
                cand_tbl = Table(show_header=True, header_style="bold #9ab6ff")
                cand_tbl.add_column("Rank", justify="right")
                cand_tbl.add_column("Pred(ms)", justify="right")
                cand_tbl.add_column("Unc", justify="right")
                cand_tbl.add_column("Patch")
                for idx, item in enumerate(candidates[:5], start=1):
                    if not isinstance(item, dict):
                        continue
                    pred = item.get("predicted_time_ms")
                    unc = item.get("uncertainty")
                    cfg = item.get("config", {}) if isinstance(item.get("config"), dict) else {}
                    cand_tbl.add_row(
                        str(idx),
                        f"{pred:.2f}" if isinstance(pred, (int, float)) else "-",
                        f"{unc:.3f}" if isinstance(unc, (int, float)) else "-",
                        self._compact_patch(cfg),
                    )
                log.write(cand_tbl)

        # Cross-step pruning trend.
        trend: Dict[str, int] = {}
        for s in sorted(self.state.base_steps.keys()):
            summary = self._read_json_cached(
                self.state.run_dir / "steps" / f"step_{s}_pruning_summary.json"
            ) if self.state.run_dir is not None else None
            if not isinstance(summary, dict):
                continue
            dropped = summary.get("dropped", {}) if isinstance(summary.get("dropped"), dict) else {}
            for reason, count in dropped.items():
                c = int(count) if isinstance(count, (int, float)) else 0
                trend[str(reason)] = trend.get(str(reason), 0) + c

        if trend:
            log.write("")
            log.write(Text("Cross-Step Drop Reasons", style="bold #63d8ff"))
            trend_tbl = Table(show_header=True, header_style="bold #63d8ff")
            trend_tbl.add_column("Reason")
            trend_tbl.add_column("Total", justify="right")
            for reason, count in sorted(trend.items(), key=lambda item: item[1], reverse=True)[:12]:
                trend_tbl.add_row(reason, str(count))
            log.write(trend_tbl)

    def _render_llm_reasoning_timeline(self) -> None:
        step = self.state.selected_step
        sig = (
            step,
            self._base_steps_signature,
            self._llm_calls_signature,
            self.state.best_iteration_ms,
        )
        if not self._should_render("right_llm_reasoning", sig, min_interval_s=0.8):
            return

        log = self.query_one("#llm-reasoning-log", RichLog)
        log.clear()
        log.write(Text("LLM Reasoning Timeline", style="bold #63d8ff"))
        log.write(
            "Covers offline strategy, online per-step decisions, and post-run distillation. "
            "Rendered as reasoning summaries without raw JSON dumps."
        )

        offline = self._load_offline_bundle()
        log.write("")
        log.write(Text("Offline Reasoning", style="bold #7ddf8d"))
        plan_status = offline.get("llm_plan_status")
        if isinstance(plan_status, dict):
            accepted = plan_status.get("accepted")
            fallback = plan_status.get("fallback_reason") or "none"
            log.write(f"- strategic plan accepted: {accepted} | fallback_reason: {fallback}")
        warm_decision = offline.get("warm_start_decision")
        if isinstance(warm_decision, dict):
            log.write(
                f"- warm-start selection: {warm_decision.get('selected_id', '-')} "
                f"because {warm_decision.get('reason', 'reason not provided')}"
            )
        warm_program = offline.get("warm_start_program")
        if isinstance(warm_program, dict):
            mode = warm_program.get("mode", "single")
            candidates = warm_program.get("candidates", []) if isinstance(warm_program.get("candidates"), list) else []
            log.write(f"- warm-start program: mode={mode}, candidates={len(candidates)}")
        pruning = offline.get("pruning_guidance")
        if isinstance(pruning, list) and pruning:
            top = [item for item in pruning if isinstance(item, dict)][:4]
            for item in top:
                log.write(
                    f"- offline pruning: {item.get('param', '?')} -> {item.get('action', '?')} "
                    f"because {_truncate_text(item.get('reason', 'not provided'), max_len=100)}"
                )
        playbook = offline.get("hypothesis_playbook")
        if isinstance(playbook, list) and playbook:
            for item in playbook[:3]:
                if not isinstance(item, dict):
                    continue
                log.write(
                    f"- offline hypothesis template: {_truncate_text(item.get('summary', 'unnamed template'), max_len=120)}"
                )

        log.write("")
        log.write(Text("Online LLM Decisions", style="bold #ffc66d"))
        rows: List[tuple[str, str, str, str, str]] = []
        for s in sorted(self.state.base_steps.keys()):
            bundle = self._load_step_bundle(s)
            decision = bundle.get("decision", {}) if isinstance(bundle.get("decision"), dict) else {}
            action = str(decision.get("action", "-"))
            lane = str(decision.get("lane_source", "schedule_fallback"))

            llm_signal = "-"
            rationale = "-"
            support = bundle.get("llm_decision_support")
            if isinstance(support, dict):
                output = support.get("output", {}) if isinstance(support.get("output"), dict) else {}
                pref = output.get("action_preference")
                rec = output.get("recommended_action", {}) if isinstance(output.get("recommended_action"), dict) else {}
                rec_kind = rec.get("kind")
                if pref or rec_kind:
                    llm_signal = f"pref={pref or 'auto'} rec={rec_kind or '-'}"
                reason_claims = rec.get("reason_claims") if isinstance(rec.get("reason_claims"), list) else []
                if reason_claims:
                    first = reason_claims[0]
                    if isinstance(first, dict):
                        rationale = str(first.get("claim", rationale))
                    else:
                        rationale = str(first)
                hyps = output.get("hypotheses", [])
                if rationale == "-" and isinstance(hyps, list) and hyps and isinstance(hyps[0], dict):
                    rationale = str(hyps[0].get("summary", "-"))
            conv = bundle.get("llm_convergence")
            if isinstance(conv, dict):
                output = conv.get("output", {}) if isinstance(conv.get("output"), dict) else {}
                conv_obj = output.get("convergence", {}) if isinstance(output.get("convergence"), dict) else {}
                conv_decision = conv_obj.get("decision")
                if conv_decision in ("continue", "stop"):
                    llm_signal = f"{llm_signal} | conv={conv_decision}" if llm_signal != "-" else f"conv={conv_decision}"
                    if rationale == "-" and conv_obj.get("reason"):
                        rationale = str(conv_obj.get("reason"))
            if rationale == "-":
                dec_bundle = bundle.get("decision_bundle", {})
                dec_record = dec_bundle if isinstance(dec_bundle, dict) and dec_bundle else bundle.get("decision_record", {})
                if isinstance(dec_record, dict):
                    why = dec_record.get("why_selected", [])
                    if isinstance(why, list) and why:
                        first = why[0]
                        if isinstance(first, dict):
                            rationale = str(first.get("claim", "-"))
                        else:
                            rationale = str(first)
            rows.append((str(s), lane, action, _truncate_text(llm_signal, max_len=80), _truncate_text(rationale, max_len=140)))

        if rows:
            tbl = Table(show_header=True, header_style="bold #ffc66d")
            tbl.add_column("Step", justify="right")
            tbl.add_column("Lane Source")
            tbl.add_column("Action")
            tbl.add_column("LLM Signal")
            tbl.add_column("Reasoning")
            for row in rows[:18]:
                tbl.add_row(*row)
            log.write(tbl)
        else:
            log.write("No online steps yet.")

        if isinstance(step, int):
            bundle = self._load_step_bundle(step)
            log.write("")
            log.write(Text(f"Selected Step {step} Convergence Reason", style="bold #9ab6ff"))
            conv_reason = None
            conv = bundle.get("llm_convergence")
            if isinstance(conv, dict):
                output = conv.get("output", {}) if isinstance(conv.get("output"), dict) else {}
                conv_obj = output.get("convergence", {}) if isinstance(output.get("convergence"), dict) else {}
                if conv_obj.get("reason"):
                    conv_reason = str(conv_obj.get("reason"))
            if conv_reason is None:
                support = bundle.get("llm_decision_support")
                if isinstance(support, dict):
                    output = support.get("output", {}) if isinstance(support.get("output"), dict) else {}
                    conv_obj = output.get("convergence", {}) if isinstance(output.get("convergence"), dict) else {}
                    if conv_obj.get("reason"):
                        conv_reason = str(conv_obj.get("reason"))
            stop_decision = bundle.get("stop_decision")
            if conv_reason is None and isinstance(stop_decision, dict):
                conv_reason = str(stop_decision.get("reason", "not provided"))
            log.write(conv_reason or "No explicit convergence reason found for the selected step.")

        log.write("")
        log.write(Text("Post-Run LLM Distillation", style="bold #7ddf8d"))
        if self.state.run_dir is not None:
            postrun_dir = self.state.run_dir / "postrun"
            convergence = self._read_json_cached(postrun_dir / "convergence.json")
            if isinstance(convergence, dict):
                log.write(
                    f"- post-run convergence: reason={convergence.get('reason', '-')} | steps={convergence.get('steps', '-')}"
                )
            rules_path = postrun_dir / "rules_distilled.jsonl"
            rules = self._read_jsonl_records(rules_path, max_records=50)
            if rules:
                log.write(f"- distilled semantic rules: {len(rules)}")
                for rule in rules[:3]:
                    if not isinstance(rule, dict):
                        continue
                    rule_id = rule.get("rule_id", "rule")
                    action = rule.get("action", {}) if isinstance(rule.get("action"), dict) else {}
                    patch = action.get("set", {}) if isinstance(action.get("set"), dict) else {}
                    effect = rule.get("effect", {}) if isinstance(rule.get("effect"), dict) else {}
                    improvement = effect.get("improvement")
                    imp_text = f"{float(improvement) * 100.0:.2f}%" if isinstance(improvement, (int, float)) else "n/a"
                    log.write(f"- {rule_id}: set {self._compact_patch(patch)} | expected improvement {imp_text}")
            else:
                log.write("- no distilled rules available yet")

    def _load_offline_bundle(self) -> Dict[str, Any]:
        if self.state.run_dir is None:
            return {}
        offline_dir = self.state.run_dir / "offline"
        if not offline_dir.exists():
            return {}
        mapping = {
            "llm_plan_status": "llm_plan_status.json",
            "llm_strategic_plan": "llm_strategic_plan.json",
            "warm_start_decision": "warm_start_decision.json",
            "warm_start_program": "warm_start_program.json",
            "pruning_guidance": "pruning_guidance.json",
            "search_space_pruning": "search_space_pruning.json",
            "hypothesis_playbook": "hypothesis_playbook.json",
        }
        sig_rows = []
        for name in mapping.values():
            path = offline_dir / name
            if not path.exists():
                continue
            try:
                stat = path.stat()
            except Exception:
                continue
            sig_rows.append((name, stat.st_mtime_ns, stat.st_size))
        signature = tuple(sig_rows)
        if signature == self._offline_bundle_signature:
            return self._offline_bundle_cache

        bundle: Dict[str, Any] = {}
        for key, name in mapping.items():
            data = self._read_json_cached(offline_dir / name)
            if data is not None:
                bundle[key] = data
        self._offline_bundle_signature = signature
        self._offline_bundle_cache = bundle
        return bundle

    def _read_jsonl_records(self, path: Path, max_records: int = 100) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        key = str(path)
        try:
            stat = path.stat()
            mtime = stat.st_mtime
            size = stat.st_size
        except Exception:
            return []
        cached = self._jsonl_cache.get(key)
        if cached and cached[0] == mtime and cached[1] == size:
            return cached[2][:max_records]
        out: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(rec, dict):
                        out.append(rec)
        except Exception:
            return []
        self._jsonl_cache[key] = (mtime, size, out)
        return out[:max_records]

    def _compact_patch(self, patch: Dict[str, Any], max_items: int = 4) -> str:
        if not isinstance(patch, dict) or not patch:
            return "-"
        parts = []
        for idx, (key, value) in enumerate(patch.items()):
            if idx >= max_items:
                parts.append("...")
                break
            parts.append(f"{key}={value}")
        return ", ".join(parts)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _selected_event(self) -> Optional[Dict[str, Any]]:
        idx = self.state.selected_event_idx
        if idx is None:
            return None
        if idx < 0 or idx >= len(self.state.events):
            return None
        return self.state.events[idx]

    def _selected_llm_call(self) -> Optional[Dict[str, Any]]:
        idx = self.state.selected_llm_idx
        if idx is None:
            return None
        if idx < 0 or idx >= len(self.state.llm_calls):
            return None
        return self.state.llm_calls[idx]

    def _find_llm_call(self, call_id: str) -> Optional[Dict[str, Any]]:
        for call in reversed(self.state.llm_calls):
            if call.get("call_id") == call_id:
                return call
        return None

    def _find_paired_tool_event(self, *, call_id: str, current_type: str) -> Optional[Dict[str, Any]]:
        target_type = "tool.result" if current_type == "tool.call" else "tool.call"
        for event in reversed(self.state.events):
            if event.get("type") != target_type:
                continue
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            if payload.get("call_id") == call_id:
                return event
        return None

    def _load_step_bundle(self, step: int) -> Dict[str, Any]:
        if self.state.run_dir is None:
            return {}
        steps_dir = self.state.run_dir / "steps"
        mapping = {
            "base": f"step_{step}.json",
            "decision": f"step_{step}_decision.json",
            "decision_record": f"step_{step}_decision_record.json",
            "decision_bundle": f"step_{step}_decision_bundle.json",
            "hypothesis": f"step_{step}_hypothesis.json",
            "hypothesis_series": f"step_{step}_hypothesis_series.json",
            "hypothesis_portfolio": f"step_{step}_hypothesis_portfolio.json",
            "hypothesis_ranked": f"step_{step}_hypothesis_ranked.json",
            "llm_decision_support": f"step_{step}_llm_decision_support.json",
            "llm_convergence": f"step_{step}_llm_convergence.json",
            "candidates": f"step_{step}_candidates.json",
            "candidates_trace": f"step_{step}_candidates_trace.json",
            "pruning_summary": f"step_{step}_pruning_summary.json",
            "context_pack": f"step_{step}_context_pack.json",
            "risk_report": f"step_{step}_risk_report.json",
            "metrics_derived": f"step_{step}_metrics_derived.json",
            "bottleneck": f"step_{step}_bottleneck.json",
            "stop_decision": f"step_{step}_stop_decision.json",
            "rollback_decision": f"step_{step}_rollback_decision.json",
        }
        signature = self._step_artifact_signature(step)
        cached = self._step_bundle_cache.get(step)
        if cached and cached[0] == signature:
            return cached[1]

        bundle: Dict[str, Any] = {}
        for key, filename in mapping.items():
            path = steps_dir / filename
            if not path.exists():
                continue
            data = self._read_json_cached(path)
            if data is not None:
                bundle[key] = data
        self._step_bundle_cache[step] = (signature, bundle)
        return bundle

    def _step_artifact_signature(self, step: int) -> tuple:
        if self.state.run_dir is None:
            return ()
        steps_dir = self.state.run_dir / "steps"
        if not steps_dir.exists():
            return ()
        names = [
            f"step_{step}.json",
            f"step_{step}_decision.json",
            f"step_{step}_decision_record.json",
            f"step_{step}_decision_bundle.json",
            f"step_{step}_hypothesis.json",
            f"step_{step}_hypothesis_series.json",
            f"step_{step}_hypothesis_portfolio.json",
            f"step_{step}_hypothesis_ranked.json",
            f"step_{step}_llm_decision_support.json",
            f"step_{step}_llm_convergence.json",
            f"step_{step}_candidates.json",
            f"step_{step}_candidates_trace.json",
            f"step_{step}_pruning_summary.json",
            f"step_{step}_context_pack.json",
            f"step_{step}_risk_report.json",
            f"step_{step}_bottleneck.json",
            f"step_{step}_metrics_derived.json",
            f"step_{step}_stop_decision.json",
            f"step_{step}_rollback_decision.json",
        ]
        out = []
        for name in names:
            path = steps_dir / name
            if not path.exists():
                continue
            try:
                stat = path.stat()
            except Exception:
                continue
            out.append((name, stat.st_mtime_ns, stat.st_size))
        return tuple(out)

    def _build_llm_impact_summary(self) -> List[str]:
        lines: List[str] = []
        if self.state.run_dir is None:
            return lines
        offline = self._read_json_cached(self.state.run_dir / "offline" / "llm_plan_status.json")
        if isinstance(offline, dict):
            accepted = offline.get("accepted")
            lines.append(f"offline strategic plan accepted: {accepted}")
            if accepted:
                lines.append("warm-start / pruning guidance came directly from LLM output")
            elif offline.get("fallback_reason"):
                lines.append(f"offline fallback reason: {offline.get('fallback_reason')}")

        used_count = 0
        conv_count = 0
        for step in sorted(self.state.base_steps.keys()):
            bundle = self._load_step_bundle(step)
            support = bundle.get("llm_decision_support")
            if isinstance(support, dict):
                if support.get("used_in_decision") is True:
                    used_count += 1
                output = support.get("output", {}) if isinstance(support.get("output"), dict) else {}
                conv = output.get("convergence", {}) if isinstance(output.get("convergence"), dict) else {}
                if conv.get("decision") in ("continue", "stop"):
                    conv_count += 1
            conv_art = bundle.get("llm_convergence")
            if isinstance(conv_art, dict):
                output = conv_art.get("output", {}) if isinstance(conv_art.get("output"), dict) else {}
                conv = output.get("convergence", {}) if isinstance(output.get("convergence"), dict) else {}
                if conv.get("decision") in ("continue", "stop"):
                    conv_count += 1
        if used_count or conv_count:
            lines.append(f"online llm advice used in decisions: {used_count} step(s)")
            lines.append(f"llm convergence decisions observed: {conv_count}")
        return lines

    def _summarize_llm_output(self, output: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        if not isinstance(output, dict):
            return lines

        if "warm_start_program" in output or "baseline_patch" in output:
            baseline = output.get("baseline_patch", {})
            warm = output.get("warm_start_program", {})
            pruning = output.get("pruning_guidance", [])
            playbook = output.get("hypothesis_playbook", [])
            lines.append(f"offline baseline patch size: {len(baseline) if isinstance(baseline, dict) else 0}")
            if isinstance(warm, dict):
                lines.append(
                    f"warm-start mode={warm.get('mode', 'single')} | "
                    f"candidates={len(warm.get('candidates', [])) if isinstance(warm.get('candidates'), list) else 0}"
                )
            lines.append(f"pruning rules={len(pruning) if isinstance(pruning, list) else 0}")
            lines.append(f"hypothesis templates={len(playbook) if isinstance(playbook, list) else 0}")
            return lines

        if "hypotheses" in output or "convergence" in output or "numeric_guidance" in output:
            conv = output.get("convergence", {}) if isinstance(output.get("convergence"), dict) else {}
            if conv:
                lines.append(
                    f"convergence={conv.get('decision', '?')} (confidence={conv.get('confidence', '-')})"
                )
                reason = conv.get("reason")
                if reason:
                    lines.append(f"reason: {_truncate_text(reason, max_len=180)}")
            pref = output.get("action_preference")
            if pref:
                lines.append(f"action preference: {pref}")
            hyps = output.get("hypotheses", [])
            if isinstance(hyps, list):
                lines.append(f"hypotheses proposed: {len(hyps)}")
                if hyps and isinstance(hyps[0], dict):
                    patch = hyps[0].get("patch", {}) if isinstance(hyps[0].get("patch"), dict) else {}
                    if patch:
                        keys = ", ".join(list(patch.keys())[:6])
                        lines.append(f"top hypothesis patch keys: {keys}")
            tool_req = output.get("tool_request", {}) if isinstance(output.get("tool_request"), dict) else {}
            if tool_req:
                lines.append(f"tool request: {tool_req.get('name', 'none')}")
            return lines

        keys = ", ".join(list(output.keys())[:8])
        lines.append(f"json keys: {keys}" if keys else "empty json response")
        return lines

    def _row_key_to_str(self, row_key: Any) -> str:
        value = getattr(row_key, "value", row_key)
        return str(value)

    def _handle_inspect_chat(self, message: str) -> None:
        msg = message.strip()
        if not msg:
            return
        low = msg.lower()

        if low.startswith("/help"):
            self._emit_chat(
                "Agent",
                "Inspect commands: /state, /best, /context, /ctxeng. "
                "Config-changing commands (/set, /setcfg, /setplan) are live-only.",
                style="bold #7ddf8d",
            )
            return

        if low.startswith("/set ") or low.startswith("/setcfg ") or low.startswith("/setplan "):
            self._emit_chat(
                "Agent",
                "Inspect mode is read-only. Use --mode live to apply runtime overrides.",
                style="bold #7ddf8d",
            )
            return

        if low.startswith("/state"):
            self._emit_chat("Agent", self._inspect_state_text(), style="bold #7ddf8d")
            return

        if low.startswith("/best"):
            self._emit_chat("Agent", self._inspect_best_text(), style="bold #7ddf8d")
            return

        if low.startswith("/context"):
            self._emit_chat("Agent", self._inspect_context_text(), style="bold #7ddf8d")
            return

        if low.startswith("/ctxeng"):
            self._emit_chat("Agent", self._inspect_ctxeng_text(), style="bold #7ddf8d")
            return

        snapshot = self._build_inspect_chat_snapshot(question=msg)
        self.state.latest_chat_context = dict(snapshot, _path="inspect://local-chat")
        self._emit_chat("Agent", self._answer_inspect_question(msg, snapshot), style="bold #7ddf8d")

    def _inspect_state_text(self) -> str:
        selected_step = self.state.selected_step
        if selected_step is None and self.state.base_steps:
            selected_step = max(self.state.base_steps.keys())
        summary = {
            "run_id": self.state.run_id,
            "phase": self.state.phase,
            "running": self.state.running,
            "events": len(self.state.events),
            "steps": len(self.state.base_steps),
            "llm_calls": len(self.state.llm_calls),
            "selected_step": selected_step,
            "selected_event_idx": self.state.selected_event_idx,
            "selected_llm_idx": self.state.selected_llm_idx,
            "best_iteration_ms": self.state.best_iteration_ms,
        }
        return f"Current state:\n```json\n{json.dumps(summary, indent=2, default=str)}\n```"

    def _inspect_best_text(self) -> str:
        best_step: Optional[int] = None
        best_ms: Optional[float] = None
        for step, rec in self.state.base_steps.items():
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics"), dict) else {}
            iter_ms = _safe_float(metrics.get("iteration_time_ms"))
            if iter_ms is None:
                continue
            if best_ms is None or iter_ms < best_ms:
                best_ms = iter_ms
                best_step = step

        if best_step is None or best_ms is None:
            return "No best config found yet."

        baseline = self.state.iteration_times_ms[0] if self.state.iteration_times_ms else None
        gain = None
        if isinstance(baseline, (int, float)) and baseline > 0:
            gain = (baseline - best_ms) / baseline * 100.0

        payload = {
            "best_step": best_step,
            "best_iteration_ms": best_ms,
            "baseline_iteration_ms": baseline,
            "gain_percent_vs_baseline": gain,
            "action": (
                self.state.base_steps.get(best_step, {}).get("action", {})
                if isinstance(self.state.base_steps.get(best_step, {}).get("action"), dict)
                else None
            ),
        }
        return f"Best record:\n```json\n{json.dumps(payload, indent=2, default=str)}\n```"

    def _inspect_context_text(self) -> str:
        step = self.state.selected_step
        if step is None and self.state.base_steps:
            step = max(self.state.base_steps.keys())
        if step is None:
            return "No step context available."
        bundle = self._load_step_bundle(step)
        context_pack = bundle.get("context_pack")
        if context_pack is None:
            return f"No context pack found for step {step}."
        return (
            f"Context Pack (step {step}):\n```json\n"
            f"{json.dumps(_compact(context_pack, max_depth=6, max_items=50), indent=2, default=str)}\n```"
        )

    def _inspect_ctxeng_text(self) -> str:
        snap = self.state.latest_chat_context
        if not isinstance(snap, dict):
            return "No chat context snapshot yet. Ask a question first."
        view = {
            "path": snap.get("_path"),
            "step": snap.get("step"),
            "question": snap.get("question"),
            "context_window": snap.get("context_window"),
            "sections": _compact(snap.get("sections", []), max_depth=5, max_items=40),
        }
        return f"Latest context-engineering snapshot:\n```json\n{json.dumps(view, indent=2, default=str)}\n```"

    def _build_inspect_chat_snapshot(self, *, question: str) -> Dict[str, Any]:
        step = self.state.selected_step
        if step is None and self.state.base_steps:
            step = max(self.state.base_steps.keys())
        step_bundle = self._load_step_bundle(step) if isinstance(step, int) else {}
        selected_call = self._selected_llm_call()
        selected_event = self._selected_event()
        summary = {
            "run_id": self.state.run_id,
            "phase": self.state.phase,
            "events": len(self.state.events),
            "steps": len(self.state.base_steps),
            "llm_calls": len(self.state.llm_calls),
            "best_iteration_ms": self.state.best_iteration_ms,
        }
        sections = [
            {"name": "run_summary", "content": summary},
            {"name": "selected_step", "content": _compact(step_bundle, max_depth=5, max_items=40)},
            {"name": "selected_llm_call", "content": _compact(selected_call, max_depth=5, max_items=40)},
            {"name": "selected_event", "content": _compact(selected_event, max_depth=5, max_items=40)},
        ]
        assembled_user_prompt = (
            "You are inspecting a completed CCL run. Use provided artifacts only.\n\n"
            f"User question: {question}\n"
        )
        approx_chars = len(assembled_user_prompt) + len(json.dumps(sections, default=str))
        token_est = max(1, approx_chars // 4)
        return {
            "mode": "inspect",
            "step": step,
            "question": question,
            "step_vs_iteration": (
                "Step = one online control decision loop; "
                "training iteration = inner workload iteration_time_ms measurements."
            ),
            "context_window": {
                "approx_tokens": token_est,
                "max_tokens_budget": 32000,
                "usage_ratio": min(1.0, token_est / 32000.0),
            },
            "sections": sections,
            "system_prompt": (
                "Answer with evidence from run artifacts. If data is missing, state that explicitly."
            ),
            "assembled_user_prompt": assembled_user_prompt,
        }

    def _answer_inspect_question(self, question: str, snapshot: Dict[str, Any]) -> str:
        q = question.lower()
        if "step" in q and "iteration" in q:
            return (
                "In this run: step means one online tuning decision cycle "
                "(propose -> execute -> evaluate -> update), while training iterations are inner workload "
                "iterations used to measure `iteration_time_ms` inside each step."
            )

        if "hypothesis" in q or "prun" in q or "reason" in q or "decision" in q:
            step = snapshot.get("step")
            if isinstance(step, int):
                return (
                    f"For step {step}, open `Online Reasoning` for step-level decision rationale, "
                    "`Pruning Lens` for offline+online pruning evidence, and `LLM Reasoning` for "
                    "offline->online->postrun reasoning timeline."
                )
            return "Select a step in the left pane to inspect hypothesis, pruning, and decision rationale."

        if "context" in q or "prompt" in q:
            return (
                "Open the `Context Engineering` tab to inspect context packs, selected LLM context windows, "
                "and the latest chat context snapshot used for this answer."
            )

        return (
            "Answered from inspect artifacts only. Use `/state`, `/best`, `/context`, or `/ctxeng` for structured "
            "views, and select an event/step/LLM call to ground follow-up questions."
        )

    def _capture_left_table_state(self, table: DataTable) -> None:
        table_id = table.id or ""
        if not table_id:
            return
        scroll_y = getattr(table, "scroll_y", None)
        if isinstance(scroll_y, (int, float)):
            self._left_scroll_y[table_id] = float(scroll_y)

    def _restore_left_table_state(self, table: DataTable) -> None:
        table_id = table.id or ""
        if not table_id:
            return

        prev_scroll_y = self._left_scroll_y.get(table_id)
        if isinstance(prev_scroll_y, (int, float)):
            current_scroll_y = getattr(table, "scroll_y", None)
            if isinstance(current_scroll_y, (int, float)) and abs(float(current_scroll_y) - float(prev_scroll_y)) < 0.1:
                return
            try:
                table.scroll_to(y=float(prev_scroll_y), animate=False)
            except Exception:
                pass

    def _emit_chat(self, who: str, text: str, *, style: str = "") -> None:
        log = self.query_one("#chat-log", RichLog)
        timestamp = _fmt_ts(time.time())
        header = Text(f"[{timestamp}] ", style="dim")
        header.append(who, style=style or "bold")
        log.write(header)
        log.write(_truncate_text(text, max_len=6000))
        log.write("")

    def _emit_system_chat(self, text: str) -> None:
        self._emit_chat("System", text, style="dim")


__all__ = ["AgentWorkbench"]
