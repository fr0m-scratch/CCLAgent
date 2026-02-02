from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rich.console import Group
    from rich.panel import Panel
    from rich.pretty import Pretty
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical
    from textual.widgets._data_table import DataTable
    from textual.widgets._footer import Footer
    from textual.widgets._header import Header
    from textual.widgets._input import Input
    from textual.widgets._rich_log import RichLog
    from textual.widgets._static import Static
    from textual.widgets._tabbed_content import TabbedContent, TabPane
except Exception as exc:  # pragma: no cover - handled by runner script
    raise RuntimeError("textual and rich are required for the TUI") from exc

from ..llm import LLMMessage, NullLLMClient, create_llm_client
from ..utils import load_env_file


@dataclass
class RunState:
    run_dir: Optional[Path] = None
    run_id: str = ""
    steps: List[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    hypothesis: Optional[Dict[str, Any]] = None
    compiled: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    microbench: Optional[Dict[str, Any]] = None
    last_update_ts: float = 0.0
    selected_step: Optional[int] = None


class AgentDashboard(App):
    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; }
    #left { width: 40%; }
    #right { width: 60%; }
    #steps { height: 1fr; }
    #status { height: auto; }
    #chat { height: 30%; }
    #chat_log { height: 1fr; }
    #chat_input { height: 3; }
    RichLog { background: $panel; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(
        self,
        artifacts_root: str = "artifacts",
        run_dir: Optional[str] = None,
        provider: str = "none",
        model: str = "",
        env_file: str = ".env.local",
        poll_interval: float = 1.0,
        tail_lines: int = 200,
    ) -> None:
        super().__init__()
        self.artifacts_root = Path(artifacts_root)
        self.run_dir_override = Path(run_dir) if run_dir else None
        self.poll_interval = poll_interval
        self.tail_lines = tail_lines
        load_env_file(env_file)
        self.llm = create_llm_client(provider, model) if provider and provider != "none" else NullLLMClient()
        self.state = RunState(steps=[])

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with Horizontal():
                with Vertical(id="left"):
                    yield Static("", id="status")
                    yield DataTable(id="steps")
                with Vertical(id="right"):
                    with TabbedContent(id="tabs"):
                        with TabPane("Metrics", id="tab_metrics"):
                            yield Static("", id="metrics")
                        with TabPane("Decision", id="tab_decision"):
                            yield RichLog(id="decision", auto_scroll=False)
                        with TabPane("Plan", id="tab_plan"):
                            yield RichLog(id="plan", auto_scroll=False)
                        with TabPane("Tools", id="tab_tools"):
                            yield RichLog(id="tools", auto_scroll=False)
                        with TabPane("Logs", id="tab_logs"):
                            yield RichLog(id="logs", auto_scroll=False)
        with Container(id="chat"):
            yield RichLog(id="chat_log", auto_scroll=False)
            yield Input(placeholder="Ask the agent...", id="chat_input")
        yield Footer()

    def on_mount(self) -> None:
        self._init_steps_table()
        self.set_interval(self.poll_interval, self.refresh_state)
        self.refresh_state()

    def _init_steps_table(self) -> None:
        table = self.query_one("#steps", DataTable)
        table.add_columns("step", "mode", "action", "iter_ms", "success", "reason")

    def action_refresh(self) -> None:
        self.refresh_state()

    def refresh_state(self) -> None:
        run_dir = self._resolve_run_dir()
        if run_dir is None:
            self._update_status("No runs found under artifacts/")
            return
        self.state.run_dir = run_dir
        self.state.run_id = run_dir.name

        self.state.microbench = self._read_json(run_dir / "offline" / "microbench_summary.json")
        self.state.plan = self._read_json(run_dir / "offline" / "initial_plan.json")

        steps = self._load_steps(run_dir / "steps")
        self.state.steps = steps
        self.state.last_update_ts = time.time()

        available_steps = [rec.get("step") for rec in steps if rec.get("step") is not None]
        selected = self.state.selected_step
        if selected is None or selected not in available_steps:
            selected = available_steps[-1] if available_steps else None
        self.state.selected_step = selected
        if selected is not None:
            self._load_step_artifacts(selected)

        self._render_status()
        self._render_steps_table()
        self._render_metrics()
        self._render_plan()
        self._render_selected_step()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        try:
            step = int(str(event.row_key))
        except Exception:
            return
        self.state.selected_step = step
        self._load_step_artifacts(step)
        self._render_selected_step()

    def _resolve_run_dir(self) -> Optional[Path]:
        if self.run_dir_override and self.run_dir_override.exists():
            return self.run_dir_override
        if not self.artifacts_root.exists():
            return None
        runs = [path for path in self.artifacts_root.iterdir() if path.is_dir()]
        if not runs:
            return None
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return runs[0]

    def _load_steps(self, steps_dir: Path) -> List[Dict[str, Any]]:
        if not steps_dir.exists():
            return []
        records = []
        for path in sorted(steps_dir.glob("step_*.json")):
            if path.name.endswith("_decision.json") or path.name.endswith("_metrics.json"):
                continue
            if "_" in path.stem[len("step_") :]:
                continue
            data = self._read_json(path)
            if data:
                records.append(data)
        return records

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _update_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _render_status(self) -> None:
        run_id = self.state.run_id or "-"
        steps = len(self.state.steps) if self.state.steps else 0
        selected = self.state.selected_step
        mode = self._infer_step_mode(selected) if selected is not None else "-"
        status = (
            f"Run: {run_id} | Steps: {steps} | Selected: {selected if selected is not None else '-'}"
            f" | Mode: {mode} | Last update: {time.strftime('%H:%M:%S')}"
        )
        self._update_status(status)

    def _render_steps_table(self) -> None:
        table = self.query_one("#steps", DataTable)
        table.clear()
        for record in self.state.steps or []:
            metrics = record.get("metrics", {})
            iter_ms = metrics.get("iteration_time_ms")
            success = metrics.get("success")
            reason = metrics.get("failure_reason") or ""
            step_idx = record.get("step")
            mode = self._infer_step_mode(step_idx) if step_idx is not None else ""
            table.add_row(
                str(record.get("step")),
                mode,
                str(record.get("action", {}).get("kind")),
                f"{iter_ms:.2f}" if isinstance(iter_ms, (int, float)) else "-",
                str(success),
                str(reason),
                key=str(record.get("step")),
            )

    def _render_metrics(self) -> None:
        panel = self.query_one("#metrics", Static)
        times = []
        algbw = []
        busbw = []
        for record in self.state.steps or []:
            metrics = record.get("metrics", {})
            if isinstance(metrics.get("iteration_time_ms"), (int, float)):
                times.append(metrics["iteration_time_ms"])
            if isinstance(metrics.get("algbw_gbps"), (int, float)):
                algbw.append(metrics["algbw_gbps"])
            if isinstance(metrics.get("busbw_gbps"), (int, float)):
                busbw.append(metrics["busbw_gbps"])
        lines = []
        if times:
            lines.append(Text("iter_ms ") + Text(self._sparkline(times)))
        if algbw:
            lines.append(Text("algbw ") + Text(self._sparkline(algbw)))
        if busbw:
            lines.append(Text("busbw ") + Text(self._sparkline(busbw)))
        if not lines:
            panel.update(Panel("No metrics yet", title="Metrics"))
            return
        detail = ""
        if self.state.selected_step is not None:
            metrics = self._read_json(
                self.state.run_dir / "steps" / f"step_{self.state.selected_step}_metrics.json"
            ) if self.state.run_dir else None
            if metrics:
                detail = json.dumps(metrics, indent=2)
        group = Group(*lines)
        if detail:
            group = Group(*lines, Text("\nSelected step metrics:\n" + detail))
        panel.update(Panel(group, title="Metrics"))

    def _render_selected_step(self) -> None:
        self._render_decision()
        self._render_tools()
        self._render_logs()

    def _render_decision(self) -> None:
        panel = self.query_one("#decision", RichLog)
        panel.clear()
        if self.state.selected_step is None:
            panel.write("No step selected")
            return
        explanation = self._explain_decision(
            self.state.selected_step,
            self.state.decision,
            self.state.hypothesis,
            self.state.compiled,
        )
        payload = {
            "step": self.state.selected_step,
            "explanation": explanation,
            "decision": self.state.decision or {},
            "hypothesis": self.state.hypothesis or {},
            "compiled": self.state.compiled or {},
        }
        panel.write(Pretty(payload, expand_all=False))

    def _render_plan(self) -> None:
        panel = self.query_one("#plan", RichLog)
        panel.clear()
        explanation = self._explain_plan(self.state.microbench, self.state.plan)
        payload = {
            "explanation": explanation,
            "microbench": self.state.microbench or {},
            "plan": self.state.plan or {},
        }
        panel.write(Pretty(payload, expand_all=False))

    def _render_tools(self) -> None:
        panel = self.query_one("#tools", RichLog)
        panel.clear()
        if self.state.selected_step is None or not self.state.run_dir:
            panel.write("No steps yet")
            return
        idx = self.state.selected_step
        tool_payload = self._tool_calls_for_step(idx)
        panel.write(Pretty(tool_payload, expand_all=False))

    def _render_logs(self) -> None:
        log_widget = self.query_one("#logs", RichLog)
        log_widget.clear()
        if self.state.selected_step is None or not self.state.run_dir:
            log_widget.write("No logs yet")
            return
        idx = self.state.selected_step
        stdout_path = self.state.run_dir / "steps" / f"step_{idx}_stdout.log"
        stderr_path = self.state.run_dir / "steps" / f"step_{idx}_stderr.log"
        stdout = self._tail_file(stdout_path)
        stderr = self._tail_file(stderr_path)
        log_widget.write("[stdout]\n" + stdout)
        if stderr:
            log_widget.write("\n[stderr]\n" + stderr)

    def _load_step_artifacts(self, step: int) -> None:
        if not self.state.run_dir:
            return
        self.state.decision = self._read_json(self.state.run_dir / "steps" / f"step_{step}_decision.json")
        self.state.hypothesis = self._read_json(self.state.run_dir / "steps" / f"step_{step}_hypothesis.json")
        self.state.compiled = self._read_json(self.state.run_dir / "steps" / f"step_{step}_compiled_config.json")

    def _infer_step_mode(self, step: Optional[int]) -> str:
        if step is None:
            return "-"
        if step == 0:
            return "initial"
        if not self.state.run_dir:
            return "apply"
        decision = self._read_json(self.state.run_dir / "steps" / f"step_{step}_decision.json")
        action = decision.get("action") if decision else None
        if action in ("hypothesis", "numeric", "numeric_fallback"):
            return action
        return "apply"

    def _tool_calls_for_step(self, step: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"step": step, "calls": []}
        record = next((rec for rec in self.state.steps or [] if rec.get("step") == step), {})
        metrics = record.get("metrics", {})
        action = record.get("action", {})
        payload["action"] = action
        payload["metrics"] = metrics

        final_env = self._read_json(self.state.run_dir / "steps" / f"step_{step}_final_env.json")
        workload_cmd = self._read_json(self.state.run_dir / "steps" / f"workload_cmd_step_{step}.json")
        training_cmd = self._read_json(self.state.run_dir / "steps" / f"training_cmd_step_{step}.json")

        calls = []
        calls.append({"tool": "nccl.apply", "config": action.get("config", {}), "delta": action.get("delta", {})})
        if self.state.compiled:
            calls.append({"tool": "compiler.compile_hypothesis", "compiled": self.state.compiled})
        if workload_cmd:
            calls.append({"tool": "workload.run", "details": workload_cmd})
        if training_cmd:
            calls.append({"tool": "training.run", "details": training_cmd})
        if not workload_cmd and not training_cmd:
            calls.append({"tool": "workload.run", "details": "simulated"})
        if final_env:
            calls.append({"tool": "final_env", "details": final_env})
        if metrics:
            calls.append({"tool": "sla.check", "sla_ok": metrics.get("raw", {}).get("sla_ok"), "violations": metrics.get("raw", {}).get("sla_violations")})
            if metrics.get("success"):
                calls.append({"tool": "surrogate.update", "iteration_time_ms": metrics.get("iteration_time_ms")})
        candidates = self._read_json(self.state.run_dir / "steps" / f"step_{step}_candidates.json")
        if candidates:
            calls.append({"tool": "numeric.candidates", "count": len(candidates), "sample": candidates[:3]})
        payload["calls"] = calls
        return payload

    def _explain_decision(
        self,
        step: int,
        decision: Optional[Dict[str, Any]],
        hypothesis: Optional[Dict[str, Any]],
        compiled: Optional[Dict[str, Any]],
    ) -> str:
        if step == 0:
            return "Initial step: apply baseline config from offline plan and collect metrics."
        action = decision.get("action") if decision else "apply"
        last_success = decision.get("last_success") if decision else None
        plateau = decision.get("plateau") if decision else None
        risk = compiled.get("risk_score") if compiled else None
        parts = [f"Step {step} action = {action}."]
        if hypothesis:
            parts.append(f"Hypothesis: {hypothesis.get('summary', 'n/a')}.")
        if risk is not None:
            parts.append(f"Risk score: {risk}.")
        if last_success is not None:
            parts.append(f"Last step success: {last_success}.")
        if plateau is not None:
            parts.append(f"Plateau detected: {plateau}.")
        return " ".join(parts)

    def _explain_plan(self, microbench: Optional[Dict[str, Any]], plan: Optional[Dict[str, Any]]) -> str:
        if not microbench or not plan:
            return "No plan artifacts found yet."
        important = [item.get("param") for item in (microbench.get("important_params") or [])]
        signals = [item.get("name") for item in (microbench.get("signals") or [])]
        subspaces = plan.get("candidate_subspaces") or []
        recommended = plan.get("recommended_search_params") or []
        notes = plan.get("notes", "")
        return (
            f"Offline plan: microbench identified important params {important} with signals {signals}. "
            f"Recommended search params: {recommended}. Candidate subspaces: {len(subspaces)}. {notes}"
        )

    def _tail_file(self, path: Path) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            return "\n".join(lines[-self.tail_lines :])
        except Exception:
            return ""

    def _sparkline(self, values: List[float]) -> str:
        if not values:
            return ""
        chars = " .:-=+*#%"
        min_v = min(values)
        max_v = max(values)
        span = max(max_v - min_v, 1e-9)
        out = []
        for v in values[-50:]:
            idx = int((v - min_v) / span * (len(chars) - 1))
            out.append(chars[idx])
        return "".join(out)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question:
            return
        event.input.value = ""
        chat_log = self.query_one("#chat_log", RichLog)
        chat_log.write(f"you> {question}")
        response = await asyncio.to_thread(self._ask_llm, question)
        chat_log.write(f"agent> {response}")

    def _ask_llm(self, question: str) -> str:
        if isinstance(self.llm, NullLLMClient):
            return "LLM disabled. Launch with --provider and --model."
        summary = {
            "run_id": self.state.run_id,
            "last_step": self.state.steps[-1] if self.state.steps else {},
            "decision": self.state.decision or {},
            "plan": self.state.plan or {},
        }
        prompt = (
            "You are monitoring a CCL tuning run. Summarize and answer the user's question.\n"
            f"State summary: {json.dumps(summary, sort_keys=True)}\n"
            f"Question: {question}\n"
            "Answer concisely."
        )
        response = self.llm.complete([LLMMessage(role="user", content=prompt)])
        return response.content.strip() or "(no response)"


__all__ = ["AgentDashboard"]
