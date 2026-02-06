import unittest

from src.agent.core import CCLAgent
from src.config import default_agent_config


class DummyPlan:
    def __init__(self) -> None:
        self.recommended_search_params = ["NCCL_ALGO", "NCCL_PROTO"]


class TestAgentRuntimeCommands(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = object.__new__(CCLAgent)

    def test_setcfg_updates_budget_field(self) -> None:
        cfg = default_agent_config()
        applied, value = self.agent._set_runtime_field(cfg, "budget.hypothesis_every", "3")
        self.assertTrue(applied)
        self.assertEqual(value, 3)
        self.assertEqual(cfg.budget.hypothesis_every, 3)

    def test_setplan_parses_python_list_literal(self) -> None:
        plan = DummyPlan()
        applied, value = self.agent._set_runtime_field(
            plan,
            "recommended_search_params",
            "['NCCL_ALGO']",
        )
        self.assertTrue(applied)
        self.assertEqual(value, ["NCCL_ALGO"])
        self.assertEqual(plan.recommended_search_params, ["NCCL_ALGO"])

    def test_rejects_unsafe_root(self) -> None:
        cfg = default_agent_config()
        applied, _ = self.agent._set_runtime_field(cfg, "__class__.__name__", "x")
        self.assertFalse(applied)

    def test_stop_command_silent_skips_chat_response(self) -> None:
        emitted = []
        self.agent._emit_chat_response = lambda step, message: emitted.append((step, message))
        self.agent._stop_requested = False
        self.agent._pause_requested = True
        cmd = type("Cmd", (), {"action": "stop", "payload": {"silent": True}})()

        self.agent._handle_command(cmd, step=3)

        self.assertTrue(self.agent._stop_requested)
        self.assertFalse(self.agent._pause_requested)
        self.assertEqual(emitted, [])

    def test_stop_command_default_emits_chat_response(self) -> None:
        emitted = []
        self.agent._emit_chat_response = lambda step, message: emitted.append((step, message))
        self.agent._stop_requested = False
        self.agent._pause_requested = True
        cmd = type("Cmd", (), {"action": "stop", "payload": {}})()

        self.agent._handle_command(cmd, step=4)

        self.assertTrue(self.agent._stop_requested)
        self.assertFalse(self.agent._pause_requested)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][0], 4)


if __name__ == "__main__":
    unittest.main()
