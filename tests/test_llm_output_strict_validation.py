import unittest

from src.agent.online_advisor import _parse_llm_json_status


class TestLLMOutputStrictValidation(unittest.TestCase):
    def test_valid_raw_json_is_marked_valid(self):
        text = (
            '{"hypotheses":[],"numeric_guidance":{},"tool_request":{"name":"none"},'
            '"action_preference":"auto","convergence":{"decision":"continue","confidence":0.8}}'
        )
        status = _parse_llm_json_status(text)
        self.assertTrue(status.raw_is_valid_json)
        self.assertFalse(status.used_partial_extraction)
        self.assertIsInstance(status.value, dict)

    def test_partial_json_is_detected(self):
        text = (
            "model says:\n"
            '{"hypotheses":[],"numeric_guidance":{},"tool_request":{"name":"none"},'
            '"action_preference":"auto","convergence":{"decision":"continue","confidence":0.8}}'
            "\nthanks"
        )
        status = _parse_llm_json_status(text)
        self.assertFalse(status.raw_is_valid_json)
        self.assertTrue(status.used_partial_extraction)
        self.assertIsInstance(status.value, dict)

    def test_invalid_json_returns_empty_dict(self):
        status = _parse_llm_json_status("not a json payload")
        self.assertFalse(status.raw_is_valid_json)
        self.assertFalse(status.used_partial_extraction)
        self.assertEqual(status.value, {})


if __name__ == "__main__":
    unittest.main()
