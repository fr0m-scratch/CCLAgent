import unittest

from src.config import default_agent_config


class ConfigValidationTest(unittest.TestCase):
    def test_default_config_valid(self):
        cfg = default_agent_config()
        params = cfg.parameter_space.default_config()
        errors = cfg.parameter_space.validate(params)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
