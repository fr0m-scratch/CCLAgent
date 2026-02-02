import unittest

from src.config import default_agent_config
from src.tools.config_compiler import ConfigCompiler
from src.types import NCCLConfig


class ConfigCompilerEnvTest(unittest.TestCase):
    def test_env_compilation(self):
        cfg = default_agent_config()
        compiler = ConfigCompiler(cfg.parameter_space)
        nccl_cfg = NCCLConfig(params=cfg.parameter_space.default_config())
        result = compiler.compile(nccl_cfg)
        self.assertTrue(result.ok)
        self.assertTrue("NCCL_ALGO" in result.env)


if __name__ == "__main__":
    unittest.main()
