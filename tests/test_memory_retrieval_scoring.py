import tempfile
import unittest

from src.memory import MemoryStore
from src.types import ContextSignature, MemoryConfig


class MemoryRetrievalTest(unittest.TestCase):
    def test_retrieve_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = MemoryStore(MemoryConfig(path=f"{tmp}/memory.json"))
            context = ContextSignature(
                workload="w",
                workload_kind="train",
                topology="t",
                scale="s",
                nodes=1,
            )
            mem.add_rule(context, {"NCCL_ALGO": "RING"}, 0.1)
            rules = mem.retrieve_rules(context, top_k=1)
            self.assertEqual(len(rules), 1)


if __name__ == "__main__":
    unittest.main()
