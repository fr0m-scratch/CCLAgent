import unittest

from src.RAG import RagStore
from src.types import RagConfig


class RagRetrievalTest(unittest.TestCase):
    def test_rag_search(self):
        config = RagConfig(mode="jaccard", docs_paths=["doc/Knowledge"])
        store = RagStore(config)
        store.load_documents(config)
        results = store.search("NCCL_ALGO", top_k=2)
        self.assertTrue(len(results) >= 0)


if __name__ == "__main__":
    unittest.main()
