import unittest

from src.observability.nccl_debug import NCCLDebugConfig, NCCLDebugParser


class TestNCCLDebugParser(unittest.TestCase):
    def test_env_overrides(self):
        cfg = NCCLDebugConfig(
            level="INFO",
            subsystems=["INIT", "GRAPH", "NET"],
            topo_dump_file="/tmp/topo.xml",
            graph_dump_file="/tmp/graph.xml",
        )
        env = cfg.build_env_overrides()
        self.assertEqual(env["NCCL_DEBUG"], "INFO")
        self.assertEqual(env["NCCL_DEBUG_SUBSYS"], "INIT,GRAPH,NET")
        self.assertEqual(env["NCCL_TOPO_DUMP_FILE"], "/tmp/topo.xml")
        self.assertEqual(env["NCCL_GRAPH_DUMP_FILE"], "/tmp/graph.xml")

    def test_parse_counts_algo_proto_channels(self):
        raw = """
NCCL INFO INIT : something
NCCL INFO GRAPH : Algo TREE Proto LL nChannels 8
NCCL WARN NET : timeout warning
NCCL INFO TUNE : algorithm RING protocol LL128 nChannels=4
"""
        parser = NCCLDebugParser()
        summary = parser.parse(raw)
        self.assertEqual(summary["line_count"], 4)
        self.assertEqual(summary["levels"]["INFO"], 3)
        self.assertEqual(summary["levels"]["WARN"], 1)
        self.assertEqual(summary["algo_hits"]["TREE"], 1)
        self.assertEqual(summary["algo_hits"]["RING"], 1)
        self.assertEqual(summary["proto_hits"]["LL"], 1)
        self.assertEqual(summary["proto_hits"]["LL128"], 1)
        ch = summary["channel_observations"]
        self.assertEqual(ch["count"], 2)
        self.assertEqual(ch["min"], 4)
        self.assertEqual(ch["max"], 8)


if __name__ == "__main__":
    unittest.main()
