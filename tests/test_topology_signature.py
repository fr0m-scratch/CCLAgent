import unittest

from src.observability.topology import build_topology_signature


class TestTopologySignature(unittest.TestCase):
    def test_build_signature(self):
        topo = """
        GPU0\tGPU1\tGPU2\tCPU Affinity
        GPU0\tX\tNV2\tPHB\t0-31
        GPU1\tNV2\tX\tPHB\t0-31
        GPU2\tPHB\tPHB\tX\t0-31
        """
        ib = """
CA 'mlx5_0'
  Port 1:
    State: Active
CA 'mlx5_1'
  Port 1:
    State: Active
"""
        sig = build_topology_signature(topo_text=topo, ib_text=ib)
        payload = sig.to_dict()
        self.assertEqual(payload["gpu_count"], 3)
        self.assertEqual(payload["nic_count"], 2)
        self.assertTrue(payload["nvlink_matrix_hash"])
        self.assertTrue(payload["numa_layout_hash"])
        self.assertTrue(payload["pcie_tree_hash"])


if __name__ == "__main__":
    unittest.main()
