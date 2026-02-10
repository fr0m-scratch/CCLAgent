import unittest

from src.models.digital_twin import DigitalTwinModel
from src.types import ContextSignature, NCCLConfig


class TestDigitalTwinModel(unittest.TestCase):
    def test_estimate_and_update(self):
        twin = DigitalTwinModel(ema_alpha=0.5)
        ctx = ContextSignature(
            workload="w",
            workload_kind="workload",
            topology="ib",
            scale="small",
            nodes=2,
            gpus_per_node=8,
        )
        cfg = NCCLConfig(params={"NCCL_ALGO": "TREE", "NCCL_PROTO": "LL"})
        pred_before = twin.estimate(
            config=cfg,
            surrogate_mean_ms=1000.0,
            surrogate_std_ms=10.0,
            context=ctx,
            profiler_summary={"p50_dur_us": 100.0, "p95_dur_us": 300.0},
            topology_signature={"nic_count": 1, "gpu_count": 8},
        )
        self.assertGreater(pred_before.mean_ms, 0.0)

        twin.update(observed_ms=900.0, predicted_ms=pred_before.mean_ms)
        pred_after = twin.estimate(
            config=cfg,
            surrogate_mean_ms=1000.0,
            surrogate_std_ms=10.0,
            context=ctx,
        )
        self.assertNotEqual(pred_before.mean_ms, pred_after.mean_ms)


if __name__ == "__main__":
    unittest.main()
