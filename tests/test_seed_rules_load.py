"""Tests for seed rule loading and injection into memory."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.memory.seed import load_seed_rules, inject_seed_rules
from src.memory.schema import Rule
from src.memory import MemoryStore
from src.types import MemoryConfig, ContextSignature


SEED_RULES_PATH = Path(__file__).resolve().parent.parent / "memory" / "seed_rules.json"


class TestLoadSeedRules(unittest.TestCase):
    """Test loading seed rules from the JSON file."""

    def test_load_default_path(self):
        """Seed rules load from the default path."""
        rules = load_seed_rules()
        self.assertGreater(len(rules), 0, "Expected at least one seed rule")

    def test_load_explicit_path(self):
        """Seed rules load from an explicit path."""
        rules = load_seed_rules(SEED_RULES_PATH)
        self.assertGreater(len(rules), 20, "Expected 20+ seed rules")

    def test_all_rules_are_rule_instances(self):
        """Every loaded seed rule is a Rule dataclass."""
        rules = load_seed_rules()
        for rule in rules:
            self.assertIsInstance(rule, Rule)

    def test_required_fields_present(self):
        """Every seed rule has the required fields populated."""
        rules = load_seed_rules()
        for rule in rules:
            self.assertTrue(rule.id, f"Rule missing id: {rule}")
            self.assertIsInstance(rule.context, dict)
            self.assertIsInstance(rule.config_patch, dict)
            self.assertIn(rule.rule_type, ("positive", "avoid"))
            self.assertGreater(rule.confidence, 0.0)
            self.assertLessEqual(rule.confidence, 1.0)

    def test_positive_rules_have_config_patches(self):
        """Positive rules have non-empty config patches."""
        rules = load_seed_rules()
        positive = [r for r in rules if r.rule_type == "positive"]
        self.assertGreater(len(positive), 0)
        for rule in positive:
            self.assertTrue(rule.config_patch, f"Positive rule {rule.id} has empty config_patch")

    def test_avoid_rules_exist(self):
        """At least some avoid rules are present."""
        rules = load_seed_rules()
        avoid = [r for r in rules if r.rule_type == "avoid"]
        self.assertGreater(len(avoid), 0, "Expected at least one avoid rule")

    def test_evidence_contains_mechanism(self):
        """Every seed rule has evidence with a mechanism explanation."""
        rules = load_seed_rules()
        for rule in rules:
            self.assertIn("mechanism", rule.evidence, f"Rule {rule.id} missing evidence.mechanism")
            self.assertTrue(rule.evidence["mechanism"], f"Rule {rule.id} has empty mechanism")

    def test_source_is_seed(self):
        """All seed rules have source indicating they are seed knowledge."""
        rules = load_seed_rules()
        for rule in rules:
            self.assertEqual(rule.source, "seed_expert_knowledge", f"Rule {rule.id} has wrong source")

    def test_tries_and_wins_zero(self):
        """Seed rules start with zero tries and wins."""
        rules = load_seed_rules()
        for rule in rules:
            self.assertEqual(rule.tries, 0, f"Rule {rule.id} has non-zero tries")
            self.assertEqual(rule.wins, 0, f"Rule {rule.id} has non-zero wins")

    def test_unique_ids(self):
        """All seed rule IDs are unique."""
        rules = load_seed_rules()
        ids = [r.id for r in rules]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate rule IDs found")

    def test_missing_file_returns_empty(self):
        """Loading from a non-existent file returns empty list."""
        rules = load_seed_rules("/nonexistent/path/seed_rules.json")
        self.assertEqual(rules, [])

    def test_malformed_json_returns_empty(self):
        """Loading from a malformed JSON file returns empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{not valid json")
            tmp = f.name
        try:
            rules = load_seed_rules(tmp)
            self.assertEqual(rules, [])
        finally:
            os.unlink(tmp)

    def test_non_array_json_returns_empty(self):
        """Loading from a JSON file that isn't an array returns empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"not": "an_array"}, f)
            tmp = f.name
        try:
            rules = load_seed_rules(tmp)
            self.assertEqual(rules, [])
        finally:
            os.unlink(tmp)


class TestInjectSeedRules(unittest.TestCase):
    """Test injecting seed rules into a MemoryStore."""

    def _make_memory(self):
        """Create a MemoryStore backed by a temporary file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.write(b"{}")
        tmp.close()
        config = MemoryConfig(path=tmp.name)
        return MemoryStore(config), tmp.name

    def test_inject_into_empty_memory(self):
        """Seed rules are injected when memory is empty."""
        store, path = self._make_memory()
        try:
            self.assertEqual(len(store.rules), 0)
            count = inject_seed_rules(store)
            self.assertGreater(count, 20)
            self.assertEqual(len(store.rules), count)
        finally:
            os.unlink(path)

    def test_skip_injection_when_rules_exist(self):
        """Seed rules are NOT injected when memory already has rules."""
        store, path = self._make_memory()
        try:
            store.rules.append(Rule(
                id="existing",
                context={},
                config_patch={"NCCL_ALGO": "RING"},
                improvement=5.0,
            ))
            count = inject_seed_rules(store)
            self.assertEqual(count, 0)
            self.assertEqual(len(store.rules), 1)
        finally:
            os.unlink(path)

    def test_injected_rules_retrievable(self):
        """Injected seed rules can be retrieved by context matching."""
        store, path = self._make_memory()
        try:
            inject_seed_rules(store)
            context = ContextSignature(
                workload="allreduce_benchmark",
                workload_kind="allreduce",
                topology="nvlink",
                scale="single_node",
                nodes=1,
            )
            rules = store.retrieve_rules(context, top_k=10)
            self.assertGreater(len(rules), 0, "Expected seed rules to be retrievable by context")
        finally:
            os.unlink(path)

    def test_avoid_rules_retrievable(self):
        """Injected avoid rules can be retrieved with include_negative."""
        store, path = self._make_memory()
        try:
            inject_seed_rules(store)
            context = ContextSignature(
                workload="benchmark",
                workload_kind="allreduce",
                topology="pcie",
                scale="single_node",
                nodes=1,
            )
            rules = store.retrieve_rules(context, top_k=10, include_negative=True)
            avoid = [r for r in rules if r.rule_type == "avoid"]
            self.assertGreater(len(avoid), 0, "Expected avoid rules to be retrievable")
        finally:
            os.unlink(path)


class TestSeedRulesJsonValid(unittest.TestCase):
    """Validate the seed_rules.json file structure directly."""

    def test_json_is_valid(self):
        """seed_rules.json is valid JSON."""
        with open(SEED_RULES_PATH, "r") as f:
            data = json.load(f)
        self.assertIsInstance(data, list)

    def test_config_patches_use_known_params(self):
        """Config patches only reference known NCCL parameters."""
        known = {
            "NCCL_ALGO", "NCCL_PROTO", "NCCL_NTHREADS", "NCCL_BUFFSIZE",
            "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_P2P_LEVEL",
            "NCCL_NET_GDR_LEVEL", "NCCL_SOCKET_NTHREADS", "NCCL_NSOCKS_PERTHREAD",
            "NCCL_IB_QPS_PER_CONNECTION", "NCCL_SHM_DISABLE", "NCCL_CROSS_NIC",
            "NCCL_NVLS_ENABLE", "NCCL_MIN_CTAS", "NCCL_MAX_CTAS", "NCCL_IB_TIMEOUT",
        }
        with open(SEED_RULES_PATH, "r") as f:
            data = json.load(f)
        for entry in data:
            for param in entry.get("config_patch", {}):
                self.assertIn(param, known, f"Unknown param '{param}' in rule {entry['id']}")

    def test_context_uses_known_fields(self):
        """Context dicts only use fields from ContextSignature."""
        known = {
            "workload", "workload_kind", "topology", "scale", "nodes",
            "model", "framework", "gpus_per_node", "gpu_type", "network",
            "nic_count", "extra",
        }
        with open(SEED_RULES_PATH, "r") as f:
            data = json.load(f)
        for entry in data:
            for field in entry.get("context", {}):
                self.assertIn(field, known, f"Unknown context field '{field}' in rule {entry['id']}")


if __name__ == "__main__":
    unittest.main()
