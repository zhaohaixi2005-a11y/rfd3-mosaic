import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[3] / "scripts/rfd3_mosaic/replay_benchmark.py"
spec = importlib.util.spec_from_file_location("benchmark_replay", path)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class BenchmarkReplayTests(unittest.TestCase):
    def test_canary_and_bulk_preserve_every_original_design_and_seed_once(self):
        for count in (1, 2, 50, 1000):
            for size in (1, 2, 5, 50):
                with self.subTest(count=count, size=size):
                    shards = replay.partition_designs(count, size)
                    self.assertEqual(shards[0], (0, 1))
                    indices = [start + local for start, stop in shards
                               for local in range(stop - start)]
                    self.assertEqual(indices, list(range(count)))
                    self.assertTrue(all(stop - start <= size for start, stop in shards))
                    self.assertEqual([9170000 + i for i in indices],
                                     list(range(9170000, 9170000 + count)))

    def test_invalid_budgets_fail_before_submission(self):
        for count, size in ((0, 5), (50, 0), (-1, 5), (50, -1)):
            with self.assertRaises(ValueError):
                replay.partition_designs(count, size)

    def test_runtime_grouping_preserves_shards_and_respects_time_budget(self):
        rows = [{"task": "slow" if i < 4 else "fast", "designs": 2,
                 "script": str(i)} for i in range(12)]
        groups = replay.group_by_runtime(rows, {"slow": 3600, "fast": 300}, 12000)
        self.assertEqual(sorted(r["script"] for g in groups for r in g["rows"]),
                         sorted(r["script"] for r in rows))
        self.assertTrue(all(g["estimated_seconds"] <= 12000 for g in groups))
        self.assertLess(len(groups), len(rows))

    def test_impossible_or_unknown_runtime_budget_fails_explicitly(self):
        rows = [{"task": "slow", "designs": 5, "script": "job"}]
        for estimate in (0, -1, float("nan"), float("inf"), 10000):
            with self.assertRaises(ValueError):
                replay.group_by_runtime(rows, {"slow": estimate}, 1000)
        with self.assertRaises(KeyError):
            replay.group_by_runtime(rows, {}, 1000)
