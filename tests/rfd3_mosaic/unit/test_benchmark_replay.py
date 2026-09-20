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
