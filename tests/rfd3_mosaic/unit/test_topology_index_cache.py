"""Dependency invalidation must include values, aliases and metadata."""
import unittest
from unittest.mock import Mock
import torch
from rfd3.inference.symmetry.topology_index_cache import TopologyIndexCache

class TopologyIndexCacheTest(unittest.TestCase):
    def test_values_aliases_shape_dtype_device_and_metadata(self):
        cache = TopologyIndexCache()
        build = Mock(side_effect=lambda: object())
        x = torch.tensor([1, 2])
        original = cache.get_or_compute((x,), ('chain_A', 8), build)
        self.assertIs(cache.get_or_compute((x.clone(),), ('chain_A', 8), build), original)
        x.data[0] = 7  # Deliberately bypasses tensor version counters.
        mutated = cache.get_or_compute((x,), ('chain_A', 8), build)
        self.assertIsNot(mutated, original)
        last = mutated
        for tensor, metadata in (
            (x.reshape(1, 2), ('chain_A', 8)),
            (x.float(), ('chain_A', 8)),
            (x, ('chain_B', 8)),
            (x, ('chain_B', 9)),
            (x.to('meta'), ('chain_B', 9)),
            (x, ('chain_B', 9)),
        ):
            result = cache.get_or_compute((tensor,), metadata, build)
            self.assertIsNot(result, last)
            last = result
        self.assertEqual(build.call_count, 8)

    def test_inference_tensor_and_failed_build(self):
        cache = TopologyIndexCache()
        with torch.inference_mode():
            x = torch.tensor([1, 2])
            a = cache.get_or_compute((x,), (), lambda: 'original')
            self.assertEqual(a, 'original')
            x[0] = 5
            with self.assertRaisesRegex(ValueError, 'failed build'):
                cache.get_or_compute((x,), (), lambda: (_ for _ in ()).throw(ValueError('failed build')))
            self.assertEqual(cache.get_or_compute((x,), (), lambda: 'rebuilt'), 'rebuilt')

    def test_separate_instances_do_not_share_entries(self):
        a, b = TopologyIndexCache(), TopologyIndexCache()
        x = torch.tensor([1])
        self.assertEqual(a.get_or_compute((x,), (), lambda: 1), 1)
        self.assertEqual(b.get_or_compute((x,), (), lambda: 2), 2)
