import math
import unittest
import torch
from rfd3.inference.symmetry.scalar_metrics import detached_scalar_metrics


class ScalarMetricsTest(unittest.TestCase):
    def test_values_order_and_graph_are_preserved(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            x = torch.tensor([0., -0., .1234567, 1e3, float('inf'), float('-inf'), float('nan')], dtype=dtype, requires_grad=True)
            mapping = {str(i): value for i, value in enumerate(x)}
            expected = {key: float(value.detach().cpu().item()) for key, value in mapping.items()}
            result = detached_scalar_metrics(mapping)
            self.assertEqual(list(result), list(expected))
            for key, value in expected.items():
                if math.isnan(value): self.assertTrue(math.isnan(result[key]))
                else:
                    self.assertEqual(result[key], value)
                    self.assertEqual(math.copysign(1, result[key]), math.copysign(1, value))
            x[:4].sum().backward()
            self.assertTrue(torch.equal(x.grad[:4], torch.ones_like(x[:4])))

    def test_heterogeneous_and_empty_use_exact_legacy_values(self):
        values = {'integer': torch.tensor(2**55+3), 'float': torch.tensor(.1), 'double': torch.tensor(.1,dtype=torch.float64)}
        self.assertEqual(detached_scalar_metrics(values), {k:float(v.item()) for k,v in values.items()})
        self.assertEqual(detached_scalar_metrics({}), {})
