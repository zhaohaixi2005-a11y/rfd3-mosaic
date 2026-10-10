"""A real improving graph proposal must not fail on invariant CA distances."""

from dataclasses import replace

import pytest
import torch
from rfd3.inference.symmetry.graph_interface_guidance import (
    GraphInterfaceEdge,
    GraphInterfaceGuidanceConfig,
    GraphInterfaceTopology,
    graph_interface_energy,
    graph_interface_proposal_acceptable,
)


def precision_graph(*, safety_only=False):
    n, count = 30, 62
    coordinates = torch.zeros(count, 3)
    coordinates[:n, 1] = torch.arange(n) * 3.8
    coordinates[n:2*n] = coordinates[:n]
    coordinates[n:2*n, 0] = 3.
    coordinates[-2] = torch.tensor([0., 0., 30.])
    coordinates[-1] = torch.tensor([20., 0., 30.])

    def edge(name, left, right):
        left_mask = torch.zeros(count, dtype=torch.bool)
        right_mask = left_mask.clone()
        left_mask[left] = True
        right_mask[right] = True
        return GraphInterfaceEdge(
            name, name, left_mask, right_mask, torch.tensor(left), torch.tensor(right),
            1, 0, 0, False, 5.5, None, None,
        )

    near = edge('near', list(range(n)), list(range(n, 2*n)))
    far = edge('far', [2*n], [2*n+1])
    topology = GraphInterfaceTopology(
        (far,) if safety_only else (near, far),
        torch.ones(count, dtype=torch.bool),
    )
    if safety_only:
        topology = replace(
            topology,
            guided_ca_mask=near.left_generated_ca_mask,
            safety_ca_mask=near.right_generated_ca_mask,
            safety_exclusions=torch.zeros(n, n, dtype=torch.bool),
        )
    config = GraphInterfaceGuidanceConfig(
        coverage_weight=0., continuity_weight=0., orientation_weight=0.,
        shape_weight=0., backbone_weight=0., interface_balance_weight=0.,
        patch_exclusivity_weight=0., clash_weight=0., distance_weight=0., pairs_per_edge=1,
    )
    return coordinates, topology, config


@pytest.mark.parametrize('safety_only', [False, True])
def test_improving_graph_trade_preserves_invariant_edge_or_global_guard(safety_only):
    coordinates, topology, config = precision_graph(safety_only=safety_only)
    candidate = coordinates + (0.125 if safety_only else 0.1)
    candidate[-1, 0] -= 1.
    before = graph_interface_energy(coordinates, topology, config)
    after = graph_interface_energy(candidate, topology, config)
    decision = {}
    assert after.total < before.total
    assert graph_interface_proposal_acceptable(before, after, config, decision=decision), decision
    # Compare the same stored coordinate values in double, not an idealized
    # translation that bypasses float32 input quantization.
    reference_before = graph_interface_energy(coordinates.double(), topology, config)
    reference_after = graph_interface_energy(candidate.double(), topology, config)
    assert graph_interface_proposal_acceptable(reference_before, reference_after, config)
    torch.testing.assert_close(after.minimum_distances.double(), reference_after.minimum_distances, atol=5e-7, rtol=0.)
    assert before.total.dtype == after.total.dtype == coordinates.dtype
    assert before.total.device == after.total.device == coordinates.device
    differentiated = candidate.clone().requires_grad_(True)
    gradient = torch.autograd.grad(graph_interface_energy(differentiated, topology, config).total, differentiated)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    assert torch.linalg.vector_norm(gradient.sum(dim=0)) < 1e-6
    tolerances = {c['rule']: c.get('tolerance') for c in decision['checks']}
    assert tolerances['interface_minimum_distance'] == 1e-6
    assert tolerances['global_minimum_distance'] == 1e-6
    assert tolerances['global_clash_regression'] == 1e-8


@pytest.mark.parametrize('safety_only', [False, True])
def test_energy_gain_cannot_buy_real_edge_or_global_clash_worsening(safety_only):
    coordinates, topology, config = precision_graph(safety_only=safety_only)
    candidate = coordinates.clone()
    candidate[-1, 0] -= 1.
    candidate[30:60, 0] -= 5e-6
    before = graph_interface_energy(coordinates, topology, config)
    after = graph_interface_energy(candidate, topology, config)
    decision = {}
    assert after.total < before.total
    assert not graph_interface_proposal_acceptable(before, after, config, decision=decision)
    assert decision['first_rejection_reason'] == (
        'global_minimum_distance' if safety_only else 'interface_minimum_distance'
    )
