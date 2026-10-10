"""Reject undefined penalty scales before expensive inference begins."""

import pytest
from rfd3.inference.symmetry.graph_interface_guidance import (
    GraphInterfaceGuidanceConfig,
)
from rfd3.inference.symmetry.scaffold_core_guidance import ScaffoldCoreGuidanceConfig


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("factory, name", [
    (ScaffoldCoreGuidanceConfig, "clash_weight"),
    (ScaffoldCoreGuidanceConfig, "routing_ownership_weight"),
    (ScaffoldCoreGuidanceConfig, "backbone_tolerance"),
    (GraphInterfaceGuidanceConfig, "weight"),
    (GraphInterfaceGuidanceConfig, "maximum_source_regression_absolute"),
    (GraphInterfaceGuidanceConfig, "capture_ca_distance"),
])
def test_nonfinite_weights_and_geometry_limits_fail_at_configuration(factory, name, value):
    with pytest.raises(ValueError, match=f"{name} must be finite"):
        factory(**{name: value})


@pytest.mark.parametrize("factory, name", [
    (ScaffoldCoreGuidanceConfig, "line_search_steps"),
    (GraphInterfaceGuidanceConfig, "line_search_steps"),
    (GraphInterfaceGuidanceConfig, "patch_blend_radius"),
])
def test_fractional_iteration_counts_do_not_reach_runtime_range(factory, name):
    with pytest.raises(ValueError, match="must be an integer"):
        factory(**{name: 1.5})
