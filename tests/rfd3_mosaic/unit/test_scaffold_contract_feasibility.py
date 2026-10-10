import sys
from pathlib import Path
import pytest
import torch
from test_reference_transport import transport_fixture
from rfd3.inference.symmetry.contract_feasibility import project_explicit_scaffold_feasibility, backbone_physical_nonregression_guard


def fixture():
    _,xyz,_,f,topology=transport_fixture()
    source=torch.tensor(xyz,dtype=torch.float64).reshape(1,-1,3)
    fixed=~topology.generated_atom_mask
    def projector(candidate):
        result=candidate.clone();result[:,fixed]=source[:,fixed];return result
    return source,topology,projector,fixed


def test_valid_state_is_exact_noop_and_preserves_rng():
    source,topology,_,_=fixture();rng=torch.random.get_rng_state().clone()
    out,record=project_explicit_scaffold_feasibility(source,topology,projector=lambda x:x)
    assert out is source and record['applied'] is False
    assert torch.equal(torch.random.get_rng_state(),rng)


@pytest.mark.parametrize('kind',['CA_envelope','N_CA_length','peptide_length'])
def test_actual_stored_backbone_checked_after_projector(kind):
    source,topology,projector,fixed=fixture();candidate=source.clone()
    if kind=='CA_envelope':candidate[:,8:12]+=torch.tensor([2.0,0,0])
    elif kind=='N_CA_length':candidate[:,12,0]+=0.3
    else:candidate[:,15,0]+=0.3
    before=candidate.clone();rng=torch.random.get_rng_state().clone()
    output,record=project_explicit_scaffold_feasibility(candidate,topology,projector=projector)
    assert record['passed'] and record['applied']
    assert torch.equal(output[:,fixed],source[:,fixed])
    assert torch.equal(candidate,before) and torch.equal(torch.random.get_rng_state(),rng)
    _,second=project_explicit_scaffold_feasibility(output,topology,projector=lambda x:x)
    assert second['applied'] is False


def test_infeasible_fixed_projector_is_explicit_failure_and_input_unchanged():
    source,topology,_,fixed=fixture();source[:,8:12]+=2.0;before=source.clone()
    def impossible(candidate):
        result=candidate.clone();result[:,fixed]+=10.0;return result
    with pytest.raises(ValueError,match='infeasible'):
        project_explicit_scaffold_feasibility(source,topology,projector=impossible)
    assert torch.equal(source,before)


def test_nonfinite_input_fails_without_projector():
    source,topology,_,_=fixture();source[0,8,0]=float('nan')
    def should_not_run(candidate):raise AssertionError('projector called')
    with pytest.raises(ValueError,match='finite'):
        project_explicit_scaffold_feasibility(source,topology,projector=should_not_run)


def test_full_backbone_guard_rejects_real_submicro_peptide_violation():
    source,topology,_,_=fixture()
    guard=backbone_physical_nonregression_guard(source,topology)
    candidate=source.clone()
    # C of residue2 to N of residue3; residue3 is generated.
    direction=candidate[0,8]-candidate[0,6]
    length=torch.linalg.vector_norm(direction)
    candidate[0,8]+=direction/length*(1.329+0.1+1e-7-length)
    checked=guard(candidate)
    assert checked['baseline_passed'] and not checked['candidate_passed']
    assert checked['passed'] is False
    assert guard(source)['passed'] is True


def test_full_backbone_guard_rejects_nonlocal_backbone_collision():
    source,topology,_,_=fixture();guard=backbone_physical_nonregression_guard(source,topology)
    candidate=source.clone();candidate[0,12]=candidate[0,3]
    checked=guard(candidate)
    assert not checked['passed'] and checked['candidate_backbone_clashes']>0


@pytest.mark.parametrize('iterations,fraction',[(-1,0.1),(True,0.1),(65,0.1),(16,float('nan'))])
def test_invalid_search_bounds_do_not_modify_input(iterations,fraction):
    source,topology,_,_=fixture();before=source.clone()
    with pytest.raises(ValueError,match='search limits'):
        project_explicit_scaffold_feasibility(source,topology,projector=lambda x:x,iterations=iterations,minimum_fraction=fraction)
    assert torch.equal(source,before)
