"""Exact diagnostic capture must never alter native proposal decisions."""

import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rfd3.inference.symmetry.exact_reference_trace import (
    MAX_TOTAL_BYTES,
    ExactReferenceTrace,
    load_reference_trace,
    replay_reference_trace,
)
from rfd3.inference.symmetry.scaffold_transport import ScaffoldReferenceTransport
from test_reference_transport import transport_fixture


def setup_case(monkeypatch, directory=None, dtype=torch.float64):
    monkeypatch.delenv('MOSAIC_TRANSPORT_TRACE_DIR', raising=False)
    monkeypatch.delenv('MOSAIC_TRANSPORT_TRACE_LIMIT', raising=False)
    if directory is not None:
        monkeypatch.setenv('MOSAIC_TRANSPORT_TRACE_DIR', str(directory))
    _, _, _, features, topology = transport_fixture()
    runtime = ScaffoldReferenceTransport(features, topology)
    original = features['motif_pos'][None].to(dtype).clone()
    candidate = original.clone()
    generated = topology.generated_atom_mask.nonzero().flatten()
    candidate[0, generated[5], 0] += 2.
    proposal = SimpleNamespace(target=original.clone(), coordinates=candidate)
    return runtime, original, proposal


def reject(runtime, original, proposal):
    with pytest.raises(ValueError, match='regresses the complete-scaffold contract') as error:
        runtime.prepare(original, proposal, projector=lambda candidate, target: candidate)
    return str(error.value)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_capture_replays_exact_cpu_guard_without_consuming_rng(monkeypatch, tmp_path, dtype):
    runtime, original, proposal = setup_case(monkeypatch, tmp_path/'trace', dtype)
    before = original.clone()
    target = proposal.target.clone()
    candidate = proposal.coordinates.clone()
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    reject(runtime, original, proposal)
    assert torch.equal(torch.get_rng_state(), torch_state)
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert random.getstate() == python_state
    assert torch.equal(original, before)
    assert torch.equal(proposal.target, target)
    assert torch.equal(proposal.coordinates, candidate)
    assert runtime.accepted == []

    path = next((tmp_path/'trace').glob('capture-*.npz'))
    captured = load_reference_trace(path)
    assert torch.equal(captured['coordinates_before'], original[0])
    assert captured['coordinates_before'].dtype == dtype
    info = captured['metadata']['tensor_descriptors']['coordinates_before']
    assert info['dtype'] == str(dtype) and info['device'] == 'cpu'
    assert captured['reference_after'].contract['residues'][0]['reference_backbone']
    replay = replay_reference_trace(path)
    assert replay['saved_guard'] == replay['replayed_guard']
    assert replay['maximum_before_residual_difference'] == 0.
    assert replay['maximum_after_residual_difference'] == 0.
    assert not replay['replayed_guard']['passed']
    assert not replay_reference_trace(path, dtype=torch.float64)['replayed_guard']['passed']
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype == np.uint8 for key in archive.files)
    manifest = json.loads((path.parent/'manifest.json').read_text())
    descriptor = manifest['snapshots'][0]
    assert descriptor['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert descriptor['bytes'] == path.stat().st_size
    diagnostic = runtime.diagnostics()['exact_reference_trace']
    assert diagnostic['captured'] == 1 and diagnostic['capture_failures'] == 0
    assert diagnostic['manifest_sha256_after_last_capture'] == hashlib.sha256((path.parent/'manifest.json').read_bytes()).hexdigest()


def test_capture_disabled_has_no_io_and_same_rejection(monkeypatch, tmp_path):
    disabled, original, proposal = setup_case(monkeypatch)
    message = reject(disabled, original, proposal)
    assert list(tmp_path.iterdir()) == []
    assert disabled.diagnostics()['exact_reference_trace'] == {'enabled': False}
    enabled, original, proposal = setup_case(monkeypatch, tmp_path/'trace')
    assert reject(enabled, original, proposal) == message


def test_valid_transaction_unchanged_and_not_captured(monkeypatch, tmp_path):
    results = []
    for path in (None, tmp_path/'trace'):
        runtime, original, _ = setup_case(monkeypatch, path)
        target = original.clone()
        target[:, :88, 0] += .1
        target[:, 88:, 0] -= .1
        prepared = runtime.prepare(original, SimpleNamespace(target=target, coordinates=None), projector=lambda candidate, target: candidate)
        runtime.commit(prepared, progress=.2)
        results.append(prepared)
    assert torch.equal(results[0][0], results[1][0])
    assert results[0][1].contract == results[1][1].contract
    assert list(tmp_path.iterdir()) == []


def test_capture_directory_bound_applies_across_transport_instances(monkeypatch, tmp_path):
    path = tmp_path/'trace'
    first, original, proposal = setup_case(monkeypatch, path)
    for _ in range(6):
        reject(first, original, proposal)
    second, original, proposal = setup_case(monkeypatch, path)
    def unexpected_serialization(*args, **kwargs):
        pytest.fail('A full shared trace directory must not serialize again')
    monkeypatch.setattr(np, 'savez_compressed', unexpected_serialization)
    monkeypatch.setattr(second._exact_reference_trace, '_write', unexpected_serialization)
    for _ in range(4):
        reject(second, original, proposal)
    assert len(list(path.glob('capture-*.npz'))) == 3
    manifest = json.loads((path/'manifest.json').read_text())
    assert len(manifest['snapshots']) == 3
    assert manifest['total_snapshot_bytes'] == sum(p.stat().st_size for p in path.glob('capture-*.npz'))
    assert manifest['total_snapshot_bytes'] <= MAX_TOTAL_BYTES
    assert first.diagnostics()['exact_reference_trace']['skipped_due_to_limit'] == 3
    assert second.diagnostics()['exact_reference_trace']['skipped_due_to_limit'] == 4
    assert second.diagnostics()['exact_reference_trace']['capture_exhausted']


def test_byte_budget_and_io_failure_do_not_change_rejection(monkeypatch, tmp_path):
    baseline, original, proposal = setup_case(monkeypatch)
    message = reject(baseline, original, proposal)
    runtime, original, proposal = setup_case(monkeypatch, tmp_path/'too_small')
    runtime._exact_reference_trace.max_bytes = 1
    assert reject(runtime, original, proposal) == message
    assert runtime.diagnostics()['exact_reference_trace']['skipped_due_to_limit'] == 1
    assert not (tmp_path/'too_small').exists()
    file_path = tmp_path/'not_a_directory'
    file_path.write_text('occupied')
    runtime, original, proposal = setup_case(monkeypatch, file_path)
    assert reject(runtime, original, proposal) == message
    assert file_path.read_text() == 'occupied'
    assert runtime.diagnostics()['exact_reference_trace']['capture_failures'] == 1
    for _ in range(3):
        assert reject(runtime, original, proposal) == message
    assert runtime.diagnostics()['exact_reference_trace']['capture_failures'] == 1
    assert runtime.diagnostics()['exact_reference_trace']['skipped_after_capture_failure'] == 3


@pytest.mark.parametrize('limit', ['0', '-1', '4', 'NaN', '2.0'])
def test_invalid_capture_limit_is_diagnostic_only(monkeypatch, tmp_path, limit):
    runtime, original, proposal = setup_case(monkeypatch, tmp_path/'trace')
    monkeypatch.setenv('MOSAIC_TRANSPORT_TRACE_LIMIT', limit)
    runtime._exact_reference_trace = ExactReferenceTrace.from_environment()
    reject(runtime, original, proposal)
    assert runtime.diagnostics()['exact_reference_trace']['capture_failures'] == 1
    assert not (tmp_path/'trace').exists()


def test_constructor_hard_caps_cannot_be_exceeded(tmp_path):
    with pytest.raises(ValueError, match='limit'):
        ExactReferenceTrace(tmp_path, limit=4)
    with pytest.raises(ValueError, match='byte budget'):
        ExactReferenceTrace(tmp_path, max_bytes=MAX_TOTAL_BYTES+1)
