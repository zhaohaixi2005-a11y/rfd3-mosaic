"""Partial batch accounting must be by frozen identity, not file count."""

import gzip
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rfd3_mosaic import experiment_worker as worker
from rfd3_mosaic.run_index import read_run_record, rebuild_run_index, update_run_state
from rfd3_mosaic.sampling_plan import DesignSamplingAssignment


def assignments(count=3):
    records = tuple(
        DesignSamplingAssignment(
            design_index=i,
            pose_index=0,
            replicate_index=i,
            pose_seed=None,
            diffusion_seed=10 + i,
        )
        for i in range(count)
    )
    return {worker._design_example_id(item): item for item in records}


def output(root, example_id, *, prefix="rfd3_input_", malformed=False, structure=True):
    path = root / f"{prefix}{example_id}_0_model_0.json"
    path.write_text("bad JSON" if malformed else json.dumps({"sample": example_id}))
    if structure:
        with gzip.open(path.with_suffix(".cif.gz"), "wt") as handle:
            handle.write("data_test\n#\n")
    return path


def test_partial_outputs_remain_auditable_and_all_designs_accounted_for(tmp_path):
    expected = assignments()
    keys = list(expected)
    ledger = worker._initial_outcome_ledger(tuple(expected.values()))
    ledger["designs"][keys[0]]["generation_status"] = "generated"
    ledger["designs"][keys[1]]["generation_status"] = "failed"
    ledger["designs"][keys[1]]["error"] = "geometry rejected"
    worker._atomic_json(tmp_path / "design_outcomes.json", ledger)
    committed = output(tmp_path, keys[0])
    result, auditable = worker._reconcile_design_outcomes(
        tmp_path, expected, inference_error="process exit 1"
    )
    assert auditable == {keys[0]: committed}
    assert result["status"] == "partial"
    assert [
        result[f"{state}_designs"] for state in ("generated", "failed", "not_run")
    ] == [1, 1, 1]
    assert result["designs"][keys[1]]["error"] == "geometry rejected"
    assert json.loads((tmp_path / "design_outcomes.json").read_text()) == result


def test_same_count_with_duplicate_identity_is_not_complete(tmp_path):
    expected = assignments(2)
    first = next(iter(expected))
    output(tmp_path, first, prefix="first_")
    output(tmp_path, first, prefix="second_")
    result, auditable = worker._reconcile_design_outcomes(
        tmp_path, expected, inference_error=None
    )
    assert not auditable
    assert result["status"] == "failed"
    assert result["failed_designs"] == result["not_run_designs"] == 1


@pytest.mark.parametrize("malformed,structure", [(True, True), (False, False)])
def test_bad_or_incomplete_output_does_not_count_as_generated(
    tmp_path, malformed, structure
):
    expected = assignments(1)
    output(tmp_path, next(iter(expected)), malformed=malformed, structure=structure)
    result, auditable = worker._reconcile_design_outcomes(
        tmp_path, expected, inference_error=None
    )
    assert not auditable
    assert result["failed_designs"] == 1
    assert result["generated_designs"] == 0


def test_truncated_gzip_does_not_count_as_generated(tmp_path):
    expected = assignments(1)
    path = output(tmp_path, next(iter(expected)))
    structure = path.with_suffix(".cif.gz")
    structure.write_bytes(structure.read_bytes()[:-8])
    result, auditable = worker._reconcile_design_outcomes(
        tmp_path, expected, inference_error=None
    )
    assert not auditable
    assert result["failed_designs"] == 1


def test_explicit_native_failure_is_not_erased_by_existing_files(tmp_path):
    expected = assignments(1)
    key = next(iter(expected))
    ledger = worker._initial_outcome_ledger(tuple(expected.values()))
    ledger["designs"][key].update(
        generation_status="failed", error="finalization failed"
    )
    worker._atomic_json(tmp_path / "design_outcomes.json", ledger)
    output(tmp_path, key)
    result, auditable = worker._reconcile_design_outcomes(
        tmp_path, expected, inference_error="failed"
    )
    assert len(auditable) == 1
    assert result["status"] == "partial"
    assert result["designs"][key]["generation_status"] == "failed"


def test_success_requires_every_expected_identity_without_unexpected_outputs(tmp_path):
    expected = assignments(2)
    for key in expected:
        output(tmp_path, key)
    result, auditable = worker._reconcile_design_outcomes(
        tmp_path, expected, inference_error=None
    )
    assert result["status"] == "completed"
    assert len(auditable) == 2
    output(tmp_path, "unexpected")
    result, _ = worker._reconcile_design_outcomes(
        tmp_path, expected, inference_error=None
    )
    assert result["status"] == "partial"
    assert len(result["unexpected_results"]) == 1


def test_audit_exception_leaves_generation_true_and_next_audit_runs(tmp_path):
    expected = assignments(2)
    assembly = SimpleNamespace(
        input_path=tmp_path / "compiled.json", semantic_audits=()
    )
    reports = (tmp_path / "audit.json",)
    outcome = SimpleNamespace(reports=reports, mobility_trajectory=None)
    with (
        patch.object(
            worker,
            "run_result_audits",
            side_effect=[ValueError("audit broken"), outcome],
        ),
        patch.object(worker, "gate_result_audits"),
        patch.object(
            worker,
            "write_advisory_screening",
            return_value={
                "contract_status": "met",
                "recommendation": "recommended_for_next_stage",
            },
        ),
        patch.object(
            worker,
            "write_decision_explanation",
            return_value=tmp_path / "decision.json",
        ),
    ):
        rows = [
            worker._audit_generated_design(
                tmp_path, key, output(tmp_path, key), item, assembly, {}
            )
            for key, item in expected.items()
        ]
    assert [row["generated"] for row in rows] == [True, True]
    assert [row["audit_status"] for row in rows] == ["failed", "completed"]
    assert [row["contract_status"] for row in rows] == ["not_evaluated", "met"]


def test_atomic_ledger_write_keeps_previous_snapshot_if_replace_fails(tmp_path):
    path = tmp_path / "design_outcomes.json"
    worker._atomic_json(path, {"status": "running"})
    with patch.object(Path, "replace", side_effect=OSError("filesystem unavailable")):
        with pytest.raises(OSError):
            worker._atomic_json(path, {"status": "completed"})
    assert json.loads(path.read_text()) == {"status": "running"}
    assert not list(tmp_path.glob("*.tmp"))


def test_partial_state_survives_worker_index_and_rebuild(tmp_path):
    run = tmp_path / "campaign" / "design" / "12345"
    run.mkdir(parents=True)
    update_run_state(
        root=tmp_path,
        job_id="12345",
        state="partial",
        experiment="design",
        campaign="campaign",
        run_directory=run,
    )
    assert read_run_record(tmp_path, "12345")["state"] == "partial"
    (run / "experiment_summary.json").write_text(
        json.dumps({"status": "partial", "experiment": "design"})
    )
    (run / "resolved_config.yaml").write_text(
        "name: design\noutput:\n  campaign: campaign\n"
    )
    assert rebuild_run_index(tmp_path)["failed"] == 0
    assert read_run_record(tmp_path, "12345")["state"] == "partial"


def test_execute_audits_all_committed_results_after_inference_process_failure(
    tmp_path, monkeypatch
):
    config = {
        "name": "partial-probe",
        "topology": {"kind": "compatibility"},
        "project_directory": str(tmp_path),
        "resources": {"checkpoint": "unused"},
        "provenance": {"render_identity": {}},
        "sampling": {
            "designs": 3,
            "seed": 42,
            "timesteps": 2,
            "execution_backend": "auto",
            "neighbour_radius": 2,
            "low_memory_mode": False,
            "dump_trajectories": False,
            "sampler": {
                "kind": "mosaic",
                "allow_realignment": False,
                "fixed_motif_finalization_mode": "restore",
                "preserve_fixed_motif_during_symmetry": True,
                "require_motif_constraint_groups": True,
                "symmetry_state_mode": "orbit_average",
                "symmetry_noise_mode": "coupled",
            },
        },
    }
    submission = tmp_path / "submission"
    submission.mkdir()
    resolved = submission / "resolved_config.yaml"
    resolved.write_text(json.dumps(config))
    (submission / "provenance.json").write_text(
        json.dumps({"resolved_config_sha256": worker._sha256(resolved)})
    )
    run = tmp_path / "run"
    monkeypatch.setattr(worker, "_verify_render_identity", lambda *a, **k: None)
    monkeypatch.setattr(worker, "collect_runtime_provenance", lambda *a, **k: {})
    monkeypatch.setattr(worker, "_record_worker_state", lambda *a, **k: None)
    monkeypatch.setattr(
        worker, "_require_compiled_pose_feasibility", lambda *a, **k: None
    )
    monkeypatch.setattr(
        worker, "_motif_mobility_runtime", lambda *a: (False, "denoiser")
    )
    for name in (
        "_graph_interface_guidance_runtime",
        "_symmetric_scaffold_packing_runtime",
    ):
        monkeypatch.setattr(worker, name, lambda *a: False)
    for name in (
        "_generated_polymer_continuity_runtime",
        "_generated_cross_chain_topology_runtime",
    ):
        monkeypatch.setattr(worker, name, lambda *a: None)
    monkeypatch.setattr(worker, "_resolved_guidance_overrides", lambda *a: [])
    monkeypatch.setattr(worker, "GeneratedCifMirror", lambda *a: nullcontext())

    def compile_assembly(topology, directory, **kwargs):
        path = directory / "rfd3_input.json"
        path.write_text(
            json.dumps({"pose": {"input": str(tmp_path / "seed.pdb"), "extra": {}}})
        )
        return SimpleNamespace(input_path=path, semantic_audits=())

    monkeypatch.setattr(worker, "compile_experiment_assembly", compile_assembly)

    def run_command(command):
        if "rfd3_mosaic.rfd3_prevalidate" in command:
            Path(command[command.index("--report") + 1]).write_text('{"passed": true}')
            return
        assert "rfd3.run_inference" in command
        merged = json.loads((run / "input" / "rfd3_input.json").read_text())
        keys = list(merged)
        output(run, keys[0])
        output(run, keys[2])
        ledger = json.loads((run / "design_outcomes.json").read_text())
        ledger["designs"][keys[1]].update(
            generation_status="failed", error="geometry rejected"
        )
        worker._atomic_json(run / "design_outcomes.json", ledger)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(worker, "_run", run_command)
    audited = []

    def audit(**kwargs):
        audited.append(kwargs["result_json"])
        return SimpleNamespace(reports=(), mobility_trajectory=None)

    monkeypatch.setattr(worker, "run_result_audits", audit)
    monkeypatch.setattr(worker, "gate_result_audits", lambda *a, **k: None)
    monkeypatch.setattr(
        worker,
        "write_advisory_screening",
        lambda *a, **k: {
            "contract_status": "met",
            "recommendation": "recommended_for_next_stage",
        },
    )
    monkeypatch.setattr(worker, "write_decision_explanation", lambda path, **k: path)
    with pytest.raises(RuntimeError, match="run partial"):
        worker.execute(resolved, run)
    summary = json.loads((run / "experiment_summary.json").read_text())
    assert len(audited) == 2
    assert summary["status"] == "partial"
    assert summary["execution_completed"] is False
    assert summary["requested_designs"] == 3
    assert summary["generated_designs"] == 2
    assert summary["failed_designs"] == 1
    assert summary["not_run_designs"] == 0
    assert summary["audit_execution"]["status"] == "completed"
    assert summary["structure_archive"]["member_count"] == 2
    assert summary["structure_archive"]["complete"] is False
    assert (
        len(json.loads((run / "sampling_manifest.json").read_text())["assignments"])
        == 3
    )


def test_main_preserves_partial_summary_and_settles_unexecuted_designs(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.yaml"
    config.write_text("name: example\n")
    expected = assignments(3)
    keys = list(expected)
    ledger = worker._initial_outcome_ledger(tuple(expected.values()))
    ledger["status"] = "partial"
    ledger["designs"][keys[0]]["generation_status"] = "generated"
    ledger["designs"][keys[1]]["generation_status"] = "running"
    worker._atomic_json(tmp_path / "design_outcomes.json", ledger)
    worker._atomic_json(
        tmp_path / "experiment_summary.json",
        {"status": "partial", "design_results": [{"generated": True}]},
    )
    monkeypatch.setattr(
        worker.sys,
        "argv",
        ["worker", "--resolved-config", str(config), "--run-dir", str(tmp_path)],
    )
    monkeypatch.setattr(
        worker,
        "execute",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stopped")),
    )
    monkeypatch.setattr(worker, "_record_worker_state", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="stopped"):
        worker.main()
    summary = json.loads((tmp_path / "experiment_summary.json").read_text())
    assert summary["status"] == "partial"
    assert summary["execution_completed"] is False
    assert len(summary["design_results"]) == 1
    assert [
        summary[f"{state}_designs"] for state in ("generated", "failed", "not_run")
    ] == [1, 1, 1]


def test_rerun_does_not_replace_existing_summary_or_ledger(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("name: example\n")
    summary = tmp_path / "experiment_summary.json"
    ledger = tmp_path / "design_outcomes.json"
    summary.write_text('{"status": "partial", "generated_designs": 2}')
    ledger.write_text('{"status": "partial"}')
    before = (summary.read_bytes(), ledger.read_bytes())
    monkeypatch.setattr(
        worker.sys,
        "argv",
        ["worker", "--resolved-config", str(config), "--run-dir", str(tmp_path)],
    )
    with pytest.raises(worker.ExistingRunError):
        worker.main()
    assert (summary.read_bytes(), ledger.read_bytes()) == before
