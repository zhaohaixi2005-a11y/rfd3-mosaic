"""Real Mosaic engine dispatch with CPU fakes for the loaded GPU model."""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rfd3.engine import RFD3InferenceEngine


def specifications(count=4, *, marked=True):
    return {
        f"engine_{index}": {
            "extra": {
                "mosaic_example_id": f"design_{index}",
                "mosaic_design_index": index,
                "mosaic_diffusion_seed": 100 + index,
                **({"mosaic_batch_protocol": 1} if marked else {}),
            }
        }
        for index in range(count)
    }


def engine_with_forward(tmp_path, failures=None, dump_failures=()):
    failures = failures or {}
    calls = []
    engine = object.__new__(RFD3InferenceEngine)
    engine.out_dir = tmp_path
    engine.pipeline = object()
    engine.trainer = SimpleNamespace(
        fabric=SimpleNamespace(
            world_size=1,
            global_rank=0,
            setup_dataloaders=lambda loader, **kwargs: loader,
        )
    )

    def forward(batch):
        key = batch["example_id"]
        calls.append(key)
        if key in failures:
            raise failures[key]

        def dump(out_dir):
            path = Path(out_dir) / f"{key}_model_0"
            with gzip.open(f"{path}.cif.gz", "wt") as handle:
                handle.write("data_test\n#\n")
            if key in dump_failures:
                raise ValueError("dump failed after writing partial output")
            Path(f"{path}.json").write_text(json.dumps({"example_id": key}))

        return [SimpleNamespace(dump=dump)]

    engine._model_forward = forward
    return engine, calls


def loader(**kwargs):
    for key, specification in kwargs["data"].items():
        yield [{"example_id": key, "specification": specification}]


def read_ledger(tmp_path):
    return json.loads((tmp_path / "design_outcomes.json").read_text())


def test_one_design_failure_does_not_cancel_following_designs(tmp_path):
    engine, calls = engine_with_forward(
        tmp_path, {"engine_1": ValueError("geometry rejected")}
    )
    with patch("rfd3.engine.assemble_distributed_inference_loader_from_json", loader):
        with pytest.raises(RuntimeError, match="failed designs"):
            engine._run_multi(specifications())
    assert calls == ["engine_0", "engine_1", "engine_2", "engine_3"]
    ledger = read_ledger(tmp_path)
    assert ledger["status"] == "partial"
    assert ledger["generated_designs"] == 3
    assert ledger["failed_designs"] == 1
    assert ledger["not_run_designs"] == 0
    assert ledger["designs"]["design_1"]["error"] == "geometry rejected"
    assert len(list(tmp_path.glob("*model_0.json"))) == 3
    assert not list(tmp_path.glob(".mosaic-design-*"))


@pytest.mark.parametrize(
    "failure",
    [
        MemoryError("allocation"),
        RuntimeError("CUDA out of memory"),
        RuntimeError("CUDA error: device-side assert triggered"),
        OSError("disk full"),
    ],
)
def test_fatal_errors_stop_without_retry_and_record_remaining(tmp_path, failure):
    engine, calls = engine_with_forward(tmp_path, {"engine_1": failure})
    with patch("rfd3.engine.assemble_distributed_inference_loader_from_json", loader):
        with pytest.raises(type(failure)):
            engine._run_multi(specifications())
    assert calls == ["engine_0", "engine_1"]
    ledger = read_ledger(tmp_path)
    assert [
        ledger[key]
        for key in ("generated_designs", "failed_designs", "not_run_designs")
    ] == [1, 1, 2]
    assert ledger["designs"]["design_1"]["fatal"] is True


def test_three_consecutive_failures_bound_broken_campaign(tmp_path):
    engine, calls = engine_with_forward(
        tmp_path, {f"engine_{i}": ValueError("bad") for i in range(3)}
    )
    with patch("rfd3.engine.assemble_distributed_inference_loader_from_json", loader):
        with pytest.raises(ValueError):
            engine._run_multi(specifications(6))
    assert len(calls) == 3
    ledger = read_ledger(tmp_path)
    assert ledger["failed_designs"] == 3
    assert ledger["not_run_designs"] == 3
    assert ledger["status"] == "failed"


def test_loader_transform_exception_is_isolated_too(tmp_path):
    engine, calls = engine_with_forward(tmp_path)

    def failing_loader(**kwargs):
        for key, specification in kwargs["data"].items():
            if key == "engine_0":
                raise ValueError("transform cannot construct input")
            yield [{"example_id": key, "specification": specification}]

    with patch(
        "rfd3.engine.assemble_distributed_inference_loader_from_json", failing_loader
    ):
        with pytest.raises(RuntimeError):
            engine._run_multi(specifications(2))
    assert calls == ["engine_1"]
    assert read_ledger(tmp_path)["generated_designs"] == 1


def test_failed_dump_never_publishes_partial_design(tmp_path):
    engine, calls = engine_with_forward(tmp_path, dump_failures={"engine_0"})
    with patch("rfd3.engine.assemble_distributed_inference_loader_from_json", loader):
        with pytest.raises(RuntimeError):
            engine._run_multi(specifications(2))
    assert not (tmp_path / "engine_0_model_0.cif.gz").exists()
    assert (tmp_path / "engine_1_model_0.cif.gz").is_file()
    assert read_ledger(tmp_path)["generated_designs"] == 1


def test_existing_user_output_is_never_overwritten(tmp_path):
    existing = tmp_path / "engine_0_model_0.json"
    existing.write_text("user data")
    engine, calls = engine_with_forward(tmp_path)
    with patch("rfd3.engine.assemble_distributed_inference_loader_from_json", loader):
        with pytest.raises(FileExistsError):
            engine._run_multi(specifications(2))
    assert existing.read_text() == "user data"
    assert calls == ["engine_0"]
    assert read_ledger(tmp_path)["not_run_designs"] == 1


def test_ordinary_upstream_still_raises_first_error(tmp_path):
    engine, calls = engine_with_forward(tmp_path, {"engine_0": ValueError("upstream")})
    with patch("rfd3.engine.assemble_distributed_inference_loader_from_json", loader):
        with pytest.raises(ValueError, match="upstream"):
            engine._run_multi(specifications(2, marked=False))
    assert calls == ["engine_0"]
    assert not (tmp_path / "design_outcomes.json").exists()


def test_success_is_complete_and_commits_exactly_one_output_per_identity(tmp_path):
    engine, calls = engine_with_forward(tmp_path)
    with patch("rfd3.engine.assemble_distributed_inference_loader_from_json", loader):
        assert engine._run_multi(specifications(2)) == {}
    ledger = read_ledger(tmp_path)
    assert ledger["status"] == "completed"
    assert ledger["generated_designs"] == ledger["requested_designs"] == 2
    assert all(len(row["result_jsons"]) == 1 for row in ledger["designs"].values())
