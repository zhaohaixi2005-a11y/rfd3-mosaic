# Contributing to RFD3-Mosaic

RFD3-Mosaic is a research extension of Foundry/RFdiffusion3. Contributions are
welcome while the project is under active development.

## Development setup

```bash
git clone --branch hx/rfd3-mosaic-product-core \
  https://github.com/Khmelinskaia-Lab/foundry.git rfd3-mosaic
cd rfd3-mosaic
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[rfd3,dev]"
```

Install a PyTorch build compatible with the target CPU or CUDA environment
before installing the project if the default resolver is unsuitable.

## Change policy

- Preserve behavior of validated workflows unless the change is an explicit
  migration.
- Add focused regression tests for bug fixes and new compiler/runtime paths.
- Keep public configuration separate from site-specific deployment settings.
- Fail closed when geometry or execution semantics are ambiguous.
- Do not commit model checkpoints, credentials, run outputs or private
  filesystem paths.
- Record experimental maturity accurately; do not label CPU-only evidence as
  GPU validation.

## Validation

Run the CPU suite:

```bash
make local-test
```

Build and test the installable artifact:

```bash
make mosaic-release-smoke
```

GPU-dependent changes should additionally include a reproducible configuration
and the task-specific audit results. Results from any compatible execution
environment are acceptable; no institutional server is required.

## Formatting

```bash
python -m ruff check src/rfd3_mosaic tests/rfd3_mosaic
python -m ruff format --check src/rfd3_mosaic tests/rfd3_mosaic
git diff --check
```

## Upstream code

Changes to shared Foundry or RFD3 code should be narrowly scoped and clearly
identified. Preserve upstream license and attribution. If a change is broadly
useful outside Mosaic, consider proposing it independently to the upstream
Foundry project.

## Pull requests

Create a descriptively named development branch in the
`Khmelinskaia-Lab/foundry` repository. Do not commit directly to its default
branch.

Describe:

- the user-visible problem;
- the scientific or execution contract affected;
- tests added or updated;
- CPU and GPU evidence actually obtained;
- compatibility risks and any required migration.

Do not include private cluster paths, user names, credentials or unpublished
run artifacts in a public pull request.
