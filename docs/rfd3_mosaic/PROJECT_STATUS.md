# RFD3-Mosaic project status

## Release stage

RFD3-Mosaic is an actively developed **research preview**. It is installable,
has a unified public CLI and executes the same compiler/sampler/audit path
independently of the host site. Direct and Slurm launch adapters are currently
built in; neither restricts the server or GPU model. The project is not yet a
stable production release.

## Supported release target

The current supported target comprises:

- Cn and Dn symmetric execution;
- exact fixed-motif and supplied-interface preservation;
- multiple rigid components and motif orbits;
- generated-interface packing guidance;
- locked and bounded component mobility, including guided translation and
  rotation;
- deterministic lowering to RFD3 inputs;
- strict replay and required post-generation audits;
- local execution and configurable Slurm execution.

These capabilities have extensive CPU regression coverage. Representative GPU
validation exists for important paths, but broader multi-seed and packing
quality calibration is still in progress.

## Experimental capabilities

The following capabilities are implemented at varying compiler or CPU
validation levels but are not currently presented as stable release features:

- tetrahedral, octahedral and icosahedral assembly paths;
- component stabilizers, cosets and quotient orbits;
- unknown-relative-pose multi-interface assembly solving;
- higher-participant interface hyperedges;
- fully automatic architecture selection.

Experimental means that the software fails closed when it cannot prove a
valid executable lowering. It does not mean that an unvalidated candidate is
silently accepted.

## Known incomplete areas

- repeated multi-seed GPU validation across diverse real inputs;
- scientific calibration of generated-interface packing quality;
- general interface-edge stabilizers and mixed multiplicities;
- fully general native polymer-path lowering;
- stable schema migration and long-term release compatibility;
- polished tutorials and a broader public example library.

Sequence design and refolding are intentionally outside the current release
scope.

## Development policy

Validated workflows are extended incrementally rather than replaced. New
features must preserve exact constraints, compile to a replayable input and
pass required result audits. Site-specific cluster results are treated as
validation evidence, not as dependencies of the public software.
