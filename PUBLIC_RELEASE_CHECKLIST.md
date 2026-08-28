# Public release checklist

RFD3-Mosaic is developed in a laboratory repository that also supports private
validation work. A public release must be created deliberately; changing the
visibility of a development repository is not a release procedure.

## Required before publication

- Confirm that `.env`, credentials, private keys, checkpoints, run outputs,
  personal presentations and laboratory records are not tracked.
- Review the complete Git history. Removing a file from the current branch does
  not remove it from earlier commits. Publish from a reviewed, squashed export
  or perform a coordinated history rewrite when old commits contain private
  material.
- Record the source, license and publication permission for every packaged
  structure fixture under `examples/rfd3_mosaic/inputs/` and
  `examples/rfd3_mosaic/lhd101_c3/inputs/`.
- Confirm that vendored Foundry/RFdiffusion3 code retains its upstream license
  and attribution.
- Run `python scripts/rfd3_mosaic/check_public_surface.py`, the complete CPU
  test suite and the wheel smoke test from a clean checkout.
- Replace development-branch installation links with a versioned tag or
  release artifact.
- Review GitHub Actions permissions, branch triggers and repository visibility
  with a laboratory maintainer.
- Pin third-party GitHub Actions to reviewed immutable commit SHAs for a stable
  public release; moving major-version tags are retained during development.

## Structure-fixture review

The following fixtures are needed by maintained examples or regression tests,
but their presence in a private development branch is not by itself permission
to redistribute them publicly:

| Fixture | Required release record |
| --- | --- |
| `lhd101_c3/inputs/7mwr_interface.pdb` | Public source identifier, transformation record and redistribution terms |
| `inputs/Prism_C3_G2_fixed_motif.pdb` | Structure owner, derivation record and explicit publication approval |
| `inputs/C4_C2_quotient_seed_backbone.pdb` | Parent structure, derivation record and explicit publication approval |
| `inputs/PI25_C3_three_participant_interface_seed.pdb` | Structure owner, derivation record and explicit publication approval |

If approval is unavailable, replace the fixture with an openly redistributable
or synthetic equivalent and update every dependent example and test before
publication.
