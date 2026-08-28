# Native RFdiffusion3 capabilities in Mosaic

RFD3-Mosaic preserves RFdiffusion3 generation capabilities where they are
compatible with an explicit assembly, fixed-geometry and exact-symmetry
contract. Options are disabled by default when they materially alter the
conditioning problem, but they remain selectable in the public design YAML.

None of the conditioning channels below is required for an ordinary Mosaic
run. Omitting `conditioning` preserves the supplied sequence and coordinates
according to the declared fixed-geometry constraints. Users should add a
channel only when it represents known scientific input, not because it exists.

## Public mappings

| RFdiffusion3 capability | Mosaic public declaration |
| --- | --- |
| Input structure | `input` |
| Generated length and motif contig | `generation` regions |
| Coordinate-fixed motif atoms | `constraints` with `fixed_xyz` |
| Joint-rigid multi-fragment motif | shared `coupling_group` |
| Unfixed motif sequence (masking) | `conditioning.sequence: mode: masked` |
| Backbone-only glycine conditioning | `conditioning.sequence: mode: glycine` |
| Ligand conditioning | `conditioning.ligands` |
| Buried / partially buried / exposed conditioning | `conditioning.buried`, `partially_buried`, `exposed` |
| Interface hotspot conditioning | `conditioning.hotspots` |
| Hydrogen-bond donor / acceptor conditioning | `conditioning.hbond_donors`, `hbond_acceptors` |
| Native motif side-chain redesign | `conditioning.redesign_motif_sidechains` |
| COM or hotspot-derived origin | `conditioning.origin_strategy` |
| Non-loopy global conditioning | `sampling.is_non_loopy` |
| pLDDT enhancement | `sampling.plddt_enhanced` |
| Low-memory inference | `sampling.low_memory_mode` |
| Denoising trajectories | `sampling.dump_trajectories` |
| Independent noise samples | `sampling.designs` and `replicates_per_pose` |
| Multiple input specifications with one model load | stochastic `initial_pose` plus `designs` |
| Cn, Dn, T, O and I symmetry | `symmetry` plus the compiled finite-orbit plan |

Selections are resolved before submission and translated to the label-chain
indices in the frozen, compiler-generated mmCIF. Invalid or ambiguous
selections fail before GPU inference.

## Which sequence mode should I use?

| Scientific intent | Declaration | Side-chain redesign |
| --- | --- | --- |
| Preserve the original motif identity | omit sequence conditioning | off |
| Hide the supplied identity but keep its backbone | `mode: masked` | optional |
| Present an all-glycine surface to RFD3 | `mode: glycine` | not allowed |

Copy-ready masked redesign:

```yaml
conditioning:
  sequence:
    - {selector: A20-35, mode: masked}
    - {selector: B40-55, mode: masked}
  redesign_motif_sidechains: true
```

Copy-ready all-glycine conditioning:

```yaml
conditioning:
  sequence:
    - {selector: A20-35, mode: glycine}
    - {selector: B40-55, mode: glycine}
  redesign_motif_sidechains: false
```

When `conditioning.redesign_motif_sidechains: true` is explicit, Mosaic keeps
protein motif backbone atoms fixed but does not re-fix their side-chain atoms
through `select_fixed_atoms`. It may be combined with `mode: masked`; this is
the native RFD3 formulation for preserving the supplied backbone while asking
the model to redesign the motif sequence and side chains. Without that opt-in,
the existing all-atom fixed-motif behavior is unchanged.
All-glycine conditioning is intentionally exclusive with side-chain redesign:
use `masked` when RFD3 should choose new identities, or `glycine` when glycine
itself is the conditioning identity.

## Atom-level conditioning example

```yaml
conditioning:
  ligands:
    - {selector: B1, coupling_group: supplied_interface}
  buried:
    - {selector: B1, atoms: ALL}
  partially_buried:
    - {selector: A42-45, atoms: TIP}
  exposed:
    - {selector: A80-85, atoms: ALL}
  hotspots:
    - {selector: A42-45, atoms: TIP}
  hbond_acceptors:
    - {selector: B1, atoms: O1,O2}
  hbond_donors:
    - {selector: A67, atoms: N}
  origin_strategy: hotspots
```

Ligands are expanded with the declared assembly and attached to the named
rigid coupling group. RASA, hotspot and hydrogen-bond selectors are remapped
after symmetry compilation. `origin_strategy: hotspots` requires a hotspot;
atom names and selector coverage are checked by `validate` before inference.

## Sampling defaults

The maintained public defaults are:

```yaml
sampling:
  timesteps: 200
  designs: 1
  replicates_per_pose: 1
  seed: 42
  low_memory_mode: true
  is_non_loopy: true
  plddt_enhanced: true
  dump_trajectories: false
```

`designs` is the total requested output count. With a stochastic initial pose,
`replicates_per_pose: 1` gives each output its own pose. Fixed arrangements
retain one exact pose and vary diffusion only. See the
[complete workflow guide](WORKFLOW_GUIDE.md) for pose examples and general
multi-design campaign guidance.

## Deliberate semantic replacements

Some native switches are not exposed as unchecked passthrough strings:

- native `n_batches` and `diffusion_batch_size` are represented by Mosaic's
  explicit design/pose/replicate plan. This prevents `designs=N` from
  accidentally meaning N copies of one hidden assembly hypothesis;
- `ori_token` is compiler-owned when exact symmetry uses quotient or
  pre-expanded coordinates. Users can request `origin_strategy` when that is
  geometrically compatible;
- ligand symmetry is derived from the declared assembly orbit rather than
  from an independent, potentially inconsistent native switch;
- generated residue initialization uses each chain's local fixed anchor for
  compiler-expanded assemblies, avoiding artificial spokes from every copy
  to one global origin.

## Fail-closed native boundaries

The following native mechanisms are intentionally not part of an exact
Mosaic motif-scaffolding run:

- `allow_realignment`: rotating the coordinate frame without conjugating the
  declared symmetry operators changes the represented assembly;
- `partial_t`: native partial diffusion changes the coordinate-conditioning
  problem and is not equivalent to preserving an exact fixed motif orbit;
- raw `ori_token`, `cif_parser_args` and
  `allow_ligand_on_existing_chain`: these are compiler/parser ownership
  controls, not portable scientific design intent;
- arbitrary native `unindex`: Mosaic uses explicit fragments, polymer paths
  and stable mapping records instead of index-free motif insertion.

For an experiment whose scientific requirement is specifically native
partial diffusion or arbitrary unindexed insertion, use RFdiffusion3 directly
rather than labeling it an exact Mosaic motif-scaffolding run.
