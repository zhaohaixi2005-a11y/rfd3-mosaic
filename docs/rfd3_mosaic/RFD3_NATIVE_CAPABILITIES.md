# Native RFdiffusion3 capabilities in Mosaic

RFD3-Mosaic preserves RFdiffusion3 generation capabilities where they are
compatible with an explicit assembly, fixed-geometry and exact-symmetry
contract. Options are disabled by default when they materially alter the
conditioning problem, but they remain selectable in the public design YAML.

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

When `conditioning.redesign_motif_sidechains: true` is explicit, Mosaic keeps
protein motif backbone atoms fixed but does not re-fix their side-chain atoms
through `select_fixed_atoms`. It may be combined with `mode: masked`; this is
the native RFD3 formulation for preserving the supplied backbone while asking
the model to redesign the motif sequence and side chains. Without that opt-in,
the existing all-atom fixed-motif behavior is unchanged.
All-glycine conditioning is intentionally exclusive with side-chain redesign:
use `masked` when RFD3 should choose new identities, or `glycine` when glycine
itself is the conditioning identity.

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
