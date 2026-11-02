"""Compare complete proposals with a frozen pre-change implementation (CPU)."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from rfd3.inference.symmetry import scaffold_core_guidance as candidate

path = Path(sys.argv[1])
name = "rfd3.inference.symmetry._frozen_core_reference"
spec = importlib.util.spec_from_file_location(name, path)
reference = importlib.util.module_from_spec(spec)
sys.modules[name] = reference
spec.loader.exec_module(reference)
records = []
for n in (2, 8, 23):
    features = {
        "atom_to_token_map": torch.arange(2*n),
        "asym_id": torch.arange(2).repeat_interleave(n),
        "residue_index": torch.arange(n).repeat(2),
        "is_ca": torch.ones(2*n, dtype=torch.bool),
        "is_protein": torch.ones(2*n, dtype=torch.bool),
    }
    for pattern in ("ends", "all_fixed", "all_generated", "alternating"):
        fixed = torch.zeros(2*n, dtype=torch.bool)
        if pattern == "ends":
            fixed[[0,n-1,n,2*n-1]] = True
        elif pattern == "all_fixed":
            fixed[:] = True
        elif pattern == "alternating":
            fixed[::2] = True
        try:
            topology = candidate.build_scaffold_core_topology(features, fixed)
        except ValueError as error:
            try:
                reference.build_scaffold_core_topology(features, fixed)
            except ValueError as old_error:
                assert str(error) == str(old_error)
                records.append({"tokens_per_chain":n,"pattern":pattern,
                                "same_validation_error":str(error)})
                continue
            raise AssertionError("candidate-only topology rejection")
        for seed in (17, 29):
            torch.manual_seed(seed)
            xyz = torch.randn(1, 2*n, 3) * .2
            xyz[0,:,0] += torch.arange(n).repeat(2) * 3.8
            xyz[0,n:,1] += 9
            config = candidate.ScaffoldCoreGuidanceConfig(intra_chain_weight=1.)
            old, od = reference.apply_scaffold_core_guidance(xyz, topology, progress=.5, config=config)
            new, nd = candidate.apply_scaffold_core_guidance(xyz, topology, progress=.5, config=config)
            assert torch.equal(old, new), (n, pattern, seed, "coordinates")
            assert od == nd, (n, pattern, seed, "diagnostics/decisions")
            records.append({"tokens_per_chain":n,"pattern":pattern,"seed":seed,
                            "bitwise_coordinates":True,"identical_diagnostics":True,
                            "accepted":nd["accepted"],"trials":len(nd["line_search_trials"])})
print(json.dumps({"device":"cpu","reference_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                  "candidate_sha256":hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
                  "cases":records}, indent=2))
