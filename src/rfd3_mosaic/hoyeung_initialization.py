"""Ho-Yeung LHD101 C3 initializer, before the unchanged Mosaic/RFD3 sampler.

Based on the public Khmelinskaia-Lab/RFdiffusion_interfaceseed initializer
(e429bab). See docs/rfd3_mosaic/HO_YEUNG_INITIALIZER_BSD_NOTICE.txt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import yaml

from rfd3_mosaic.schema import UserDesignSpec, load_user_design

AUTHOR_COMMIT = "e429babbd32ba64394eb1039a3ff952c0f0560c9"
LHD101_INPUTS = frozenset({
    "c4022c0ec09b9bee13bb281ee80d24ab6e28c6e93f0ebf5db317aa5af0c47a63",
    "2048eb976ad3dfa070b4aa78bd11376a9ce886193b1513d0996f266618161d82",
})


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lhd101(path: Path):
    if digest(path) not in LHD101_INPUTS:
        raise ValueError("Ho-Yeung preparation supports only the pinned public LHD101 interface")
    keys, xyz = [], []
    for line in path.read_text().splitlines():
        if line.startswith("ATOM  "):
            keys.append((line[21], int(line[22:26]), line[12:16].strip()))
            xyz.append([float(line[i:i+8]) for i in (30,38,46)])
    if len(keys) != 496 or len(set(keys)) != 496:
        raise ValueError("Unexpected LHD101 atom map")
    return keys, np.array(xyz, dtype=np.float32)


def xyz_rotation(degrees):
    x, y, z = np.deg2rad(degrees)
    cx, cy, cz = np.cos((x,y,z))
    sx, sy, sz = np.sin((x,y,z))
    return np.array(((cy*cz, sx*sy*cz-cx*sz, cx*sy*cz+sx*sz),
                     (cy*sz, sx*sy*sz+cx*cz, cx*sy*sz-sx*cz),
                     (-sy, sx*cy, cx*cy)))


def draw_geometry(key_xyz, rng, minimum, maximum):
    """Port of sample_init's geometry, not the author's RFD1 model or sampler."""
    keys, xyz = key_xyz
    angles = [rng.randint(-180,180) for _ in range(3)]
    shift = rng.randint(20,30)
    r = xyz_rotation(angles)
    rotated = np.einsum("nj,kj->nk",xyz,r.astype(np.float32))
    rotated[:,0] += shift
    copies = np.stack([np.einsum("nj,kj->nk",rotated,xyz_rotation((0,0,120*i)).astype(np.float32))
                       for i in range(3)])
    a = [i for i,(c,_,atom) in enumerate(keys) if c=="A" and atom=="N"]
    b = [i for i,(c,_,atom) in enumerate(keys) if c=="B" and atom=="N"]
    a0 = copies[0,a].mean(axis=0)
    d_plus = float(np.sqrt(np.sum((a0-copies[1,b].mean(axis=0))**2)))
    d_minus = float(np.sqrt(np.sum((a0-copies[2,b].mean(axis=0))**2)))
    offset = -1 if d_plus < d_minus else 1  # B(k)→A(k±1), not A→B.
    length = rng.randint(minimum,maximum)
    center = r @ xyz.astype(np.float64).mean(axis=0) + np.array((shift,0,0))
    radial = np.hypot(*center[:2])
    direction = [float(center[0]/radial),float(center[1]/radial),0.] if radial else [1.,0.,0.]
    pose = {"radius":{"minimum":float(radial),"maximum":float(radial)},
            "axial_offset":{"minimum":float(center[2]),"maximum":float(center[2])},
            "radial_direction":direction,
            "orientation":{"method":"fixed","rotation_deg":angles}}
    return {"rotation_deg": angles, "x_translation_angstrom": shift,
            "d_plus_angstrom": d_plus, "d_minus_angstrom": d_minus,
            "B_to_A_orbit_offset": offset, "linker_length": length,
            "initial_pose": pose, "coordinates": copies}


def _scope(design: UserDesignSpec):
    payload = design.model_dump(mode="json",exclude_none=True)
    segments = payload["generation"]
    fixed = [s for s in payload["constraints"] if s["kind"]=="fixed_xyz"]
    if (payload["symmetry"]!="C3" or payload["task"]!="preserve_supplied_geometry"
        or len(segments)!=1 or segments[0]["kind"]!="between"
        or segments[0]["from_selector"]!="B211-241" or segments[0]["to_selector"]!="A165-194"
        or len(fixed)!=2 or {c["selector"] for c in fixed}!={"A165-194","B211-241"}
        or not fixed[0].get("coupling_group") or fixed[0]["coupling_group"]!=fixed[1].get("coupling_group")):
        raise ValueError("Expected LHD101 C3 with a joint A/B rigid seed and B→A linker")
    sampling = payload["sampling"]
    if sampling.get("initial_poses") or sampling.get("scaffold_artifact"):
        raise ValueError("Multi-seed pose and complete scaffold modes are not supported here")
    length = segments[0]["length"]
    if type(length["minimum"]) is not int or type(length["maximum"]) is not int:
        raise ValueError("Expected integer linker length bounds")
    return payload, length["minimum"], length["maximum"]


def prepare_hoyeung_poses(config: Path, output: Path, *, count: int, seed: int):
    """Prepare every draw, without scoring, deduplication, or replacement."""
    if not 1 <= count <= 4096 or seed < 0:
        raise ValueError("Require 1 <= count <= 4096 and seed >= 0")
    design = load_user_design(config)
    template, minimum, maximum = _scope(design)
    key_xyz = read_lhd101(design.input)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_path = output/"7mwr_interface.pdb"
    seed_path.write_bytes(design.input.read_bytes())
    input_sha = digest(seed_path)
    if input_sha not in LHD101_INPUTS:
        raise ValueError("Copied seed failed input digest")
    plan = {"mode":"hoyeung_lhd101_c3", "source_commit":AUTHOR_COMMIT,
            "source_input_sha256":input_sha, "template_sha256":digest(config),
            "requested_count":count, "pairing_convention":"B(k) to A(k+offset)",
            "rng_convention":"separate Python random.Random(base seed + index) per design",
            "pose_scope":"one draw per design", "atom_order":key_xyz[0], "attempts":[]}
    path = output/"pose_tasks.json"
    with (output/"attempts.jsonl").open("x") as ledger:
        for idx in range(count):
            draw_seed = seed+idx
            row = {"index":idx, "pose_seed":draw_seed}
            try:
                trial = draw_geometry(key_xyz,random.Random(draw_seed),minimum,maximum)
                xyz = trial.pop("coordinates")
                row.update(trial)
                data = json.loads(json.dumps(template))
                length = trial["linker_length"]
                data["generation"][0].update(orbit_offset=trial["B_to_A_orbit_offset"],
                                             length={"minimum":length,"maximum":length})
                data["sampling"].update(initial_pose=trial["initial_pose"], designs=1,
                                        seed=template["sampling"]["seed"]+idx)
                data["sampling"].pop("replicates_per_pose",None)
                task_name = f"{design.name}-hy-{idx:04d}"
                data.update(name=task_name,input=str(seed_path))
                if data.get("output"):
                    data["output"]["campaign"]=task_name
                UserDesignSpec.model_validate(data)
                folder = output/f"trial-{idx:04d}"
                folder.mkdir()
                task = folder/"mosaic.yaml"
                task.write_text(yaml.safe_dump(data,sort_keys=False))
                coords = folder/"expected_pre_model_seed.npy"
                np.save(coords,xyz,allow_pickle=False)
                row.update(status="staged",task=str(task),task_sha256=digest(task),
                           coordinates_sha256=digest(coords))
            except (OSError,TypeError,ValueError) as err:
                row.update(status="failed",reason=f"{type(err).__name__}: {err}")
            plan["attempts"].append(row)
            ledger.write(json.dumps(row,allow_nan=False)+"\n")
            ledger.flush()
    path.write_text(json.dumps(plan,indent=2,allow_nan=False)+"\n")
    return plan
