"""Bounded, reference-free CCD sidechain chemistry and clash candidate polish."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree


def relax_sidechains(candidate, baseline, features, *, maximum_step=1.5, iterations=100, require_absolute_clearance=False):
    import torch
    from atomworks.ml.encoding_definitions import AF3SequenceEncoding
    from rfd3.inference.symmetry.output_geometry_guard import native_labels, _ideal, bond_targets
    xyz=candidate.detach().double().cpu().numpy().reshape(-1,3)
    original=baseline.detach().double().cpu().numpy().reshape(-1,3)
    tokens=features['atom_to_token_map'].detach().cpu().numpy()
    virtual=features['is_virtual'].detach().cpu().numpy().astype(bool)
    fixed=features['is_motif_atom_with_fixed_coord'].detach().cpu().numpy().astype(bool)
    labels=native_labels(features);protein=features['is_protein'].detach().cpu().numpy().astype(bool)
    names=AF3SequenceEncoding().decode(features['restype'].detach().cpu().numpy()).tolist()
    movable=np.zeros(len(xyz),dtype=bool);bonds=[];angles=[];chiral=[]
    for token in np.unique(tokens):
        ids=np.flatnonzero((tokens==token)&~virtual)
        if not protein[token] or fixed[tokens==token].any():continue
        for i in ids:movable[i]=labels[i] not in ('N','CA','C','O','OXT')
        lookup={labels[i]:int(i) for i in ids};ideal,definition=_ideal(names[token]);neighbors={}
        from rfd3.inference.symmetry.chemistry_joint_states import joint_bond_states,choose_joint_state
        physical={a:xyz[i] for a,i in lookup.items()};joint_targets={}
        for pairs,states,kind in joint_bond_states(names[token],physical,ideal):
            state,_,_=choose_joint_state(pairs,states,physical)
            joint_targets.update({frozenset(pair):target for pair,target in zip(pairs,state)})

        for a,b in definition:
            if a not in lookup or b not in lookup:continue
            i,j=lookup[a],lookup[b]
            neighbors.setdefault(a,[]).append(b);neighbors.setdefault(b,[]).append(a)
            if movable[i] or movable[j]:
                targets=(joint_targets[frozenset((a,b))],) if frozenset((a,b)) in joint_targets else bond_targets(names[token],a,b,float(np.linalg.norm(ideal[a]-ideal[b])))
                target=min(targets,key=lambda t:abs(t-np.linalg.norm(xyz[i]-xyz[j])))
                bonds.append((i,j,target))
        for center,ends in neighbors.items():
            for n,a in enumerate(ends):
                for b in ends[n+1:]:
                    i,j,k=lookup[a],lookup[center],lookup[b]
                    if not (movable[i] or movable[j] or movable[k]):continue
                    u,v=ideal[a]-ideal[center],ideal[b]-ideal[center]
                    angles.append((i,j,k,float(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))))
        if all(a in lookup for a in ('N','CA','C','CB')):
            ids4=tuple(lookup[a] for a in ('N','CA','C','CB'))
            sign=float(np.sign(np.dot(np.cross(ideal['N']-ideal['CA'],ideal['C']-ideal['CA']),ideal['CB']-ideal['CA'])))
            chiral.append((*ids4,sign))
    ids=np.flatnonzero(movable)
    if not len(ids):return candidate,{'applicable':False,'reason':'no_movable_physical_sidechains'}
    index={int(i):j for j,i in enumerate(ids)}
    chains=features['asym_id'].detach().cpu().numpy();positions=features['residue_index'].detach().cpu().numpy()
    physical=np.flatnonzero(~virtual)
    # Cartesian box bounds permit sqrt(3)*maximum_step per atom.
    search_radius=2.02+2*np.sqrt(3)*maximum_step if require_absolute_clearance else 4.0
    pairset=set(cKDTree(xyz[physical]).query_pairs(search_radius)) | set(cKDTree(original[physical]).query_pairs(2.0))
    pairs=[]
    for a,b in sorted(pairset):
        i,j=int(physical[a]),int(physical[b]);ti,tj=tokens[i],tokens[j]
        if not (movable[i] or movable[j]) or (chains[ti]==chains[tj] and abs(positions[ti]-positions[tj])<=1):continue
        target=2.02 if require_absolute_clearance else min(2.02,float(np.linalg.norm(original[i]-original[j]))+.005)
        pairs.append((i,j,target))
    center=xyz[ids].copy();count=3*len(ids)
    dependencies=[(i,) for i in ids for _ in range(3)]+[(a,b) for a,b,_ in bonds]+[(a,b,c) for a,b,c,_ in angles]+[(a,b) for a,b,_ in pairs]+[row[:4] for row in chiral]
    sparsity=lil_matrix((len(dependencies),count),dtype=int)
    for row,atoms in enumerate(dependencies):
        for atom in atoms:
            if atom in index:sparsity[row,3*index[atom]:3*index[atom]+3]=1
    def residual(flat):
        value=xyz.copy();value[ids]=flat.reshape(-1,3)
        result=[*(.05*(value[ids]-center)).reshape(-1)]
        result.extend(40*(np.linalg.norm(value[a]-value[b])-target) for a,b,target in bonds)
        for a,b,c,target in angles:
            u,v=value[a]-value[b],value[c]-value[b]
            cosine=np.dot(u,v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-12)
            result.append(5*(cosine-target))
        result.extend(100*max(0,target-np.linalg.norm(value[a]-value[b])) for a,b,target in pairs)
        for a,b,c,d,sign in chiral:
            volume=np.dot(np.cross(value[a]-value[b],value[c]-value[b]),value[d]-value[b])*sign
            result.append(20*max(0,.2-volume))
        return np.asarray(result)
    solution=least_squares(residual,center.reshape(-1),bounds=((center-maximum_step).reshape(-1),(center+maximum_step).reshape(-1)),jac_sparsity=sparsity.tocsr(),max_nfev=iterations,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    value=xyz.copy();value[ids]=solution.x.reshape(-1,3)
    return torch.as_tensor(value,dtype=candidate.dtype,device=candidate.device).reshape_as(candidate),{'applicable':True,'evaluations':solution.nfev,'solver_success':bool(solution.success),'maximum_sidechain_step':float(np.linalg.norm(value-xyz,axis=-1).max()),'physical_sidechain_atoms':len(ids),'pair_constraints':len(pairs),**({'requires_absolute_heavy_clearance':True} if require_absolute_clearance else {}),'scope':'CCD sidechain bonds/angles/chirality plus individual nonadjacent heavy-pair lower bounds; backbone and fixed atoms pinned; candidate requires actual projected absolute backbone and chemistry non-regression guards'}
