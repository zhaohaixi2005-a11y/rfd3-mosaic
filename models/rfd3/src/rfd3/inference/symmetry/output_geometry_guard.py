"""Final physical identities, sidechain chemistry and pairwise non-regression.

Pre-existing predicted defects remain visible; lower total clash counts never
permit a new or worsened individual pair. This does not certify a folded core.
"""
from functools import lru_cache
import numpy as np
from scipy.spatial import cKDTree


@lru_cache(maxsize=32)
def _ideal(name):
    from biotite.structure.info import residue, bonds_in_residue
    aa = residue(name)
    return dict(zip(aa.atom_name.tolist(), aa.coord.astype(float))), bonds_in_residue(name)


def _angle(a, b, c):
    u, v = a-b, c-b
    denominator = np.linalg.norm(u)*np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v)/max(denominator, 1e-15), -1, 1))))


def _volume(g):
    return float(np.dot(np.cross(g['N']-g['CA'], g['C']-g['CA']), g['CB']-g['CA']))


def bond_targets(residue_name, a, b, ccd_target):
    """Equivalent carboxyl oxygens and protonation/resonance alternatives.

    CCD ideal free residues can encode an acid O-H or a guanidinium valence
    form. Protein predictions need not use that protonation or atom naming.
    Keep the existing 0.1A tolerance around explicit alternative targets.
    """
    pair={a,b}
    if (residue_name=='ASP' and pair in ({'CG','OD1'},{'CG','OD2'})) or (residue_name=='GLU' and pair in ({'CD','OE1'},{'CD','OE2'})):
        return (1.23,1.26,1.34)
    if residue_name=='ARG' and pair in ({'NE','CZ'},{'CZ','NH1'},{'CZ','NH2'}):
        return (1.33,float(ccd_target))
    return (float(ccd_target),)


def native_labels(features):
    encoded = features['ref_atom_name_chars']
    if encoded.ndim == 2 and encoded.shape[-1] == 256:
        encoded = encoded.reshape(-1, 4, 64)
    codes = encoded.argmax(-1) if encoded.ndim == 3 else encoded
    return [''.join(chr(int(c)+32) for c in row).strip() for row in codes.detach().cpu().tolist()]


def audit_bound_geometry(coordinates, features, baseline=None):
    """One batch, physical labels bound to the final exporter policy."""
    from atomworks.ml.encoding_definitions import AF3SequenceEncoding
    from rfd3.inference.symmetry.interface_backbone_guard import interface_backbone_geometry_guard
    xyz = coordinates.detach().double().cpu().numpy().reshape(-1, 3)
    before = xyz if baseline is None else baseline.detach().double().cpu().numpy().reshape(-1, 3)
    tokens = features['atom_to_token_map'].detach().cpu().numpy()
    fixed = features['is_motif_atom_with_fixed_coord'].detach().cpu().numpy().astype(bool)
    virtual = features['is_virtual'].detach().cpu().numpy().astype(bool)
    protein = (features['is_protein'].detach().cpu().numpy().astype(bool)
               if 'is_protein' in features else np.ones(len(features['asym_id']), dtype=bool))
    restype = features['restype']; restype = restype.argmax(-1) if restype.ndim == 2 else restype
    residues = AF3SequenceEncoding().decode(restype.detach().cpu().numpy()).tolist()
    names = native_labels(features)
    chains = features['asym_id'].detach().cpu().numpy()
    positions = features['residue_index'].detach().cpu().numpy()
    failures, regressions, rings = [], [], []
    generated = np.zeros(len(tokens), dtype=bool)
    for token in np.unique(tokens):
        ids = np.flatnonzero((tokens == token) & ~virtual)
        if not protein[token] or not len(ids):
            continue
        if fixed[ids].all():
            continue
        generated[ids] = True
        labels = [names[i] for i in ids]
        if len(set(labels)) != len(labels):
            raise ValueError('Duplicate physical atom in final identity binding')
        g, old = dict(zip(labels, xyz[ids])), dict(zip(labels, before[ids]))
        ideal, bonds = _ideal(residues[token])
        from rfd3.inference.symmetry.chemistry_joint_states import joint_bond_states,choose_joint_state
        for a, b in bonds:
            if a not in g or b not in g or (a in ('N','CA','C','O','OXT') and b in ('N','CA','C','O','OXT')):
                continue
            target = float(np.linalg.norm(ideal[a]-ideal[b]))
            length, previous = float(np.linalg.norm(g[a]-g[b])), float(np.linalg.norm(old[a]-old[b]))
            targets=bond_targets(residues[token],a,b,target)
            row = {'token': int(token), 'kind': 'sidechain_bond', 'atoms': [a,b], 'length': length, 'CCD_target': target, 'chemically_allowed_targets':list(targets)}
            error, olderror = min(abs(length-t) for t in targets),min(abs(previous-t) for t in targets)
            if error > .1:
                failures.append(row)
            if error > max(.1, olderror)+1e-5:
                regressions.append(row)
        for pairs,states,kind in joint_bond_states(residues[token],g,ideal):
            state,error,lengths=choose_joint_state(pairs,states,g)
            _,olderror,_=choose_joint_state(pairs,states,old)
            row={'token':int(token),'kind':'joint_sidechain_bond_state','chemical_group':kind,'bonds':[list(p) for p in pairs],'lengths':lengths,'allowed_joint_states':[list(s) for s in states],'minimum_joint_max_deviation_A':error,'tolerance_A':.1}
            if error>.1:failures.append(row)
            if error>max(.1,olderror)+1e-5:regressions.append(row)
        neighbors={}
        for a,b in bonds:
            if a not in g or b not in g:continue
            neighbors.setdefault(a,[]).append(b);neighbors.setdefault(b,[]).append(a)
        for center,ends in neighbors.items():
            if center=='CA':continue # Existing stricter CA-sidechain audit below.
            for n,a in enumerate(ends):
                for c in ends[n+1:]:
                    if all(v in ('N','CA','C','O','OXT') for v in (a,center,c)):continue
                    target=_angle(ideal[a],ideal[center],ideal[c]);actual=_angle(g[a],g[center],g[c])
                    error=abs(actual-target);olderror=abs(_angle(old[a],old[center],old[c])-target)
                    row={'token':int(token),'kind':'sidechain_bond_angle','atoms':[a,center,c],'angle_degrees':actual,'CCD_ideal_angle_degrees':target,'tolerance_degrees':20}
                    if error>20:failures.append(row)
                    if error>max(20,olderror)+1e-4:regressions.append(row)
        if residues[token] == 'PRO':
            if not all(a in g for a in ('N','CD')):
                failures.append({'token': int(token), 'kind': 'missing_proline_ring_atom'})
                regressions.append(failures[-1])
            else:
                length = float(np.linalg.norm(g['N']-g['CD']))
                row = {'token': int(token), 'N_CD_length': length}
                rings.append(row)
                if abs(length-1.47) > .1:
                    failures.append({'kind': 'proline_ring_closure', **row})
                    regressions.append(failures[-1])
        if all(a in g and a in ideal for a in ('N','CA','C','CB')):
            if _volume(g)*_volume(ideal) <= 0:
                row = {'token': int(token), 'kind': 'CA_chirality'}
                failures.append(row); regressions.append(row)
            for a in ('N','C'):
                target = _angle(ideal[a], ideal['CA'], ideal['CB'])
                error = abs(_angle(g[a],g['CA'],g['CB'])-target)
                olderror = abs(_angle(old[a],old['CA'],old['CB'])-target)
                row = {'token': int(token), 'kind': 'CA_sidechain_angle', 'atoms': [a,'CA','CB'], 'deviation_degrees': error}
                if error > 15:
                    failures.append(row)
                if error > max(15, olderror)+1e-4:
                    regressions.append(row)
    physical = np.flatnonzero(~virtual)
    # Union includes new clashes as well as existing pairs, independent of totals.
    pairs = set(cKDTree(xyz[physical]).query_pairs(2.0)) | set(cKDTree(before[physical]).query_pairs(2.0))
    clashes, pair_regressions = [], []
    for a,b in sorted(pairs):
        i,j = int(physical[a]),int(physical[b]); ti,tj = tokens[i],tokens[j]
        if not (generated[i] or generated[j]) or (chains[ti] == chains[tj] and abs(positions[ti]-positions[tj]) <= 1):
            continue
        distance, prior = float(np.linalg.norm(xyz[i]-xyz[j])),float(np.linalg.norm(before[i]-before[j]))
        if distance < 2:
            clashes.append({'atoms': [i,j], 'distance': distance})
        if max(0,2-distance) > max(0,2-prior)+1e-5:
            pair_regressions.append({'atoms': [i,j], 'distance': distance, 'baseline_distance': prior})
    bb_guard = interface_backbone_geometry_guard(coordinates.reshape(1,-1,3), features)
    bb = bb_guard(coordinates.reshape(1,-1,3)) if bb_guard is not None else None
    return {'identity_bound': True, 'accepted': not regressions and not pair_regressions,
            'absolute_backbone_passed': bb is None or bb['candidate_geometry_passed'],
            'absolute_sidechain_geometry_passed': not failures,
            'sidechain_failure_count': len(failures), 'sidechain_failures_first100': failures[:100],
            'sidechain_regressions': regressions, 'proline_ring_checks': rings,
            'nonadjacent_allheavy_clash_pair_count': len(clashes),
            'allheavy_pair_regression_count': len(pair_regressions),
            'allheavy_pair_regressions_first100': pair_regressions[:100],
            'backbone_check': bb, 'scope': 'Final physical identity; CCD bond/CA-angle/chirality, PRO closure and each nonadjacent heavy pair at 2A; no core/fold certificate'}


def _export_residue_positions(keys):
    """Resolve insertion variants into distinct peptide positions.

    With no insertions this is exactly the declared numeric position. A
    standard ordered insertion advances one position; missing numeric or
    insertion labels retain a gap. Unsupported ambiguous ordering is unknown,
    never a relaxed geometric pass.
    """
    positions = []
    previous = {}
    for chain, number, insertion in keys:
        old = previous.get(chain)
        if old is None:
            position = number
        else:
            old_number, old_insertion, old_position = old
            if number > old_number:
                position = old_position + number - old_number
            elif number == old_number:
                rank = lambda code: 0 if not code else ord(code)-ord('A')+1 if len(code)==1 and 'A' <= code <= 'Z' else -1
                before, after = rank(old_insertion), rank(insertion)
                if before < 0 or after <= before:
                    raise ValueError('Unsupported insertion ordering')
                position = old_position + after - before
            elif insertion or old_insertion or old_position != old_number:
                raise ValueError('Nonmonotonic insertion residue ordering')
            else:
                position = number
        positions.append(position)
        previous[chain] = (number, insertion, position)
    return positions


def audit_exported_geometry(atom_array):
    """Audit the actual cleaned output, rather than assuming native slot names."""
    import torch
    from atomworks.ml.encoding_definitions import AF3SequenceEncoding
    from rfd3_mosaic.validation.generated_backbone import STANDARD_AMINO_ACIDS
    xyz = torch.as_tensor(np.asarray(atom_array.coord)).reshape(1,-1,3)
    keys = list(zip(atom_array.chain_id.tolist(),atom_array.res_id.tolist(),np.char.strip(atom_array.ins_code.astype(str)).tolist()))
    starts = [key for i,key in enumerate(keys) if i==0 or key!=keys[i-1]]
    if len(starts) != len(set(starts)):
        return {'stage':'after_virtual_cleanup_export','identity_bound':False,'accepted':False,'reason':'noncontiguous_duplicate_residue_identity'}
    unique = list(dict.fromkeys(keys)); lookup = {k:i for i,k in enumerate(unique)}
    tokens = np.array([lookup[k] for k in keys]); first = [keys.index(k) for k in unique]
    names = atom_array.res_name[first].tolist()
    try:
        positions = _export_residue_positions(unique)
    except ValueError as error:
        return {'stage':'after_virtual_cleanup_export','identity_bound':False,'accepted':False,'reason':'unsupported_export_residue_topology','error':str(error)}
    categories = atom_array.get_annotation_categories()
    protein = np.asarray(atom_array.is_protein)[first] if 'is_protein' in categories else np.array([n in STANDARD_AMINO_ACIDS for n in names])
    unresolved = [i for i,n in enumerate(names) if protein[i] and n not in STANDARD_AMINO_ACIDS]
    if unresolved:
        return {'stage':'after_virtual_cleanup_export','identity_bound':False,'accepted':False,'reason':'final_export_identity_unresolved','tokens':unresolved}
    chain_names = list(dict.fromkeys(atom_array.chain_id.tolist()))
    features = {'atom_to_token_map':torch.as_tensor(tokens),
        'is_motif_atom_with_fixed_coord':torch.as_tensor(np.asarray(atom_array.is_motif_atom_with_fixed_coord).astype(bool) if 'is_motif_atom_with_fixed_coord' in categories else np.zeros(len(keys),dtype=bool)),
        'is_virtual':torch.zeros(len(keys),dtype=torch.bool), 'is_protein':torch.as_tensor(protein),
        'asym_id':torch.tensor([chain_names.index(k[0]) for k in unique]),
        'residue_index':torch.tensor(positions),
        'restype':torch.as_tensor(AF3SequenceEncoding().encode(np.array(names))),
        'ref_atom_name_chars':torch.tensor([[ord(c)-32 for c in label.ljust(4)] for label in atom_array.atom_name.tolist()])}
    result = audit_bound_geometry(xyz,features)
    result['stage'] = 'after_virtual_cleanup_export'
    result['accepted'] = result['accepted'] and result['absolute_backbone_passed'] and result['absolute_sidechain_geometry_passed'] and result['nonadjacent_allheavy_clash_pair_count']==0
    return result
