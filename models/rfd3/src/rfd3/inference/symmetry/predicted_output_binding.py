"""Bind the exact batch identity and association policy used by final export."""
import numpy as np


def bind_predicted_output(features, sequence_indices, *, batch=0, association_scheme='dense'):
    import torch
    from atomworks.ml.encoding_definitions import AF3SequenceEncoding
    from rfd3.constants import ATOM14_ATOM_NAMES,association_schemes
    from rfd3_mosaic.validation.generated_backbone import STANDARD_AMINO_ACIDS
    required=('atom_to_token_map','ref_atom_name_chars','restype','is_motif_atom_with_fixed_coord')
    if any(key not in features for key in required):
        return None,{'bound':False,'reason':'no_full_final_identity_binding','missing':[key for key in required if key not in features]}
    encoding=AF3SequenceEncoding();tokens=features['atom_to_token_map'].detach().cpu().numpy()
    encoded=features['ref_atom_name_chars']
    if encoded.ndim==2 and encoded.shape[-1]==256:encoded=encoded.reshape(-1,4,64)
    codes=encoded.argmax(-1) if encoded.ndim==3 else encoded
    raw=[''.join(chr(int(c)+32) for c in row).strip() for row in codes.detach().cpu().tolist()]
    restype=features['restype'];restype=restype.argmax(-1) if restype.ndim==2 else restype
    conditioning=encoding.decode(restype.detach().cpu().numpy()).tolist();count=len(conditioning)
    fixed=features.get('is_motif_atom_with_fixed_seq',features['is_motif_atom_with_fixed_coord']).detach().cpu().numpy().astype(bool)
    protein=features.get('is_protein',torch.ones(count,dtype=torch.bool)).detach().cpu().numpy().astype(bool)
    prediction=None
    if sequence_indices is not None:
        indices=sequence_indices.detach().cpu().numpy()
        if indices.ndim==3:indices=indices.argmax(-1)
        if indices.ndim==2:indices=indices[batch]
        if indices.shape!=(count,):raise ValueError('Final predicted sequence must bind every native token')
        prediction=encoding.decode(indices.astype(int)).tolist()
    names=[];unknown=[]
    for token,original in enumerate(conditioning):
        keep=bool(fixed[tokens==token].all())
        name=original if keep or prediction is None else prediction[token]
        if protein[token] and (name not in STANDARD_AMINO_ACIDS or (not keep and prediction is None and 'final_output_uses_sequence_head' in features)):unknown.append(token)
        names.append(name)
    if unknown:
        return None,{'bound':False,'reason':'final_output_identity_unresolved','tokens':unknown}
    if association_scheme not in association_schemes:raise ValueError('Unknown final export association scheme')
    slots={name:i for i,name in enumerate(ATOM14_ATOM_NAMES.tolist())};labels=[];virtual=[]
    original_virtual=features.get('is_virtual',torch.zeros(len(tokens),dtype=torch.bool)).detach().cpu().numpy().astype(bool)
    for i,label in enumerate(raw):
        token=tokens[i]
        if not protein[token]:labels.append(label);virtual.append(bool(original_virtual[i]));continue
        scheme=association_schemes[association_scheme][names[token]]
        if label in slots:
            physical=scheme[slots[label]]
            labels.append(physical.strip() if physical is not None else 'VX');virtual.append(physical is None)
        else:
            # Actual exported-name fixtures already use physical atom labels.
            labels.append(label);virtual.append(label not in {v.strip() for v in scheme if v is not None})
    bound=dict(features)
    bound['restype']=torch.as_tensor(encoding.encode(np.array(names)),device=restype.device)
    bound['ref_atom_name_chars']=torch.tensor([[ord(c)-32 for c in label.ljust(4)] for label in labels],device=codes.device)
    bound['is_virtual']=torch.tensor(virtual,dtype=torch.bool,device=features['atom_to_token_map'].device)
    return bound,{'bound':True,'batch':batch,'association_scheme':association_scheme,'identity_source':'final sequence_indices_I with fixed-sequence tokens retained' if prediction is not None else 'fully known conditioning','residue_names':names}
