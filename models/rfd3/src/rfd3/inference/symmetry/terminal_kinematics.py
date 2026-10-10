"""Bounded exact covalent-axis proposals for one-anchor protein termini."""
import math
import torch


def terminal_kinematic_candidate(coordinates, gradient, features, topology, config, *, line_search_scale=1.):
    encoded=features.get('ref_atom_name_chars')
    if encoded is None:return None,{'reason':'no_real_backbone_binding'}
    if not torch.isfinite(coordinates).all() or not torch.isfinite(gradient).all():
        raise ValueError('Terminal kinematics requires finite coordinates and gradient')
    if encoded.ndim==2 and encoded.shape[-1]==256:encoded=encoded.reshape(-1,4,64)
    codes=encoded.argmax(-1) if encoded.ndim==3 else encoded
    names=[''.join(chr(int(i)+32) for i in row).strip() for row in codes.detach().cpu().tolist()]
    token_ids=features['atom_to_token_map'];chain_ids=features['asym_id'];positions=features['residue_index'];fixed=features['is_motif_atom_with_fixed_coord'];virtual=features['is_virtual']
    binding={}
    for i,name in enumerate(names):
        if name in ('N','CA','C','O') and not bool(virtual[i]):binding.setdefault(int(token_ids[i]),{})[name]=i
    if not binding:return None,{'reason':'no_real_backbone_binding'}
    if any(len(ids)!=4 for ids in binding.values()):
        raise ValueError('Terminal kinematics requires complete N/CA/C/O')
    protein=features.get('is_protein',torch.ones(len(chain_ids),dtype=torch.bool,device=chain_ids.device))
    generated={t:bool(protein[t]) and not bool(fixed[ids['CA']]) for t,ids in binding.items()}
    for t in binding:
        atom_mask=token_ids==t
        if bool(fixed[atom_mask].any()) and not bool(fixed[atom_mask].all()):
            generated[t]=False
    guided=set(token_ids[topology.generated_atom_mask].detach().cpu().tolist())
    work=coordinates.detach().double();grad=gradient.detach().double();hinges=[]
    def add(member_tokens,left,right,exclude=None,include=None):
        mask=torch.isin(token_ids,torch.tensor(member_tokens,device=token_ids.device))&~fixed
        if exclude is not None:mask[exclude]=False
        if include is not None and not bool(fixed[include]):mask[include]=True
        ids=torch.nonzero(mask).flatten()
        if not len(ids):return
        origin=work[:,left];axis=work[:,right]-origin;norm=torch.linalg.vector_norm(axis,dim=-1,keepdim=True)
        if bool((norm<1e-8).any()):return
        axis=axis/norm
        tangent=torch.cross(axis[:,None].expand_as(work[:,ids]),work[:,ids]-origin[:,None],dim=-1)
        derivative=(grad[:,ids]*tangent).sum((1,2))
        if bool((derivative.abs()>1e-10).any()):hinges.append((ids,left,right,derivative))
    for chain in torch.unique(chain_ids):
        tokens=torch.nonzero(chain_ids==chain).flatten();tokens=tokens[torch.argsort(positions[tokens])].detach().cpu().tolist()
        for terminus,ordered in [('N',tokens),('C',list(reversed(tokens)))]:
            branch=[]
            for t in ordered:
                if not generated.get(t,False):break
                branch.append(t)
            if not branch or len(branch)==len(tokens) or not guided.intersection(branch):continue
            anchor=ordered[len(branch)]
            if anchor not in binding or not bool(fixed[token_ids==anchor].all()):continue
            path=branch+[anchor]
            if any(abs(int(positions[a])-int(positions[b]))!=1 for a,b in zip(path,path[1:])):continue
            if terminus=='N':
                for i,t in enumerate(branch):
                    atom=binding[t]
                    add(branch[:i+1],atom['CA'],atom['C'],exclude=atom['O'])
                    if i>0:add(branch[:i],atom['N'],atom['CA'])
                anchor=ordered[len(branch)];atom=binding[anchor]
                add(branch,atom['N'],atom['CA'])
            else:
                increasing=list(reversed(branch))
                for i,t in enumerate(increasing):
                    atom=binding[t]
                    add(increasing[i:],atom['N'],atom['CA'])
                    if i+1<len(increasing):add(increasing[i+1:],atom['CA'],atom['C'],include=atom['O'])
    if not hinges:return None,{'reason':'no_terminal_gradient_direction'}
    angular_max=torch.stack([h[3].abs() for h in hinges]).max().clamp_min(1e-12)
    angles=[-h[3]/angular_max*math.radians(config.maximum_patch_rotation_degrees) for h in hinges]
    def rotate(scale):
        result=work.clone()
        for (ids,left,right,_),angle in zip(hinges,angles):
            origin=result[:,left].clone();axis=result[:,right]-origin;axis=axis/torch.linalg.vector_norm(axis,dim=-1,keepdim=True).clamp_min(1e-12)
            v=result[:,ids]-origin[:,None];a=axis[:,None];theta=(angle*scale)[:,None,None]
            result[:,ids]=origin[:,None]+v*theta.cos()+torch.cross(a.expand_as(v),v,dim=-1)*theta.sin()+a*(a*v).sum(-1,keepdim=True)*(1-theta.cos())
        return result
    scale=1.
    for iteration in range(10):
        proposed=rotate(scale);observed=float(torch.linalg.vector_norm(proposed-work,dim=-1).max())
        if observed<=config.maximum_token_step*.999:break
        scale*=config.maximum_token_step*.995/max(observed,1e-12)
    candidate=rotate(scale*line_search_scale).to(coordinates.dtype)
    maximum=float(torch.linalg.vector_norm(candidate-coordinates,dim=-1).max())
    if maximum>config.maximum_token_step+1e-5:return None,{'reason':'global_trust_cap_not_met','maximum_atom_step':maximum}
    return candidate,{'hinge_count':len(hinges),'maximum_atom_step':maximum,'common_angle_scale':scale*line_search_scale,'largest_per_joint_rotation_deg':float(angular_max*0+config.maximum_patch_rotation_degrees*scale*line_search_scale),'scope':'Exact recomputed-axis phi/psi, same0.25A atom trust cap and2deg per-joint bound; per-joint rotation bound, not a whole-patch rigid-angle bound'}
