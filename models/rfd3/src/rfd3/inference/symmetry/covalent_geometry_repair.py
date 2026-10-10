import numpy as np


def _bounded_residue_transport(atoms, origin, target, rotation, maximum_step):
    """Rigidly transport a residue without spending the whole-atom trust budget.

    A modest backbone rotation can move a distal sidechain atom much farther
    than any backbone atom.  If that full rotation already exceeds the existing
    absolute atom budget, retain the residue's original orientation and apply
    only the CA translation.  The downstream sidechain solver and unchanged
    absolute chemistry/clearance guards still decide whether the candidate can
    commit.
    """
    rigid=(atoms-origin)@rotation+target
    rigid_step=float(np.linalg.norm(rigid-atoms,axis=-1).max())
    if rigid_step<=maximum_step+1e-12:
        return rigid,{'mode':'full_rigid','full_rigid_maximum_atom_step':rigid_step}
    translated=atoms+(target-origin)
    return translated,{
        'mode':'translation_fallback',
        'full_rigid_maximum_atom_step':rigid_step,
        'translation_maximum_atom_step':float(np.linalg.norm(translated-atoms,axis=-1).max()),
    }


def _repair_bound_covalent_geometry(coordinates, features, *, projector, candidate_validator=None, candidate_transformer=None, maximum_step=None, force_project_unchanged=False):
    """Repair actual generated covalent geometry with a bounded fallback.

    Only accepted, fully projected, absolutely feasible backbones are published.
    Backbone coordinates deform; remaining slots share a proper CA-anchored
    rigid transform. Known proline N-CD closure is checked; this is not full sidechain packing certification. This uses the network's conformation, without a reference
    scaffold or a new model loss. Failure retains the original output and its
    explicit failed physical diagnostic.
    """
    import torch
    from rfd3.inference.symmetry.interface_backbone_guard import interface_backbone_geometry_guard
    encoded=features.get('ref_atom_name_chars')
    if encoded is None:
        return coordinates, {'applicable':False,'reason':'no_full_backbone_binding'}
    if encoded.ndim==2 and encoded.shape[-1]==256:encoded=encoded.reshape(-1,4,64)
    codes=encoded.argmax(-1) if encoded.ndim==3 else encoded
    if codes.ndim!=2 or codes.shape[1]!=4:
        raise ValueError('Peptide repair requires native N/CA/C/O atom names')
    atom_names=[''.join(chr(int(i)+32) for i in row).strip() for row in codes.detach().cpu().tolist()]
    tokens=features['atom_to_token_map'].detach().cpu().numpy()
    fixed=features['is_motif_atom_with_fixed_coord'].detach().cpu().numpy().astype(bool)
    virtual=features.get('is_virtual',torch.zeros(len(tokens),dtype=torch.bool)).detach().cpu().numpy().astype(bool)
    chains=features['asym_id'].detach().cpu().numpy()
    positions=features['residue_index'].detach().cpu().numpy()
    protein=features.get('is_protein',torch.ones(len(chains),dtype=torch.bool)).detach().cpu().numpy().astype(bool)
    binding={}
    for i,name in enumerate(atom_names):
        if protein[tokens[i]] and not virtual[i] and name in ('N','CA','C','O'):
            if name in binding.setdefault(int(tokens[i]),{}):raise ValueError('Duplicate native backbone atom')
            binding[int(tokens[i])][name]=i
    order=sorted(binding,key=lambda t:(chains[t],positions[t]))
    if not order:return coordinates,{'applicable':False,'reason':'no_protein_backbone'}
    if any(len(binding[t])!=4 for t in order):raise ValueError('Peptide repair requires complete real N/CA/C/O')
    indices=np.array([[binding[t][name] for name in ('N','CA','C','O')] for t in order])
    residues=[{'chain_id':str(chains[t]),'residue_number':int(positions[t]),'fixed':bool(fixed[indices[j]].all())} for j,t in enumerate(order)]
    movable=np.array([not fixed[tokens==t].any() for t in order])
    identities=['ALA']*len(chains)
    if 'restype' in features:
        from atomworks.ml.encoding_definitions import AF3SequenceEncoding
        restype=features['restype'];restype=restype.argmax(-1) if restype.ndim==2 else restype
        decoded=AF3SequenceEncoding().decode(restype.detach().cpu().numpy()).tolist()
        from rfd3_mosaic.validation.generated_backbone import STANDARD_AMINO_ACIDS
        if any(name not in STANDARD_AMINO_ACIDS for i,name in enumerate(decoded) if protein[i]):
            raise ValueError('Covalent repair requires resolved final protein identities')
        identities=decoded
    names=[identities[t] for t in order]
    from rfd3.inference.symmetry.covalent_backbone_relaxation import relax_backbone
    has_cylindrical=bool(features.get('cylindrical_keep_mask') is not None and features['cylindrical_keep_mask'].any())
    maximum_step=(9.5 if has_cylindrical else 3.0) if maximum_step is None else maximum_step
    original=coordinates.detach().double().cpu().numpy()
    proposed=original.copy();details=[];guards=[]
    for batch in range(len(original)):
        guard=interface_backbone_geometry_guard(coordinates[batch:batch+1],features);guards.append(guard)
        bb=original[batch,indices]
        if guard is not None and guard(coordinates[batch:batch+1])['candidate_geometry_passed'] and not force_project_unchanged:
            details.append({'already_feasible':True});continue
        repaired=bb.copy();chain_details=[];transport_details=[];canonical=None
        for chain in dict.fromkeys(r['chain_id'] for r in residues):
            ids=np.array([j for j,r in enumerate(residues) if r['chain_id']==chain]);cyl=None
            if has_cylindrical:
                native_indices=indices[ids]
                cyl={'axis':features['cylindrical_axis'].detach().cpu().numpy().copy(),'center':np.stack([torch.as_tensor(transform[1]).detach().cpu().numpy() for transform in features['sym_transform'].values()]).astype(float).mean(axis=0),'keep_mask':features['cylindrical_keep_mask'].detach().cpu().numpy()[native_indices].reshape(-1,3),'reference':features['cylindrical_reference'].detach().cpu().numpy()[native_indices].reshape(-1,3)}
            reused=False
            if canonical is not None:
                ref_ids,ref_part=canonical
                if len(ref_ids)==len(ids) and np.array_equal(np.array(names)[ref_ids],np.array(names)[ids]) and np.array_equal(movable[ref_ids],movable[ids]):
                    p=bb[ref_ids].reshape(-1,3);q=bb[ids].reshape(-1,3);left,_,right=np.linalg.svd((p-p.mean(0)).T@(q-q.mean(0)));rotation=left@right
                    if np.linalg.det(rotation)<0:left[:,-1]*=-1;rotation=left@right
                    residual=float(np.linalg.norm((p-p.mean(0))@rotation+q.mean(0)-q,axis=-1).max())
                    if residual<0.002:
                        part=(ref_part-p.mean(0))@rotation+q.mean(0);part[~movable[ids]]=bb[ids[~movable[ids]]];d={'symmetry_consistent_reuse':True,'fit_max_angstrom':residual};reused=True
            if not reused:
                obstacle_ids=np.array([i for i,name in enumerate(atom_names) if fixed[i] and not virtual[i] and name not in ('N','CA','C','O','OXT')],dtype=int)
                obstacles=None
                if len(obstacle_ids):
                    obstacle_tokens=tokens[obstacle_ids];current_tokens=np.array(order)[ids]
                    eligible=(chains[current_tokens,None]!=chains[obstacle_tokens][None,:]) | (abs(positions[current_tokens,None]-positions[obstacle_tokens][None,:])>1)
                    obstacles={'coordinates':original[batch,obstacle_ids], 'eligible':eligible}
                part,d=relax_backbone(bb[ids],np.array(names)[ids],np.repeat((~movable[ids])[:,None],4,axis=1),cylindrical=cyl,maximum_step=6.0 if has_cylindrical else 2.0,iterations=400 if has_cylindrical else 150,fixed_sidechain_obstacles=obstacles,require_absolute_obstacle_clearance=force_project_unchanged)
                if canonical is None:canonical=(ids,part)

            repaired[ids]=part;chain_details.append(d)
        details.append({'chains':chain_details,'residue_transport_fallbacks':transport_details})
        for j,t in enumerate(order):
            if not movable[j] or np.array_equal(bb[j],repaired[j]):continue
            origin=bb[j,1];target=repaired[j,1]
            u,_,v=np.linalg.svd((bb[j]-origin).T@(repaired[j]-target))
            rotation=u@v
            if np.linalg.det(rotation)<0:u[:,-1]*=-1;rotation=u@v
            if names[j]=='PRO':
                a=bb[j,0]-origin;b=repaired[j,0]-target;a/=np.linalg.norm(a);b/=np.linalg.norm(b);cross=np.cross(a,b);cosine=float(np.dot(a,b))
                if cosine<-1+1e-8:
                    axis=np.cross(a,np.eye(3)[np.argmin(np.abs(a))]);axis/=np.linalg.norm(axis);rotation=2*np.outer(axis,axis)-np.eye(3)
                else:
                    vx,vy,vz=cross;skew=np.array([[0,-vz,vy],[vz,0,-vx],[-vy,vx,0]]);rotation=(np.eye(3)+skew+skew@skew/(1+cosine)).T
            atom_mask=tokens==t
            transported,transport=_bounded_residue_transport(
                original[batch,atom_mask],origin,target,rotation,maximum_step
            )
            proposed[batch,atom_mask]=transported
            if transport['mode']!='full_rigid':
                transport_details.append({
                    'token':int(t),'chain_id':residues[j]['chain_id'],
                    'residue_number':residues[j]['residue_number'],
                    'residue_name':names[j],**transport,
                })
            proposed[batch,indices[j]]=repaired[j]
    diagnostic={'applicable':True,'applied':False,'batches':details,'maximum_allowed_atom_step':maximum_step,
                'scope':'Reference-free sparse covalent geometry relaxation; backbone atoms adjusted independently; remaining residue atoms use rigid CA transport with translation-only trust-region fallback; hard projections and absolute full-backbone feasibility checked after actual all-atom transport; not folding/function certification'}
    if np.array_equal(proposed,original) and not force_project_unchanged:
        diagnostic['absolute_backbone_passed']=all(g is None or g(coordinates[b:b+1])['candidate_geometry_passed'] for b,g in enumerate(guards))
        diagnostic['reason']='unchanged'
        return coordinates,diagnostic
    candidate=torch.as_tensor(proposed,dtype=coordinates.dtype,device=coordinates.device)
    candidate=projector(candidate)
    step=float(torch.linalg.vector_norm(candidate-coordinates,dim=-1).max().item())
    audits=[g(candidate[b:b+1]) for b,g in enumerate(guards) if g is not None]
    diagnostic.update(maximum_atom_step=step,absolute_backbone_passed=all(a['candidate_geometry_passed'] for a in audits),actual_projected_backbone_checks=audits)
    if step>maximum_step+1e-6 or not diagnostic['absolute_backbone_passed']:
        diagnostic['reason']='projected_candidate_not_absolutely_feasible'
        return coordinates,diagnostic
    if not torch.equal(candidate[:,torch.as_tensor(fixed,device=coordinates.device)],coordinates[:,torch.as_tensor(fixed,device=coordinates.device)]):
        diagnostic['reason']='fixed_atom_changed'
        return coordinates,diagnostic
    if candidate_transformer is not None:
        try:
            candidate, refinement = candidate_transformer(candidate.detach().clone(), maximum_step)
            diagnostic['candidate_refinement'] = refinement
            if not refinement.get('applied', False):
                diagnostic['reason'] = 'candidate_refinement_rejected'
                return coordinates, diagnostic
            if candidate.shape != coordinates.shape or candidate.dtype != coordinates.dtype or candidate.device != coordinates.device or not torch.isfinite(candidate).all():
                raise ValueError('Refinement changed native coordinate identity')
            step=float(torch.linalg.vector_norm(candidate-coordinates,dim=-1).max().item())
            audits=[g(candidate[b:b+1]) for b,g in enumerate(guards) if g is not None]
            if step>maximum_step+1e-6 or not all(a['candidate_geometry_passed'] for a in audits) or not torch.equal(candidate[:,torch.as_tensor(fixed,device=coordinates.device)],coordinates[:,torch.as_tensor(fixed,device=coordinates.device)]):
                raise ValueError('Refinement failed final native guards')
            diagnostic.update(maximum_atom_step=step, actual_projected_backbone_checks=audits)
        except Exception as error:
            diagnostic.update(reason='candidate_refinement_error', error_type=type(error).__name__)
            return coordinates, diagnostic
    if candidate_validator is not None:
        extra=candidate_validator(candidate);diagnostic['task_guard']=extra
        if not extra.get('accepted',extra.get('passed',False)):
            diagnostic['reason']='task_guard_rejected'
            return coordinates,diagnostic
    rings=[]
    for j,t in enumerate(order):
        if names[j]!='PRO' or residues[j]['fixed']:continue
        delta=[i for i in np.flatnonzero(tokens==t) if atom_names[i]=='CD' and not virtual[i]]
        if len(delta)>1:raise ValueError('Duplicate proline CD atom')
        if not delta:continue
        lengths=torch.linalg.vector_norm(candidate[:,indices[j,0]]-candidate[:,delta[0]],dim=-1)
        rings.append({'token':int(t),'N_CD_lengths':lengths.detach().cpu().tolist()})
        if bool(((lengths-1.47).abs()>.1).any()):
            diagnostic.update(reason='proline_ring_closure_failed',proline_ring_checks=rings)
            return coordinates,diagnostic
    diagnostic['proline_ring_checks']=rings
    diagnostic.update(applied=True,reason='absolute_projected_geometry_feasible')
    return candidate,diagnostic


def repair_native_covalent_geometry(coordinates, features, *, projector, candidate_validator=None, candidate_transformer=None, per_sample_candidate_validator=None, maximum_step=None, sequence_indices=None, association_scheme=None, require_absolute_clearance=False):
    """Bind each final predicted batch and reject unresolved or regressing candidates."""
    from rfd3.inference.symmetry.predicted_output_binding import bind_predicted_output
    from rfd3.inference.symmetry.output_geometry_guard import audit_bound_geometry
    scheme = association_scheme or features.get('final_output_association_scheme', 'dense')
    working = coordinates.clone(); diagnostics=[]
    for batch in range(len(coordinates)):
        bound,binding = bind_predicted_output(features,sequence_indices,batch=batch,association_scheme=scheme)
        if bound is None:
            diagnostics.append({'applied':False,'absolute_backbone_passed':False,'reason':binding['reason'],'identity_binding':binding}); continue
        sidechain_polish=[]
        def project(value):
            full=working.clone(); full[batch:batch+1]=value
            projected=projector(full)[batch:batch+1]
            chemistry=audit_bound_geometry(projected,bound,coordinates[batch:batch+1])
            if chemistry["accepted"] and chemistry["absolute_sidechain_geometry_passed"] and (not require_absolute_clearance or chemistry["nonadjacent_allheavy_clash_pair_count"] == 0):
                return projected
            from rfd3.inference.symmetry.sidechain_geometry_relaxation import relax_sidechains
            polished,polish=relax_sidechains(projected,coordinates[batch:batch+1],bound,require_absolute_clearance=require_absolute_clearance)
            sidechain_polish.append(polish)
            full=working.clone();full[batch:batch+1]=polished
            return projector(full)[batch:batch+1]
        def validate(value):
            chemistry=audit_bound_geometry(value,bound,coordinates[batch:batch+1])
            full=working.clone(); full[batch:batch+1]=value
            task=per_sample_candidate_validator(value,batch) if per_sample_candidate_validator is not None else candidate_validator(full) if candidate_validator is not None else {'accepted':True}
            return {'accepted':chemistry['accepted'] and chemistry['absolute_backbone_passed'] and chemistry['absolute_sidechain_geometry_passed'] and (not require_absolute_clearance or chemistry['nonadjacent_allheavy_clash_pair_count']==0) and task.get('accepted',task.get('passed',False)), 'final_identity_geometry':chemistry,'original_task_guard':task}
        def transform(value, budget):
            return candidate_transformer(value, budget, batch, bound, binding)
        try:
            initial_chemistry=audit_bound_geometry(working[batch:batch+1],bound)
            force_polish=require_absolute_clearance and (not initial_chemistry['absolute_sidechain_geometry_passed'] or initial_chemistry['nonadjacent_allheavy_clash_pair_count']!=0)
            value,detail=_repair_bound_covalent_geometry(working[batch:batch+1],bound,projector=project,candidate_validator=validate,candidate_transformer=transform if candidate_transformer is not None else None,maximum_step=maximum_step,force_project_unchanged=force_polish)
            detail['sidechain_polish']=sidechain_polish
            detail['identity_binding']=binding
            # Audit unchanged outputs too: an unchanged defective prediction is not a repair.
            detail['final_identity_geometry']=audit_bound_geometry(value,bound,coordinates[batch:batch+1])
        except Exception as error:
            if candidate_transformer is None:
                raise
            value=coordinates[batch:batch+1].clone()
            detail={'applied':False,'absolute_backbone_passed':False,
                'reason':'per_design_covalent_repair_error','error_type':type(error).__name__,
                'error':str(error),'identity_binding':binding,'sidechain_polish':sidechain_polish}
        working[batch:batch+1]=value;diagnostics.append(detail)
    return working, {**(diagnostics[0] if len(diagnostics)==1 else {}), 'applicable':True,'applied':any(d.get('applied',False) for d in diagnostics),
        'absolute_backbone_passed':all(d.get('absolute_backbone_passed',False) for d in diagnostics),
        'reason':diagnostics[0].get('reason') if len(diagnostics)==1 else 'per_batch_final_identity_checks','batches':diagnostics,
        'scope':'Final predicted identities and actual export association slots; each heavy clash pair guarded; failed candidates retained as original outputs'}
