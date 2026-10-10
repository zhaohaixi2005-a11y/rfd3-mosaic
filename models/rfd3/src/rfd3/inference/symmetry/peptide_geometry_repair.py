"""Bounded reference-free peptide closure with absolute projected backbone checks."""
import numpy as np
from rfd3_mosaic.validation.generated_backbone import audit_generated_backbone, default_backbone_policy, _dihedral, _angle


def _repair_backbone(backbone, residues, names, *, iterations=80, maximum_step=.5, movable_mask=None):
    x = np.asarray(backbone, dtype=float).copy(); original=x.copy()
    generated=np.array([not r['fixed'] for r in residues]);movable=generated if movable_mask is None else generated & np.asarray(movable_mask,dtype=bool);chains=np.array([r['chain_id'] for r in residues]);names=np.array(names)
    relevant=(chains[:-1]==chains[1:])&(generated[:-1]|generated[1:])
    contract={'residues':residues,'backbone_policy':default_backbone_policy(),'helix_blocks':[],'support_edges':[], 'limits':{'contact_distance':8.,'geometry_tolerance':.001,'minimum_helix_contact_fraction':.5,'maximum_unsupported_run':8}}
    def audit(v):return audit_generated_backbone(contract=contract,backbone_coordinates=v,residue_names=names)
    before=audit(x); history=[]
    for iteration in range(iterations):
        report=audit(x)
        cn=np.linalg.norm(x[1:,0]-x[:-1,2],axis=-1);target=np.where(names[1:]=='PRO',1.341,1.329)
        if report['geometry_passed'] and np.max(np.abs(cn[relevant]-target[relevant]),initial=0)<.095:
            break
        shift=np.zeros((len(x),3));counts=np.zeros(len(x))
        for i in np.flatnonzero(relevant & (np.abs(cn-target)>.095)):
            j=i+1;direction=(x[j,0]-x[i,2])/max(cn[i],1e-12)
            error=cn[i]-target[i]
            correction=np.clip(np.sign(error)*(abs(error)-.09),-.08,.08)*direction
            if movable[i] and movable[j]:shift[i]+=.5*correction;shift[j]-=.5*correction;counts[[i,j]]+=1
            elif movable[i]:shift[i]+=correction;counts[i]+=1
            elif movable[j]:shift[j]-=correction;counts[j]+=1
        for clash in report['nonlocal_backbone_clashes']:
            i=clash['left_residue_index'];j=clash['right_residue_index'];a=['N','CA','C','O'].index(clash['left_atom']);b=['N','CA','C','O'].index(clash['right_atom'])
            d=x[j,b]-x[i,a];distance=np.linalg.norm(d)
            if distance<1e-8:continue
            correction=min(2.05-distance,.1)*d/distance
            if movable[i] and movable[j]:shift[i]-=.5*correction;shift[j]+=.5*correction;counts[[i,j]]+=1
            elif movable[i]:shift[i]-=correction;counts[i]+=1
            elif movable[j]:shift[j]+=correction;counts[j]+=1
        shift/=np.maximum(counts,1)[:,None]
        x+=shift[:,None,:]
        def rotate_residue(index,pivot,axis,theta):
            u=x[index]-pivot
            return pivot+u*np.cos(theta)+np.cross(axis,u)*np.sin(theta)+np.sum(u*axis,axis=-1)[:,None]*axis*(1-np.cos(theta))
        # Repair angular deficits created by length closure using whole-residue
        # rotations about the actual junction, never moving fixed atoms.
        for kind, target_angle in [('O_C_N_angle',123.),('CA_C_N_angle',116.2),('C_N_CA_angle',121.7)]:
            for i in np.flatnonzero(relevant):
                j=i+1
                def points():
                    if kind=='O_C_N_angle':return x[i,3],x[i,2],x[j,0]
                    if kind=='CA_C_N_angle':return x[i,1],x[i,2],x[j,0]
                    return x[i,2],x[j,0],x[j,1]
                a,b,c=points();angle=float(_angle(a,b,c))
                if abs(angle-target_angle)<=14.8:continue
                pivot=b.copy();axis=np.cross(a-b,c-b);norm=np.linalg.norm(axis)
                if norm<1e-8:continue
                axis/=norm
                index=(j if movable[j] else i) if kind=='C_N_CA_angle' else (i if movable[i] else j)
                if not movable[index]:continue
                saved=x[index].copy();x[index]=rotate_residue(index,pivot,axis,1e-4)
                derivative=(float(_angle(*points()))-angle)/1e-4;x[index]=saved
                if abs(derivative)<1e-6:continue
                goal=target_angle+np.sign(angle-target_angle)*14.8
                theta=np.clip((goal-angle)/derivative,-.035,.035)
                x[index]=rotate_residue(index,pivot,axis,theta)
        omega=_dihedral(x[:-1,1],x[:-1,2],x[1:,0],x[1:,1])
        for i in np.flatnonzero(relevant & (np.minimum(abs(omega),abs(180-abs(omega)))>19.5)):
            j=i+1;index=j if movable[j] else i;axis=x[j,0]-x[i,2];axis/=max(np.linalg.norm(axis),1e-12);pivot=x[j,0] if index==j else x[i,2]
            sign=np.sign(omega[i]);goal=sign*19.5 if abs(omega[i])<90 else sign*160.5
            if not movable[index]:continue
            saved=x[index].copy();x[index]=rotate_residue(index,pivot,axis,1e-4)
            perturbed=float(_dihedral(x[i,1],x[i,2],x[j,0],x[j,1]));x[index]=saved
            derivative=((perturbed-omega[i]+180)%360-180)/1e-4
            theta=np.clip(((goal-omega[i]+180)%360-180)/derivative,-.06,.06)
            x[index]=rotate_residue(index,pivot,axis,theta)
        step=float(np.linalg.norm(x-original,axis=-1).max())
        history.append({'iteration':iteration,'geometry_failures':report['geometry_failure_count'],'clashes':report['nonlocal_backbone_clash_count'],'maximum_total_backbone_step':step})
        if step>maximum_step:
            return original, {'accepted':False,'reason':'maximum_displacement','before':before,'last':audit(x),'history':history}
    after=audit(x)
    accepted=after['geometry_passed']
    return x if accepted else original, {'accepted':bool(accepted),'before':before,'last':after,'history':history,'maximum_backbone_displacement':float(np.linalg.norm(x-original,axis=-1).max())}



def repair_native_generated_peptides(coordinates, features, *, projector, candidate_validator=None, maximum_step=.5):
    """Close real generated junctions with bounded rigid residue transport.

    Only accepted, fully projected, absolutely feasible backbones are published.
    All residue atoms, including sidechains and virtual slots, share one proper
    rigid transform. This uses the network's conformation, without a reference
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
        identities=[name if name in STANDARD_AMINO_ACIDS else 'ALA' for name in decoded]
    names=[identities[t] for t in order]
    original=coordinates.detach().double().cpu().numpy()
    proposed=original.copy();details=[];guards=[]
    for batch in range(len(original)):
        guard=interface_backbone_geometry_guard(coordinates[batch:batch+1],features);guards.append(guard)
        bb=original[batch,indices]
        repaired,detail=_repair_backbone(bb,residues,names,maximum_step=maximum_step,movable_mask=movable)
        details.append({'solver_accepted':detail['accepted'],'reason':detail.get('reason'),'iterations':len(detail['history']),
                        'before_geometry_failures':detail['before']['geometry_failure_count'],
                        'before_backbone_clashes':detail['before']['nonlocal_backbone_clash_count'],
                        'proposed_geometry_failures':detail['last']['geometry_failure_count'],
                        'proposed_backbone_clashes':detail['last']['nonlocal_backbone_clash_count']})
        if not detail['accepted']:continue
        for j,t in enumerate(order):
            if not movable[j] or np.array_equal(bb[j],repaired[j]):continue
            origin=bb[j].mean(axis=0);target=repaired[j].mean(axis=0)
            u,_,v=np.linalg.svd((bb[j]-origin).T@(repaired[j]-target))
            rotation=u@v
            if np.linalg.det(rotation)<0:u[:,-1]*=-1;rotation=u@v
            atom_mask=tokens==t
            proposed[batch,atom_mask]=(original[batch,atom_mask]-origin)@rotation+target
    diagnostic={'applicable':True,'applied':False,'batches':details,'maximum_allowed_atom_step':maximum_step,
                'scope':'Reference-free local rigid-residue peptide closure and backbone clash resolution; hard projections and absolute full-backbone feasibility checked after actual all-atom transport; not folding/function certification'}
    if np.array_equal(proposed,original):
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
    if candidate_validator is not None:
        extra=candidate_validator(candidate);diagnostic['task_guard']=extra
        if not extra.get('accepted',extra.get('passed',False)):
            diagnostic['reason']='task_guard_rejected'
            return coordinates,diagnostic
    diagnostic.update(applied=True,reason='absolute_projected_geometry_feasible')
    return candidate,diagnostic
