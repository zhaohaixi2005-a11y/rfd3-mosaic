"""Experimental conditional observations for partially constrained known CAs.

+Uses only declared known motif CA radius/z and a free angular proposal. It
+never imports generated reference coordinates or pins final angular values.
"""
import numpy as np
from scipy.optimize import least_squares


def condition_partial_cylindrical_cas(coordinates, features, *, target_ca_distance=3.8, ca_tolerance=.6):
    import torch
    keep=features.get('cylindrical_keep_mask')
    if keep is None:return coordinates,features,{'applicable':False,'reason':'no_cylindrical_constraints'}
    keep=keep.detach().cpu().numpy().astype(bool)
    if not keep.any():return coordinates,features,{'applicable':False,'reason':'no_active_cylindrical_constraints'}
    if coordinates.shape[0]!=1:raise ValueError('Experimental partial CA conditioning currently requires one diffusion batch')
    tokens=features['atom_to_token_map'].detach().cpu().numpy();ca=features['is_ca'].detach().cpu().numpy().astype(bool)
    seq_fixed=features['is_motif_atom_with_fixed_seq'].detach().cpu().numpy().astype(bool)
    fixed=features['is_motif_atom_with_fixed_coord'].detach().cpu().numpy().astype(bool)
    active=ca&seq_fixed&~fixed&keep[:,0]&keep[:,2]&~keep[:,1]
    if not active.any():return coordinates,features,{'applicable':False,'reason':'no_known_CA_radius_axial_partial_observations'}
    if np.any(keep.any(1)&~active&~fixed):raise ValueError('Experimental conditioning only supports known CA radius+axial with free azimuth')
    axis=features['cylindrical_axis'].detach().double().cpu().numpy();axis/=np.linalg.norm(axis)
    transformations=features['sym_transform'];center=np.stack([torch.as_tensor(t[1]).detach().double().cpu().numpy() for t in transformations.values()]).mean(0)
    for rotation,translation in transformations.values():
        if not np.allclose(torch.as_tensor(rotation).detach().double().cpu().numpy()@axis,axis,atol=1e-5):raise ValueError('Experimental partial cylinder conditioning requires axis-preserving symmetry')
    basis=np.eye(3)[np.argmin(abs(axis))];u=np.cross(axis,basis);u/=np.linalg.norm(u);v=np.cross(axis,u)
    xyz=coordinates.detach().double().cpu().numpy()[0];reference=features['cylindrical_reference'].detach().double().cpu().numpy()
    asym=features['asym_id'].detach().cpu().numpy();positions=features['residue_index'].detach().cpu().numpy();groups=[]
    for chain in dict.fromkeys(asym.tolist()):
        ids=np.flatnonzero(active&(asym[tokens]==chain));ids=sorted(ids,key=lambda i:positions[tokens[i]])
        block=[]
        for i in ids:
            if block and positions[tokens[i]]!=positions[tokens[block[-1]]]+1:groups.append(block);block=[]
            block.append(i)
        if block:groups.append(block)
    value=xyz.copy();details=[];canonical={}
    for ids in groups:
        ids=np.array(ids);source=reference[ids]-center;target_z=source@axis;radial=source-target_z[:,None]*axis;r=np.linalg.norm(radial,axis=-1);phi0=np.arctan2(radial@v,radial@u)
        proposed=xyz[ids]-center;desired=np.arctan2(proposed@v,proposed@u)
        delta=np.angle(np.exp(1j*(desired-phi0)));phase=float(np.angle(np.mean(np.exp(1j*delta))))
        signature=(len(ids),tuple(np.round(r,3)),tuple(np.round(target_z,3)))
        reused=signature in canonical
        if reused:angles=phi0+canonical[signature]
        else:
            def render(angles):return center+target_z[:,None]*axis+r[:,None]*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v)
            def residual(angles):
                points=render(angles);steps=np.linalg.norm(np.diff(points,axis=0),axis=-1)
                deviation=np.maximum(abs(steps-target_ca_distance)-ca_tolerance,0)
                parts=[.01*np.angle(np.exp(1j*(angles-desired))),deviation*100]
                if len(ids)>2:
                    left,right=np.triu_indices(len(ids),2);distances=np.linalg.norm(points[left]-points[right],axis=-1)
                    parts.append(np.maximum(3.2-distances,0)*100)
                return np.concatenate(parts)
            solution=least_squares(residual,phi0+phase,max_nfev=60,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            angles=solution.x;canonical[signature]=angles-phi0
        points=center+target_z[:,None]*axis+r[:,None]*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v)
        steps=np.linalg.norm(np.diff(points,axis=0),axis=-1)
        if len(steps) and np.any(abs(steps-target_ca_distance)>ca_tolerance+1e-4):
            # A coherent free global phase is a feasible initialization from
            # the supplied known CAs; not a lock on the allowed angular DOFs.
            angles=phi0+phase;points=center+target_z[:,None]*axis+r[:,None]*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v)
            steps=np.linalg.norm(np.diff(points,axis=0),axis=-1)
            canonical[signature]=angles-phi0
        if len(steps) and np.any(abs(steps-target_ca_distance)>ca_tolerance+1e-4):raise ValueError('Known partial CAs do not admit the existing peptide CA-distance range')
        value[ids]=points;details.append({'tokens':tokens[ids].tolist(),'symmetry_reused':reused,'maximum_CA_step_deviation':float(abs(steps-target_ca_distance).max()) if len(steps) else 0,'maximum_free_azimuth_change_radians':float(abs(np.angle(np.exp(1j*(angles-phi0)))).max())})
    noisy=coordinates.clone();active_gpu=torch.as_tensor(active,device=coordinates.device);noisy[0,active_gpu]=torch.as_tensor(value[active],device=coordinates.device,dtype=coordinates.dtype)
    network=dict(features);network['is_motif_atom_with_fixed_coord']=features['is_motif_atom_with_fixed_coord'].clone();network['is_motif_atom_with_fixed_coord'][active_gpu]=True
    network['motif_pos']=features['motif_pos'].clone();network['motif_pos'][active_gpu]=noisy[0,active_gpu].to(network['motif_pos'].dtype)
    count=int(tokens.max())+1;net_fixed=network['is_motif_atom_with_fixed_coord'];atom_tokens=features['atom_to_token_map']
    sizes=torch.bincount(atom_tokens,minlength=count);known=torch.bincount(atom_tokens,weights=net_fixed.float(),minlength=count)
    network['is_motif_token_with_fully_fixed_coord']=known==sizes
    return noisy,network,{'applicable':True,'active_CA_atoms':int(active.sum()),'groups':details,'scope':'Experimental constrained-CA noise chart and learned fixed-CA conditional view only; original hard masks unchanged; azimuth proposed from existing noise and remains free; no generated reference coordinates or backbone copied; initializer must be refreshed for this network view'}
