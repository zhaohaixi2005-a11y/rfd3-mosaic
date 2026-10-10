"""Reference-free sparse covalent relaxation probe with exact cylindrical CA DOFs."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from rfd3_mosaic.validation.generated_backbone import _angle, _dihedral


def relax_backbone(backbone, names, fixed_atoms, *, cylindrical=None, maximum_step=2., iterations=150, fixed_sidechain_obstacles=None, require_absolute_obstacle_clearance=False):
    source=np.asarray(backbone,dtype=float);n=len(source);flat=source.reshape(-1,3)
    fixed=np.asarray(fixed_atoms,dtype=bool).reshape(-1)
    generated=~np.asarray(fixed_atoms,dtype=bool).all(axis=1)
    dependencies=[[] for _ in flat];initial=[];lower=[];upper=[];cart=[];polar=[]
    if cylindrical is not None:
        axis=np.asarray(cylindrical['axis'],dtype=float);axis/=np.linalg.norm(axis)
        center=np.asarray(cylindrical['center'],dtype=float)
        basis=np.eye(3)[np.argmin(abs(axis))];u=np.cross(axis,basis);u/=np.linalg.norm(u);v=np.cross(axis,u)
    for i,point in enumerate(flat):
        if fixed[i]:continue
        keep=np.zeros(3,dtype=bool) if cylindrical is None else np.asarray(cylindrical['keep_mask'])[i]
        if not keep.any():
            ids=list(range(len(initial),len(initial)+3));dependencies[i]=ids;cart.append((i,ids));initial.extend([0.]*3);lower.extend([-maximum_step]*3);upper.extend([maximum_step]*3)
        else:
            target=np.asarray(cylindrical['reference'])[i]-center;current=point-center
            target_z=np.dot(target,axis);target_r=np.linalg.norm(target-target_z*axis);target_phi=np.arctan2(np.dot(target,v),np.dot(target,u))
            current_z=np.dot(current,axis);current_r=np.linalg.norm(current-current_z*axis);current_phi=np.arctan2(np.dot(current,v),np.dot(current,u))
            values=[target_r if keep[0] else current_r,target_phi if keep[1] else current_phi,target_z if keep[2] else current_z];ids=[]
            for j in range(3):
                if keep[j]:ids.append(None);continue
                k=len(initial);ids.append(k);dependencies[i].append(k);initial.append(0.)
                bound=.5 if j==1 else maximum_step;lower.append(max(-bound,-values[j]) if j==0 else -bound);upper.append(bound)
            polar.append((i,values,ids))
    if not initial:return source.copy(),{'success':True,'reason':'no_free_coordinates','evaluations':0}
    cart_atoms=np.array([i for i,ids in cart],dtype=int);cart_variables=np.array([ids for i,ids in cart],dtype=int).reshape(-1,3)
    def render(parameters):
        xyz=flat.copy();xyz[cart_atoms]+=parameters[cart_variables]
        for i,values,ids in polar:
            r,phi,z=[base+(0 if j is None else parameters[j]) for base,j in zip(values,ids)]
            xyz[i]=center+r*(np.cos(phi)*u+np.sin(phi)*v)+z*axis
        return xyz.reshape(n,4,3)
    groups=[]
    def rows(atom_sets):
        groups.extend(atom_sets)
    residue_ids=np.flatnonzero(generated);edge_ids=np.flatnonzero(generated[:-1]|generated[1:])
    for a,b in [(0,1),(1,2),(2,3)]:rows([[i*4+a,i*4+b] for i in residue_ids])
    for a,b,c in [(0,1,2),(1,2,3)]:rows([[i*4+a,i*4+b,i*4+c] for i in residue_ids])
    rows([[i*4+2,(i+1)*4] for i in edge_ids])
    for atoms in [(1,2,4),(2,4,5),(3,2,4),(1,2,4,5),(1,2,4,3)]:rows([[i*4+a for a in atoms] for i in edge_ids])
    candidates=cKDTree(flat).query_pairs(2+2*maximum_step,output_type='ndarray')
    if len(candidates):
        left=candidates[:,0];right=candidates[:,1]
        relevant=(abs(left//4-right//4)>1)&(~fixed[left]|~fixed[right]);pairs=candidates[relevant]
    else:pairs=np.empty((0,2),dtype=int)
    rows(pairs.tolist())
    ca_pairs=cKDTree(source[:,1]).query_pairs(3.3+2*np.sqrt(3)*maximum_step,output_type='ndarray')
    if len(ca_pairs):ca_pairs=ca_pairs[(abs(ca_pairs[:,0]-ca_pairs[:,1])>1)&(generated[ca_pairs[:,0]]|generated[ca_pairs[:,1]])]
    ca_atom_pairs=ca_pairs*4+1
    rows(ca_atom_pairs.tolist())
    obstacle_atoms=np.empty(0,dtype=int);obstacle_xyz=np.empty((0,3));obstacle_targets=np.empty(0)
    if fixed_sidechain_obstacles is not None:
        candidates=[]
        for i,point in enumerate(flat):
            if fixed[i]:continue
            for j in cKDTree(fixed_sidechain_obstacles['coordinates']).query_ball_point(point,2+2*np.sqrt(3)*maximum_step):
                if not fixed_sidechain_obstacles['eligible'][i//4,j]:continue
                candidates.append((i,j))
        if candidates:
            obstacle_atoms=np.array([a for a,b in candidates]);obstacle_xyz=np.array([fixed_sidechain_obstacles['coordinates'][b] for a,b in candidates])
            prior=np.linalg.norm(flat[obstacle_atoms]-obstacle_xyz,axis=-1)
            obstacle_targets=np.full_like(prior,2.03) if require_absolute_obstacle_clearance else np.minimum(2.03,prior+.02)
            rows([[i] for i in obstacle_atoms])
    for i in np.flatnonzero(~fixed):rows([[i]]*3)
    sparse=lil_matrix((len(groups),len(initial)),dtype=np.int8)
    for row,atoms in enumerate(groups):
        deps=[j for atom in atoms for j in dependencies[atom]]
        if deps:sparse[row,deps]=1
    names=np.asarray(names);nca=np.where(names=='GLY',1.451,np.where(names=='PRO',1.466,1.458));cac=np.where(names=='GLY',1.516,1.525);cn=np.where(names[1:]=='PRO',1.341,1.329)
    def residual(parameters):
        x=render(parameters);parts=[]
        for a,b,target in [(0,1,nca),(1,2,cac),(2,3,np.full(n,1.231))]:parts.append((np.linalg.norm(x[:,a]-x[:,b],axis=-1)[residue_ids]-target[residue_ids])/.03)
        parts.extend([(_angle(x[:,0],x[:,1],x[:,2])[residue_ids]-111.2)/5,(_angle(x[:,1],x[:,2],x[:,3])[residue_ids]-120.8)/5])
        parts.append((np.linalg.norm(x[:-1,2]-x[1:,0],axis=-1)[edge_ids]-cn[edge_ids])/.03)
        parts.extend([(_angle(x[:-1,1],x[:-1,2],x[1:,0])[edge_ids]-116.2)/5,(_angle(x[:-1,2],x[1:,0],x[1:,1])[edge_ids]-121.7)/5,(_angle(x[:-1,3],x[:-1,2],x[1:,0])[edge_ids]-123.)/5])
        for angle in [_dihedral(x[:-1,1],x[:-1,2],x[1:,0],x[1:,1]),_dihedral(x[:-1,1],x[:-1,2],x[1:,0],x[:-1,3])]:parts.append(np.sin(np.radians(angle[edge_ids]))/np.sin(np.radians(10.)))
        all_atoms=x.reshape(-1,3)
        parts.append(np.maximum(2.10-np.linalg.norm(all_atoms[pairs[:,0]]-all_atoms[pairs[:,1]],axis=-1),0)/.03)
        parts.append(np.maximum(3.3-np.linalg.norm(all_atoms[ca_atom_pairs[:,0]]-all_atoms[ca_atom_pairs[:,1]],axis=-1),0)/.03)
        parts.append(np.maximum(obstacle_targets-np.linalg.norm(all_atoms[obstacle_atoms]-obstacle_xyz,axis=-1),0)/.005)
        parts.append(((all_atoms-flat)[~fixed]*.1).reshape(-1))
        result=np.concatenate(parts)
        if not np.isfinite(result).all():raise ValueError('Nonfinite covalent relaxation residual')
        assert len(result)==len(groups)
        return result
    result=least_squares(residual,np.array(initial),bounds=(lower,upper),jac_sparsity=sparse.tocsr(),max_nfev=iterations,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    candidate=render(result.x)
    return candidate,{'success':bool(result.success),'message':result.message,'evaluations':result.nfev,'cost':result.cost,'maximum_backbone_step':float(np.linalg.norm(candidate-source,axis=-1).max()),'variables':len(initial),'residuals':len(groups),'reference_backbone_used':False,'hard_cylindrical_parameterization':cylindrical is not None}
