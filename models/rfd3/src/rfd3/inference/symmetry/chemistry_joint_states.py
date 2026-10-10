"""Joint protonation/resonance states; equivalent labels remain interchangeable."""
import numpy as np


def joint_bond_states(name, physical, ideal):
    """Atoms, correlated target vectors, and unchanged 0.1A bond tolerance."""
    if name in ('ASP','GLU'):
        carbon,ends=('CG',('OD1','OD2')) if name=='ASP' else ('CD',('OE1','OE2'))
        if all(a in physical for a in (carbon,*ends)):
            return [([(carbon,ends[0]),(carbon,ends[1])],[(1.26,1.26),(1.23,1.34),(1.34,1.23)],'carboxylate_or_single_protonated_acid')]
    if name=='ARG' and all(a in physical for a in ('CZ','NE','NH1','NH2')):
        pairs=[('CZ','NE'),('CZ','NH1'),('CZ','NH2')]
        ccd=tuple(float(np.linalg.norm(ideal[a]-ideal[b])) for a,b in pairs)
        return [(pairs,[(1.33,1.33,1.33),ccd,(ccd[0],ccd[2],ccd[1])],'guanidinium_resonance_or_CCD_valence_with_equivalent_NH_labels')]
    return []


def choose_joint_state(pairs,states,xyz):
    lengths=np.array([np.linalg.norm(xyz[a]-xyz[b]) for a,b in pairs])
    # Minimax uses the existing per-bond acceptance tolerance jointly.
    errors=[float(np.abs(lengths-np.asarray(s)).max()) for s in states]
    index=int(np.argmin(errors))
    return states[index],errors[index],lengths.tolist()
