"""Backbone-only partial-diffusion input must preserve supplied coordinates."""
import numpy as np
import pytest
from biotite.structure import AtomArray
from rfd3.transforms.virtual_atoms import PadTokensWithVirtualAtoms


def backbone():
    atoms = AtomArray(4)
    atoms.atom_name = np.array(['N', 'CA', 'C', 'O'])
    atoms.res_name[:] = 'ALA'
    atoms.chain_id[:] = 'A'
    atoms.res_id[:] = 1
    atoms.element = np.array(['N', 'C', 'C', 'O'])
    atoms.coord = np.array([[0,0,0],[1,1,0],[2,1,0],[3,1,0]],dtype=np.float32)
    for name, value in {'token_id':0,'is_protein':True,'atomize':False,
                        'is_motif_atom_with_fixed_seq':False,
                        'is_motif_atom_unindexed':False,'is_motif_atom':False,
                        'occupancy':1.0}.items():
        atoms.set_annotation(name,np.full(4,value))
    atoms.set_annotation('gt_atom_name',atoms.atom_name.copy())
    return atoms


def test_backbone_only_inference_pads_virtual_slots_without_moving_atoms():
    atoms=backbone(); before=atoms.coord.copy()
    result=PadTokensWithVirtualAtoms(14,'CB','dense').forward({'atom_array':atoms,'is_inference':True})['atom_array']
    np.testing.assert_array_equal(result.coord[:4],before)
    np.testing.assert_array_equal(result.coord[4:],np.repeat(before[1:2],10,axis=0))
    assert list(result.atom_name[:5])==['N','CA','C','O','CB']
    assert not result.is_motif_atom[4:].any()
    assert len(result)==14


def test_missing_backbone_is_not_silently_repaired():
    atoms=backbone()[:3]
    with pytest.raises(AssertionError):
        PadTokensWithVirtualAtoms(14,'CB','dense').forward({'atom_array':atoms,'is_inference':True})


def test_sequence_fixed_backbone_is_not_padded():
    atoms=backbone();atoms.is_motif_atom_with_fixed_seq[:]=True
    result=PadTokensWithVirtualAtoms(14,'CB','dense').forward({'atom_array':atoms,'is_inference':True})['atom_array']
    assert len(result)==4
