import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from biotite.structure.info import residue
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from rfd3.inference.symmetry.output_geometry_guard import audit_exported_geometry
from rfd3.inference.symmetry.final_output_geometry import finalize_generated_output


def physical_residue(name):
    reference=residue(name);aa=reference[reference.element!='H'].copy();aa.chain_id[:]='A';aa.res_id[:]=1;aa.add_annotation('is_protein',bool);aa.is_protein[:]=True
    return aa


def test_serine_linear_CA_CB_OG_with_correct_bonds_is_not_safe_export():
    aa=physical_residue('SER');lookup={n:i for i,n in enumerate(aa.atom_name)};ca,cb,og=[lookup[a] for a in ('CA','CB','OG')]
    vector=aa.coord[cb]-aa.coord[ca];length=np.linalg.norm(aa.coord[og]-aa.coord[cb]);aa.coord[og]=aa.coord[cb]+vector/np.linalg.norm(vector)*length
    audit=audit_exported_geometry(aa)
    assert not audit['accepted'] and not audit['absolute_sidechain_geometry_passed']
    assert any(r['kind']=='sidechain_bond_angle' and abs(r['angle_degrees']-180)<.1 for r in audit['sidechain_failures_first100'])


@pytest.mark.parametrize('name,carbon,ends',[('ASP','CG',('OD1','OD2')),('GLU','CD',('OE1','OE2'))])
def test_both_long_carboxyl_bonds_reject_without_rejecting_legal_joint_states(name,carbon,ends):
    aa=physical_residue(name);lookup={n:i for i,n in enumerate(aa.atom_name)};center=lookup[carbon];directions=[aa.coord[lookup[a]]-aa.coord[center] for a in ends]
    directions=[v/np.linalg.norm(v) for v in directions]
    for lengths in [(1.26,1.26),(1.23,1.34),(1.34,1.23)]:
        legal=aa.copy()
        for a,v,length in zip(ends,directions,lengths):legal.coord[lookup[a]]=legal.coord[center]+v*length
        audit=audit_exported_geometry(legal)
        assert audit['absolute_sidechain_geometry_passed']
        assert audit['accepted']
    for a,v in zip(ends,directions):aa.coord[lookup[a]]=aa.coord[center]+v*1.43
    audit=audit_exported_geometry(aa)
    assert not audit['accepted'] and not audit['absolute_sidechain_geometry_passed']
    assert any(r['kind']=='joint_sidechain_bond_state' for r in audit['sidechain_failures_first100'])


@pytest.mark.parametrize('mode',['peptide_only','covalent_only','both'])
def test_actual_finalizer_CO_failure_rolls_back_and_refreshes_every_claim(mode):
    aa=physical_residue('SER');x=torch.tensor(aa.coord).unsqueeze(0);n=len(aa)
    f={'atom_to_token_map':torch.zeros(n,dtype=torch.long),'is_motif_atom_with_fixed_coord':torch.zeros(n,dtype=torch.bool),'is_virtual':torch.zeros(n,dtype=torch.bool),'is_protein':torch.ones(1,dtype=torch.bool),'asym_id':torch.zeros(1,dtype=torch.long),'residue_index':torch.zeros(1,dtype=torch.long),'restype':torch.tensor(AF3SequenceEncoding().encode(np.array(['SER']))),'ref_atom_name_chars':torch.tensor([[ord(c)-32 for c in name.ljust(4)] for name in aa.atom_name])}
    lookup={name:i for i,name in enumerate(aa.atom_name)};candidate=x+.1
    def finalize(value):
        result=value.clone()
        if result[0,lookup['N'],0]>x[0,lookup['N'],0]+.05:
            c,o=lookup['C'],lookup['O'];direction=result[0,o]-result[0,c];result[0,o]=result[0,c]+direction/torch.linalg.vector_norm(direction)*1.8
        return result
    peptide={'applied':True,'absolute_backbone_passed':True} if mode in ('peptide_only','both') else None
    covalent={'applied':True,'absolute_backbone_passed':True} if mode in ('covalent_only','both') else None
    y,checks=finalize_generated_output(candidate,x,[(f,{'bound':True})],finalizer=finalize,peptide_diagnostics=peptide,covalent_diagnostics=covalent)
    assert torch.equal(x,y)
    assert not checks[0]['rejected_repair_check']['absolute_backbone_passed']
    assert checks[0]['absolute_backbone_passed'] and checks[0]['absolute_sidechain_geometry_passed']
    for detail in (peptide,covalent):
        if detail is None:continue
        assert not detail['applied'] and detail['reason']=='finalize_absolute_geometry_guard_rollback'
        assert detail['absolute_backbone_passed']==checks[0]['absolute_backbone_passed']
        assert detail['after_finalize_checks']==checks
