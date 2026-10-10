"""Opt-in terminal caller for the R3 native snapshot transaction.

Integration scope is indexed all-protein designs in explicit all-copy mode.
Each diffusion-batch member owns a private transaction; unsupported modes fail before sampler denoising.
The adapter owns private features, graph patches, runtime and temporary exports.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from dataclasses import replace
import numpy as np
import torch
from rfd3.inference.symmetry.physical_candidate_refinement import (
    PublishedCandidate, coordinate_identity, refine_physical_candidate,
)


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(snapshot(v) for v in value)
    return copy.deepcopy(value)


def validate_refinement_scope(sampler, features, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Final refinement requires a positive batch size')
    if type(sampler.final_covalent_graph_refinement_steps) is not int or sampler.final_covalent_graph_refinement_steps < 1:
        raise ValueError('Enabled final refinement requires a positive integer step budget')
    if sampler.symmetry_state_mode != 'orbit_average' or getattr(sampler, '_uses_local_symmetry_neighbourhood', True):
        raise ValueError('Final refinement currently requires explicit all-copy orbit_average')
    protein = features.get('is_protein')
    if protein is None or not bool(protein.all()):
        raise ValueError('Final refinement currently supports all-protein designs only')
    unindexed = features.get('is_motif_atom_unindexed')
    if unindexed is not None and bool(unindexed.any()):
        raise ValueError('Final refinement currently requires indexed residues')
    if not sampler.enable_generated_peptide_geometry_repair or not sampler.enable_generated_covalent_geometry_repair:
        raise ValueError('Final refinement requires generated peptide and covalent repair enabled')


def exported_identity(array):
    """Bind physical atoms and chain/residue topology with stable labels.

    Chain labels and a constant residue-number offset per chain are gauges.
    Chain partition, residue order/gaps, insertion codes, names and actual
    coordinate order are not gauges. Keep them in the certificate.
    """
    chain_ordinals = {}
    chain_origins = {}
    chain_residues = {}
    seen_residues = set()
    keys = []
    last = None
    residue_key = None
    for atom in array:
        chain = str(atom.chain_id)
        number = int(atom.res_id)
        insertion = str(atom.ins_code).strip()
        current = (chain, number, insertion)
        if current != last:
            if current in seen_residues:
                return {'schema_version':2,'identity_bound':False,
                        'reason':'noncontiguous_duplicate_residue_identity'}
            seen_residues.add(current)
            ordinal = chain_ordinals.setdefault(chain, len(chain_ordinals))
            origin = chain_origins.setdefault(chain, number)
            residues = chain_residues.setdefault(chain, [])
            residue_key = [ordinal, len(residues), number-origin, insertion]
            residues.append(residue_key[1:])
            last = current
        keys.append(residue_key + [str(atom.res_name), str(atom.atom_name)])
    topology = [chain_residues[chain] for chain in chain_ordinals]
    coords = np.asarray(array.coord, dtype=np.float32)
    digest = lambda value: hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()
    return {'schema_version':2,'identity_bound':True,'atom_count':len(array),
            'chain_count':len(chain_ordinals),'residue_count':len(seen_residues),
            'keys_sha256':digest(keys),'topology_sha256':digest(topology),
            'coordinates_sha256':hashlib.sha256(coords.tobytes()).hexdigest()}


def build_native_export_input(coordinates,features,bound):
    from biotite.structure import AtomArray
    from atomworks.ml.encoding_definitions import AF3SequenceEncoding
    from rfd3.inference.symmetry.output_geometry_guard import native_labels
    tokens = features['atom_to_token_map'].detach().cpu().numpy()
    array = AtomArray(len(tokens))
    array.coord = coordinates[0].detach().cpu().float().numpy().copy()
    array.atom_name = native_labels(features)
    array.res_name = np.asarray(AF3SequenceEncoding().decode(bound['restype'].detach().cpu().numpy()))[tokens]
    # Preserve declared residue topology, including numbering gaps. Chain
    # labels/constant numbering offsets may change during serialization.
    array.res_id = features['residue_index'].detach().cpu().numpy()[tokens]
    array.chain_id = features['asym_id'].detach().cpu().numpy()[tokens].astype(str)
    array.element[:] = 'X'
    array.add_annotation('gt_atom_name', 'U6'); array.gt_atom_name = native_labels(bound)
    for name in ('is_motif_atom_with_fixed_seq', 'is_motif_atom_with_fixed_coord'):
        array.add_annotation(name, bool)
        array.set_annotation(name, features.get(name,features['is_motif_atom_with_fixed_coord']).detach().cpu().numpy())
    array.add_annotation('is_motif_atom_unindexed',bool); array.is_motif_atom_unindexed[:] = False
    array.add_annotation('is_protein',bool); array.is_protein[:] = True
    array.add_annotation('token_id',int); array.token_id=tokens
    array.add_annotation('is_ligand',bool); array.is_ligand[:]=False
    array.add_annotation('src_component','U1'); array.src_component[:]=''
    return array


def prepare_export(coordinates, features, bound, scheme):
    """Use the actual trainer association cleanup and actual CIF round trip."""
    from biotite.structure import AtomArray
    from biotite.structure.io import save_structure, load_structure
    from atomworks.ml.encoding_definitions import AF3SequenceEncoding
    from rfd3.inference.symmetry.output_geometry_guard import native_labels, audit_exported_geometry
    from rfd3.trainer.trainer_utils import _cleanup_virtual_atoms_and_assign_atom_name_elements
    array = build_native_export_input(coordinates,features,bound)
    cleaned = _cleanup_virtual_atoms_and_assign_atom_name_elements(array, association_scheme=scheme)
    expected_indices = np.flatnonzero(~bound['is_virtual'].detach().cpu().numpy())
    expected_labels = np.asarray(native_labels(bound))[expected_indices]
    association_ok = (len(cleaned) == len(expected_indices) and np.array_equal(cleaned.atom_name,expected_labels)
                      and np.array_equal(cleaned.coord,coordinates[0,expected_indices].detach().cpu().float().numpy()))
    chemical = audit_exported_geometry(cleaned)
    with tempfile.TemporaryDirectory(prefix='mosaic-final-candidate-') as scratch:
        path = Path(scratch)/'candidate.cif'
        save_structure(str(path),cleaned)
        reread = load_structure(str(path))
        reread.add_annotation('is_protein',bool); reread.is_protein[:] = True
        readback = audit_exported_geometry(reread)
        identical = exported_identity(cleaned) == exported_identity(reread)
        cif_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'accepted': bool(association_ok and chemical.get('accepted') and readback.get('accepted') and identical and exported_identity(cleaned).get('identity_bound') is True),
            'association_matches_bound_identity':association_ok, 'association_atom_count':len(cleaned), 'bound_atom_count':len(expected_indices), 'association_label_examples':[[str(a),str(b)] for a,b in zip(cleaned.atom_name,expected_labels) if a!=b][:20], 'association_label_mismatches':int(np.count_nonzero(cleaned.atom_name!=expected_labels)) if len(cleaned)==len(expected_indices) else None, 'association_coordinate_mismatches':int(np.count_nonzero(cleaned.coord!=coordinates[0,expected_indices].detach().cpu().float().numpy())) if len(cleaned)==len(expected_indices) else None, 'actual_export':chemical,
            'CIF_readback_geometry':readback,'CIF_component_identity':identical,
            'prepared_CIF_sha256':cif_hash,'export_identity':exported_identity(cleaned)}


class FinalCandidateAdapter:
    def __init__(self,sampler,features,baseline,runtime,topology,config,patch_state,ca_topology,ca_config):
        from rfd3.inference.symmetry.constraint_runtime import MosaicConstraintRuntime
        from rfd3.inference.symmetry.scaffold_core_guidance import scaffold_geometry_guard
        self.children = None
        if len(baseline) > 1:
            from types import SimpleNamespace
            self.children = [FinalCandidateAdapter(sampler,features,baseline[b:b+1],
                SimpleNamespace(fixed_target=runtime.fixed_target[b:b+1],fixed_mask=runtime.fixed_mask),
                topology,config,patch_state,ca_topology,ca_config) for b in range(len(baseline))]
            return
        self.features = snapshot(features)
        self.baseline = baseline.detach().clone()
        self.sampler = copy.copy(sampler)
        self.sampler._exact_symmetry_orbit_layout = snapshot(sampler._exact_symmetry_orbit_layout)
        self.target = runtime.fixed_target.detach().clone()
        self.fixed = runtime.fixed_mask.detach().clone()
        self.topology = snapshot(topology)
        self.config = replace(config,contact_prior_weight=0.0) if config is not None else None
        self.assignments = snapshot(patch_state.assignments) if patch_state is not None else None
        self.ca_guard = scaffold_geometry_guard(self.baseline,snapshot(ca_topology),ca_config) if ca_topology is not None else None
        self.steps = sampler.final_covalent_graph_refinement_steps
        self.certificate = None
        self.transaction = None
        self.runtime_factory = lambda: MosaicConstraintRuntime(
            projector=self.sampler._joint_projector(self.features),
            fixed_target=self.target.clone(),fixed_mask=self.fixed.clone(),
            cylindrical_projector=self.sampler._cylindrical_projector(self.features,self.baseline))
        self.scratch = self.runtime_factory()

    def project(self,value):
        return self.scratch._project(value,label='Isolated terminal refinement')

    def transform(self,candidate,budget,batch,bound,binding):
        from rfd3.inference.symmetry.output_geometry_guard import audit_bound_geometry
        from rfd3.inference.symmetry.graph_interface_guidance import (
            GraphInterfacePatchState,GraphInterfaceStepContext,apply_graph_interface_guidance,
            graph_interface_energy,graph_interface_proposal_acceptable,
        )
        if self.children is not None:
            return self.children[batch].transform(candidate,budget,0,bound,binding)
        if batch != 0 or (self.topology is not None and (self.config is None or self.assignments is None)):
            return self.baseline.clone(),{'applied':False,'reason':'missing_original_graph_binding'}
        bound = snapshot(bound)
        if self.topology is not None:
            initial = graph_interface_energy(self.baseline,self.topology,self.config,patch_assignments=self.assignments)
            state = GraphInterfacePatchState(assignments=snapshot(self.assignments),locked=True,lock_reason='terminal_original_assignment_snapshot')
            context = GraphInterfaceStepContext(progress=1.,source_config=self.config,effective_config=self.config,
                patch_assignments=snapshot(self.assignments),adaptive_phase='original_final_covalent_repair',
                time_scheduled_target_ca_distance=self.config.target_ca_distance,target_ca_distance=self.config.target_ca_distance,
                contact_prior_schedule_scale=0.)
        def physics(value):
            chemistry = audit_bound_geometry(value,bound,self.baseline)
            ca = self.ca_guard(value) if self.ca_guard is not None else {'accepted':True}
            fixed = torch.equal(value[:,self.fixed],self.target[:,self.fixed])
            cylinder = self.scratch.cylindrical_projector.maximum_error(value) if self.scratch.cylindrical_projector is not None else 0.
            self.scratch.projector.validate_closure(value,'Terminal native candidate')
            return {'accepted':bool(chemistry.get('accepted') and chemistry.get('absolute_backbone_passed') and chemistry.get('absolute_sidechain_geometry_passed') and chemistry.get('nonadjacent_allheavy_clash_pair_count') == 0 and ca.get('accepted') and fixed and cylinder<=1e-5),
                    'chemistry':chemistry,'CA':ca,'fixed_XYZ':fixed,'cylindrical_maximum_error':float(cylinder)}
        def policy(value):
            if self.topology is None:
                return self.validate_original_policy(value,batch)
            after=graph_interface_energy(value,self.topology,self.config,patch_assignments=self.assignments)
            decision={};graph_interface_proposal_acceptable(initial,after,self.config,decision=decision,evaluate_all=True)
            return decision
        def propose(value,index):
            if self.topology is None:
                return value.detach().clone(),{'applied':False,'reason':'no_declared_graph_identity_proposal'}
            return apply_graph_interface_guidance(value,bound,self.topology,progress=1.,config=self.config,
                projector=self.project,patch_state=state,candidate_validator=physics,step_context=context)
        def publish(value):
            temporary=self.runtime_factory();temporary.initialize_state(self.baseline)
            final=temporary.finalize(value)
            exported=prepare_export(final,self.features,bound,binding['association_scheme'])
            return PublishedCandidate(final,exported)
        result,transaction=refine_physical_candidate(self.baseline,candidate,proposer=propose,
            projector=self.project,physical_validator=physics,original_policy_validator=policy,
            publication_validator=publish,maximum_atom_step=budget,maximum_steps=self.steps)
        transaction['scope']='actual terminal sampler caller; isolated private state; per-design indexed all-protein explicit orbit_average'
        self.transaction=transaction
        if transaction.get('applied'):
            self.certificate={'native_identity':coordinate_identity(result),
                'publication':snapshot(transaction['evaluations'][-1]['publication'])}
        return result,transaction

    def validate_original_policy(self,value,batch):
        if self.children is not None:
            return self.children[batch].validate_original_policy(value,0)
        from rfd3.inference.symmetry.graph_interface_guidance import graph_interface_energy,graph_interface_proposal_acceptable
        ca=self.ca_guard(value) if self.ca_guard is not None else {'accepted':True}
        if self.topology is None:
            return ca
        if self.config is None:
            return {'accepted':False,'reason':'missing_original_graph_config'}
        before=graph_interface_energy(self.baseline,self.topology,self.config,patch_assignments=self.assignments)
        after=graph_interface_energy(value,self.topology,self.config,patch_assignments=self.assignments)
        decision={};graph_interface_proposal_acceptable(before,after,self.config,decision=decision,evaluate_all=True)
        return {'accepted':bool(ca['accepted'] and decision['accepted']),'ca_guard':ca,'original_graph_acceptance':decision}

    def prepare_final(self,value):
        if self.children is not None:
            return torch.cat([child.prepare_final(value[b:b+1]) for b,child in enumerate(self.children)],dim=0)
        if self.certificate is not None and coordinate_identity(value)==self.certificate['native_identity']:
            return value.detach().clone()
        return self.project(value.detach().clone())

    def bind_final_diagnostics(self,value,rows,covalent):
        if self.children is not None:
            batches=(covalent or {}).get('batches',[])
            for b,child in enumerate(self.children):
                child.bind_final_diagnostics(value[b:b+1],[rows[b]],batches[b] if b<len(batches) else None)
            return
        committed = bool(self.certificate is not None and coordinate_identity(value)==self.certificate['native_identity'] and (covalent or {}).get('applied'))
        if self.transaction is not None:
            self.transaction['committed_to_sampler']=committed
            if not committed:
                self.transaction['sampler_commit_reason']='final_sampler_snapshot_not_committed'
                if self.transaction.get('applied'):
                    self.transaction['prepared_transaction_reason']=self.transaction.get('reason')
                    self.transaction.update(applied=False,reason='final_sampler_snapshot_not_committed')
                self.transaction.pop('committed_native_identity',None)
        for row in rows:
            row['final_candidate_refinement']={'enabled':True,'committed_to_sampler':committed,
                'reason':self.transaction.get('reason') if self.transaction else 'no_covalent_candidate'}
            if committed:
                row['refinement_export_certificate']=snapshot(self.certificate)


def verify_actual_export(array,final_record):
    certificate=final_record.get('refinement_export_certificate')
    if certificate is None:
        return None
    identity=exported_identity(array)
    expected=certificate.get('publication',{}).get('export_identity')
    matched=bool(isinstance(expected,dict) and expected.get('schema_version')==2 and expected.get('identity_bound') is True and identity.get('identity_bound') is True and identity==expected)
    return {'accepted':matched,
            'actual_export_identity':identity,'expected_export_identity':expected,
            'reason':'matched_verified_native_physical_atoms_and_topology' if matched else 'export_identity_mismatch'}
