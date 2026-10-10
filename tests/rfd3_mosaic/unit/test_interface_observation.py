import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from rfd3.inference.symmetry.interface_observation_trace import InterfaceObservationTrace
from rfd3.inference.symmetry.graph_interface_guidance import GraphInterfaceGuidanceConfig, apply_graph_interface_guidance
import test_graph_interface_guidance as fixture_helpers


def fixture():
    features = {'atom_to_token_map': torch.arange(6), 'asym_id': torch.tensor([0,0,0,1,1,1]),
                'residue_index': torch.tensor([0,1,2,0,1,2]), 'is_ca': torch.ones(6,dtype=torch.bool),
                'is_virtual': torch.zeros(6,dtype=torch.bool)}
    xyz = torch.tensor([[[0.,0.,0.],[0.,3.8,0.],[0.,7.6,0.],[12.,0.,0.],[12.,3.8,0.],[12.,7.6,0.]]])
    return xyz,features,fixture_helpers.GraphInterfaceGuidanceTestCase._three_by_three_topology()


def test_disabled_trace_does_not_touch_features_or_rng():
    class Untouchable(dict):
        def items(self): raise AssertionError('Disabled recorder must not inspect features')
    state=torch.random.get_rng_state().clone()
    with patch.dict(os.environ,{},clear=True):
        assert InterfaceObservationTrace.from_environment(seed=17,features=Untouchable()) is None
    assert torch.equal(state,torch.random.get_rng_state())


def test_exact_storage_gradient_and_features_roundtrip_without_rng_mutation():
    xyz,features,_=fixture();before=xyz.clone();state=torch.random.get_rng_state().clone()
    with tempfile.TemporaryDirectory() as directory:
        with patch.dict(os.environ,{'MOSAIC_INTERFACE_TRACE_DIR':directory}):
            trace=InterfaceObservationTrace.from_environment(seed=17,features=features)
        trace.record('raw_denoiser_prediction',xyz,gradient=xyz*2,step=3)
        arrays=np.load(next(trace.directory.glob('interface-*.npz')),allow_pickle=False)
        meta=json.loads(bytes(arrays['metadata_json']).decode())
        restored=torch.from_numpy(arrays['coordinates_storage'].copy()).view(torch.float32).reshape(meta['shape'])
        assert torch.equal(restored,xyz)
        np.testing.assert_array_equal(arrays['gradient'],(xyz*2).numpy())
        assert trace.diagnostics()['errors']==[]
    assert torch.equal(xyz,before);assert torch.equal(state,torch.random.get_rng_state())


def test_graph_observation_preserves_output_and_records_rejected_projected_trials():
    xyz,features,topology=fixture();config=GraphInterfaceGuidanceConfig()
    def guard(value):return {'accepted':False,'checks':[{'rule':'deliberate_negative_control','passed':False}]}
    baseline,diag=apply_graph_interface_guidance(xyz,features,topology,progress=.5,config=config,candidate_validator=guard)
    with tempfile.TemporaryDirectory() as directory:
        trace=InterfaceObservationTrace(directory,seed=17)
        state=torch.random.get_rng_state().clone()
        observed,actual=apply_graph_interface_guidance(xyz,features,topology,progress=.5,config=config,candidate_validator=guard,observer=trace.record)
        assert torch.equal(baseline,observed);assert diag==actual
        assert torch.equal(state,torch.random.get_rng_state())
        arrays=[np.load(p,allow_pickle=False) for p in sorted(trace.directory.glob('interface-*.npz'))]
        metadata=[json.loads(bytes(a['metadata_json']).decode()) for a in arrays]
        assert metadata[0]['stage']=='graph_input';assert 'gradient' in arrays[0]
        trials=[m for m in metadata if m['stage']=='graph_trial']
        assert len(trials)==config.line_search_steps
        assert all(not m['extra']['decision']['accepted'] for m in trials)
        assert trace.errors==[]


def test_trace_limit_and_io_error_do_not_mutate_coordinates():
    xyz,features,_=fixture();before=xyz.clone()
    with tempfile.TemporaryDirectory() as directory:
        trace=InterfaceObservationTrace(directory,seed=17,limit=1)
        trace.record('a',xyz);trace.record('b',xyz)
        assert trace.count==1 and trace.skipped==1
        blocked=Path(directory)/'blocked';blocked.write_text('file')
        trace=InterfaceObservationTrace(blocked,seed=17)
        trace.record('a',xyz)
        assert trace.count==0 and len(trace.errors)==1
    assert torch.equal(xyz,before)


def test_nested_actual_symmetry_transforms_are_serialized():
    xyz,features,_=fixture()
    features['sym_transform']={'0':(torch.eye(3),torch.zeros(3))}
    with tempfile.TemporaryDirectory() as directory:
        trace=InterfaceObservationTrace(directory,seed=17)
        trace.save_features(features)
        assert trace.errors==[]
        arrays=np.load(trace.directory/'features.npz',allow_pickle=False)
        meta=json.loads(bytes(arrays['metadata_json']).decode())
        assert meta['sym_transform']['0'][0]==torch.eye(3).tolist()
