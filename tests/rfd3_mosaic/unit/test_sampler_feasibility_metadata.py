import json
from rfd3.trainer.rfd3 import _copy_sampler_diagnostics


def test_actual_metadata_copy_and_json_roundtrip_preserve_recovery_cost():
    feasibility = {'schema_version': 1, 'steps': [
        {'stage': 'model_prediction', 'step_num': 0, 'fraction_retained': 0.1431686654,
         'maximum_atom_displacement': 1.489599, 'passed': True},
        {'stage': 'model_prediction', 'step_num': 1, 'fraction_retained': None,
         'method': 'local_CA_envelope_translation', 'passed': True},
    ]}
    trace = {'enabled': True, 'captured': 73, 'errors': []}
    metadata = {0: {'seed': 17}, 1: {'seed': 29}}
    _copy_sampler_diagnostics({'scaffold_feasibility_diagnostics': feasibility,
                              'phase_feasibility_trace': trace}, metadata)
    written = json.loads(json.dumps(metadata))
    for item in written.values():
        assert item['scaffold_feasibility_diagnostics'] == feasibility
        assert item['phase_feasibility_trace'] == trace


def test_absent_diagnostics_do_not_invent_recovery_fields():
    metadata = {0: {'seed': 17}}
    _copy_sampler_diagnostics({'scaffold_feasibility_diagnostics': None}, metadata)
    assert metadata == {0: {'seed': 17}}
