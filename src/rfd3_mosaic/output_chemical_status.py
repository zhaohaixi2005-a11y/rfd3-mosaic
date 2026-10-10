"""Keep final chemical validity separate from declared task-contract status."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping


def joint_task_and_chemical_status(
    contract_status: str,
    chemical_geometry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Report the joint production verdict without redefining ``accepted``.

    A declared task failure or an independent chemical failure is terminal.
    Missing evidence remains unknown; only two explicit passes produce a joint
    pass.  This keeps the historical contract-audit compatibility field intact
    while giving downstream consumers one unambiguous selection field.
    """

    contract_passed = True if contract_status == "met" else (
        False if contract_status == "flagged" else None
    )
    chemical_passed = (
        chemical_geometry.get("passed")
        if isinstance(chemical_geometry, Mapping)
        and type(chemical_geometry.get("passed")) is bool
        else None
    )
    if contract_passed is False or chemical_passed is False:
        status, passed = "failed", False
    elif contract_passed is True and chemical_passed is True:
        status, passed = "passed", True
    else:
        status, passed = "not_evaluated", None
    return {
        "status": status,
        "passed": passed,
        "contract_passed": contract_passed,
        "chemical_passed": chemical_passed,
    }

def final_chemical_status(metadata: Mapping[str, Any]) -> dict[str, Any]:
    export=metadata.get('final_export_geometry_diagnostics')
    stages=metadata.get('final_output_geometry_diagnostics')
    if export is None and stages is None:
        return {'status':'not_evaluated','passed':None,'reason':'final_chemical_diagnostics_missing'}
    records=[]
    if isinstance(export,dict): records.append(('actual_export',export))
    # Each result JSON describes one exported design. Legacy producers copied
    # the entire diffusion batch into every design; without an explicit binding
    # a multi-record list cannot identify this design and must remain unknown.
    scope=metadata.get('final_output_geometry_diagnostic_scope')
    scoped_valid=(scope is None or isinstance(scope,dict) and scope.get('binding')=='per_design'
        and type(scope.get('batch_index')) is int and type(scope.get('batch_count')) is int
        and 0<=scope['batch_index']<scope['batch_count'])
    stage_bound=isinstance(stages,list) and len(stages)==1 and isinstance(stages[0],dict) and scoped_valid
    if stage_bound:records.append(('final_output_this_design',stages[0]))
    checks=[];incomplete=not isinstance(export,dict) or not stage_bound
    if not stage_bound:checks.append({'source':'final_output_this_design','rule':'per_design_batch_binding','passed':None})
    for source,r in records:
        verdict=r.get('identity_bound')
        checks.append({'source':source,'rule':'final_identity_bound','passed':verdict if type(verdict) is bool else None})
        incomplete |= type(verdict) is not bool
        for key in ('absolute_backbone_passed','absolute_sidechain_geometry_passed'):
            verdict=r.get(key)
            checks.append({'source':source,'rule':key,'passed':verdict if type(verdict) is bool else None})
            incomplete |= type(verdict) is not bool
        count=r.get('nonadjacent_allheavy_clash_pair_count')
        valid_count=type(count) is int and count>=0
        checks.append({'source':source,'rule':'zero_nonadjacent_allheavy_clashes','passed':count==0 if valid_count else None,'count':count})
        incomplete |= not valid_count
        # Bound accepted denotes non-regression; export accepted also requires
        # absolute chemical validity. Never interchange the two meanings.
        if source=='actual_export':
            verdict=r.get('accepted')
            checks.append({'source':source,'rule':'actual_export_accepted','passed':verdict if type(verdict) is bool else None})
            incomplete |= type(verdict) is not bool
    failed=any(c['passed'] is False for c in checks)
    status='failed' if failed else 'incomplete' if incomplete else 'passed'
    return {'status':status,'passed':False if failed else None if incomplete else True,
            'checks':checks,'scope':'final recorded N/CA/C/O, physical sidechain identity and nonadjacent heavy pairs; no folding certificate'}

def read_final_chemical_status(result_json: str | Path) -> dict[str, Any]:
    path=Path(result_json)
    metadata=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(metadata,dict):raise ValueError('Result metadata must be an object')
    return {'source_result_json':str(path),**final_chemical_status(metadata)}
