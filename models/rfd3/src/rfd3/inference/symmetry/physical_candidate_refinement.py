"""Unwired prototype: commit the exact publication-verified native snapshot."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable
import torch

Verdict = Callable[[torch.Tensor], dict[str, Any]]


@dataclass(frozen=True)
class PublishedCandidate:
    """Finalized native coordinates plus the verdict/artifact bindings for them.

    The callback certifies this exact snapshot, not its input. The core clones
    it immediately, rechecks all native guards and returns that private clone.
    Artifact publication is still the caller's prepare/verify/commit duty.
    """
    coordinates: torch.Tensor
    diagnostics: dict[str, Any]


def coordinate_identity(value: torch.Tensor) -> dict[str, Any]:
    storage=value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(storage).hexdigest()}


@dataclass(frozen=True)
class _Evaluation:
    working: torch.Tensor | None
    verified: torch.Tensor | None
    working_feasible: bool
    accepted: bool


def refine_physical_candidate(
    original: torch.Tensor,
    repaired: torch.Tensor,
    *,
    proposer: Callable[[torch.Tensor, int], tuple[torch.Tensor, dict[str, Any]]],
    projector: Callable[[torch.Tensor], torch.Tensor],
    physical_validator: Verdict,
    original_policy_validator: Verdict,
    publication_validator: Callable[[torch.Tensor], PublishedCandidate],
    maximum_atom_step: float,
    maximum_steps: int = 0,
    observer: Callable[..., None] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bounded coordinate transaction with observer-independent acceptance.

    All callbacks receive clones. Callers must additionally isolate closure
    feature/patch/RNG/runtime and scratch-artifact side effects. Publication
    must return its actual finalized snapshot; legacy boolean/dict callbacks
    are unknown and can never commit. No NN/model/task constants live here.
    """
    if type(maximum_steps) is not int or maximum_steps<0:
        raise ValueError('maximum_steps must be a non-negative integer')
    if not math.isfinite(maximum_atom_step) or maximum_atom_step<0:
        raise ValueError('maximum_atom_step must be finite and non-negative')
    diagnostic: dict[str, Any]={'applied':False,'maximum_steps':maximum_steps,
        'maximum_atom_step':maximum_atom_step,'proposal_calls':0,'evaluations':[],
        'scope':'isolated unwired prototype; production sampler has no caller'}
    if maximum_steps==0:
        diagnostic['reason']='disabled'
        return original,diagnostic
    baseline=original.detach().clone()
    if not torch.isfinite(baseline).all():raise ValueError('baseline must be finite')

    def emit(stage,coordinates,**extra):
        if observer is not None:
            # No live records, solver state or decision data escape to logging.
            observer(stage,coordinates.detach().clone(),**copy.deepcopy(extra))

    def valid(value):
        return (isinstance(value,torch.Tensor) and value.shape==baseline.shape
            and value.dtype==baseline.dtype and value.device==baseline.device
            and bool(torch.isfinite(value).all()))

    def check(validator,coordinates):
        try:
            verdict=validator(coordinates.detach().clone())
            if not isinstance(verdict,dict):return {'accepted':False,'reason':'invalid_verdict'}
            # A callback cannot change an earlier verdict through shared dicts.
            return copy.deepcopy(verdict)
        except Exception as error:
            return {'accepted':False,'reason':'validator_error','error_type':type(error).__name__,'error':str(error)}

    def evaluate(value,index,proposal):
        record={'index':index,'proposal':copy.deepcopy(proposal),'accepted':False}
        diagnostic['evaluations'].append(record)
        if not valid(value):
            reason='nonfinite_coordinates' if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()) else 'coordinate_identity_mismatch'
            record.update(reason=reason,not_executed=['projection','physical','original_policy','publication'])
            return _Evaluation(None,None,False,False)
        emit('refinement_candidate_before_projection',value,index=index)
        projected=projector(value.detach().clone())
        if not valid(projected):
            record.update(reason='invalid_projected_coordinates',not_executed=['physical','original_policy','publication'])
            return _Evaluation(None,None,False,False)
        projected=projected.detach().clone()
        emit('refinement_candidate_after_projection',projected,index=index)
        step=float(torch.linalg.vector_norm(projected-baseline,dim=-1).max().item())
        pre_physical=check(physical_validator,projected)
        pre_policy=check(original_policy_validator,projected)
        record['pre_publication']={'maximum_atom_step_from_original':step,
            'physical':pre_physical,'original_policy':pre_policy}
        working_feasible=bool(pre_physical.get('accepted') is True and step<=maximum_atom_step+1e-6)
        native=None
        try:
            published=publication_validator(projected.detach().clone())
            if not isinstance(published,PublishedCandidate):
                record['publication']={'accepted':False,'reason':'missing_verified_native_snapshot'}
            elif not valid(published.coordinates):
                record['publication']={'accepted':False,'reason':'invalid_verified_native_snapshot'}
            elif not isinstance(published.diagnostics,dict):
                record['publication']={'accepted':False,'reason':'invalid_publication_verdict'}
            else:
                native=published.coordinates.detach().clone()
                record['publication']=copy.deepcopy(published.diagnostics)
        except Exception as error:
            record['publication']={'accepted':False,'reason':'validator_error','error_type':type(error).__name__,'error':str(error)}
        accepted=False
        if native is not None:
            record['publication_native_identity']=coordinate_identity(native)
            final_step=float(torch.linalg.vector_norm(native-baseline,dim=-1).max().item())
            record['maximum_atom_step_from_original']=final_step
            record['step_passed']=final_step<=maximum_atom_step+1e-6
            record['physical']=check(physical_validator,native)
            record['original_policy']=check(original_policy_validator,native)
            accepted=bool(record['step_passed'] and record['physical'].get('accepted') is True
                and record['original_policy'].get('accepted') is True and record['publication'].get('accepted') is True)
        else:
            record['not_executed']=['final_native_physical','final_native_original_policy']
        record['accepted']=accepted
        emit('refinement_candidate_evaluated',native if native is not None else projected,index=index,diagnostic=record)
        # The decision is a private local bool in a frozen evaluation object;
        # never read it back from anything handed to an observer.
        return _Evaluation(projected,native if accepted else None,working_feasible,accepted)

    try:
        emit('refinement_original_baseline',baseline)
        evaluation=evaluate(repaired,-1,{'kind':'initial_repaired_candidate'})
        if evaluation.accepted:
            diagnostic.update(applied=True,reason='initial_candidate_fully_accepted',
                committed_native_identity=coordinate_identity(evaluation.verified))
            emit('refinement_committed',evaluation.verified,index=-1,diagnostic=diagnostic)
            return evaluation.verified,diagnostic
        if not evaluation.working_feasible:
            diagnostic['reason']='initial_candidate_not_physically_feasible'
        else:
            current=evaluation.working
            for index in range(maximum_steps):
                diagnostic['proposal_calls']+=1
                value,proposal=proposer(current.detach().clone(),index)
                evaluation=evaluate(value,index,proposal)
                if evaluation.accepted:
                    diagnostic.update(applied=True,reason='refined_candidate_fully_accepted',
                        committed_native_identity=coordinate_identity(evaluation.verified))
                    emit('refinement_committed',evaluation.verified,index=index,diagnostic=diagnostic)
                    return evaluation.verified,diagnostic
                if evaluation.working_feasible:
                    if torch.equal(evaluation.working,current):
                        diagnostic['reason']='no_changed_feasible_proposal'
                        break
                    current=evaluation.working
            else:diagnostic['reason']='proposal_budget_exhausted'
    except Exception as error:
        diagnostic.update(applied=False,reason='transaction_error',error_type=type(error).__name__,error=str(error))
        diagnostic.pop('committed_native_identity',None)
    try:emit('refinement_rejected_original_retained',baseline,diagnostic=diagnostic)
    except Exception as error:diagnostic['observer_error']={'error_type':type(error).__name__,'error':str(error)}
    return original,diagnostic
