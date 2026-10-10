"""Absolute final-state checks, rollback, and truthful repair diagnostics."""


def finalize_generated_output(coordinates, baseline, bindings, *, finalizer, isolate_batches=False, peptide_diagnostics=None, covalent_diagnostics=None):
    from rfd3.inference.symmetry.output_geometry_guard import audit_bound_geometry
    def audit(value):
        rows=[]
        for batch,(bound,binding) in enumerate(bindings):
            row=audit_bound_geometry(value[batch:batch+1],bound,baseline[batch:batch+1]) if bound is not None else {'identity_bound':False,'accepted':False,'reason':binding['reason']}
            row.update(identity_binding=binding,stage='after_final_runtime_projection')
            rows.append(row)
        return rows
    def passed(rows):
        return all(r.get('identity_bound',False) and r.get('accepted',False) and r.get('absolute_backbone_passed',False) and r.get('absolute_sidechain_geometry_passed',False) for r in rows)
    result=finalizer(coordinates)
    checks=audit(result)
    repaired=any((d or {}).get('applied',False) for d in (peptide_diagnostics,covalent_diagnostics))
    rolled_back=repaired and not passed(checks)
    if rolled_back:
        rejected=checks
        # Keep original predictions and original hard constraints, even if the
        # original itself remains infeasible; never turn failure into a pass.
        if isolate_batches:
            import torch
            restored=result.clone()
            bad=[not passed([row]) for row in checks]
            for batch,reject in enumerate(bad):
                if reject:restored[batch:batch+1]=baseline[batch:batch+1]
            result=finalizer(restored)
            for detail in (peptide_diagnostics,covalent_diagnostics):
                if detail is None:continue
                batches=detail.get('batches',[])
                for batch,reject in enumerate(bad):
                    if reject and batch<len(batches):
                        batches[batch].update(applied=False,reason='per_design_final_geometry_rollback')
                if batches:detail['applied']=any(row.get('applied',False) for row in batches)
        else:
            result=finalizer(baseline.clone())
        checks=audit(result)
        for batch,(row,failed) in enumerate(zip(checks,rejected)):
            if not isolate_batches or bad[batch]:
                row.update(stage='finalize_rollback_to_original_prediction',rejected_repair_check=failed)
    for detail in (peptide_diagnostics,covalent_diagnostics):
        if detail is None:continue
        detail['after_finalize_checks']=checks
        detail['absolute_backbone_passed']=all(r.get('absolute_backbone_passed',False) for r in checks)
        detail['final_absolute_sidechain_geometry_passed']=all(r.get('absolute_sidechain_geometry_passed',False) for r in checks)
        detail['final_geometry_passed']=passed(checks)
        if (rolled_back or (detail.get('applied',False) and not passed(checks))) and not isolate_batches:
            detail.update(applied=False,reason='finalize_absolute_geometry_guard_rollback' if rolled_back else 'finalize_absolute_geometry_check_failed')
    return result,checks
