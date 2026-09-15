"""Expose recorded decisions without recomputing or inventing acceptance rules."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

DECISION_POLICY = {
    "version": 1,
    "rules_document": "docs/rfd3_mosaic/DECISION_RULES.zh-CN.md",
    "parameter_authority": "Instantiated runtime config and per-step effective_config; presets alone are insufficient.",
    "levels": {
        "hard_contract": "Declared geometry and execution requirements; failure flags the contract.",
        "controller_proxy": "Heuristic sampling objective; lower loss does not establish designability.",
        "advisory": "Backbone measurements for review; all generated structures are retained.",
    },
    "limitations": [
        "Controller weights and default quality cutoffs have not been established as universal scientific thresholds.",
        "RFD3 learned predictions are not explained mechanistically by the controller formulas.",
        "Missing diagnostics mean not recorded, not proof that a mechanism was inactive.",
    ],
}

_MECHANISMS = {
    "constraint_runtime_diagnostics": "Exact constraints and projection",
    "motif_mobility_diagnostics": "Bounded rigid-component motion and joint acceptance",
    "graph_interface_guidance_diagnostics": "Generated interface attraction, patch selection and acceptance",
    "scaffold_core_guidance_diagnostics": "Generated core, continuity and routing guidance",
    "generated_polymer_continuity_diagnostics": "Final generated-backbone continuity projection",
}


def write_decision_explanation(
    output: Path,
    *,
    result_json: Path,
    compiled_input: Path,
    reports: Iterable[Path],
    screening: Mapping[str, Any],
) -> Path:
    """Write a portable evidence index plus a readable companion per design.

    Raw trajectories remain in the result JSON. Audit snapshots retain nested
    thresholds/checks verbatim; this layer never derives a second verdict.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    def evidence(path: Path) -> dict[str, str]:
        return {
            "path": os.path.relpath(path.resolve(), output.parent.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    metadata = json.loads(result_json.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected result metadata object in {result_json}")
    mechanisms = {}
    for key, name in _MECHANISMS.items():
        diagnostics = metadata.get(key)
        record: dict[str, Any] = {
            "name": name, "recorded": isinstance(diagnostics, dict),
            "result_json_key": key,
        }
        if isinstance(diagnostics, dict):
            # Keep all resolved settings and final evidence; reference the
            # original for potentially large per-step arrays.
            record["summary"] = {
                field: value for field, value in diagnostics.items()
                if field not in {"steps", "trajectory"}
            }
            steps = diagnostics.get("steps")
            if isinstance(steps, list):
                record["recorded_step_count"] = len(steps)
                record["step_evidence"] = key + ".steps"
        mechanisms[key] = record
    audits = [
        {**evidence(path), "payload": json.loads(path.read_text(encoding="utf-8"))}
        for path in reports
    ]
    payload = {
        "schema_version": 1, "policy": DECISION_POLICY,
        "result": evidence(result_json), "compiled_input": evidence(compiled_input),
        "mechanisms": mechanisms, "audits": audits, "screening": dict(screening),
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# 本设计的判定说明", "",
        "这里只转述本次实际记录的参数和审计结论，不重新打分。",
        "控制器默认权重和代理阈值属于工程设定；达到它们不等于可折叠或实验成功。", "",
        f"- 合同状态：`{screening['contract_status']}`",
        f"- 筛选建议：`{screening['recommendation']}`",
        "- 未记录某个机制时，不能据此断言它关闭了。",
        "- 完整公式、触发条件和依据：仓库 `docs/rfd3_mosaic/DECISION_RULES.zh-CN.md`。",
        "- 本目录 `decision_explanation.json` 保存审计原文、参数摘要、证据相对路径和 SHA-256。",
        "- 逐步数值在原始结果 JSON 的各 `*_diagnostics.steps` 中；",
        "  界面/核心步的 `line_search_trials` 记录候选步长、条件与拒绝原因。", "",
        "## 筛选证据", "", "```json",
        json.dumps(dict(screening), indent=2, ensure_ascii=False), "```", "",
    ]
    for record in mechanisms.values():
        lines.extend([
            "## " + record["name"], "", "```json",
            json.dumps(record, indent=2, ensure_ascii=False), "```", "",
        ])
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return output
