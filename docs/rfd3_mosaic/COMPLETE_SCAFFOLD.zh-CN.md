# 完整骨架准备与 partial diffusion

这个入口用于已有目标骨架参考、希望保留单体内部折返与螺旋支撑关系的任务。
它把 **contig 长度、完整骨架、interface seed 摆放、RFD3 初始化和最终验收** 绑定成一个任务。
普通从噪声生成、移动 seed 等已有入口仍然保留。

当前支持锁定或有界刚体移动的 seed、普通完整 Cn/Dn 对称轨道、两端有固定片段的生成区。
参考骨架必须提供每个残基的 N/CA/C/O。可有多个 ASU 聚合物路径；每条路径必须明确对应完整对称副本。
这是有参考的几何构建与受限细化功能，不能据此承诺任意长度均可构建、一定形成真正 coiled-coil、一定可折叠或高生成成功率。

## 使用入口

在已有 RFD3-Mosaic 环境补充 CPU 构建依赖：

```bash
python -m pip install -e '.[scaffold]'
rfd3-mosaic prepare-scaffold design.yaml \
  --blueprint scaffold.yaml \
  --output-dir prepared-task
```

`design.yaml` 仍是普通公开任务文件，声明原始 seed、固定策略、对称群、生成长度和 design 数量。
准备器读取其 **实际编译的 contig 长度**，整体拟合每个 joint seed，并写一个新的固定初始 pose。
随后将所有 seed 的刚体 pose 和所有生成区扭转角放进同一个全装配求解问题，独立验收后再冻结。它不会改变原文件，也不会通过改变生成长度来让任务通过。

`scaffold.yaml` 声明参考及可审查的几何界限。下面示例对应 PI25 的一个 C3 骨架；数值是这个示例的工程界限，**不是通用、已校准的成功率阈值**。
若使用这个示例，对应 `design.yaml` 的生成区必须明确写成 90，而不是让程序把已有的 100 改短。
参考结构不随软件分发，需使用自己有权使用的 PDB/CIF。

```yaml
schema_version: 1
template: PI25_design_backbone.pdb
# 每一行对应一个编译后的 ASU polymer path。
# 每一列按编译器 registry_transform_order 对应一个完整对称副本。
# C3 示例为 e, r1, r2；Dn 是完整 2n 个副本，不能只给一个环。
chains: [[A, C, B]]
partial_t: 2.0
maximum_template_seed_rmsd: 1.0
maximum_template_symmetry_rmsd: 0.1

# 以下编号是目标完整单体中的一基序号，不是原 seed 的 PDB 编号。
# helix_blocks 来自参考结构的二级结构分析，是显式的目标块声明。
helix_blocks:
  - {entity: 0, id: h0, start: 3, end: 20}
  - {entity: 0, id: h1, start: 46, end: 62}
  - {entity: 0, id: h2, start: 75, end: 84}
  - {entity: 0, id: h3, start: 97, end: 110}
  - {entity: 0, id: h4, start: 123, end: 140}
support_edges:
  - {left: h0, right: h1}
  - {left: h1, right: h2}
  - {left: h2, right: h3}
  - {left: h3, right: h4}

limits:
  maximum_ca_deviation: 1.0
  contact_distance: 8.0
  minimum_helix_contact_fraction: 0.2
  minimum_interchain_segment_distance: 1.0
  minimum_ca_bond_distance: 3.3
  maximum_ca_bond_distance: 4.3
  fixed_ca_tolerance: 0.01
  geometry_tolerance: 0.001
  maximum_unsupported_run: 8  # 示例要求；必须显式给出，非普适经验值

construction:
  attempts: 2
  max_nfev: 300

closure:
  reference_shape_weight: 1.0
  maximum_reference_ca_deviation: 2.0
```

这些阈值分三类：参考兼容性、CPU 重建相对原参考的偏差、diffusion 相对已验证重建骨架的偏差。
`closure.maximum_reference_ca_deviation` 与 `limits.maximum_ca_deviation` 因而不是同一个阶段。
`minimum_helix_contact_fraction` 对每条支撑边的两端分别检查；只统计同一物理链上的不同声明块，避免借另一条链“补足”单体内部支撑。
距离单位为 Å，比例无量纲，形状权重单位为 Å⁻²。

`sampling.scaffold_packing` 原本控制生成区之间的**链间新界面**，与这里的单体内部
块间支撑不同。它取 `off` 不会关闭完整 scaffold 合同；后者在这个模式中必须执行。
`plan --format json` 的 `complete_scaffold` 会显示实际 artifact、阈值、支撑边与 partial 参数。

## 产物与执行

成功目录含：

- `design.yaml`：可交给现有 `validate/run/submit` 的任务入口；包含冻结 pose 及相对 artifact 路径。
- `seed_input.*`：原始 seed 的逐字节副本。
- `complete_scaffold.cif`：完整、对称展开、固定 seed 已恢复的初始骨架。
- `scaffold_artifact.json`：与编译图、固定原子、长度和 pose 绑定的哈希与显式映射。
- `preparation_report.json`：参考哈希、对称拟合、整个 joint seed 拟合、每个连接区求解和逐项验收证据。
- `reference_template.*`、`blueprint.yaml`：准备过程所用参考与可重放的几何声明。

```bash
rfd3-mosaic validate prepared-task/design.yaml
rfd3-mosaic run prepared-task/design.yaml
```

后两步需要完整 RFD3 环境；`run` 才开始模型采样。准备器本身不提交任何 GPU 作业。
移动整个目录即可携带结构依赖；执行服务器的 `resources.profile` 与 `output.root` 仍需按其环境设置。

同一任务的所有 designs 共用同一个完整骨架和初始 pose，随后使用不同的生成噪声。
锁定模式保持 seed 初始位置；移动模式允许每个 design 在声明界限内独立更新整个 joint seed，同时更新生成区参考，逐项接受或整体回退。
修改 `sampling.designs` 不重新抽取 pose。要比较不同 pose/fold prior，应分别准备任务并保留各自的报告；不同任务名称本身不保证几何多样性。
已存在的输出目录拒绝覆盖，避免把两个 pose 混成同一个任务。

## 长度不同与失败的含义

同长度参考直接提供内部坐标先验。若目标生成长度不同，必须给每个生成区明确的
`secondary_structure` 和 `reference_secondary_structure`，各为 `[entity][run]` 的 H/L 字符串。
长度必须分别等于目标和参考生成残基数；H/L 块顺序必须一致。
多个变长生成区还需要 `reference_generated_lengths: [[...], ...]` 消除片段对应歧义。

重建插值的是对应块内的**扭转角**，不拉伸 XYZ，也不修改固定 seed。
H/L 只是初始化，完整骨架仍须闭合并通过碰撞、支撑、形状与对称检查。
变长重建没有天然的一一残基对应，因此在相同 H/L 块内选取 `min(旧长度, 新长度)` 个互不重复的离散地标。形状和接触残差只作用于这些显式对应，不插值 XYZ；新增位点由肽链内部坐标与完整装配验收约束。映射与搜索预算写入报告。

`unresolved` 表示给定先验和有限 CPU 搜索预算内没有得到通过验收的骨架。
它既不是“这类结构不存在”的证明，也不是可提交采样的成功输入。
CLI 以非零状态退出，只保存 `status: unresolved` 的 `preparation_report.json`，不生成可运行的任务 YAML。
不会降低阈值、缩短生成区、修改 seed 或自动转到 GPU 搜索。

## 验收的实际边界

完整骨架模式以参考形状、同链支撑和实际链段间距替代原来的直线连接走廊规则。
原子的避碰、固定 seed、对称性、聚合物连续性及其它适用的审计仍然执行。
CPU 全装配还检查非相邻 N/CA/C/O 的距离，包含其它 seed 和其它链；它不是侧链能量或全原子打分。
生成结果必须独立通过相同的结构合同；参考成立不代表生成结果自动通过。

新准备任务使用 v2 合同：从实际 N/CA/C/O 推断 alpha 螺旋，检查生成螺旋是否被目标声明覆盖、伙伴是否属于同链另一条真实螺旋、接触覆盖率和最长无支撑段是否合格。生成骨架及接头的键长、键角、肽平面和非局部骨架碰撞也是必须通过的验收项。旧 v1 合同仍可读取，但明确标记完整骨架质量未评估。

这仍不能认证真正 coiled-coil、侧链疏水核心、序列设计性或实验稳定性。
本入口当前不支持 local-neighbourhood、quotient/mixed 轨道、端部生成区、配体、柱坐标约束、辅助 residue conditioning 或 motif-sidechain redesign；组合使用会明确拒绝。
这些功能在普通模式仍保留，不能把普通模式的成熟度当作新组合已经验证。

完整公式、系数来源、触发条件与实现位置见
[判定规则文档](DECISION_RULES.zh-CN.md)第 25 节。

## 移动模式与结果状态

移动策略仍在普通 `design.yaml` 的 `constraints[].pose` 中声明；属于同一个 `coupling_group` 的片段使用相同策略。例如：

```yaml
pose:
  mode: bounded_mobile
  subspace: bounded_se3
  proposal: scaffold_objectives
  max_translation: 2.0
  max_rotation_deg: 10.0
```

这些是示例运动预算。准备器记录每个固定原子的归属和生成区两端 seed，运行时按参考弧长对两端刚体变换作 SE(3) 插值。seed、参考和生成区干净候选一起通过几何、对称、运动边界与参考分离检查后才提交；后处理重放移动记录，不能只根据日志里的 `passed` 判断。

Mosaic 批任务使用逐 design 的 `design_outcomes.json`。单个普通异常不会立即丢弃其余设计；连续三次普通异常或 CUDA/OOM/文件系统错误会停止并明确记录未执行项。`partial` 表示未完成请求数量；后处理审计不会把它变成 `completed`。`generated` 表示有完整结构与元数据，并不表示结构合同通过。原始失败结构可留作分析，但不会计为合格。

这一轮只验证 CPU 构建、输入交接、采样循环控制逻辑和独立审计；没有新增 GPU 生成成功率证据。
