# RFD3 Multi-Interface Seed：功能保留、升级与新功能实施方案

> **统一的复现基线（2026-07-22）**：一个 interface seed 是从已验证 PPI
> 两侧提取并保持相对几何的完整双片段 motif。对称复制后，同一 seed 的两侧
> 继续形成跨 protomer 界面；一个最终 protomer 由相邻 seed 各贡献一侧并由新
> scaffold 连接。邻接方向由几何确定，不能普遍写死为 `k+1`。在当前静态
> RFD3 复现阶段，所有 seed 原子的坐标和序列必须完全冻结（`ALL`），只有
> scaffold 可生成；任何部分固定或端到端坐标漂移都视为失败。
>
> **Pose sampling 实现更新（2026-07-22）**：C3 基线不再使用固定
> `[0°,0°,0°] + 25 Å`。完整全原子 seed 现在通过单位 quaternion 进行
> Haar-uniform SO(3) 刚体旋转，并在 20–30 Å 采样半径。每个 pose seed
> 均可复现并记录 quaternion、旋转矩阵和实际半径。GPU 推理前先通过
> CPU pose ensemble 批量过滤 clash、界面、linker 可达性和 required
> objectives；只允许通过静态门槛的候选进入 RFD3。

## 1. 项目重新定义

本项目不应被定义为“把 RFD1 的 A/B 双链 Interface-Seed 搬到 RFD3”。

更准确的定义是：

> 保留 Interface-Seed 1.0 的完整能力，并将其重构为一个面向 RFD3 的多片段、多界面、全原子几何条件控制框架。A/B 双链仅作为 legacy benchmark；新框架从数据模型开始就支持任意数量的 motif fragments、多个非等价 interfaces、多个刚体或柔性组，以及显式的 symmetry/interface connectivity。

建议项目名称：

```text
RFD3 Multi-Interface Seed
```

建议方法版本：

```text
Interface-Seed 2.0
```

## 2. 最重要的设计原则

### 2.1 A/B 是特例，不是核心模型

旧版：

```text
固定假设：chain A + chain B
```

新版：

```text
N 个 fragments
M 个 interfaces
G 个 rigid/flexible groups
S 个 symmetry orbits
```

例如，一个任务可以同时包含：

- fragment A/B：主要寡聚化界面；
- fragment C/D：第二个非等价界面；
- fragment E：内部功能 motif；
- ligand L：小分子或金属；
- loop F：允许局部柔性的支持片段。

### 2.2 不再用 chain letter 表达算法语义

不能继续依赖：

```text
A/B, C/D, E/F
```

chain ID 只是输入文件标识，不应决定谁与谁形成界面。新版使用稳定的逻辑 ID：

```text
ring_left
ring_right
secondary_top
secondary_bottom
functional_core
ligand_site
```

真实 PDB/CIF chain、residue、atom selection 只存放在 selection 字段中。

### 2.3 显式 graph 代替字符串猜测

旧版通过 X/Y 替换及 previous/next 距离判断隐式决定拓扑。

新版内部首先构建：

```text
InterfaceGraph
```

其中：

- node = motif fragment 或 rigid group；
- edge = 必须满足的 interface relationship；
- orbit = symmetry 作用下的一组等价 nodes/edges；
- scaffold link = 需要由 RFD3 生成的连接区域。

## 3. 必须保留的 Interface-Seed 1.0 能力

| 1.0 能力 | 2.0 中的保留方式 |
|---|---|
| A/B 双链输入 | legacy adapter 将 A/B 转成两个 `FragmentSpec` |
| whole-seed rotation | 一个或多个 `RigidGroup` 的 pose sampling |
| initial distance sampling | 几何定义明确的 radial/axial placement |
| cyclic expansion | `SymmetryOrbit` 展开 |
| previous/next 自动判断 | graph inference，可被显式配置覆盖 |
| X/Y contig 重建 | graph-to-RFD3 specification 编译器 |
| symmetry-tied linker length | `ScaffoldLink.tie_group` |
| motif dragging | group-level SE(3) pose controller |
| sampled pose metadata | 结构化 JSON trajectory |

legacy 模式必须可运行：

```yaml
interface_seed:
  mode: legacy_rfd1
```

新版模式：

```yaml
interface_seed:
  mode: multi_interface_se3
```

## 4. 通用核心数据模型

### 4.1 `FragmentSpec`

表示一个具有独立逻辑含义的 motif fragment：

```yaml
fragments:
  ring_left:
    source: inputs/design_seed.cif
    selection: A/PROTEIN/165-194/*
    role: interface

  ring_right:
    source: inputs/design_seed.cif
    selection: B/PROTEIN/211-241/*
    role: interface

  catalytic_core:
    source: inputs/design_seed.cif
    selection: C/PROTEIN/10-18/*
    role: functional_motif

  ligand:
    source: inputs/ligand.sdf
    selection: "*"
    role: ligand
```

每个 fragment 至少保存：

- 唯一逻辑 ID；
- source structure；
- atom/residue selection；
- polymer/ligand 类型；
- reference coordinates；
- local frame；
- flexibility mode；
- fixed atoms；
- sequence/index conditioning；
- symmetry membership。

### 4.2 `MotionGroup`

多个 fragments 可以属于同一个运动组：

```yaml
motion_groups:
  primary_seed:
    members: [ring_left, ring_right]
    mode: rigid

  functional_group:
    members: [catalytic_core, ligand]
    mode: rigid

  support_group:
    members: [supporting_helix]
    mode: soft_rigid
```

支持模式：

| 模式 | 含义 |
|---|---|
| `fixed` | 坐标始终固定 |
| `rigid` | 组内几何不变，整体 SE(3) 可变化 |
| `soft_rigid` | 允许有限平移和旋转 |
| `flexible_backbone` | 后期功能，允许局部 backbone 变化 |
| `diffused` | 交给 RFD3 生成或重设计 |

第一版实现：

```text
fixed + rigid + soft_rigid
```

不要第一版就实现自由 backbone deformation。

### 4.3 `InterfaceEdge`

显式描述两个 fragments/groups 应满足的关系：

```yaml
interfaces:
  ring_interface:
    left: ring_left
    right: ring_right
    copy_relation: previous
    required: true

    geometry:
      target_distance: 8.0
      distance_tolerance: 2.0
      target_twist_deg: null
      target_tilt_deg: null

    contacts:
      min_heavy_atom_contacts: 12

  secondary_interface:
    left: secondary_top
    right: secondary_bottom
    copy_relation: secondary_axis
    required: true
```

`copy_relation` 可以是：

```text
same
previous
next
offset(k)
secondary_axis
explicit_transform
auto
```

### 4.4 `SymmetryOrbit`

不是所有 fragments 都必须使用相同对称操作：

```yaml
symmetry:
  groups:
    ring_orbit:
      id: C3
      members: [ring_left, ring_right]

    unsym_functional_group:
      id: none
      members: [catalytic_core, ligand]
```

这允许：

- 对称蛋白框架 + 非对称 ligand；
- 对称重复界面 + 单个功能 motif；
- 多个 symmetry-related interface families；
- 后续扩展 bifacial building blocks。

### 4.5 `ScaffoldLink`

描述需要生成的蛋白片段和拓扑连接：

```yaml
scaffold_links:
  protomer_linker:
    from: ring_right
    to: ring_left
    length:
      min: 70
      max: 100
    tie_group: protomer_lengths
    chain_break: false

  functional_insert:
    from: catalytic_core
    to: support_group
    length:
      min: 15
      max: 25
    tie_group: null
```

`tie_group` 相同的 link 只采样一次长度，保证 symmetry-equivalent chains 等长。

## 5. 建议的统一配置格式

```yaml
interface_seed:
  schema_version: 2
  mode: multi_interface_se3
  random_seed: 42

  fragments:
    ring_left:
      source: inputs/design_seed.cif
      selection: A/PROTEIN/165-194/*
      role: interface
      flexibility: rigid

    ring_right:
      source: inputs/design_seed.cif
      selection: B/PROTEIN/211-241/*
      role: interface
      flexibility: rigid

    secondary_top:
      source: inputs/design_seed.cif
      selection: C/PROTEIN/40-55/*
      role: interface
      flexibility: soft_rigid

    secondary_bottom:
      source: inputs/design_seed.cif
      selection: D/PROTEIN/80-95/*
      role: interface
      flexibility: soft_rigid

    functional_core:
      source: inputs/design_seed.cif
      selection: E/PROTEIN/10-18/*
      role: functional_motif
      flexibility: fixed

  motion_groups:
    primary_interface_seed:
      members: [ring_left, ring_right]
      mode: rigid

    secondary_interface_seed:
      members: [secondary_top, secondary_bottom]
      mode: soft_rigid
      max_translation: 1.0
      max_rotation_deg: 5.0

    functional_site:
      members: [functional_core]
      mode: fixed

  symmetry:
    groups:
      primary_orbit:
        id: C3
        members: [primary_interface_seed, secondary_interface_seed]
        axis: [0.0, 0.0, 1.0]
        center: [0.0, 0.0, 0.0]

      functional_orbit:
        id: none
        members: [functional_site]

  interfaces:
    primary_ring:
      left: ring_left
      right: ring_right
      copy_relation: previous
      required: true

    secondary_contact:
      left: secondary_top
      right: secondary_bottom
      copy_relation: next
      required: true

  scaffold_links:
    primary_linker:
      from: ring_right
      to: ring_left
      length: {min: 70, max: 100}
      tie_group: primary_protomer

  initialization:
    center_method: interface_heavy_atom_com
    rotation_method: uniform_so3

    placement:
      radius: {mean: 25.0, range: 5.0}
      axial_offset: {mean: 0.0, range: 2.0}
      azimuth_deg: [-180.0, 180.0]
      tilt_deg: [-20.0, 20.0]
      twist_deg: [-180.0, 180.0]

  guidance:
    enabled: true
    controller: rigid_group_se3
    aggregation: pose_graph

    schedule:
      early_fraction: 0.35
      middle_fraction: 0.45
      late_fraction: 0.20

    terms:
      distance: {early: 2.0, middle: 1.0, late: 0.1}
      orientation: {early: 1.5, middle: 1.0, late: 0.1}
      contacts: {early: 0.1, middle: 1.0, late: 0.5}
      clashes: {early: 0.5, middle: 2.0, late: 2.0}
      motif_fidelity: {early: 0.2, middle: 1.0, late: 3.0}

    limits:
      max_translation_per_step: 0.5
      max_rotation_deg_per_step: 2.0
      freeze_pose_after_fraction: 0.85

  output:
    save_resolved_config: true
    save_pose_trajectory: true
    save_guidance_trajectory: true
    save_interface_metrics: true
```

## 6. 软件架构

```text
rfd3_multi_interface_seed/
├── pyproject.toml
├── README.md
├── configs/
│   ├── legacy/
│   ├── single_interface/
│   └── multi_interface/
├── src/rfd3_multi_interface_seed/
│   ├── schema/
│   │   ├── fragment.py
│   │   ├── motion_group.py
│   │   ├── interface_edge.py
│   │   ├── symmetry_orbit.py
│   │   └── scaffold_link.py
│   ├── parsing/
│   │   ├── structure_loader.py
│   │   ├── atomworks_selection.py
│   │   └── legacy_ab_adapter.py
│   ├── geometry/
│   │   ├── frames.py
│   │   ├── se3.py
│   │   ├── so3_sampling.py
│   │   ├── placement.py
│   │   └── symmetry_expansion.py
│   ├── topology/
│   │   ├── graph.py
│   │   ├── pairing.py
│   │   ├── pose_graph.py
│   │   └── scaffold_compiler.py
│   ├── adapters/
│   │   ├── rfd3_input.py
│   │   ├── rfd3_sampler.py
│   │   └── output_metadata.py
│   ├── guidance/
│   │   ├── controller.py
│   │   ├── aggregation.py
│   │   ├── schedules.py
│   │   └── terms/
│   │       ├── distance.py
│   │       ├── orientation.py
│   │       ├── contacts.py
│   │       ├── clashes.py
│   │       ├── motif_fidelity.py
│   │       └── ligand_geometry.py
│   ├── validation/
│   │   ├── schema.py
│   │   ├── geometry.py
│   │   ├── topology.py
│   │   └── rfd3_prevalidation.py
│   ├── diagnostics/
│   │   ├── metrics.py
│   │   ├── trajectory.py
│   │   └── report.py
│   └── cli.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
└── examples/
    ├── lhd101_c3/
    ├── two_interface_d2/
    └── ligand_mediated/
```

原则：

```text
Geometry/Topology 层不依赖 RFD3
Adapter 层负责 RFD3 API
Guidance 层通过 sampler adapter 工作
Diagnostics 层不改变生成状态
```

## 7. 正确的多片段 SE(3) 初始化

### 7.1 每个 motion group 建立局部 frame

对 interface group：

- origin：所选 interface heavy atoms 的 COM；
- primary axis：两个 fragment COM 的连线；
- interface normal：通过 PCA 或选定 anchor atoms 建立；
- third axis：叉积并正交化。

对单 fragment functional motif：

- origin：固定 atoms COM；
- axes：PCA 或用户指定 anchors；
- frame definition 必须写入 metadata。

### 7.2 使用标准 SE(3)

对 group `g`：

```text
X_g(t) = R_g(t) X_g(reference) + t_g(t)
```

必须保证：

- 刚体组内部任意 atom-pair distance 不变；
- symmetry copies 来自同一个 master pose；
- translation 与 rotation 分开记录；
- 不允许 atom-slot-wise 非刚体漂移。

### 7.3 支持多个独立 pose

```text
T_primary(t)
T_secondary(t)
T_functional(t)
...
```

不同组可以：

- 完全独立；
- symmetry tied；
- relative-pose restrained；
- hierarchical：child pose 相对 parent pose 表示。

## 8. 多界面 pose graph controller

多界面任务不能逐条 edge 顺序移动坐标，否则不同约束可能互相覆盖。

建议每一步：

1. 对所有 interface edges 同时计算误差；
2. 将每条 edge 的误差转换为 group-level SE(3) twist proposal；
3. 按权重和 schedule 聚合 proposal；
4. 解决同一 group 上相互冲突的约束；
5. 对 translation/rotation step 分别 clamp；
6. 更新 master poses；
7. 从 master poses 重新生成 symmetry copies；
8. 验证 fixed atoms、connectivity 和 masks 一致。

SE(3) 小步更新可表示为：

```text
delta_xi_g = [delta_translation_g, delta_rotation_g]
T_g(t-1) = Exp(alpha(t) * delta_xi_g) T_g(t)
```

第一版不需要实现复杂全局优化器，但接口应允许后续使用：

- weighted least squares；
- robust loss；
- constrained pose-graph optimization；
- differentiable guidance。

## 9. Guidance terms 的进一步完善

### 9.1 Distance

不要只用全局 COM。支持：

- fragment COM distance；
- anchor-atom distance；
- plane-to-plane distance；
- axis-to-axis distance；
- symmetry-axis radius；
- ligand-to-pocket distance。

### 9.2 Orientation

使用 local frames 测量：

- geodesic rotation error；
- interface normal angle；
- twist；
- tilt；
- crossing angle；
- handedness。

### 9.3 Contacts

RFD3 模式默认使用 inter-group heavy-atom contacts：

- 排除 self-contact；
- 排除同一 rigid group 内部 contact；
- 区分 motif–scaffold、motif–motif、ligand–protein；
- 支持 soft contact kernel，避免硬 8 Å 阈值不连续；
- 可按 atom/residue importance 加权。

### 9.4 Clashes

分别记录：

- seed–scaffold clashes；
- seed–seed clashes；
- symmetry-copy clashes；
- ligand clashes；
- chain crossing proxy。

### 9.5 Motif fidelity

支持多层级：

- rigid-group internal distance RMSD；
- backbone RMSD；
- selected functional atom RMSD；
- sidechain chi/geometry；
- metal coordination distance/angle；
- ligand pose RMSD。

### 9.6 Connectivity feasibility

多片段任务增加：

- fragment termini distance；
- linker length feasibility；
- chain direction compatibility；
- impossible loop closure penalty；
- scaffold link crossing detection。

## 10. Guidance schedule

建议最初使用可解释的 piecewise schedule：

| 阶段 | 主要目标 | pose step |
|---|---|---|
| Early | radius、distance、orientation、拓扑展开 | 大 |
| Middle | contacts、packing、clash、connectivity | 中 |
| Late | motif fidelity、局部精修、冻结 pose | 小或零 |

所有权重均应：

- 写入 resolved config；
- 每一步记录实际值；
- 支持按 interface edge 单独设置；
- 支持按 motion group 单独设置；
- 可在 ablation 中关闭。

## 11. RFD3 Adapter 的职责

RFD3 adapter 必须明确维护：

- atom coordinates；
- reference/conditioning coordinates；
- fixed-atom masks；
- sequence conditioning；
- residue/index conditioning；
- chain/entity IDs；
- transform/symmetry IDs；
- ligand/CCD metadata；
- trajectory coordinates；
- output mapping。

不能只修改一个坐标 tensor，而忽略 conditioning state。

### 11.1 静态 adapter

第一阶段只生成：

- pre-symmetrized CIF/PDB；
- 标准 RFD3 JSON/YAML；
- mapping JSON；
- pose metadata。

### 11.2 动态 sampler adapter

静态版本成功后再寻找 RFD3 正确 hook：

```text
before_denoising_step
after_model_prediction
after_denoising_step
```

最终只选择一个语义清楚、不会被 sampler realignment 覆盖的位置进行 pose update。

RFD3 的 `center_option`、`allow_realignment`、`ori_token`、symmetry handling 必须与自定义 placement/controller 明确分工。

## 12. 全原子与 ligand-aware 扩展

多 fragment schema 应允许 protein、ligand、metal、PTM 同时存在。

建议新增：

```yaml
functional_constraints:
  metal_site:
    type: coordination
    center: ZN1
    coordinating_atoms:
      - HIS_A:NE2
      - HIS_B:NE2
      - ASP_C:OD1
    target_distances: [2.1, 2.1, 2.0]

  hbond_1:
    type: hydrogen_bond
    donor: fragment_A:NZ
    acceptor: ligand:O4
```

可利用 RFD3 已有的：

- fixed atoms；
- atom-level hotspots；
- donor/acceptor conditioning；
- ligand input；
- motif sidechain redesign；
- RASA conditioning。

## 13. Negative design

多界面设计不能只奖励目标结构。建议将 negative states 也作为 specification：

```yaml
negative_states:
  wrong_pairing:
    penalize_interfaces: [ring_left:ring_left]

  undesired_symmetry:
    symmetry: C2

  apo_assembly:
    remove_fragments: [ligand]
    max_allowed_contacts: 4
```

分阶段实现：

1. inference 后过滤；
2. reranking score；
3. sampler guidance；
4. 必要时再进入 fine-tuning。

## 14. 输出与可诊断性

每个 design 输出：

```text
resolved_config.yaml
input_mapping.json
initial_pose_graph.json
pose_trajectory.jsonl
guidance_trajectory.jsonl
interface_metrics.json
presymmetrized_input.cif
final_design.cif
RFD3 native outputs
```

每个 timestep 至少记录：

- 每个 group 的 translation/quaternion；
- 每个 interface edge 的 distance/orientation error；
- contact/clash counts；
- motif fidelity；
- applied step 和 clamp 状态；
- symmetry deviation；
- NaN/Inf 检查。

禁止用 `nan_to_num` 静默吞掉错误。发现非有限值时应保存 debug snapshot 并终止当前 sample。

## 15. 测试体系

### 15.1 Schema tests

- 任意 N fragments；
- 任意 M interfaces；
- dangling fragment 检测；
- 重复 edge 检测；
- symmetry orbit 冲突；
- cyclic dependency；
- selection 为空；
- fixed/flexible 配置冲突。

### 15.2 Geometry tests

- SO(3) matrix 正交且 determinant = 1；
- rigid transform 保持所有内部 pairwise distances；
- inverse transform 恢复原坐标；
- sampled radius 等于到 symmetry axis 的距离；
- symmetry copies 符合 group operations；
- 多 group 独立变换不互相污染。

### 15.3 Graph tests

- C3/C4/C5 previous/next pairing；
- D2/D3 两类 symmetry edges；
- 多个 non-equivalent interfaces；
- edge orbit 无遗漏、无重复；
- tied linker lengths 一致；
- graph-to-contig/specification mapping 可逆追踪。

### 15.4 Guidance tests

- distance error 单步下降；
- orientation error 单步下降；
- clash term 不会把 groups 拉得更近；
- step-size clamp 生效；
- late freeze 生效；
- 多 edge 冲突不会依赖遍历顺序；
- symmetry copies 始终来自 master pose。

### 15.5 Integration tests

1. LHD101 C3 legacy；
2. LHD101 C3 corrected SE(3)；
3. C4/C5 cyclic generalization；
4. 一个 D2 双界面任务；
5. 一个三片段任务；
6. 一个 ligand-mediated task；
7. RFD3 prevalidation；
8. 单设计 end-to-end smoke test。

## 16. 实施阶段

### Phase 0 — 固定版本与基线

- 固定 RFD1 legacy outputs；
- 固定 RFD3 commit、checkpoint、环境和入口；
- 建立版本清单和 hash；
- 不修改共享安装。

### Phase 1 — 通用多片段几何核心

实现：

```text
FragmentSpec
MotionGroup
InterfaceEdge
SymmetryOrbit
ScaffoldLink
InterfaceGraph
SE(3) geometry
```

虽然先用 LHD101 测试，但 API 不允许出现硬编码 A/B。

### Phase 2 — 静态 RFD3 MVP

- 多 fragment parser；
- 正确 SE(3) placement；
- cyclic expansion；
- graph compilation；
- pre-symmetrized CIF；
- RFD3 InputSpecification writer；
- prevalidation；
- 一个设计完成。

### Phase 3 — Legacy compatibility

- A/B adapter；
- legacy Euler；
- legacy translation；
- legacy pairing；
- legacy contact metric；
- 与 RFD1 reference set 对照。

### Phase 4 — Rigid-group dynamic controller

- translation controller；
- rotation controller；
- schedule；
- contact/clash/fidelity terms；
- pose trajectory。

### Phase 5 — Multi-interface controller

- 多 group poses；
- 多 edge aggregation；
- pose graph；
- conflict handling；
- per-edge/per-group schedules。

### Phase 6 — Controlled flexibility

- fixed；
- rigid；
- soft-rigid；
- fragment-specific bounds；
- 后续再评估 flexible backbone。

### Phase 7 — All-atom functional constraints

- ligand；
- metal；
- hydrogen bonds；
- functional atom fidelity；
- sidechain redesign。

### Phase 8 — Higher-order assemblies

顺序：

```text
C3 LHD101
→ C4/C5
→ D2/D3 multi-interface
→ bifacial building blocks
→ external polyhedral expansion
→ cage
```

官方 RFD3 symmetry sampler 当前主要支持 C/D；T/O/I 或任意 cage topology 应作为独立扩展，不能在 MVP 阶段承诺原生支持。

## 17. 实验与消融矩阵

| 组别 | 目的 |
|---|---|
| RFD1 Interface-Seed | 原方法 baseline |
| Native RFD3 fixed symmetric motif | RFD3 官方 baseline |
| RFD3 legacy adapter | 分离模型平台升级 |
| RFD3 corrected SE(3) | 验证几何修复 |
| RFD3 + rigid pose controller | 验证动态控制 |
| RFD3 + scheduled guidance | 验证时间调度 |
| RFD3 + soft-rigid fragments | 验证局部柔性 |
| RFD3 + multiple interfaces | 验证核心新能力 |
| RFD3 + ligand-aware constraints | 验证全原子优势 |
| RFD3 + negative design | 验证状态选择性 |

核心指标：

- motif backbone/functional-atom RMSD；
- interface distance/orientation error；
- heavy-atom contacts；
- buried SASA；
- clashes；
- symmetry deviation；
- connectivity feasibility；
- designable backbone rate；
- MPNN sequence success；
- RF3/AF3 refolding RMSD；
- pTM/ipTM/PAE；
- GPU cost per accepted design。

## 18. 何时考虑 fine-tuning

初期不训练。只有在以下证据出现后才考虑：

- RFD3 系统性忽略多 interface constraints；
- 强 sampler guidance 明显破坏 backbone quality；
- 多 fragment simultaneous satisfaction 始终很低；
- ligand/metal geometry 无法靠现有 conditioning 保持；
- zero-shot higher-order assembly 成功率不可接受。

训练阶段应新增对应的 multi-interface conditioning task sampler，而不是直接用少量成功结构盲目 fine-tune。

## 19. 第一批交付物

### Deliverable 1 — Audit and reference

- RFD1 Interface-Seed 审计；
- RFD1 reference outputs；
- RFD3 version manifest。

### Deliverable 2 — Multi-fragment geometry library

- 通用 schema；
- selection parser；
- SE(3) placement；
- interface graph；
- symmetry expansion；
- 单元测试。

### Deliverable 3 — RFD3 static MVP

- pre-symmetrized CIF；
- RFD3 JSON/YAML；
- mapping/metadata；
- LHD101 C3 smoke test；
- 一个三片段 smoke test。

### Deliverable 4 — Dynamic multi-interface prototype

- sampler adapter；
- rigid-group pose controller；
- scheduled guidance；
- 多 edge aggregation；
- trajectory diagnostics。

## 20. 当前立即执行的步骤

1. 记录服务器 RFD3 环境、源码 commit、checkpoint hash 和入口；
2. 将 RFD3 源码复制/fork 到个人开发目录；
3. 建立独立 `rfd3_multi_interface_seed` Git 仓库；
4. 先实现 schema 与纯几何层，不调用 RFD3；
5. 同时写 LHD101 双片段测试和一个人工三片段测试；
6. 完成 graph → pre-symmetrized CIF → RFD3 specification；
7. 通过 `prevalidate_inputs=True`；
8. 完成单设计 smoke test；
9. 再定位 sampler hook，开发动态 controller。

## 21. 汇报用定义

### 原方法

> Interface-Seed 1.0 extended RFdiffusion1 at inference time to symmetrize an asymmetric two-chain interface seed, reconstruct cyclic cross-interface contigs, and heuristically reposition the motif during denoising.

### 本项目

> RFD3 Multi-Interface Seed generalizes this two-chain cyclic heuristic into a graph-based, all-atom framework for an arbitrary number of motif fragments and non-equivalent interfaces, with SE(3)-correct initialization, symmetry-aware pose control, controlled fragment flexibility, and extensible functional constraints.

### 当前补充：联合分层 pose exploration（2026-07-22）

单纯增加 Haar-SO(3) 随机样本并用一个全局 compactness 分数排序，会使榜单
集中到小半径和相似倾角。当前实现因此增加两层彼此分离的覆盖机制：

1. 使用 joint Latin-hypercube 同时覆盖 radius、axial offset 和 Shoemake
   SO(3) 的三个单位变量，改善有限样本的空间填充；
2. 从完整 rigid interface seed 的坐标计算确定性的最长 PCA 主轴，记录其
   相对于 symmetry axis 的 0–90° principal tilt，并在 radius × tilt 每个
   occupied cell 内独立保留最佳候选。

该分层不改变 interface seed 内部 PPI，也不把 tilt bin 当作生物学过滤阈值。
它解决的是探索覆盖与 GPU 预算分配问题；最终候选仍须通过 fixed-seed、
linker、clash、RFD3 生成和后续结构验证。

## 22. 边界声明

- “任意数量 fragments/interfaces”指软件 schema 和几何/topology 层不硬编码数量；实际可运行规模仍受 GPU 显存、RFD3 token/atom limits 和约束可满足性影响。
- 第一阶段仍使用 LHD101，是为了验证对 Interface-Seed 1.0 的兼容，而不是把框架限制为双链。
- RFD3 原生 symmetry sampler 当前主要支持 C/D；更高阶 polyhedral/cage 支持需要外部 expansion 或 sampler 扩展。
- 服务器私有 `RFdiffusion_asy` 仍需单独审计，不能默认等同于公开 fork。
