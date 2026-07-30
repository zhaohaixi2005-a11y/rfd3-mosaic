# RFD3 Multi-Interface Seed（Interface-Seed 2.0）最终架构与实施方案

> **当前已落实的静态采样基线（2026-07-22）**：完整 interface seed 使用
> `fixed_atoms: all` 保持全原子坐标和序列；其整体 pose 使用 Haar-uniform
> SO(3) quaternion sampling，LHD101 C3 半径在 20–30 Å 内采样。采样变换
> 对 seed 两侧共同施加，内部 PPI 几何保持不变。每个 pose 具有可复现 seed
> 和完整 provenance，并先经过 CPU pose-ensemble 静态过滤，再进入 RFD3。
> 首轮 64 个 pose 全部通过，证明了采样链路可用，也暴露了旧筛选过松：
> 70–100 aa contour 只能判断“理论上够长”，不能判断 linker 是否容易形成；
> 未配置 objective 时 penalty 恒为零；按 copy 间距越大排序会错误偏好松散组装。
> 当前已改为在 hard clash、interface 和 contour gates 之后，以最坏 linker
> endpoint span 与中心轴空洞作为显式 soft shortlist 指标。该分数只用于节省
> GPU 与选择多样化候选，不被解释为可折叠性或最终 designability 证明。
> 第二轮 64-pose 排序进一步显示，中心轴间隙也不能无界地“越小越好”：
> seed 1010 的间隙仅 2.267 Å，却因旧方向被奖励。LHD101 示例现改用
> 6–14 Å 的可配置 soft window；这是该示例的启发式窗口，不是通用 Cn
> 定律。当前静态几何的暂定首选为 seed 1058（最大 linker span 24.623 Å，
> axis clearance 10.639 Å），仍须经过姿态多样性选择和 RFD3 验证。
> 为避免把几乎相同的旋转姿态重复送入 GPU，现增加独立的
> `rfd3_mosaic.pose_select`：它保持几何评分顺序，同时以 quaternion 的
> sign-invariant SO(3) geodesic angle 去除近重复姿态；候选池大小、shortlist
> 数量、最小角分离以及未来多 group 情况下的 diversity group 均为显式参数。
> v3 前十名全部落在 25 Å 以下，进一步证明单一 compactness 排名会把
> radius 与 orientation 的联合分布压缩到“小半径、短 linker”的局部区域。
> 当前 ensemble 改用 joint Latin-hypercube 覆盖 radius、axial offset 与
> Shoemake Haar-SO(3) 的三个单位变量，并记录完整 unit samples。为了把肉眼
> 所见的“倾斜”变成可审计指标，每个完整 rigid seed 用最长 PCA 主轴定义
> sign-invariant 0–90° principal tilt。随后通过独立的
> `rfd3_mosaic.pose_stratify` 在 radius × tilt 每个 occupied cell 内选择最佳
> 候选，避免某一种紧凑倾角垄断全局榜单。分层用于探索覆盖，不是生物学
> 合格阈值；PPI seed 内部仍保持全原子刚体不变。
> v4 的 256 个候选实现了 16/16 radius × tilt cell 覆盖。为确保 GPU 真正
> 使用 CPU 搜索得到的同一个 LHS pose，RFD3 adapter 现可直接读取 candidate
> manifest，校验 config SHA256、恢复精确 unit samples，并要求重建 CIF 的
> SHA256 与候选完全一致；仅传相同整数 seed 不再被视为足够 provenance。
> 在不改变 Haar SO(3) 与 joint LHS 的前提下，当前进一步加入实验性的
> `rfd3_mosaic.pose_qd`。它沿用 ensemble ranking 作为静态调度优先级，同时
> 使用中心轴间隙与轴向/径向形态比建立 ring/cage morphology cells，并在
> 每个 cell 内保留最佳候选；全局再施加 quaternion SO(3) 最小角分离。
> 为避免稀有但低质量的极端 cell 强行占用 GPU，只有 ensemble 排名前
> 25% 的 accepted candidates 默认有资格进入 QD archive。
> 这使实验能够同时比较优先候选与形态覆盖，而不会让单一
> compactness 分数把候选全部压缩到相似的扁平、小半径构型。形态描述符
> 是 GPU 预算分配工具，不是可折叠性或 designability 结论。

### 当前候选筛选的严格逻辑与证据边界

设计者未声明目标 ring/cage 尺寸时，不存在“generated-scaffold endpoint
越短越好”
或“空洞越大越好”的普适优化方向。当前 CPU 阶段必须区分三件事：

1. **admissibility（硬门槛）**：无 hard motif clash、required interface
   geometry 通过、待生成 scaffold segment 的端点 span 不超过最大 contour、
   required constraints 全部满足。失败者不得进入 GPU 实验。
2. **coverage（实验覆盖）**：合格姿态按 axis clearance 与
   axial/radial aspect ratio 分箱，并施加 quaternion SO(3) 最小角分离。
   未配置目标尺寸时，各 cell 是并列的实验条件，而不是从小到大的质量等级。
3. **empirical quality（经验质量）**：同一 pose 必须用相同的一组 diffusion
   seeds 做配对 50-step 实验。只有 fixed-seed、exact symmetry、zero chain
   breaks 和 zero hard clashes 同时通过，才记为一次成功。pose 质量由重复成功率
   定义，而不能由 CPU endpoint span 单独定义。

当前配置中的静态 penalty 为：

```text
P_static = d_max / 25
         + 0.25 * (distance(c_axis, [6, 14]) / 4)^2
```

其中 `d_max` 是同一 protomer 内、待生成 scaffold segment 两端固定片段之间
的最大 endpoint distance；它不是亚基间 linker。`c_axis` 是最小 axis
clearance。第一项仅表达“在相同待生成残基预算下减少几何闭合负担”的启发式，
不是生物物理最优性；第二项表达 LHD101 示例的可配置空洞窗口，也不是通用
Cn 定律。因此 `P_static` 只能用于 CPU 调度优先级，不能被汇报为设计质量分数。

该拓扑必须表述为：

```text
fixed interface pair k
    -> generated protomer segment k
    -> fixed interface pair k+1
```

每个 fixed interface pair 的两侧属于两个不同 protomers，并负责相邻 units
之间的自组装；每段 70--100 aa 生成区域则把两个相邻 interface 位置上、属于
同一个 protomer 的固定片段连接为完整 unit。不存在连接三个 units 的独立
柔性 linker。

正式 pose-position 对照采用配对实验。对 pose `i` 和共享 diffusion seed
`s`，定义：

```text
H(i,s) = 1
```

当且仅当 seed integrity、declared-transform symmetry、continuity 和 hard-clash
四个 audit 同时通过，否则 `H(i,s) = 0`。若共有 `S` 个共享 diffusion seeds：

```text
success_rate(i) = sum_s H(i,s) / S
```

只有 `success_rate` 和通过后的结构指标能够支持“该 seed position 更优”的
结论。推荐第一阶段对每个 morphology cell 的代表使用同一组至少三个 diffusion
seeds；随后只将重复成功且形态符合设计目标的 pose 提升到 200 steps。若设计者
之后明确给出目标直径、空洞或厚度，则应把这些目标写成显式 window objective，
而不是继续使用隐含的单调 compactness 假设。

### GPU 前可计算的 unit-boundary 几何

GPU 前并非只能检查 endpoint distance。对每一段同一 protomer 内的
`C-fixed -> generated scaffold -> N-fixed` 边界，standalone compiler 还应
报告：

- C 端 `CA->C` continuation tangent 与 endpoint chord 的夹角；
- N 端 `N->CA` continuation tangent 与 endpoint chord 的夹角；
- 两端 continuation tangents 的相对夹角；
- 两个 terminal peptide-plane normals 的 sign-invariant 相对夹角；
- endpoint chord 相对于 symmetry plane 的轴向分量和角度；
- endpoint chord 到 symmetry axis 的最小距离；
- 去除本 protomer 两个端点 fragments 后，chord 中央 80% 到其余 fixed
  motif atoms 的最小距离。

这些指标分别描述 terminal-frame compatibility、生成主体的空间跨越方向和
直线路径风险。它们可以在配置中作为显式 window/threshold objective，也可以
作为 QD exploration coordinates。由于 70--100 aa 主体可以折叠成非直线路径，
chord clearance 只能作为风险代理，不能单独硬判 folded scaffold 是否成功。
在首批配对 50-step 数据建立成功区间前，默认只记录并分层这些指标，不擅自设置
“越小”或“越大”的普适方向。

> 文档状态：最终设计基线（implementation baseline）
> 目标平台：RFD3
> 兼容基线：RFdiffusion1 Interface-Seed
> 核心定位：面向任意数量 motif fragments 与多个非等价 interfaces 的、对称感知的 SE(3) pose-graph 条件控制框架

---

## 0. 文档用途与证据等级

本文同时服务于三件事：

1. 指导代码实现；
2. 定义可验证的阶段性交付物；
3. 为升级汇报提供准确边界。

文中的陈述分为三个等级：

| 标记 | 含义 | 汇报时的表达 |
|---|---|---|
| `CONFIRMED` | 已由本地 RFD1 fork、示例或运行结果确认 | 可以描述为原方法事实 |
| `TO_VERIFY` | 依赖服务器上的 RFD3 版本、checkpoint 或 API | 只能描述为待验证兼容性 |
| `PROPOSED` | 本项目计划新增的方法或软件能力 | 只能描述为设计/开发目标 |

在任何实验完成前，不应把 `PROPOSED` 写成“RFD3 已经支持”或“我们已经实现”。

相关审计文件：

- `INTERFACE_SEED_RFD1_UPGRADE_AUDIT.md`：RFD1 Interface-Seed 改动审计；
- 本文：RFD3 升级的最终架构与执行路线。

---

## 1. 最终项目定义

### 1.1 项目不是双链移植

本项目不定义为：

```text
把 A/B 双链 Interface-Seed 从 RFD1 搬到 RFD3
```

最终定义为：

> 保留 Interface-Seed 1.0 的推理期界面种子控制能力，将其重构为独立于生成模型的几何与拓扑编译层，并进一步发展为 RFD3 上的多片段、多界面、全原子、对称感知的 SE(3) pose-graph conditioning framework。

其中：

- LHD101 A/B 是 legacy regression benchmark；
- 双链是通用模型在 `N=2` 时的特例；
- schema、geometry、topology 和 provenance 层不得硬编码链数；
- 真正可运行的最大规模仍受 GPU、RFD3 输入限制和约束可满足性影响。

建议名称：

```text
Project: RFD3 Multi-Interface Seed
Method:  Interface-Seed 2.0
```

### 1.2 方法级表述

> Interface-seeded assembly design is formulated as an SE(3) pose-graph conditioning problem with symmetry tying, explicit interface ports, all-atom geometric constraints, and scaffold-connectivity constraints, with RFD3 acting as the structure generator.

对应关系：

| 数学/图模型对象 | 软件对象 | 生物学含义 |
|---|---|---|
| node | `MotionGroupInstance` | 一个可独立运动的 motif 组实例 |
| port | `InterfacePortInstance` | 运动组上的特定界面及局部坐标系 |
| edge | `InterfaceEdgeInstance` | 两个 port 应满足的相对几何 |
| group action | `SymmetryTransform` | Cn、Dn 或显式变换 |
| sequence link | `ScaffoldLinkInstance` | 需要生成的 N/C 端连接 |
| state | `MotionGroupState` | 当前扩散步的 master SE(3) pose |
| generator | RFD3 adapter | 全原子结构生成 |
| controller | pose-graph guidance | 随时间变化的几何控制 |

---

## 2. 范围与边界

### 2.1 必须实现

- 任意 `N` 个 fragments；
- 任意 `G` 个 motion-group specs；
- 任意 `M` 个 interface-edge specs；
- 一个或多个 symmetry orbits；
- 显式 interface ports 与稳定 local frames；
- 完整 provenance mapping；
- RFD1 A/B legacy 配置适配；
- 静态 pre-symmetrized RFD3 input 编译；
- 分阶段加入 single-interface 和 multi-interface dynamic control；
- protein、ligand、metal 和 functional atoms 的可扩展表示。

“任意数量”表示软件层不硬编码 `2`，不表示可以忽略实际计算资源。

### 2.2 第一版不承诺

- 自由 backbone deformation；
- 任意 cage topology 的 RFD3 原生采样；
- 对所有 RFD3 版本通用的 sampler hook；
- 主体严格对称但单个蛋白 motif 不对称的 native symmetry sampling；
- 未经实验验证的 fine-tuning 收益；
- 一次性同时实现所有多界面、配体、金属和 negative design 功能。

### 2.3 平台能力必须现场验证

以下项目必须针对服务器上实际使用的 RFD3 commit、checkpoint 和入口验证：

- 输入 specification 的正式字段；
- fixed atom、sequence、ligand、hotspot、donor/acceptor 等 conditioning；
- symmetry 的支持范围和语义；
- centering/realignment 行为；
- trajectory 中可安全修改的状态；
- prevalidation 和输出格式；
- token/atom/residue limits。

在验证前统一标记为 `TO_VERIFY`，不得仅依据文档印象写死 adapter。

---

## 3. 从 Interface-Seed 1.0 继承什么、修正什么、新增什么

### 3.1 保留能力

| RFD1 Interface-Seed 能力 | 2.0 保留方式 |
|---|---|
| A/B asymmetric seed input | `LegacyABAdapter` 转换为通用 specs |
| whole-seed pose sampling | master `MotionGroupState` 的 SE(3) sampling |
| cyclic symmetry expansion | `SymmetryOrbitSpec` + transform registry |
| previous/next pairing | 用户 shortcut，编译为明确 group element |
| X/Y contig reconstruction | 显式 scaffold topology compiler |
| tied linker lengths | `ScaffoldLinkSpec.tie_group` |
| motif dragging | 可选的 group-level pose controller |
| pose metadata | resolved config + mapping + JSONL trajectory |

### 3.2 必须修正

- 从 Euler 顺序依赖转为标准 SO(3)/SE(3)；
- 从含糊的三轴平移转为 radial、axial 或全局向量定义；
- 从 chain letter 语义转为稳定逻辑 ID；
- 从 fragment/group 混合节点转为唯一的 `MotionGroupInstance` 节点；
- 从顺序处理 interfaces 转为同时聚合 edge residuals；
- 从只存距离转为完整相对变换或显式几何约束集合；
- 从“graph-to-spec 完全可逆”转为完整 provenance traceability；
- 从静默数值修复转为 fail-fast + debug snapshot。

### 3.3 新能力

- 多个非等价 interfaces；
- 多个独立或层级 motion groups；
- interface port 局部坐标系；
- reference-preserving 与 exploratory 两类界面目标；
- pose-graph conflict handling；
- per-edge/per-group guidance schedules；
- soft-rigid bounds；
- ligand/metal/functional-atom constraints；
- negative-state filtering、reranking，后续可扩展为 guidance；
- 与生成模型解耦的 standalone compiler。

---

## 4. 核心语义：Spec、Instance、State 必须分离

一个 Python class 不得同时承担用户配置、对称展开结果和扩散时动态状态。

### 4.1 Spec：用户意图

```text
FragmentSpec
MotionGroupSpec
InterfacePortSpec
InterfaceEdgeSpec
SymmetryOrbitSpec
ScaffoldLinkSpec
NegativeStateSpec
```

Spec 应尽量不可变，负责回答“用户想设计什么”。

### 4.2 Instance：编译后的真实对象

```text
FragmentInstance
MotionGroupInstance
InterfacePortInstance
InterfaceEdgeInstance
ScaffoldLinkInstance
```

Instance 包含：

- symmetry copy index；
- resolved atom/residue indices；
- resolved transform ID；
- resolved chain/entity/index mapping；
- 从 reference structure 计算出的 frame/transform；
- 指向源 Spec 的 provenance ID。

### 4.3 State：运行时变量

```text
MotionGroupState
PoseGraphState
GuidanceState
```

State 包含：

- 当前 master pose；
- 当前 timestep；
- edge residuals；
- proposed/applied twists；
- clamp 和 freeze 状态；
- 当前 RFD3 coordinate/mask mapping。

依赖方向必须是：

```text
Spec -> Instance -> State
```

禁止 State 反向修改 Spec。

---

## 5. 最终数据模型

### 5.1 `FragmentSpec`：它是什么

`FragmentSpec` 只描述结构身份和条件信息，不决定它如何运动。

```yaml
fragments:
  left:
    source: inputs/7mwr_interface.pdb
    selection: "A/165-194/*"
    entity_type: protein
    role: interface_motif
    fixed_atoms: all

  ligand:
    source: inputs/ligand.sdf
    selection: "*"
    entity_type: ligand
    role: functional_component
```

必须字段：

- `id`（由 mapping key 给出）；
- `source`；
- `selection`；
- `entity_type`；
- `role`。

可选字段：

- `fixed_atoms`；
- sequence conditioning；
- atom-level annotations；
- CCD/component metadata；
- provenance tags。

复现基线中的 `interface_motif` 必须使用 `fixed_atoms: all`，并关闭 motif
side-chain redesign。这样可同时冻结主链、侧链和序列。`backbone`、`TIP` 或
显式原子子集只保留给后续经过单独验证的软约束实验，不得用于 LHD101 基线。

不得在这里放 `rigid/soft_rigid/fixed` 等运动学模式。

### 5.2 `MotionGroupSpec`：它如何运动

```yaml
motion_groups:
  primary_seed:
    members: [left, right]
    mode: rigid

  catalytic_unit:
    members: [catalytic_fragment, ligand]
    mode: soft_rigid
    bounds:
      max_translation: 1.0
      max_rotation_deg: 5.0
```

第一阶段支持：

| mode | 语义 |
|---|---|
| `fixed` | master pose 不更新 |
| `rigid` | 内部几何严格不变，整体 SE(3) 可更新 |
| `soft_rigid` | 整体 pose 可在边界内更新 |

后续评估：

- `flexible_backbone`；
- `diffused`。

关键语义：

```text
left 与 right 在同一 rigid group
=> 两者之间的相对几何永远固定

left 与 right 在不同 motion groups
=> controller 可以优化两者的相对 pose
```

parser 必须检测 fragment 是否重复属于互斥 group，或完全没有合法 owner。

### 5.3 `InterfacePortSpec`：界面选择 + 局部坐标系

interface edge 不直接连接 fragment，也不直接连接整个 group，而是连接 group 上的 port。

```yaml
ports:
  left_port:
    group: primary_seed
    fragments: [left]
    atoms: "heavy"
    frame:
      method: anchors
      origin: ["left:A165:CA", "left:A166:CA", "left:A167:CA"]
      x_axis: ["left:A165:CA", "left:A175:CA"]
      xy_plane: ["left:A165:CA", "left:A175:CA", "left:A185:CA"]

  right_port:
    group: primary_seed
    fragments: [right]
    atoms: "heavy"
    frame:
      method: reference_interface_pca
```

port 必须包含：

- owner motion group；
- contributing fragments/atoms；
- 稳定、确定性的右手局部 frame；
- frame construction provenance；
- 可选 contact/functional atom subsets。

支持的 frame method：

1. `anchors`：用户明确给出 origin/axis/plane atoms，最可解释；
2. `reference_interface_pca`：用参考界面原子构造，但必须固定符号规则；
3. `principal_axis_with_anchor`：PCA + anchor 解决轴翻转；
4. `precomputed`：直接读取 4×4 transform。

验证要求：

- anchors 存在且不共线；
- 旋转矩阵正交；
- determinant 接近 `+1`；
- 相同输入重复编译得到相同 frame；
- 不允许 PCA 符号随机翻转。

### 5.4 `InterfaceEdgeSpec`：两个 ports 的目标关系

```yaml
interfaces:
  ring_interface:
    left_port: left_port
    right_port: right_port
    copy_relation:
      orbit_offset: -1
    required: true
    target_geometry:
      mode: reference_transform
      from_reference_seed: true
      translation_tolerance: 2.0
      rotation_tolerance_deg: 10.0
```

内部 pose-graph edge 必须连接：

```text
MotionGroupInstance/InterfacePortInstance
    ->
MotionGroupInstance/InterfacePortInstance
```

而不是混用 fragment node 与 group node。

#### 模式 A：reference-preserving

适合已有界面 seed：

```yaml
target_geometry:
  mode: reference_transform
  from_reference_seed: true
  translation_tolerance: 2.0
  rotation_tolerance_deg: 10.0
```

目标变换：

```text
T_target(left->right) = inverse(T_world_left_ref) * T_world_right_ref
```

distance、twist、tilt、normal angle 都作为该 transform 的派生诊断指标。

#### 模式 B：exploratory constraints

适合没有完整参考 pose 的新界面探索：

```yaml
target_geometry:
  mode: geometric_constraints
  distance:
    type: plane_to_plane
    target: 8.0
    tolerance: 2.0
  normal_angle_deg: {target: 180.0, tolerance: 20.0}
  twist_deg: {range: [-40.0, 40.0]}
  contacts: {min_heavy_atom_contacts: 12}
```

两种模式互斥；parser 不允许同时定义完整 target transform 和互相冲突的绝对几何目标。

### 5.5 `SymmetryOrbitSpec` 与 transform registry

```yaml
symmetry:
  transform_sets:
    ring_c3:
      type: cyclic
      order: 3
      axis: [0.0, 0.0, 1.0]
      center: [0.0, 0.0, 0.0]

  orbits:
    primary_orbit:
      transform_set: ring_c3
      master_groups: [primary_seed]
```

内部统一使用明确 transform ID：

```text
C3:e
C3:r1
C3:r2
D2:e
D2:rx
D2:ry
D2:rz
```

用户可写：

```yaml
copy_relation: {orbit_offset: -1}
```

parser 必须将其编译为具体 group element/transform ID。`previous` 和 `next` 只作为 Cn 的可读 shortcut，不进入核心算法。

第一版 MVP 只承诺所有 protein motif groups 使用同一 C3 orbit。非对称 ligand 或 functional component 可以被 schema 表达，但能否进入 RFD3 native symmetry sampling 标记为 `TO_VERIFY`。

### 5.6 `ScaffoldLinkSpec`：有方向的蛋白连接

```yaml
scaffold_links:
  protomer_linker:
    from: {fragment: right, terminus: C}
    to: {fragment: left, terminus: N}
    length: {min: 70, max: 100}
    tie_group: protomer_length
    chain_break: false
```

必须验证：

- 两个 termini 存在；
- `C -> N` 方向合法；
- 所需长度与端点距离基本相容；
- symmetry-equivalent links 的方向与采样长度一致；
- 不重复占用同一 terminus；
- 不产生明显非法环或跨链连接；
- graph 连通不等于 protein topology 可生成，二者分别验证。

### 5.7 `MappingRegistry`：完整来源追踪

不要求 graph 到 RFD3 specification 严格数学可逆，因为 symmetry expansion 会把高层对象展开为多个实体。

硬性要求是：每个输出 atom/residue/chain 都能追踪到：

- source file；
- source chain/residue/atom；
- `FragmentSpec`；
- `MotionGroupSpec` 与 instance；
- symmetry copy 与 transform ID；
- port membership；
- interface-edge membership；
- scaffold-link membership；
- RFD3 entity/chain/residue/atom index。

建议核心 API：

```python
registry.source_to_output(...)
registry.output_to_source(...)
registry.group_atoms(group_instance_id)
registry.port_atoms(port_instance_id)
registry.edge_atoms(edge_instance_id)
registry.rfd3_indices(fragment_instance_id)
```

映射缺失、重复或一对多关系未经声明时，编译必须失败。

---

## 6. 编译生命周期

整个过程分为五层：

```text
User YAML
  -> validated Specs
  -> resolved reference geometry
  -> symmetry-expanded Instances
  -> RFD3-independent CompiledInterfaceSeed
  -> RFD3 adapter inputs
  -> optional dynamic PoseGraphState
```

### 6.1 第一条 end-to-end API

```python
compiled = compile_interface_seed(
    config="configs/single_interface/lhd101_c3.yaml"
)
```

返回：

```python
CompiledInterfaceSeed(
    structure=...,
    rfd3_agnostic_specification=...,
    mapping_registry=...,
    initial_pose_graph=...,
    resolved_config=...,
    validation_report=...,
)
```

此函数不得导入、安装或运行 RFD3。

### 6.2 编译步骤

```text
load_fragments
-> resolve_selections
-> build_motion_group_specs
-> build_interface_ports_and_frames
-> calculate_reference_transforms
-> sample_master_poses
-> expand_symmetry_orbits
-> instantiate_interface_edges
-> sample_tied_link_lengths
-> compile_scaffold_topology
-> build_mapping_registry
-> validate_compilation
-> write_presymmetrized_cif
-> write_agnostic_specification
```

任何步骤失败，都必须带对象 ID、source selection 和建议修复方式。

---

## 7. LHD101 C3 静态 MVP 配置

第一版只包含一个 interface family，不同时引入 secondary interface、functional motif 或 ligand。

```yaml
interface_seed:
  schema_version: 2
  mode: se3_static
  random_seed: 42

  fragments:
    left:
      source: inputs/7mwr_interface.pdb
      selection: "A/165-194/*"
      entity_type: protein
      role: interface_motif
      fixed_atoms: all

    right:
      source: inputs/7mwr_interface.pdb
      selection: "B/211-241/*"
      entity_type: protein
      role: interface_motif
      fixed_atoms: all

  motion_groups:
    primary_seed:
      members: [left, right]
      mode: rigid

  ports:
    left_port:
      group: primary_seed
      fragments: [left]
      atoms: heavy
      frame: {method: reference_interface_pca}

    right_port:
      group: primary_seed
      fragments: [right]
      atoms: heavy
      frame: {method: reference_interface_pca}

  symmetry:
    transform_sets:
      ring_c3:
        type: cyclic
        order: 3
        axis: [0.0, 0.0, 1.0]
        center: [0.0, 0.0, 0.0]
    orbits:
      primary_orbit:
        transform_set: ring_c3
        master_groups: [primary_seed]

  interfaces:
    ring_interface:
      left_port: left_port
      right_port: right_port
      copy_relation: {orbit_offset: -1}
      required: true
      target_geometry:
        mode: reference_transform
        from_reference_seed: true
        translation_tolerance: 2.0
        rotation_tolerance_deg: 10.0

  scaffold_links:
    protomer:
      from: {fragment: right, terminus: C}
      to: {fragment: left, terminus: N}
      length: {min: 70, max: 100}
      tie_group: protomer_length
      chain_break: false

  initialization:
    center_method: interface_heavy_atom_com
    orientation:
      method: assembly_dofs
      tilt_deg: [-20.0, 20.0]
      twist_deg: [-180.0, 180.0]
    placement:
      radius: {mean: 25.0, range: 5.0}
      axial_offset: {mean: 0.0, range: 2.0}

  output:
    directory: outputs/lhd101_c3
    save_mapping: true
    save_validation: true
```

这里 `left` 和 `right` 同属一个 rigid group，因为 LHD101 MVP 保留输入 seed 的内部相对几何，并让整个 seed 作为 master pose 移动。若研究目标是改变两者的相对 pose，应拆成两个 groups，并用 edge 约束它们。

第一版输出：

```text
resolved_config.yaml
presymmetrized_input.cif
agnostic_specification.json
mapping.json
initial_pose_graph.json
validation_report.json
```

通过 RFD3 adapter 后再额外输出：

```text
rfd3_input.json
rfd3_adapter_manifest.json
RFD3 native outputs
```

---

## 8. 初始化几何

### 8.1 orientation parameterization 必须互斥

模式 A：无偏全局旋转搜索

```yaml
orientation:
  method: uniform_so3
```

模式 B：可解释 assembly DOFs

```yaml
orientation:
  method: assembly_dofs
  azimuth_deg: [-180.0, 180.0]
  tilt_deg: [-20.0, 20.0]
  twist_deg: [-180.0, 180.0]
```

不得先采样 uniform SO(3)，再对同一 pose 额外采样 azimuth/tilt/twist，除非显式声明为两阶段 perturbation 并记录组合顺序。

MVP 推荐 `assembly_dofs`，因为 radius、axial offset、azimuth、tilt 和 twist 可以分别做成功率分析。

### 8.2 标准 SE(3)

对 master group `g`：

```text
X_g = R_g X_g_ref + t_g
T_g = [R_g, t_g; 0, 1]
```

必须满足：

- `R^T R = I`；
- `det(R) = +1`；
- rigid group 内所有 pairwise distances 不变；
- translation 与 rotation 独立记录；
- 不允许 atom-slot-wise dragging 破坏刚性。

### 8.3 master pose 与 symmetry copies

对于 group element `S_k`：

```text
T_(g,k) = S_k * T_g_master
X_(g,k) = T_(g,k) * X_g_ref
```

所有 copies 必须每一步从 master pose 重建，不能各自累计更新。这样可以避免 copy drift，并降低状态维数。

---

## 9. RFD3-independent validation gates

只有全部通过才生成 RFD3 input。

### Gate A：schema

- 所有 ID 唯一；
- selection 非空；
- 每个 fragment 的 owner 合法；
- orientation modes 不冲突；
- edge ports 存在；
- tie groups 类型一致。

### Gate B：geometry

- port frames 稳定且为右手系；
- reference transforms 可计算；
- symmetry matrices 合法；
- rigid-distance invariant 通过；
- 无 NaN/Inf；
- 初始严重 clash 可识别。

### Gate C：graph

- pose-graph nodes 统一为 motion-group instances；
- edge orbit 展开无遗漏、无重复；
- transform IDs 存在；
- required interfaces 可实例化；
- 不依赖 edge 遍历顺序。

### Gate D：protein topology

- termini 存在且方向正确；
- linker length 基本可行；
- symmetry-tied lengths 一致；
- chain breaks 显式；
- 不存在重复占用或非法环。

### Gate E：provenance

- 每个输出 atom 有来源；
- 每个 generated region 有 scaffold-link 来源；
- 每个 symmetry copy 有 transform ID；
- RFD3-independent indices 与输出 CIF 一致。

---

## 10. RFD3 adapter 边界

### 10.1 adapter 唯一职责

将 `CompiledInterfaceSeed` 转换成服务器当前 RFD3 版本接受的输入，并保存双向映射。

它必须统一维护：

- coordinates/reference coordinates；
- fixed-atom masks；
- sequence conditioning；
- residue/index conditioning；
- chain/entity identifiers；
- ligand/CCD metadata；
- symmetry/transform identifiers；
- generated-region definition；
- input/output provenance。

不能只改 coordinate tensor 而忽略 masks、indices 或 conditioning state。

### 10.2 adapter manifest

每次运行保存：

```yaml
rfd3:
  source_path: ...
  git_commit: ...
  environment: ...
  entrypoint: ...
  checkpoint_path: ...
  checkpoint_sha256: ...
  adapter_version: ...
  verified_capabilities: [...]
```

环境、代码目录、运行入口和 checkpoint 必须成套记录，禁止混搭。

### 10.3 静态优先

第一阶段 adapter 只负责：

```text
presymmetrized CIF
-> RFD3 native input specification
-> prevalidation
-> one completed design
```

静态 MVP 没有成功前，不进入 sampler hook 修改。

---

## 11. Dynamic pose-graph controller

### 11.1 分阶段，不直接跳到多界面

第一版 dynamic controller 只处理：

- 一个 master rigid group；
- 一个 interface-edge orbit；
- translation + rotation update；
- 无 ligand、无 flexible backbone。

验证 controller 不会破坏 RFD3 状态后，再扩展到多 groups、多 edges。

### 11.2 同时聚合 residuals

每个 timestep：

1. 从 RFD3 state 同步 master/group coordinates；
2. 对所有 active interface edges 同时计算 residual；
3. 将 residual 转成各 group 的 SE(3) twist proposal；
4. 按 edge weight、group mobility 和 schedule 联合聚合；
5. 使用 robust loss/conflict policy 处理不一致约束；
6. 分别 clamp translation 和 rotation；
7. 每个 master pose 只更新一次；
8. 从 master pose 重新生成全部 symmetry copies；
9. 同步坐标、masks、indices 和 conditioning；
10. 验证并记录 trajectory。

不能使用：

```text
edge 1 更新坐标 -> edge 2 再更新 -> edge 3 再更新
```

因为这会导致结果依赖 edge 顺序。

### 11.3 更新形式

```text
delta_xi_g = [delta_translation_g, delta_rotation_g]
T_g(t-1) = Exp(alpha_g(t) * delta_xi_g) * T_g(t)
```

第一版可使用加权 proposal aggregation；接口预留：

- weighted least squares；
- robust M-estimator；
- constrained pose-graph optimizer；
- differentiable guidance。

### 11.4 目标项

| term | 使用对象 | 说明 |
|---|---|---|
| relative transform | port pair | translation + SO(3) geodesic error |
| distance | anchors/planes/axes | 不局限于 COM |
| contacts | inter-group atoms | 排除组内自接触 |
| clashes | seed/scaffold/copy/ligand | 分类别记录 |
| connectivity | scaffold links | termini distance、方向、闭合可行性 |
| symmetry | expanded instances | copies 必须来自 master pose |
| functional geometry | ligand/metal atoms | 后续阶段加入 |

### 11.5 fidelity 必须随 motion mode 定义

| motion mode | fidelity 定义 |
|---|---|
| `fixed` | 与 reference 坐标一致 |
| `rigid` | internal geometry 由变换构造天然保持；监测错误漂移即可 |
| `soft_rigid` | master pose 偏离初始 pose 的 translation/rotation |
| `flexible_backbone` | backbone/functional-atom RMSD 与内部几何 |
| ligand/metal site | coordination distance、angle、chirality、pose |

对严格 rigid group 反复优化最佳叠合后的内部 RMSD 没有实际意义；若非零，通常代表实现错误。

### 11.6 guidance schedule

| 阶段 | 主要目标 | pose step |
|---|---|---|
| Early | placement、relative transform、拓扑 | 较大 |
| Middle | contacts、packing、clash、connectivity | 中等 |
| Late | functional fidelity、局部精修、冻结 pose | 很小或零 |

权重必须支持：

- per-edge；
- per-group；
- timestep schedule；
- ablation 开关；
- resolved-config 和 trajectory 记录。

### 11.7 sampler hook 的选择流程

必须实测候选位置：

```text
before denoising step
after model prediction
after denoising step
```

选择标准：

- 修改不会被随后 realignment 覆盖；
- 修改后 state 内坐标与 conditioning 一致；
- 不破坏 sampler 的噪声/时间语义；
- 可通过单步 round-trip test；
- 失败时可以完全关闭 controller，回到 native RFD3。

---

## 12. 多界面与全原子扩展

### 12.1 multi-interface 配置原则

多个 interfaces 可以：

- 连接同一对 groups 的不同 ports；
- 连接不同 groups；
- 属于不同 edge orbits；
- 使用不同 target-geometry modes；
- 使用不同 schedules 和优先级。

示意：

```yaml
interfaces:
  ring_edge:
    left_port: ring_out
    right_port: ring_in
    copy_relation: {orbit_offset: -1}
    target_geometry: {mode: reference_transform, from_reference_seed: true}

  axial_edge:
    left_port: top_face
    right_port: bottom_face
    copy_relation: {transform: "D2:rx"}
    target_geometry:
      mode: geometric_constraints
      distance: {type: plane_to_plane, target: 9.0, tolerance: 2.0}
```

### 12.2 conflict handling

多 edge 约束可能不可同时满足。controller 必须输出：

- 每条 edge residual；
- aggregate objective；
- proposal cosine/conflict matrix；
- 哪些 updates 被 clamp；
- required/optional edge 的满足状态；
- infeasible 标记，而不是无限增大 guidance。

### 12.3 ligand/metal constraints

在 static single-interface 和 dynamic single-interface 都稳定后加入：

```yaml
functional_constraints:
  zinc_site:
    type: coordination
    center: "ligand:ZN"
    coordinating_atoms:
      - "his1:NE2"
      - "his2:NE2"
      - "asp1:OD1"
    distances: [2.1, 2.1, 2.0]
    angle_tolerance_deg: 15.0
```

需要分别验证 RFD3 是否能保持：

- ligand identity；
- covalent/non-covalent relation；
- donor/acceptor conditioning；
- atom-level fixed masks；
- coordination chirality。

### 12.4 asymmetric functional components

schema 可以表达：

```text
symmetric protein framework + asymmetric ligand/component
```

但第一版 RFD3 symmetry MVP 只实现所有 protein motif groups 使用同一 C3 orbit。其余组合必须先做 adapter capability test，再决定采用：

- native mixed-symmetry input；
- explicit pre-expansion；
- non-symmetric RFD3 generation + external constraints；
- post-assembly docking/design。

---

## 13. Negative design

按风险由低到高实施：

1. generation 后过滤；
2. 多状态 reranking；
3. sampler guidance；
4. 有充分证据后再进入训练。

```yaml
negative_states:
  wrong_homotypic_pairing:
    penalize_port_pairs: [[ring_out, ring_out]]

  undesired_c2:
    transform_set: C2

  apo_state:
    remove_fragments: [ligand]
    max_allowed_contacts: 4
```

必须区分：

- target state 未形成；
- negative state 也稳定；
- target 和 negative 都不稳定。

不能只报告 target score。

---

## 14. 软件包结构

```text
rfd3_multi_interface_seed/
├── pyproject.toml
├── README.md
├── configs/
│   ├── legacy/
│   ├── single_interface/
│   ├── multi_interface/
│   └── functional/
├── src/rfd3_multi_interface_seed/
│   ├── schema/
│   │   ├── specs.py
│   │   ├── instances.py
│   │   ├── states.py
│   │   └── validation.py
│   ├── parsing/
│   │   ├── structure_loader.py
│   │   ├── selection.py
│   │   └── legacy_ab_adapter.py
│   ├── geometry/
│   │   ├── frames.py
│   │   ├── se3.py
│   │   ├── sampling.py
│   │   ├── transforms.py
│   │   └── symmetry_registry.py
│   ├── topology/
│   │   ├── pose_graph.py
│   │   ├── interface_expansion.py
│   │   ├── scaffold_graph.py
│   │   └── compiler.py
│   ├── provenance/
│   │   └── mapping_registry.py
│   ├── adapters/
│   │   ├── rfd3_static.py
│   │   ├── rfd3_sampler.py
│   │   └── manifest.py
│   ├── guidance/
│   │   ├── controller.py
│   │   ├── aggregation.py
│   │   ├── schedules.py
│   │   └── terms/
│   ├── validation/
│   │   ├── geometry.py
│   │   ├── topology.py
│   │   ├── provenance.py
│   │   └── rfd3_prevalidation.py
│   ├── diagnostics/
│   │   ├── metrics.py
│   │   ├── trajectory.py
│   │   └── debug_snapshot.py
│   ├── compile.py
│   └── cli.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── fixtures/
└── examples/
    ├── lhd101_c3/
    ├── three_fragment/
    ├── two_interface_d2/
    └── ligand_mediated/
```

依赖规则：

```text
schema <- parsing/geometry/topology/provenance
                  <- compiler
                  <- RFD3 adapters
                  <- dynamic guidance
```

geometry、topology、provenance 和 compiler 不得 import RFD3。

---

## 15. 测试计划

### 15.1 Schema tests

- `N=1,2,3,...` 不触发硬编码链假设；
- 多 fragments、多 groups、多 ports、多 edges；
- fragment owner 冲突；
- port 引用错误；
- geometry mode 冲突；
- dangling objects；
- tied-link 类型冲突；
- fixed/mobility 语义冲突。

### 15.2 Frame 与 SE(3) tests

- frame 正交、右手、确定性；
- anchor 共线时明确失败；
- transform/inverse round trip；
- composition 顺序正确；
- rigid pairwise distances 不变；
- assembly DOFs 的每个变量可独立恢复；
- uniform SO(3) 与 assembly DOFs 互斥。

### 15.3 Symmetry tests

- C3/C4/C5 closure；
- D2/D3 transform table；
- master-copy reconstruction；
- group element composition；
- orbit offset 到 transform ID 的确定性编译；
- copy drift 始终为零（数值容差内）。

### 15.4 Graph/topology tests

- pose-graph node 仅为 motion-group instance；
- port edge 展开无遗漏、无重复；
- 多个非等价 interfaces；
- edge 顺序打乱不改变聚合结果；
- N/C termini 与 linker feasibility；
- tied linker lengths；
- illegal cycles/cross-links；
- infeasible constraints 可诊断。

### 15.5 Provenance tests

- source -> compiled -> RFD3 indices；
- RFD3 output -> source/spec/instance；
- symmetry-expanded atoms 全覆盖；
- generated linker residues 可追踪；
- 无未映射 atom；
- 映射序列化 round trip。

### 15.6 Guidance tests

- 单 edge translation error 下降；
- 单 edge rotation error 下降；
- clashes 不把 groups 拉近；
- step clamp 与 freeze 生效；
- required/optional edge 权重正确；
- 多 edge 聚合与遍历顺序无关；
- 不相容约束被标记；
- 禁用 controller 时与 native sampler 一致。

### 15.7 Integration/regression tests

1. RFD1 repaired LHD101 reference；
2. LHD101 C3 static corrected SE(3)；
3. RFD3 prevalidation；
4. RFD3 单设计 smoke test；
5. legacy adapter 对照；
6. single-interface dynamic controller；
7. 人工 three-fragment compile test；
8. C4/C5 static expansion；
9. D2 two-interface compile test；
10. ligand-mediated static capability test。

---

## 16. 分阶段实施与退出标准

### Phase 0 — 可复现基线

完成：

- RFD1 exact audit；
- RFD1 repaired runnable baseline；
- reference inputs/outputs；
- RFD3 source commit；
- environment export；
- checkpoint SHA256；
- entrypoint 和完整命令；
- GPU/runtime manifest。

退出标准：另一台同配置服务器能够按 manifest 运行同一 smoke test。

### Phase 1 — Schema、provenance 与 compiler skeleton

实现：

```text
FragmentSpec
MotionGroupSpec
InterfacePortSpec + InterfacePortFrame
InterfaceEdgeSpec
SymmetryOrbitSpec
ScaffoldLinkSpec
Spec/Instance/State separation
MappingRegistry
```

退出标准：LHD101 YAML 能解析、resolve、序列化；所有 provenance unit tests 通过。

### Phase 2 — Pure geometry/topology engine

实现：

- local frames；
- SO(3)/SE(3)；
- assembly DOF sampling；
- transform registry；
- master pose；
- C3 expansion；
- relative-transform objectives；
- directed scaffold topology；
- standalone validation。

退出标准：不安装 RFD3 即可生成并验证 LHD101 `presymmetrized_input.cif`、mapping 和 pose graph。

### Phase 3 — Static RFD3 MVP

只完成：

```text
LHD101 A/B
-> corrected SE(3)
-> C3 expansion
-> scaffold compilation
-> presymmetrized CIF
-> RFD3 input
-> RFD3 prevalidation
-> one completed design
```

退出标准：至少一个 RFD3 job 正常结束，输出可通过 mapping 回溯，且无未解释的 chain/mask/index 错误。

### Phase 4 — Legacy-compatible static adapter

加入：

- legacy Euler convention；
- legacy translation convention；
- auto previous/next；
- legacy linker sampling；
- legacy contact metrics。

目的：分离以下两类贡献：

```text
RFD3 model/platform improvement
vs
geometry/controller improvement
```

退出标准：同一 reference set 能运行 native/legacy/corrected 三种静态配置并得到可比较结果。

### Phase 5 — Single-interface dynamic controller

加入：

- 一个 master rigid group；
- 一个 edge orbit；
- SE(3) update；
- schedule；
- trajectory；
- sampler hook round-trip test。

退出标准：controller 的开/关对照可复现；几何误差下降；无 mask/index/copy drift。

### Phase 6 — Multi-interface pose graph

加入：

- 多个 master groups；
- 多个 ports/edge families；
- simultaneous aggregation；
- conflict detection；
- per-edge/per-group schedules；
- C4/C5 与 D2/D3 测试。

退出标准：至少一个非 LHD101 的多界面任务完成，且 edge-order invariance test 通过。

### Phase 7 — Soft-rigid 与全原子功能约束

依次加入：

1. soft-rigid bounds；
2. ligand pose；
3. metal coordination；
4. donor/acceptor；
5. functional atom fidelity；
6. sidechain redesign capability test。

退出标准：每种功能都具有独立最小测试和 ablation，不把多种新变量绑在一个实验里。

### Phase 8 — Higher-order architectures 与 negative design

顺序：

```text
C3
-> C4/C5
-> D2/D3
-> bifacial building block
-> explicit polyhedral expansion
-> cage
```

negative design 按 filter -> rerank -> guidance 顺序推进。

退出标准：每类拓扑均有 capability matrix，明确 native、external expansion、unsupported 三种状态。

---

## 17. 实验与消融设计

### 17.1 基础对照

| 实验组 | 回答的问题 |
|---|---|
| RFD1 original/repaired | 原方法基线是什么 |
| Native RFD3 fixed motif | 不加本方法时 RFD3 能做到什么 |
| RFD3 legacy adapter | 仅换生成模型的收益 |
| RFD3 corrected static SE(3) | 几何修正的收益 |
| + single-interface controller | 动态 pose control 的收益 |
| + schedule | 时间调度的收益 |
| + multi-interface pose graph | 同时满足多个界面的收益 |
| + soft-rigid | 有限柔性的收益/代价 |
| + ligand/all-atom terms | 全原子条件的收益 |
| + negative-state ranking | 选择性的收益 |

### 17.2 必须报告的指标

- motif backbone RMSD；
- selected functional-atom RMSD；
- interface translation/rotation error；
- heavy-atom contact count；
- buried SASA；
- clash count，按类别拆分；
- symmetry deviation；
- termini/linker feasibility；
- required edge satisfaction rate；
- all-edge simultaneous satisfaction rate；
- backbone acceptance/designability；
- sequence-design success；
- RF3/AF3 等独立结构预测指标；
- GPU hours、失败率和 accepted design cost。

### 17.3 不能混淆的结论

- 编译成功不等于设计成功；
- RFD3 job 完成不等于 interface 满足；
- 单界面成功不等于多界面同步成功；
- 对称 RMSD 低不等于 scaffold connectivity 合理；
- refolding confidence 高不等于 negative state 被排除；
- schema 能表达不等于当前 adapter/native sampler 能运行。

---

## 18. 诊断与输出规范

每个 design 保存：

```text
resolved_config.yaml
version_manifest.yaml
mapping.json
initial_pose_graph.json
presymmetrized_input.cif
rfd3_input.json
rfd3_adapter_manifest.json
pose_trajectory.jsonl
guidance_trajectory.jsonl
interface_metrics.json
validation_report.json
final_design.cif
debug_snapshot/           # 仅失败时
RFD3_native_outputs/
```

每个 dynamic timestep 至少记录：

- master group translation/quaternion；
- expanded-copy symmetry deviation；
- 每条 edge 的 translation/rotation/contact residual；
- 每条 scaffold link 的 feasibility；
- proposed twist 与 applied twist；
- weights、clamp、freeze；
- clash categories；
- non-finite check；
- RFD3 state synchronization status。

禁止用 `nan_to_num` 静默吞掉异常。发现 NaN/Inf 时：

1. 停止当前 sample；
2. 写 debug snapshot；
3. 保存最近一个有限状态；
4. 报告对象 ID、edge ID、timestep 和触发项。

---

## 19. Fine-tuning 决策门槛

初期不训练。只有同时满足以下条件才评估 fine-tuning：

1. static compiler 与 adapter 已稳定；
2. sampler controller 已有系统 ablation；
3. 失败不是输入、mapping、hook 或几何错误；
4. RFD3 在足够样本上系统性忽略某类条件；
5. 强 guidance 明显损害 backbone/all-atom quality；
6. 已定义训练数据、任务采样和独立测试集。

如果进入训练，应新增 multi-interface conditioning task/data sampler，而不是用少量成功样本直接微调。

---

## 20. 第一批交付物

### Deliverable A — Reproducibility manifest

- RFD1 repaired baseline；
- RFD3 environment/source/checkpoint/entrypoint manifest；
- LHD101 reference set。

### Deliverable B — Standalone compiler

- 最终 schema；
- `InterfacePortFrame`；
- Spec/Instance/State；
- MappingRegistry；
- SE(3) geometry；
- C3 transform registry；
- directed scaffold topology；
- validation report。

### Deliverable C — Static RFD3 MVP

- LHD101 C3 pre-symmetrized input；
- native RFD3 input；
- mapping/manifest；
- one completed design；
- native vs legacy vs corrected 对照。

### Deliverable D — Dynamic prototype

- single-interface controller；
- sampler hook validation；
- trajectory diagnostics；
- multi-interface pose-graph extension。

---

## 21. 立即执行顺序

1. 固定服务器 RFD3 的环境、代码目录、commit、checkpoint、入口和完整命令；
2. 运行最小 native RFD3 smoke test，确认当前安装本身可用；
3. 完成 RFD1 repaired LHD101 baseline，冻结输入/输出/hash；
4. 建立独立 `rfd3_multi_interface_seed` 仓库，不直接修改共享软件；
5. 实现 Specs 与 schema validation；
6. 实现 `InterfacePortFrame` 和 SE(3) tests；
7. 实现 transform registry、symmetry expansion 和 MappingRegistry；
8. 实现 directed scaffold compiler；
9. 生成 LHD101 standalone compiled outputs；
10. 写当前 RFD3 版本的 static adapter；
11. 通过 prevalidation 并完成一个 design；
12. 完成静态对照后，再定位 sampler hook；
13. 先做 single-interface controller，再做 multi-interface controller；
14. 最后加入 soft-rigid、ligand/metal 和 negative design。

第一个代码 milestone 不是 sampler，而是：

```python
compiled = compile_interface_seed(
    config="configs/single_interface/lhd101_c3.yaml"
)
```

---

## 22. 汇报口径

### 22.1 原方法

> Interface-Seed 1.0 modified RFdiffusion1 inference to expand an asymmetric two-fragment interface seed under cyclic symmetry, reconstruct the corresponding scaffold topology, sample the seed pose, and heuristically reposition the motif during denoising without changing the network architecture or retraining the checkpoint.

该句最终仍应以 RFD1 审计文件中的逐项证据为准。

### 22.2 本次升级目标

> Interface-Seed 2.0 generalizes the original two-fragment cyclic heuristic into a generator-independent, graph-based framework for an arbitrary number of motif fragments and non-equivalent interfaces. It introduces explicit interface ports and local frames, SE(3)-correct master-pose control, symmetry transform registries, directed scaffold connectivity, complete provenance, and an extensible RFD3 adapter, followed by time-dependent pose-graph guidance and all-atom functional constraints.

### 22.3 分阶段汇报，避免过度声明

完成 Phase 2 后只能说：

> We implemented a generator-independent compiler for symmetry-aware interface seeds.

完成 Phase 3 后可以说：

> We demonstrated static RFD3 scaffolding of a pre-symmetrized LHD101 interface seed.

完成 Phase 5 后可以说：

> We integrated a single-interface SE(3) pose controller into the verified RFD3 sampling path.

完成 Phase 6 并通过非 LHD101 实验后，才可以说：

> We generalized interface-seeded design to multiple non-equivalent interfaces using a symmetry-aware pose graph.

完成 ligand/metal 实验后，才可以声明 all-atom functional extension。

---

## 23. 最终架构决策摘要

| 问题 | 最终决策 |
|---|---|
| 核心是否以 A/B 为模型 | 否；A/B 仅为 regression benchmark |
| pose-graph node | 统一为 `MotionGroupInstance` |
| interface 如何挂载 | 通过 `InterfacePortInstance` |
| port 是否只有 atom selection | 否；必须包含稳定 `InterfacePortFrame` |
| fragment 是否定义 flexibility | 否；运动学只由 `MotionGroupSpec` 管理 |
| 界面是否只存距离 | 否；支持完整 relative transform 或 constraint set |
| uniform SO(3) 与 assembly DOFs | 互斥模式 |
| previous/next 是否进入核心算法 | 否；编译成明确 transform ID |
| scaffold link 是否有方向 | 是；明确 `C -> N` |
| graph-to-spec 是否要求严格可逆 | 否；要求完整 provenance traceability |
| symmetry copy 是否独立移动 | 否；始终由 master pose 生成 |
| 多界面是否逐条更新 | 否；residual 同时计算并联合聚合 |
| 第一段代码是否修改 sampler | 否；先做 standalone compiler |
| 第一版是否承诺 mixed symmetry/cage | 否；先 capability test |
| 何时 fine-tune | 排除工程与 inference 控制问题之后 |

---

## 24. 成功标准

这个项目的成功不只是“RFD3 能跑”。最终应依次满足：

```text
可复现
-> 可编译
-> 可追踪
-> 几何正确
-> 拓扑可生成
-> 静态 RFD3 可运行
-> 动态控制不破坏 sampler
-> 单界面有效
-> 多界面可同时满足
-> 全原子功能约束有效
-> 在独立任务上优于清晰定义的 baselines
```

LHD101 是第一道回归测试，不是终点。真正的方法贡献是在保持旧功能可复现的基础上，把双片段启发式脚本升级为数量无硬编码、接口显式、对称严格、来源可追踪、可独立测试并能接入 RFD3 的通用多界面设计框架。

---

## 26. C5/C6/C7 native cyclic capability extension

C3 端到端通过后，下一组 cyclic capability 实验固定方法学，只改变
symmetry order。C5、C6、C7 均使用同一个 Interface-Seed compiler、
Haar SO(3)、joint LHS、QD、native RFD3 symmetry sampler 和结果审计。

不能直接复制 C3 的半径区间。为保持相邻 copy 的弦长尺度，采用：

```text
R_n = R_3 * sin(pi / 3) / sin(pi / n)
```

对应配置中心半径约为 C5 `36.84 A`、C6 `43.30 A`、C7 `49.90 A`，
并同比缩放半径采样范围。该缩放只保持初始相邻 copy 的几何尺度，不保证
折叠成功。

为避免把 C3 的绝对 cavity 尺度错误套到更高阶环，absolute
axis-clearance objective window 与 scale 使用同一个阶数缩放因子；
QD 分箱则使用无量纲的
`minimum_axis_clearance / sampled_radius`。因此更高阶环不会仅因半径更大
而被系统性惩罚或全部落入同一个 clearance cell。

完整运行说明见
`docs/rfd3_mosaic/C5_C6_C7_200STEP_RUNBOOK.md`。C5/C6/C7 只有在
200-step 结果同时通过 interface-seed、连续性、hard clash、compactness
和 declared-transform symmetry gate 后，才能声明 native cyclic
capability 已扩展；目前这些仍是待验证实验。

### 26.1 高阶 Cn 的表达能力不等于 native RFD3 运行能力

当前 schema、symmetry registry 与 instance compiler 的阶数是参数化的，
因此可以表达 C12/C20。native symmetric-motif inference 则有两道独立边界：
Mosaic adapter 对 multiplicity `> 10` fail closed，官方 Foundry RFD3 在
motif-frame recovery 中同样定义 `MAX_TRANSFORMS = 10`。所以当前 native
输入的理论上限是 C10 / D5，而仓库实际追踪的 GPU cyclic 路径仍是
C3、C5、C6、C7。

不能把两处常数删除后称为高阶支持。显式 all-copy token-pair state 随
assembly size 二次增长；checkpoint relative-chain encoding、超过单字符的
chain-ID 路径、高阶 seed pairing 和 clash audit 也都尚未验证。C12/C20
因此属于后续 ASU-only 或 local-neighborhood 高阶架构任务，不进入当前
P100 50/200-step 矩阵。

---

## 25. 2026-07-22 通用化优先级修订

### 25.1 软件能力与 benchmark 必须分离

项目不以某一个 C3 ring 或 D3 barrel 为产品模型。C3、D2、D3 和
LHD101 都只是 regression/capability fixtures。核心 API 不得包含具体
PDB、链名、残基编号或固定 symmetry order。

通用能力的验收问题是：用户更换 fragments、ports、interfaces、Cn/Dn
order 和 scaffold relations 后，是否仍可经过同一条编译、诊断和 adapter
路径，而不是某一个案例是否视觉上漂亮。

### 25.2 新的实现顺序

在 dynamic sampler controller 之前，顺序调整为：

```text
generic objective/scoring API
-> static pose candidate generator
-> symmetry feasibility screening
-> multi-interface conflict diagnostics
-> RFD3 timestep controller
```

必须先建立 objective API，再实现 pose search，避免把 clash、linker、
interface、cavity 等打分逻辑写死在搜索循环中。Static search 和未来
dynamic controller 应复用相同 objective 定义与逐项报告格式。

### 25.3 Objective API 的边界

第一版 objective 是 backend-independent 的标量评价层，至少支持：

```yaml
objectives:
  no_hard_clashes:
    metric: clashes.total_hard_clashes
    mode: at_most
    threshold: 0
    required: true
    scale: 1

  prefer_larger_cavity:
    metric: cavities.minimum_central_void_radius
    mode: maximize
    required: false
    weight: 0.5
    scale: 10
```

每个 term 必须独立输出 raw value、penalty、weight、required/satisfied
状态；总排名首先按 required constraint failure 数量，再按加权 penalty。
不得只输出一个无法解释的混合总分。

Pose search 必须能够保留并评分 infeasible candidates。因此 standalone
compiler 应同时提供 strict validation 和 diagnostic/relaxed 模式；后者只
用于搜索和故障分析，不能直接送入 RFD3 adapter。

### 25.4 对自动化能力的准确命名

- `symmetry feasibility screening`：比较候选 Cn/Dn 的几何、碰撞、拓扑、
  linker 和 backend capacity；不得声称从单一 seed 预测生物学正确 symmetry。
- `topology compilation`：展开并验证用户声明的连接；不得擅自推断应该共价
  连接哪些蛋白。
- `constraint conflict diagnostics`：报告局部残差、相反更新方向和 tolerance
  冲突；除非有严格证明，不得声称全局无解。

### 25.5 当前项目定位

在 dynamic controller 完成前，准确表述为：

> A backend-independent, symmetry-aware interface-seed geometry and topology compiler with native Cn/Dn support and an RFD3 adapter.

只有在 sampler hook、time-dependent update、multi-interface aggregation 和
independent benchmarks 全部验证后，才升级表述为 control framework。
