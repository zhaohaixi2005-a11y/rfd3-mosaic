# Mosaic 的操作、触发条件、公式与筛选依据

本文对应 `refactor/product-core-v1` 的本地实现，说明 Mosaic 在 RFD3 外增加的决策。
公式描述代码行为，不代表这些权重和阈值已经经过实验或基准验证。
**必须同时看本次实例化参数：显式设置、预设和编译器会覆盖默认值。**
本文没有把 RFD3 学习到的去噪预测解释成一组人工规则，也没有声称覆盖仓库全部实验分支。

**本文中的 energy/“能量”指 Mosaic 人为定义的几何引导损失。** 它由坐标、拓扑和
配置计算，不是结合自由能、分子力场能量、RFD3 置信度或实验成功概率。分数降低只表示
当前几何偏好得到改善；line search 保证按指定规则选步，不能证明评分规则本身正确。

阅读时可按问题直接定位：

- “为什么拉近、总分怎么算？”：第 5 节及第 6 节的总公式、逐项公式和数值例子。
- “为什么这一步被拒绝？”：第 6 节的接受规则及日志复算方法。
- “为什么 seed 移动但内部不变？”：第 4、7 节。
- “怎样选起点、保留不同 pose？”：第 3 节及 [任务与 pose](TASK_POSES.zh-CN.md)。
- “生成了文件为什么仍判失败？”：第 9 节。

每个决策应能追溯到：**作用对象 → 实际参数 → 公式 → 触发条件 → 判定 → 证据**。
表中的默认值不是本次运行值；未记录的决策不能靠默认值补成“已证明通过”。

### 覆盖范围与阅读索引

对照基线为本仓库保留的上游 `production@551a901`，覆盖 `e20a04e` 及本次 benchmark 整改；
这不是与上游最新版本的比较，也不能把分支中的全部新增实现都归因于最近一次整改。
本文覆盖上述差异中的重要几何公式、搜索规则、采样控制和结果判断；工程模块说明其
输入校验、版本和状态判定，不将其包装成科学算法。逐文件索引另列所有对应源码入口。

| 模块 | 公式与判断位置 | 对应实现 |
|---|---|---|
| 界面 packing | 第 5、6、16 节 | `graph_interface_guidance.py` |
| 精确对称、固定约束及柱坐标 | 第 4、13 节 | `symmetry_utils.py`、`constraint_runtime.py`、`cylindrical_projector.py` |
| 多组件、稳定子及商轨道 | 第 11 节 | `geometry/symmetry_registry.py`、`topology/stabilizer_cosets.py`、`seed_stabilizer.py` |
| 聚合物连接、图搜索与长度绑定 | 第 3、12 节 | `topology/`、`graph_search.py`、`feasibility_restoration.py` |
| 刚体运动及联合接受 | 第 7、15 节 | `motif_mobility.py`、`scaffold_guidance.py` |
| 初始化、核心、路径与连续性 | 第 8、17 节 | `input_parsing.py`、`scaffold_core_guidance.py` |
| 用户目标和功能/形态几何 | 第 18 节 | `objectives/`、`functional_geometry.py`、`validation/assembly_morphology.py` |
| pose 采样、搜索与多样性 | 第 3、14 节 | `compile.py`、`pose_*.py` |
| 局部邻域、attention 和精度 | 第 20 节 | `local_neighbourhood.py`、`model/layers/block_utils.py`、`alignment.py` |
| 最终审计、统计和证据 | 第 9、19、21 节 | `rfd3_*_audit.py`、`validation/`、运行及报告模块 |

部分算法属于独立工具或研究路径，是否参与某次任务应由编译配置和运行记录确认。
本文的覆盖不表示所有模式均已通过 GPU 验证，也不表示每次生成都会同时运行所有算法。

## 1. 先区分四种“标准”

| 层次 | 示例 | 不满足时的含义 |
|---|---|---|
| 输入/几何可行性 | linker 最大长度不足；固定块相撞；显式必需界面失败 | 对当前声明的构型不应继续编译或选择 |
| 运行约束 | 对称复制、固定块内部几何、允许运动范围、链连续性 | 合同/执行存在问题，需要检查；不是实验失败标签 |
| 控制器代理目标 | 接触势、覆盖率、紧凑度、方向、形状损失 | 用于提出和比较坐标更新，默认阈值主要是工程设定 |
| 结果建议/科学验证 | 骨架审计建议；后续序列设计、回折叠、实验 | 骨架建议只支持下一阶段选择，不能代替后续验证 |

“硬约束”也有两种执行时机：投影每步强制满足的约束，和生成后核查的合同。
生成后检查失败不会倒过来证明采样时每一步都满足了它。
`generated`、`contract_met`、`recommendation` 是三个不同字段。

## 2. 参数从哪里来，如何查本次真实值

1. 用户设计声明确定组件、端口、连接、必需界面、运动范围和形状范围。
2. `design_preferences.py` 把 loose/balanced/tight、界面面积和运动偏好解析为数值。
3. 编译器按结构、对称群和连接拓扑生成物理界面、遮罩及可行性要求。
4. worker 将编译结果及解析的覆盖项传给 sampler；显式专家设置可覆盖相应预设。
5. sampler 实例化配置，再根据进度和当前几何得到每步有效权重、目标和步长。

查询入口：

- `plan --format json` 的 `resolved_preferences`、`constraint_plan`、`sampling_plan`、
  `assembly_lowering` 和 `decision_policy`：采样前声明及解析结果。
- 每个设计的 `audits/<design_id>/decision_explanation.md`：实际参数和结论摘要。
- 同目录 `decision_explanation.json`：原始结果/编译输入的相对路径、SHA-256、
  运行摘要、全部审计原文（包括嵌套阈值）和筛选原因。
- 原始结果 JSON 的 `*_diagnostics.steps`：逐步记录。界面步骤的 `effective_config`
  是当步有效配置；`line_search_trials` 是候选及判定；刚体联合更新另有
  `packing_decision`、`failed_conditions` 和联合能量前后值。
- 旧输出不会自动拥有新日志。缺字段表示未记录，不能解释为机制关闭或检查通过。

下文 Å 表示埃，角度字段注明度；各代理损失使用不同归一化，并非统一物理能量单位，
总分不能解释成 kcal/mol 或成功概率。

### 符号与比较范围

| 符号 | 定义 |
|---|---|
| `X`、`x_i` | 被评价的坐标状态及第 i 个原子的坐标；界面损失主要取生成残基的 Cα |
| `d_ij` | `||x_i-x_j||₂`；将距离的 Å 数值代入公式 |
| `p` | 当前扩散进度，取值 0–1；不是接触概率 |
| `D_e` | 某条物理界面边在本步的有效距离目标，受边配置和阶段调度影响 |
| `P_e` | 当前界面片段内选中的残基接触对；不是所有原子对 |
| `[z]+`、`sigmoid(z)` | `max(z,0)`、`1/(1+exp(-z))` |
| `L_*`、`w_*` | 未乘权重的损失项、对应有效权重；权重为零时该项不贡献总分 |
| `E_s` | 一个用户声明界面的物理副本边分数的平均值 |

一次 line search 的前后比较使用同一有效配置、同一目标和同一组片段身份。
不同扩散步可能更换片段、阶段、权重和目标，所以不能把所有步的 energy 当成同一个
固定函数的下降曲线；不同任务之间也不能直接用原始 energy 排名。

代码中至少有三种不同的总分：界面 `E_graph`、核心 `E_core`，以及刚体联合更新的
`E_joint`。第 6、8、7 节分别定义它们，不能只看到日志字段名 energy 就认为含义相同。

## 3. 采样前：初始位置为什么被保留或排除

2026-09-16 起，一项任务仅实现一个起始 pose，全部 `designs` 共用；刚体运动模式
独立决定生成时能否移动。`prepare-poses` 在 CPU 上按几何评分筛选不同起点并输出
独立任务，完整公式和去重限制见 [任务与 pose](TASK_POSES.zh-CN.md)。
运行前的 wedge、tangent、轴线和直线走廊描述现在均为建议项，不能独立判定失败。


来源：`src/rfd3_mosaic/pose_optimizer.py`、`output/standalone.py`。

两个连接端点的距离为 `d`，中间生成残基数的几何下界为：

```text
n_min = max(0, ceil(d / 3.8 Å) - 1)
可行的必要条件：n_min <= linker.maximum_length
```

例如端点相距 38 Å，下界为 9 个中间残基。它来自近似 Cα 步长和三角不等式；
满足下界不保证能避障、满足端点方向或折叠。直线走廊被挡也不能证明柔性链无法绕行。
当前优化器把直线走廊障碍（内部弦距固定原子小于 2 Å）放在软排序中。

硬失败计数包含固定块碰撞、必需界面失败、linker 长度不足和必需输出目标失败。
排序是按元组逐项比较：先可行性和硬失败，再连接内部弦拥挤比例、走廊障碍、长度不足惩罚、目标惩罚、
端点方向/净空/距离等。它不是“降低一点距离分，就可以抵消一个硬失败”的加权排名。
`prepare-poses` 按已排名候选贪心保留满足距离谱差异下界的姿态；它排除整体
旋转、平移和副本编号造成的假差异，但可能合并不同形状。旧的 SO(3) 夹角筛选
工具仍属于独立研究工具，不等于此处的任务级多样性保证。

实际 `score` 元组按以下顺序升序比较；只有前项相等才比较后一项：

```text
hard_count = 固定块碰撞数 + 失败必需界面数 + 不可行连接数 + 必需输出目标失败数
feasible = (hard_count == 0)
score = (not feasible, hard_count, 失败必需界面数, 不可行连接数,
         固定块碰撞数, 必需输出目标失败数, 内部连接弦拥挤比例,
         被挡直线走廊数, sum_linkers [所需最少残基数-允许最大残基数]+,
         用户目标加权惩罚, 最大端点切向夹角, -最小走廊净空,
         最大端点距离, -最小固定组间距离)
```

后三类角度/走廊/端点指标缺失时相应字段为 `+inf`；最后的组间距离缺失时字段为
`-inf`，这是当前实现的缺失值排序约定，不是物理测量值。轴线净空仅报告，不参与该排名。

任务级去重则收集不同刚体实例之间的 Cα 距离，排序为向量 `s`，计算：

```text
D_pose(A,B) = sqrt(mean((s_A-s_B)²))
保留条件：对每个已保留 pose B，都有 D_pose(A,B) >= minimum_separation
```

默认 `minimum_separation=1 Å` 是描述符分辨率。整体平移/旋转不制造差异，但不同形状
可能有同样的距离谱；它不是生成骨架 RMSD，也不保证后续可移动模式的最终多样性。

## 4. 扩散中，什么是始终被约束的

来源：`models/rfd3/src/rfd3/model/inference_sampler.py` 及 `inference/symmetry/`。

精确群作用路径把对称副本逆变换到代表坐标、聚合，再按群作用复制，并耦合同一轨道的噪声。
固定原子恢复为当前约束目标。允许运动的固定块以整个刚体更新目标，因此“内部几何固定”
不意味着“绝对空间位置固定”。最终写出坐标再次经过约束处理。
官方/legacy sampler 与 exact Mosaic sampler 的机制不同，不能由同一设计名称推断开启项。

典型调用顺序是：去噪模型给出干净坐标预测 → 允许时提出刚体/界面联合更新 →
去噪预测上的界面与核心引导 → 原噪声日程下的扩散积分更新 → 末端界面 polish →
最终生成链连续性修正 → 最终约束投影和审计。
具体是否进入某分支取决于 sampler 类型、编译拓扑、开关、进度和可移动组件。

刚体更新可写为 `x_i'=R*x_i+t`，其中 `RᵀR=I`、`det(R)=1`，同一固定块使用同一个
`R,t`，因此 `||x_i'-x_j'||=||x_i-x_j||`。`locked` 不提交刚体位姿变化；可移动模式
只允许声明自由度内的变化。跨不同刚体的距离可以变，不能将其当作“seed 内部固定”
的同一条约束。对称副本按相应群变换生成，最终仍须审计数值误差。

几何引导是去噪预测上的有界局部修正，不是经过证明的精确后验采样；保留原有
噪声日程、进度窗口与步长上限。日志 `coordinate_space` 区分去噪预测和最终结构。
在去噪预测上评价几何的相关依据可参考 [FK protein steering 方法](https://arxiv.org/html/2511.09216v2)，
其结果不能直接转述为本项目 Cn/Dn 的成功率。

graph/core 局部修正与最终连续性修复使用逐对几何回退检查：对非相邻 Cα 对和
跨链 Cα 线段，令 `v=max(0,3.2 Å-d)`；对主链相邻 Cα，令
`v=max(0,abs(d-3.8 Å)-tolerance)`。仅当每一对的违约量均不增加超过
`1e-6 Å` 数值容差时才接受候选，不能用一处冲突改善抵消另一处新增冲突。
距离参数复用核心/连续性配置；上述值为默认值，3.2 Å 属于粗粒度控制器阈值。
该检查包括生成–固定 Cα 对，排除全部固定的约束对。它不等于全原子排斥或完整缠绕检测，
也不限制去噪模型本身每一步的变化；最终输出仍须独立审计。

末端界面 polish 可以与核心/连续性拓扑共存，并经过同一几何回退检查；配置的步数
是上限，已达到目标会提前停止。连续性修复若引入几何退化，保留修复前坐标并报告
`geometry_regression`，未修好的断链不会因此变为通过。

## 5. “拉近”之一：远距离接触势

来源：`graph_interface_guidance.py::rf_oligomer_contact_prior`。
作用于声明/自动拓扑选中的界面两侧生成 Cα；不等于任意两个块都互相吸引。

```text
q(d) = 1 / (1 + ((d - d0) / r0)^6)
E_contact = -sum_ij q(d_ij) / min(N_left, N_right)
r0 = 8 Å, d0 = 2 Å
```

这是最小化目标：在 d > d0 时减小距离会使该项更负。
单个距离的 q(10 Å)=0.5，q(18 Å)=1/65；这两个数是平滑接触强度，不是成功概率。
该形式受 [RFdiffusion 原始 contact potential](https://github.com/RosettaCommons/RFdiffusion/blob/main/rfdiffusion/potentials/potentials.py)
启发；Mosaic 使用负号、长度归一化及自己的调度。原实现不证明 Mosaic 的参数组合已校准。

实际接触势系数还乘 `guide_scale * (1-p)^decay_power`，默认 `2*(1-p)^2`；
p 是扩散步序号归一化进度，不是当前距离。公共 balanced 预设基础权重为 0.10，
所以早期有效系数接近 0.20、后期趋零；仍需满足整个引导的激活条件。
配置类本身的 contact_prior_weight 默认是 0，不能把预设与类默认混为一谈。
单独使用接触势会偏好靠近，因此还配有碰撞、覆盖、方向和步长约束。

## 6. “拉近”之二：局部界面片段及阶段切换

来源：`graph_interface_guidance.py` 的 patch selection、energy、schedule 和 apply 函数。
片段按双方相互匹配的连续窗口选择；离散选择不求导，选中距离重新参与梯度计算。
最终片段/残基 ID 在步骤记录中保留。面积目标必须看拓扑的 target 字段和容量预检；
`pairs_per_edge=8` 不是“所有界面一定有 8 个残基”的声明。

默认起点 a=0.05，终点 b=0.80，尾部强度 f=0.8，令 u=(p-a)/(b-a)：

```text
W(p) = 0                                    p <= a
       sin(pi*u)                            a < p < b, u <= 0.5
       f + (1-f)*sin(pi*u)                   a < p < b, u > 0.5
       f                                    p >= b
```

因此 end_fraction=0.80 不表示后 20% 关闭，而是进入尾部强度。
时间距离目标从 12 Å 以 smoothstep 收到 8 Å：`12 + (3u²-2u³)*(8-12)`，两端截断。
12 Å 是扩大捕获范围的工程选择，不是已接触的判定。

几何阶段另行判断，优先顺序如下：

| 阶段 | 触发 | 对时间目标与基础配置的调整 |
|---|---|---|
| capture | 任一界面选中距离均值 > 最终基础目标 + 2 Å | 距离至少 target+2；覆盖×0.75，连续×0.60，方向/形状×0.50 |
| expand | 距离条件未触发，但任一侧覆盖数或连续覆盖数不足 | 距离夹在 target 与 target+2；吸引×0.85，覆盖×1.25，连续×1.35，形状×0.80 |
| polish | 上述距离与覆盖条件都不触发 | 使用最终目标；吸引×0.60，连续×1.15，方向/形状×1.50，步长/旋转上限×0.50 |

单条边最终 Cα 目标还取 `min(config.target_ca_distance, max(config.clash_ca_distance+0.5,
edge.contact_cutoff+2.5))`，再叠加调度超过基础目标的部分。
所以 8 Å 不是所有边最终审计的接触截止值。Cα 控制距离和原子级接触距离也不是同一个量。

进度达到 0.50 只允许尝试锁片段；还必须在最终目标半径下满足双方覆盖和连续覆盖，
才能 `patch_locked=true`。锁片段用于减少反复换接触对象，不意味着结果已高质量。

### 界面各能量项

#### 总分：代码究竟在最小化什么

来源：[graph_interface_energy](../../models/rfd3/src/rfd3/inference/symmetry/graph_interface_guidance.py)。
输入为 `coordinates`、编译后的 `topology`、本步 `effective_config`、目标距离及片段身份。
输出 `GraphInterfaceEnergy` 包含以下总分及未加权分项：

```text
E_graph(X) = w_contact   * L_contact
           + w_attract  * L_attraction
           + w_coverage * L_coverage
           + w_cont     * L_continuity
           + w_orient   * L_orientation
           + w_shape    * L_shape
           + w_backbone * L_backbone
           + w_balance  * L_balance
           + w_exclusive* L_exclusivity
           + w_clash    * L_clash
           + w_distance * L_distance
```

公共 balanced 预设在阶段、面积及其他覆盖项调整前对应：

```text
E_base = .10*contact + attraction + coverage + continuity
       + .25*orientation + .50*shape + .10*backbone
       + .50*balance + exclusivity + 8*clash + .25*distance
```

这只是基础系数展开，**实际每步必须使用 effective_config**。例如基础接触权重 .10
在进度 `p=.5` 时成为 `.10*2*(1-.5)^2=.05`；capture 阶段的基础覆盖权重 1 变为 .75。
时间窗口 `W(p)` 控制激活和移动幅度，不要再把它误乘到日志记录的总分上。

定义平滑损失（代码使用 `smooth_l1_loss`）：

```text
H_beta(z) = z²/(2*beta)          abs(z) < beta
            abs(z)-beta/2       其他情况
```

边损失先对同一声明界面的物理副本取平均，再对声明界面取平均，记为 `A(...)`：

```text
A(l_e) = mean_s(mean_{e 属于 s}(l_e))
```

这样副本数量多的界面不会仅因对称阶数更高就获得更多权重。
全局 junction、global_safety_clash 和 exclusivity 另行计算，不在每条边重复相加。

#### 各项的公式、对象与理由

| 项 | 代码中的量 | 设置理由与局限 |
|---|---|---|
| attraction | 选中距离的 mean H_1([d-target]+) | 使选中接触靠近；目标内不额外奖励此项 |
| coverage | 双方满足目标数量的最近距离的 H_1 超距损失 | 避免只靠一个点；数量需结合拓扑和面积偏好 |
| continuity | 最优连续窗内 `(max(m-sum sigmoid((target-d)/s),0)/m)^2 + 0.25 mean(([d-target]+/target)^2)`；s=0.75 | 偏好连续片段；不是二级结构判别 |
| orientation | 两侧切向 t 与接近方向 n 的 mean `[abs(t·n)-0.65]+²` | 避免端对端接触；短/退化片段此项可为 0 |
| shape | 最近距离方差/target² + mean `([abs(d-max(3.5,target-1))-0.5]+/target)^2` | 控制距离均匀程度；不是已发表的表面互补性 Sc |
| backbone | 连续 Cα 步的 mean H_0.5(`[abs(d-3.8)-0.5]+`)，另加固定/生成接点项 | 减少局部链拉断；不是完整立体化学模型 |
| clash | sum `[3.5-d]+²`/两侧残基数；另加全局安全集合碰撞/引导 Cα 数 | 防止只优化选中片段却撞上旁边链；有拓扑排除项 |
| distance | 有显式质心距离目标时 H_1(`[abs(d_COM-target)-tolerance]+`) | 实现用户距离意图；未声明时为零 |
| exclusivity | 对每残基不同参与方的软占有率累加，mean `[sum occupancy-1]+²`；同参与方先取 max | 减少多个界面抢同一片段；属于分配启发式 |
| balance | 多个来源界面时 logsumexp(E_source)-log(N_source)，单来源为零 | 更关注较差界面；不是绝对的最差项优化 |

上表中的 `continuity` 指**接触残基在序列上的连续性**；肽链是否拉断由 backbone/junction
等其他项处理。`shape` 只是 Cα 距离分布，不是全原子表面互补性或结合自由能。

为便于复算，几个容易混淆的求和范围和归一化展开如下：

```text
# 接触势在两侧生成 Cα 的全部配对上计算，公式见第 5 节。
L_contact = A(-sum_{i∈left,j∈right} q(d_ij) / min(N_left,N_right))

# 吸引只取当前成对片段中选出的接触对 P_e。
L_attraction = A(mean_{(i,j)∈P_e} H_1([d_ij-D_e]+))

# r 是选中片段每个残基到对侧片段的最近距离。
# 左右各选要求数量的最小 r；拼接为 r_selected 后计算。
L_coverage = A(mean H_1([r_selected-D_e]+))

# 连续窗口的目标接触数为 m，窗口宽度 min(m,可用残基数)。
J(window) = ([m-sum sigmoid((D_e-r)/.75)]+/m)²
            + .25*mean(([r-D_e]+/D_e)²)
L_continuity = A(.5*(min_left_windows J + min_right_windows J))

# t 为片段单位切向，n 为两片段中心间的单位方向；退化片段不贡献。
L_orientation = A(mean_{左右有效切向 t} [abs(t·n)-.65]+²)
r_star = max(3.5,D_e-1)
L_shape = A(mean(((r-mean(r))/D_e)²)
            + mean(([abs(r-r_star)-.5]+/D_e)²))

# 主链内部先按连续段平均，再按左右两侧和声明界面平均。
L_local_backbone = A(.5*(B_left+B_right))
B_side = mean_runs(mean_adjacent H_.5([abs(d-3.8)-.5]+))
L_junction = mean_{topology.junction_ca_pairs} H_.5([abs(d-3.8)-.5]+)
L_backbone = L_local_backbone + L_junction

# 单边碰撞按残基数归一化，不按 N_left*N_right 归一化。
L_edge_clash = sum_{left,right} [3.5-d_ij]+² / (N_left+N_right)
L_global_clash = sum_{安全集合中未排除的配对} [3.5-d_ij]+² / N_guided_CA
L_clash = A(L_edge_clash) + L_global_clash

# 一个 token 对同一参与方的重复记录先取最大占用量。
o_i,k = max_records sigmoid((D_e-r_i)/.75)
L_exclusivity = mean_{有多个参与方的 token i} [sum_k o_i,k-1]+²

# 用户没有为某边声明质心距离目标时，该边 distance 项为零。
L_distance = A(H_1([abs(d_COM-D_user)-tolerance]+))
```

这些表达式中的距离、软化尺度、目标和阈值使用默认数值展示；复算时换成本步参数。
可选项没有合格对象时按实现取零；界面两侧完全没有可配对的生成 Cα 时会报错，
不会按零分通过。未声明数量要求且没有启用自动界面
质量的边，不启用片段吸引、覆盖、连续性、方向、形状及局部 backbone 项；
接触势、碰撞、显式距离项和全局安全项仍按各自配置计算。

界面平衡项使用的 `E_s` 先由单边加权分数取副本平均；**单边分数不包含**全局 junction、
global_safety_clash、exclusivity 或 balance 本身，因此没有递归计算：

```text
E_edge = w_contact*contact_e + w_attract*attraction_e + w_coverage*coverage_e
       + w_cont*continuity_e + w_orient*orientation_e + w_shape*shape_e
       + w_backbone*backbone_e + w_clash*clash_e + w_distance*distance_e
E_s = mean_{e 属于 s}(E_edge)
L_balance = log(sum_s exp(E_s))-log(S)     S > 1
            0                            S = 1
```

balance 额外强调较差界面，但并不等于“最大损失减最小损失”，也不是概率。

#### 权重的来源和数值例子

公共基础预设如下，尚未乘阶段/面积调节：

| packing | 接触势 | 吸引 | 覆盖 | 连续 | 方向 | 形状 | 平衡 | 距离 | token 最大步长 Å |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| loose | .06 | .80 | .80 | .85 | .20 | .35 | .40 | .20 | .20 |
| balanced | .10 | 1 | 1 | 1 | .25 | .50 | .50 | .25 | .25 |
| tight | .15 | 1.15 | 1.25 | 1.30 | .40 | .80 | .70 | .35 | .20 |

面积 small/auto/large 分别设置 pairs=4/8/12，覆盖再乘 .85/1/1.25，连续再乘 .90/1/1.15。
基础安全项 clash=8、backbone=.1、exclusivity=1；专家覆盖项仍可能改变允许修改的数值。
这些比例是工程调节策略，不能解释成 tight 在科学意义上“更好”。

上述 .25、.50、8 等相对权重，以及方向、形状和覆盖偏好，是工程选择；它们不是
从统一分子力场推导出的常数，也不是模型训练得到的参数。距离量、平方距离量和
归一化量混合在同一目标中，总分不具有统一的物理能量单位。

以下只是手算示例，不是本项目 benchmark 结果：

| 场景 | 代入公式 | 含义 |
|---|---|---|
| 当步目标 D=8，某选中接触距离 12→10 Å | `H_1(4)=3.5` → `H_1(2)=1.5` | 该接触的吸引损失减少 2；之后还要平均并乘权重 |
| D=8，接触距离 8→7 Å | 吸引项均为 0 | 这一项不再奖励靠近，其他项仍可能变化 |
| 某非成键 Cα 距离 3.6→3.3 Å | 原始重叠平方 `0→(3.5-3.3)²=.04` | 产生碰撞惩罚；之后才做归一化和权重相乘 |
| 某 junction 距离 3.8→4.6 Å | `H_.5(0)=0` → `H_.5(.3)=.09` | 该连接对出现惩罚；整体 junction 是配对平均值 |

因此“总分 10→9”只能解释为该步这套几何目标下降 1，不能解释为稳定性提高 10%。

### 为什么接受这一步，而拒绝另一步

梯度先聚合为生成 token 的整体平移，平滑相邻更新，融合受限片段刚体运动。
默认 token 上限 .25 Å、片段旋转上限 2°；未满足代理目标且梯度非零但过小时，
可放大至阶段步长上限的 .50 倍。不是按原始梯度大小无限移动。
随后按 1、.5、.25、.125、.0625 试步（默认 5 次），**先投影真实约束，再评价**。
如调用方提供第 4 节的 `candidate_validator`，**先做逐对几何回退检查**；不通过时
记 `geometry_regression` 并尝试更小步长，此时下面的总分检查尚未执行。
通过后 `graph_interface_proposal_acceptable` 按以下顺序短路检查：

1. 总能量有限，且 `E_after < E_before - 1e-10`。
2. 界面及全局最小 Cα 距离：原来 ≥3.5 Å 则不能降到 3.5 Å 以下；原来更小则不能再缩小，容差 1e-6。
3. 来源界面数组形状一致；最差来源的能量不能增加超过 .002。
4. 任何来源能量增量不能超过 `max(.002, abs(worst_before)*.02)`。
5. junction 损失若原来 ≤.02，之后最多 .02；否则不能继续增加，容差 1e-8。
6. exclusivity 使用同样逻辑，但参考值是 .05。
7. 全局安全碰撞能量不能增加超过 1e-8。

第一组全部通过的候选被接受；都失败则保留此次局部修正前的坐标。
“全部候选被拒绝”不表示 RFD3 的整个扩散时间步停止；后续去噪/积分仍按采样器执行。

把距离条件写成公式更准确：

```text
required = d_clash          d_min_before >= d_clash
           d_min_before     d_min_before < d_clash
通过条件：d_min_after >= required - 1e-6 Å
```

这里分别对界面距离集合的整体最小值、全局安全距离最小值检查，不是每条界面的
最小值均严格单调；第 4 节的逐对 guard 是另一层检查，默认半径为 3.2 Å。

| 前→后（默认参数） | 该条检查的结论 |
|---|---|
| 总分 10→9，最小距离 3.6→3.3 Å | 距离检查拒绝；总分改善不能抵消越界 |
| 最小距离 3.6→3.55 Å | 距离检查允许；是否接受仍要看其他条件 |
| 最小距离 3.3→3.2 Å | 距离检查拒绝；已经越界的状态不能继续缩小 |
| junction .01→.015 | junction 检查允许，因为仍不超过 .02 |
| junction .03→.031 | junction 检查拒绝，因为原来已超过 .02 |
| exclusivity .01→.04 | exclusivity 检查允许，因为仍不超过 .05 |

`.002` 与 `2%` 也**允许少量单界面退步**；2% 的基数是 `abs(worst_before)`，
不是各个界面自己的分数。不能将这些条件统称为“任何指标都不能恶化”。

### 怎样从运行记录复算一次决策

1. 从 `decision_explanation.json` 的 `result.path` 找到原始结果 JSON，核对 SHA-256。
2. 进入 `graph_interface_guidance_diagnostics.steps`，选择需要解释的步骤。
3. 先看 `coordinate_space`、`progress`、`adaptive_phase`、`effective_config`、
   `scheduled_target_ca_distance` 和 `patch_assignments`，确认评价的是哪组坐标和片段。
4. `energy_before` 是局部修正前的总分；`energy_after`/`energy` 和同层分项是最终保留
   坐标上的值。如果全被拒绝，保留的就是修正前状态；这些分项不是被拒绝候选的分项。
5. 逐个检查 `line_search_trials` 中的 `scale`、`geometry_guard`、`checks` 和
   `first_rejection_reason`。短路后没有执行的检查不能称作通过。

下面的代码只用标准库，复算**已记录步骤最终保留状态的总分**。在仓库外也能运行；
将 `result.json` 替换为上述原始结果路径：

```python
import json
import math
from pathlib import Path

terms = {
    "contact_prior": "contact_prior_weight",
    "attraction": "weight",
    "coverage": "coverage_weight",
    "continuity": "continuity_weight",
    "orientation": "orientation_weight",
    "shape": "shape_weight",
    "backbone": "backbone_weight",
    "interface_balance": "interface_balance_weight",
    "patch_exclusivity": "patch_exclusivity_weight",
    "clash": "clash_weight",
    "distance": "distance_weight",
}
result = json.loads(Path("result.json").read_text())
steps = result["graph_interface_guidance_diagnostics"]["steps"]
for i, step in enumerate(steps):
    config = step.get("effective_config")
    if config is None or "energy_after" not in step:
        print(i, "该记录没有可复算的完整总分", step.get("reason"))
        continue
    contributions = {term: step[term] * config[key] for term, key in terms.items()}
    computed = sum(contributions.values())
    recorded = step["energy_after"]
    print(i, contributions, "复算:", computed, "记录:", recorded,
          "数值近似一致:", math.isclose(computed, recorded, rel_tol=1e-5, abs_tol=1e-5))
```

这里的 `1e-5` 只是复算时浮点累加差异的比较容差，不是采样接受阈值。
不要再次乘 `contact_prior_schedule_scale`，因为它已包含在 `effective_config` 中；
不要额外加 junction/global_safety_clash，它们已分别包含在 backbone/clash 中。

当前记录的边界也必须公开：graph 的每个被拒绝 trial **没有统一保存全部损失分项**，
主要保存实际执行的检查及数值；core 的 trial 则保存 `metrics`。总分字段本身不能
反推出所有坐标，也不足以独立复算每一个被拒绝候选。缺少候选坐标或分项时，应明确
写“证据不足以完整复算”，不能声称已经记录了所有中间状态。

## 7. “拉近”之三：刚体朝生成骨架核心移动

来源：`motif_mobility.py`、`scaffold_guidance.py`、`scaffold_core_guidance.py`。
只在启用刚体运动、存在相应拓扑、处于允许窗口/更新步时发生；锁定块不能因此移动。
允许平移/旋转方向、每步上限和累计上限来自组件声明及 sampler 配置。

刚体基础能量包括：

```text
E_junction = mean_junctions delta² * (sqrt(1 + ((d-3.8)/delta)²) - 1), delta=.25
E_clash = mean [d_clash-d]+²，排除相邻成键关系
E_tilt = [cos(theta_max)-abs(dot(rotated_principal_axis, symmetry_axis))]+²
E_prior = ||t/translation_scale||²
          + ||R-I||_F² / (2*(2*sin(rotation_scale/2))²)
```

配置类基础权重 junction=1、clash=1、tilt=.25、prior=.05，基础 clash 距离3 Å、
最大倾角20°；以运行的 `scaffold_guidance_config` 为准。
先验限制偏离初始姿态，倾角项表达轴向偏好；两者都不是普适生物学要求。

可选 denoiser proposal 通过逆变换并平均对称副本、拟合刚体变化来获得更新；
scaffold-objective proposal 则从上述能量的受限 SE(3) 梯度提出更新。
早期还有带种子的多候选探索，所以不是完全确定的“向最近点走一步”。
在组件允许窗口内归一化进度，默认前40% capture、中间至80% settle、最后 polish；
capture 响应最多按基础响应×5并封顶1，之后收小，几何运动上限仍然生效。

**robust capture** 是另外一个早期目标：双端锚定任务按编译后的固定端点归属，
绑定每个刚体副本实际连接的生成链核心；采样中不按几何距离更换邻居。
没有任何双端锚定生成段的 terminal-only 任务仍使用最近两个生成链的兼容规则。
双端任务中，没有连接端点的刚体不参与此捕获。该归属规则不依赖 Cn/Dn 的阶数。
从普通中心平滑过渡到接触支持加权中心：

```text
c_target = (1-u)*mean(c_plain_j for j in neighbours) + u*mean(c_supported_j for j in neighbours)
E_capture = mean ||(c_rigid - c_target)/contact_distance||²
```

只有启用此功能、存在生成骨架、允许运动且处于捕获窗口时加入。
双端任务使用绑定的所有不同链核心；未绑定链的刚体不参与。terminal-only 兼容规则在不足两个核心时为零。
端点身份绑定不随坐标变化；只有 terminal-only 的最近核心选择依赖瞬时几何。
默认捕获权重1，早期窗口取运动窗口前40%，但以实际配置为准。
它表达“朝有多链支撑的位置移动”的启发式，不能保证正确装配；空腔/细长/松散目标尤其需要审查。

联合更新实际比较的总分为：

```text
E_scaffold = w_junction*J + w_clash*(C_scaffold+C_inter_orbit)
           + sum_{可移动刚体轨道 g}(w_tilt,g*T_g + w_prior,g*P_g)
E_extra = E_core + w_capture*mean_g(E_capture,g)   对应拓扑/目标启用时
E_joint = E_graph + E_scaffold + E_extra
```

不启用的附加目标不贡献分数；capture 只在其窗口内加入。几何项在联合 scaffold 中
计算一次，各轨道姿态项逐个相加。graph 的 junction 和 scaffold 的 junction 使用
不同损失形式与作用集合，可能对同一连接提供重复约束；相加不表示它们是独立的物理能量。

刚体与界面联合事务同时比较 graph、scaffold 和启用的 core/capture 总能量。
必须真的有变化、界面改善、联合总能量改善、界面接受条件及安全条件都满足，才提交。
失败则回退坐标、刚体姿态和片段状态；proposal-only 模式即使候选通过也不提交。
因此 `E_joint` 降低但 `E_graph` 没降低时，也不能通过联合接受规则。

## 8. 核心、走线和最终链连续性

来源：`scaffold_core_guidance.py`。核心引导只更新生成 token；默认窗口 .05–.90，
窗内 `min(1,4u,4(1-u))`、窗外0。总能量为零也跳过。

长程接触强度为 `sigmoid((8-d)/.75)`；同链序列间隔至少8的接触用于长程支持。
`Rg=sqrt(mean ||x-mean(x)||² + 1e-8)`，对整条链的 Cα 计算，长度归一化用 `Rg/N_chain^.38`。
长程接触仅取至少含一个生成残基的非重复配对；配对计数与每残基支持计数不同。

```text
L_long = [1.5 - soft_contact_sum/N_generated]+²
L_rg = [Rg/N_chain^.38 - 2.6]+²
L_support = mean [2 - support_per_generated_residue]+²
L_worst = T*(logsumexp(window_mean_deficits/T)-log(number_of_windows)), T=.25
```

worst 使用长度 min(8,N_generated) 的连续窗口，强调最缺乏接触支持的一段。
基本核心能量是 `intra*(.75 L_long + .35 L_rg + L_support + L_worst)`，
再加启用的跨链接触过量惩罚、8倍 Cα/链段碰撞、2倍连续性和走线项。
完整总式为：

```text
E_core = w_intra*(w_long*L_long + w_rg*L_rg + w_support*L_support + w_worst*L_worst)
       + inter_chain_excess_penalty*inter_chain_excess_weight*L_inter_excess
       + clash_weight*(L_CA_clash+L_segment_clash)
       + continuity_weight*L_backbone_continuity
       + routing_ownership_weight*L_routing
L_inter_excess = mean_chain_pairs(([sum soft_contacts-.08*N_min]+/N_min)²)
L_backbone_continuity = mean_chains(mean_adjacent [abs(d-3.8)-.55]+²)
```

`N_min` 是该链对两侧生成 Cα 数量的较小值，仅对非空链对计算；.08 来自默认
`incidental_inter_chain_fraction`。Cα/线段碰撞项是对合格配对的 `[3.2-d]+²` 分组平均。
此处连续性默认容差 .55 Å，不能与 graph 的 .5 Å 或末端投影的 .5 Å 混为一谈。
`intra=0` 时紧凑化项关闭，但安全/走线项可以独立存在。公共创建界面任务通常解析为 intra=1。
上述 1.5、2、2.6、.38 都不能直接视为适合全部长度和折叠类型的质量判据。

走线项比较生成 Cα 到自己连接端点弦和其他链端点弦的距离，
惩罚 `([d_own-d_nearest_other]+/3.8)²`。它鼓励空间归属，
不要求链笔直，也不是数学上的无穿链/无打结证明。

核心候选做梯度限幅、两遍邻居平滑和三遍相邻步差限制（默认 .08 Å），
默认 token 上限 .20 Å 乘窗口。五次减半试步要求：
总能量下降超过1e-8，clash、跨链段碰撞、continuity 不增加超过1e-7；
启用 routing 时 routing 也不得超过该容差。失败保留原坐标，日志保存逐次条件。

最终生成链连续性投影只在末端执行，默认 Cα 目标3.8 Å、容差.5 Å、最多64轮，
通过移动整个生成 token 调整相邻距离，不移动固定 token；随后再做硬约束投影。
已经全部在容差内则不改。达到迭代上限可能仍未达标，必须看最终坐标审计。
这不是全原子肽键几何优化，也不保证所有 N–C 距离、键角、二面角正确。

## 9. 最终如何筛选，为什么会有 FLAG

来源：`advisory_screening.py`、`validation/scaffold_validity.py` 和各 `rfd3_*_audit.py`。
筛选转述现有审计，没有一个通用“总分≥多少就通过”的函数。

| 检查 | 当前解释 |
|---|---|
| 固定几何、对称轨道、运动范围、坐标合同 | 不满足记 contract flag |
| 主链原子完整性、链连续性、声明对称性 | 不满足记 contract flag |
| 当前语义的显式 required 界面、用户声明形状范围 | 不满足记 contract flag |
| 控制器执行、配置/标识/最终指标一致性 | 不满足记 contract flag；不是生物学结论 |
| 自动界面覆盖/方向/形状代理未达标 | 默认 advisory flag |
| 粗粒度碰撞、紧凑度、肽键几何提示 | 默认 advisory flag；生成文件保留 |
| scaffold 审计的跨链拓扑/穿插检查失败 | contract flag；不能仅因为存在生成文件就判通过 |
| 显式 required 的核心代理目标 | 用户要求未满足记 contract flag；仍只是代理目标 |
| 无审计文件 | not_evaluated，不能称合同通过 |

`geometric_constraints.contacts` 显式接触关系的有效最小接触数为
`max(1, declared_minimum)`；`reference_transform` 分支允许声明最小接触数为 0。
报告保留相应声明值和判定参数，详见第 19.6 节。一个接触只证明按该距离定义存在接触，
不证明界面质量足够。
没有捕获到可锁定片段，与锁定之后片段身份改变分别报告；前者不是执行标识错误。
控制器可以合法拒绝全部候选，但必须有完整、可解释的拒绝记录，不能缺失执行证据。
用户显式 required 的核心质量即使关闭核心引导也会保留审计；纯测量模式不要求发生位移。

界面代理束默认还检查：双方覆盖/连续覆盖数，orientation≤.05、shape≤.08、
backbone/junction≤.02、exclusivity≤.05、最小 Cα 距离≥3.5 Å。
这些是控制器阈值，不是已发表接口质量标准。

骨架审计函数默认参考包括：相邻 C–N 1–2 Å，Cα 最大步长4.5 Å，
链 Cα Rg≤25 Å，非相邻 Cα 小于3 Å碰撞提示，跨链 Cα 段距离1 Å参考，
对称坐标 RMSD .01 Å、最大误差 .03 Å。实际阈值随调用参数改变并写入审计。
**运行中的3.5 Å碰撞半径与审计中的3 Å不是同一标准；3.8±.5 Å投影与4.5 Å审计上限也不同。**
前者是运动保护/修正设置，后者是独立结果观察；需要分别展示，不能只报“安全”。

最终映射：有合同问题→`review_contract`；只有建议问题→`review_advisory_metrics`；
两者都没有且有审计→`recommended_for_next_stage`。
`screening=off` 输出 `not_screened` 并隐藏建议项，仍核查合同并保留审计证据。
历史 `accepted` 是兼容字段；新的用户结论应使用 generated/contract_status/recommendation。

## 10. 是否合理：目前能支持的结论

- 对称投影、刚体内部几何和轮廓长度下界有清晰几何依据；实现仍受浮点容差和路径覆盖影响。
- 限步长、回溯、总能量下降及碰撞不恶化是合理的数值保护；不保证全局最优或设计成功。
- 捕获半径、阶段比例、权重、接触支持、Rg及形状代理阈值，是有目的的工程启发式；
  当前不能宣称对不同长度、对称群、空腔目标和折叠类型都合理。
- Cα 碰撞代理不是 MolProbity clashscore，距离形状损失不是 Lawrence–Colman Sc，
  绝对 Rg≤25 Å不能直接用于所有长度，单结构建议不能冒充队列中位数筛选。
- 本文的公式解释本身不构成效果验证，也没有重新标定上述参数。测试、GPU 推理和
  benchmark 的结论须注明对应源码版本、输入与配置；不能由文档更新推断它们已通过。
  后续验证应冻结数据/参数、比较有无各项引导、报告不同任务的失败与收益，才能支持调参结论。

### 规则与实现的对应入口

| 要核对的规则 | 源码入口 |
|---|---|
| packing 预设与面积调节 | [design_preferences.py](../../src/rfd3_mosaic/design_preferences.py) |
| 界面损失、阶段切换、回溯及接受条件 | [graph_interface_guidance.py](../../models/rfd3/src/rfd3/inference/symmetry/graph_interface_guidance.py)：`graph_interface_energy`、`_phase_guidance_config`、`apply_graph_interface_guidance`、`graph_interface_proposal_acceptable` |
| 刚体联合损失与提交/回退 | [motif_mobility.py](../../models/rfd3/src/rfd3/inference/symmetry/motif_mobility.py)：`_joint_scaffold_energy`、`update_orbits_with_interface_packing` |
| 核心、路径及逐对保护 | [scaffold_core_guidance.py](../../models/rfd3/src/rfd3/inference/symmetry/scaffold_core_guidance.py)：`scaffold_core_energy`、`scaffold_geometry_guard` |
| 采样过程的调用时机 | [inference_sampler.py](../../models/rfd3/src/rfd3/model/inference_sampler.py) |
| pose 可行性和排序 | [pose_optimizer.py](../../src/rfd3_mosaic/pose_optimizer.py) |
| 同任务起点冻结与跨任务去重 | [pose_tasks.py](../../src/rfd3_mosaic/pose_tasks.py) |
| 最终合同与建议映射 | [advisory_screening.py](../../src/rfd3_mosaic/advisory_screening.py) |
| 每个设计的证据索引 | [decision_explanation.py](../../src/rfd3_mosaic/decision_explanation.py) |

进一步的指标出处与限制见 [STRUCTURE_METRIC_PROVENANCE.md](STRUCTURE_METRIC_PROVENANCE.md)
和 [BACKBONE_EVALUATION_EVIDENCE.md](BACKBONE_EVALUATION_EVIDENCE.md)。
源码是执行规则的最终依据；本文及 `decision_policy.version` 的变化需与规则修改一起审查。

## 11. 刚体变换、群作用、稳定子和装配数量

**调用范围：**装配编译与对称展开；稳定子识别仅用于声明了相应商轨道/组件作用的路径。
来源：[geometry/se3.py](../../src/rfd3_mosaic/geometry/se3.py)、
[symmetry_registry.py](../../src/rfd3_mosaic/geometry/symmetry_registry.py)、
[stabilizer_cosets.py](../../src/rfd3_mosaic/topology/stabilizer_cosets.py)、
[seed_stabilizer.py](../../src/rfd3_mosaic/seed_stabilizer.py)。

### 11.1 坐标、旋转及参考系

本文统一用列向量；源码有的批量数组用行向量存储，因此对应乘 `R.T`。

```text
T = [[R,t],[0,1]]，T(x)=Rx+t，RᵀR=I，det(R)=1
T1∘T2 = (R1 R2, R1 t2+t1)
T⁻¹ = (Rᵀ, -Rᵀt)
K=[a]×，Kx=a×x，||a||=1
R(a,θ)=I+sin(θ)K+(1-cos(θ))K²
绕中心 c 旋转：t=c-Rc
```

零轴、非有限矩阵、非刚体矩阵和不匹配形状会被拒绝。角度传入 Rodrigues 公式时为弧度。
由坐标定义局部 frame 时，对方向归一化并正交化；共线/零长度轴不能定义完整 frame。
固定 Euler 姿态按源码的 `Rz @ Ry @ Rx` 组合，不能随意调换三个旋转的顺序。

### 11.2 Cn、Dn、T/O/I 的实际群作用

```text
Cn：R_k=R(a,2πk/n)，k=0..n-1，|G|=n
Dn：{R_k} ∪ {S R_k}，S=R(b,π)，a·b=0，|G|=2n
T/O/I：|G|=12/24/60；标准正旋转经 frame Q 共轭为 Q R_g Qᵀ
```

Dn 的另一组是绕垂直轴的 180° **正旋转**，不是镜像反射。未给第二轴时使用确定的垂直
方向；给了却不垂直时拒绝。T/O/I 从标准顶点/旋转集合构造，检查元素数量、单位元和
乘法闭包。关系 ID 的组合依据矩阵乘法，不把不同群都当成环形编号取模。

```text
群闭包：对每个 g,h，必须在注册表中找到 k，使 T_g T_h≈T_k
组件稳定子 H⊆G：保持该组件集合不变的变换，允许声明的参与方置换
物理组件副本数 M=|G|/|H|
左陪集 gH 两两不交，且其并集必须等于 G
```

`M` 不整除群阶或找不到所需阶数的子群时，不存在该商轨道候选。
**M 整除 |G| 只是一项必要条件**；还要验证实际 seed 是否具有该稳定子作用。

### 11.3 seed 是否真的具有内部对称性

参与方必须有相同的有序骨架原子签名。将一方坐标 `P` 拟合到另一方 `Q`：

```text
P0=P-mean(P)，Q0=Q-mean(Q)，H=P0ᵀQ0=UΣVᵀ
R=V diag(1,1,det(VUᵀ)) Uᵀ
t=mean(Q)-R mean(P)
RMSD=sqrt(mean_i ||Rp_i+t-q_i||²)
```

默认拟合 RMSD≤.25 Å。拟合得到的变换还要组成闭群：对每个乘积，按旋转误差、平移误差、
索引的字典序选最近元素，要求旋转误差≤2°、平移误差≤.5 Å。

```text
δθ=acos(clip((tr(R1 R2ᵀ)-1)/2,-1,1))
δt=||t1-t2||
共同中心：最小二乘求解堆叠的 (I-R_g)c=t_g
中心残差=sqrt(mean((Ac-b)²))，默认必须≤.1 Å
```

中心不唯一时沿零空间取靠近 seed 质心的解。随后核对乘法表同构、群元素阶数，以及将
拟合旋转系映射到标准子群的共轭 frame Q。frame 的残差是
`max_g ||Q R_standard,g Qᵀ-R_fitted,mapping(g)||_F`，选择残差最小候选且要求≤`.05`；
这是无量纲矩阵 Frobenius 范数，不是 .05°。找不到匹配就拒绝。这些容差是几何识别容差，
不能把两个普通不同片段仅凭“数量相同”当成一个内禀对称组件。

### 11.4 多组件和界面的关联数量

来源：[component_incidence.py](../../src/rfd3_mosaic/topology/component_incidence.py)。
为左右组件分别选择稳定子 `H_L,H_R` 后，将每个 `g∈G` 映射为陪集代表对：

```text
edge(g)=(rep(gH_L),rep(gH_R))
物理边集合=unique{edge(g)}
每条物理边的作用类大小 × 物理边数 = |G|
组件数=|G|/|H_component|
每侧各组件的实际 interface degree 必须一致
```

稳定子阶数和实际界面度数分别报告，不混用。请求界面数不整除群阶、度数不一致、
作用类大小不一致或超出枚举上限时拒绝。组合上成立仍需下游几何与 GPU 验证。

## 12. 连接图、闭合关系与 linker 可行性

**调用范围：**用户声明连接的编译、显式图搜索和接口种子装配；图搜索不是每次运行都自动执行。
来源：[scaffold_graph.py](../../src/rfd3_mosaic/topology/scaffold_graph.py)、
[interface_seed_graph.py](../../src/rfd3_mosaic/topology/interface_seed_graph.py)、
[polymer_path_solver.py](../../src/rfd3_mosaic/topology/polymer_path_solver.py)。

### 12.1 “装配成环”和“肽链成环”是不同图

连续聚合物边是有方向的 C→N 连接；不允许同一端口被不兼容地重复使用，连续链图不能
有分叉或未支持的有向环。显式 chain break 不视为连续肽键。
interface seed 中已有的两侧接触是非共价边，不自动变成需要生成的肽链连接。

将连续连接的片段合并成 protein unit 后，再构造“interface—unit”二部图。
对声明为一个连通 interface-seed 装配的任务，必须满足：

```text
同一个 supplied interface 的两侧不能直接连成一条 scaffold 边；
同一个 supplied interface 的两侧不能经其他连续边归入同一个 protein unit；
interface/unit 图的连通分量数=1；没有结构/映射违规。
```

例如 C3 的装配可以形成非共价闭环，同时每条蛋白质链仍是开放的线性链。
这个检查只作用于声明了相应 seed 装配语义的任务；普通多链固定输入不应被强迫满足它。

### 12.2 连接关系是否覆盖整个对称群

来源：[symmetry_connectivity.py](../../src/rfd3_mosaic/topology/symmetry_connectivity.py)。
从单位元开始，反复左右乘声明关系，直到没有新群元素：

```text
S0={e}，S_{k+1}=S_k ∪ {rs,sr | r∈relations,s∈S_k}
生成整个群的条件：S_final=G
Cn 单一偏移 k：可达副本数=n/gcd(n,k)
```

例如 C6 使用偏移 2 只覆盖 3 个元素，不能单凭这一关系宣称连接整个 C6。
Dn 只有层内旋转关系时也不能连接另一个陪集。不同组件/商轨道的实际连通性还要检查
展开后的图，不能仅用上述生成元检查替代。

### 12.3 拓扑候选如何枚举，何时停止

- 二元 seed 环枚举副本次序与两侧方向，以循环移位和整体反向的字典序最小表达去重。
- 超边配对把每个参与侧恰好使用一次，只允许不同 seed 的侧配对；总侧数必须是偶数。
- 多面 protein unit 路径不允许包含同一个 seed 的两侧，单位长度在用户声明范围内。

设总侧数 `S`，每条 unit 允许 `[m_min,m_max]` 个面，每个 seed 的侧数为 `k_j`：

```text
unit_min=max(max_j k_j, ceil(S/m_max))
unit_max=min(floor(S/m_min), floor(S/2))
unit_min>unit_max ⇒ 无候选
```

普通完整匹配超过 `max_candidates` 会报错，不偷偷返回“完整解集”。多面路径搜索则
有预算截断：每种 unit 数共享候选预算，分配尝试上限为 `20*max_candidates`；触及预算
将 `search_complete=false`。这些候选标记 `executable=false`，只代表组合假设。

### 12.4 linker 长度绑定，而不是偷偷扩大用户范围

来源：[feasibility_restoration.py](../../src/rfd3_mosaic/feasibility_restoration.py)。
第 3 节的 `n_min` 对每个物理副本计算；同一连接取最严格要求。绑定组计算：

```text
l=max_j l_j，u=min_j u_j              # tie group 允许区间交集
r=max_{组内所有物理实例} n_min
n_selected=max(floor((l+u)/2),r)
l>u 或 n_selected>u ⇒ 拒绝
```

区间中点是默认长度偏好，达到轮廓长度下界是几何必要条件。绑定后所有同组连接使用同一
确定长度。结果带出选择策略和每个物理实例的下界；此操作不证明柔性链能避障或折叠。

### 12.5 关系兼容性与图候选排名

来源：[relation_compatibility.py](../../src/rfd3_mosaic/relation_compatibility.py)、
[graph_search.py](../../src/rfd3_mosaic/graph_search.py)。相对变换 `T=(R,t)` 的旋转角
`θ=acos((tr R-1)/2)`，对候选 Cn 偏移 k：

```text
θ_expected=min(2πk/n,2π-2πk/n)
m=n/gcd(n,k)，闭合变换=T^m
score=角度误差/角度容差 + 螺旋平移/平移容差
     + 闭合旋转误差/闭合旋转容差 + 闭合平移误差/闭合平移容差
```

螺旋平移是 `(I-R)c=t` 最小二乘残差在旋转轴上的绝对投影。所有误差先满足各自容差
才入候选；零容差按源码处理为误差≤`1e-12` 时归零，否则无穷大。排名为
`(score,未观测陪集数,群阶,等价偏移)`。单个 120° 关系既可兼容 C3，也可兼容 C6
的子群，所以此工具不自动推断“真实装配就是 C3”。

显式 graph neighbour search 枚举允许群元素的笛卡尔积，自界面不能选择单位元；
组合数量 `∏_edge |options_edge|` 超过上限先报错。候选经编译/严格重放后，按以下元组升序：

```text
(not accepted,失败必需界面数,不可行连接数,必需目标失败数,硬碰撞数,
 未满足输出目标数,-4.5 Å 内界面原子接触数,目标惩罚,
 最大 linker 端点距离,平均界面质心距离,-最小固定组间距,candidate_id)
```

编译异常的候选排在失败项；缺失距离按实现用 ±inf 排序。此排名与第 3 节的普通 pose
排名不同，不能将一种工具的优选标准说成所有入口通用标准。

## 13. 精确对称、耦合噪声、固定目标与柱坐标

**调用范围：**精确 Mosaic sampler；legacy/官方 ASU 复制路径并不采用下面全部规则。
来源：[symmetry_utils.py](../../models/rfd3/src/rfd3/inference/symmetry/symmetry_utils.py)、
[constraint_runtime.py](../../models/rfd3/src/rfd3/inference/symmetry/constraint_runtime.py)、
[joint_projector.py](../../models/rfd3/src/rfd3/inference/symmetry/joint_projector.py)。

### 13.1 轨道投影与噪声

对一组同源原子副本 `x_g`，变换为 `(R_g,t_g)`：

```text
y_bar=mean_g R_gᵀ(x_g-t_g)
P_sym(X)_g=R_g*y_bar+t_g
z∼N(0,I)，ε_g=R_g z                  # 位移不加 t_g
Cov(ε_g,ε_h)=R_g R_hᵀ              # 同一对应原子的副本间协方差
X_initial=X_target+σ_initial*ε，固定原子 ε=0
```

若从非单位元 ASU 噪声起步，先乘其逆旋转还原到标准坐标。此法保留一个副本的噪声尺度，
不是把 n 个独立高斯噪声取平均后导致方差变成 `1/n`。投影在正确群作用和映射下具有
幂等性；错映射不能靠平均补救。

浮点闭合阈值按坐标尺度计算：

```text
numeric_floor=32*eps_dtype*max(sqrt(mean(X²)),1)
effective_tolerance=max(configured_tolerance,numeric_floor)
```

frame 因序列化出现微小非正交时，SVD 极分解投影到最近正旋转；逐元素最大修正超过
`1e-3` 就拒绝。几何状态使用足够工作精度，避免 bf16 量化损坏固定目标及旋转矩阵。

### 13.2 固定约束如何组合

顺序为 `对称投影 → 恢复当前固定目标 → 验证对称闭合 → 可选柱坐标投影 → 再验闭合`。
固定组遮罩必须覆盖每个固定原子，且不能把生成原子纳入固定组。多个组重叠时，重叠
原子的目标必须在声明容差内一致；冲突时报错，不按组输入顺序覆盖。

移动模式更新的是刚体目标 `X_target`，更新被提交时同时刷新模型条件；不能只改输出坐标。
生命周期必须为 `created→running→finalized`，重复初始化或终结后继续更新报错。
最终固定目标检查：

```text
e_i=||X_final,i-X_current_target,i||
RMSD_fixed=sqrt(mean_i e_i²)，max_i e_i≤1e-5 Å
```

这个 `1e-5` 是刚执行坐标恢复之后的内部一致性门槛，不是与最初 seed 的拟合容差。

### 13.3 柱坐标自由度

来源：[cylindrical_projector.py](../../models/rfd3/src/rfd3/inference/symmetry/cylindrical_projector.py)。
给定单位轴 `a`、轴上一点 `c`，按 token 声明保留半径、方位或轴向分量：

```text
v=x-c，z=v·a，r_vec=v-za，r=||r_vec||，u=r_vec/r
x'=c+r_selected*u_selected+z_selected*a
```

锁定的分量取 reference，其余取当前状态。参考点在轴上却要求锁方位时拒绝；当前点
落轴时方位采用参考方向回退，数值零判断用 `32*eps_dtype`。这不是整个 seed 的刚体
拟合，不能把逐 token 柱坐标约束误称为保持所有内部距离。

记录误差是半径绝对误差、轴向绝对误差、方位单位向量差范数的最大值；最后一项无量纲，
不能把这个混合 `maximum_error` 全部称为 Å。runtime 默认门槛 `1e-5`；柱坐标审计还
检查声明 ID、原子键、保留分量及初始化/终结次数，并明确依赖 runtime 记录。

## 14. 初始摆放、独立 pose 搜索及多样性工具

**调用范围：**初始 pose 声明、显式准备/搜索命令。它们不会在每个 design 内重新随机选起点。
来源：[compile.py](../../src/rfd3_mosaic/compile.py)、
[pose_optimizer.py](../../src/rfd3_mosaic/pose_optimizer.py)、
[pose_select.py](../../src/rfd3_mosaic/pose_select.py)、
[pose_qd.py](../../src/rfd3_mosaic/pose_qd.py)。

### 14.1 均匀旋转和轴向锥采样

对于独立均匀数 `u1,u2,u3∈[0,1)`，四元数按 `(x,y,z,w)` 为：

```text
q=(sqrt(1-u1) sin(2πu2), sqrt(1-u1) cos(2πu2),
   sqrt(u1) sin(2πu3),   sqrt(u1) cos(2πu3))
```

由单位四元数转换到 SO(3)；它表达均匀旋转，不是独立均匀抽三个 Euler 角。
轴向锥模式用 `cos θ=1-u1*(1-cos θ_max)`、方位 `φ=2πu2` 采样锥内方向，再用
`2πu3` 绕该方向旋转。即使 `θ_max=0`，仍有绕轴 roll 多样性。半径/轴向范围按
`v=min+u*(max-min)` 取样；这不意味着三维体积均匀。

### 14.2 自动全局起点与局部优化

独立全局候选生成器测量组件重原子质心及最大半径 `r_component`，设
`chord=2*max(r_component)+8 Å`、物理槽位数 `N=|G|*组件数`：

```text
环状初始半径 R=chord/(2 sin(π/max(N,2)))
多面体起点 R=chord/(2 sin(min(π,sqrt(4π/max(N,2)))/2))
没有显式直径范围时，R 截到 [12,120] Å 后乘预设 initial_radius_scale
显式直径范围 [Dmin,Dmax] 时，R=(Dmin+Dmax)/4
```

多面体公式来自等面积角间距近似；它不是 packing 最优半径。随后以黄金比例序列、
不同轴向偏移、倾角和 roll 生成确定候选。Dn 赤道起点额外加非零层偏移，以免落在
二重轴稳定子上重合；有显式稳定子/陪集而无对应全局放置 frame 的输入会报不支持。

这个序列的常数属于搜索启发式，不能只用“多样性采样”略过其实际构造。令样本索引
s、组件索引 j 都从 0 开始，组件数 m，`g=.6180339887498949`，`frac(x)=x-floor(x)`：

```text
R_s=R*(1+.4*(frac(s*g)-.5))
slot=(Cn/Dn 时 360°/n，否则 360°)/m
phase=(frac((s+1)*g)-.5)*slot
tilt=(-24°,-12°,0°,12°,24°)[(s+2j) mod 5]
roll=((137s+71j) mod 360)°-180°
Cn/Dn：azimuth=j*slot+phase
       h_j=min(8 Å,.35*r_component,j)，z=(((s+j) mod 3)-1)*h_j
       Dn 且 |z|≤1e-8 时，z=(s+j 偶数时 +.5，否则 -.5)*h_j
T/O/I：q=s*m+j，u=2*clip(frac((q+.5)*g),.05,.95)-1
       azimuth=360°*frac((q+1)*g)，径向半径=R_s*sqrt(1-u²)，z=R_s*u
```

这使名义半径波动约 ±20%，并不保证不同候选经后续优化仍不同；最终仍需编译可行性
检查和第 14.3 节的去重/分层。主轴定向用中心化坐标协方差 `XᵀX/N` 的最大特征向量；
若最大特征值≤`1e-12` 或最大与次大之差≤`1e-8*max(最大特征值,1)`，编译器将主轴视为
退化并返回不可用。非退化轴令绝对值最大分量为正，以固定符号，不引入额外生物学方向。

`optimize_design_poses` 对半径、方位、轴向和三旋转角做有界直接搜索，层级 l 的
平移/旋转步长是初始步长除以 `2^l`。先尝试多组件一起变化，再逐组件逐变量尝试。
实际位移和相对旋转必须在总边界内，且第 3 节的 score **字典序严格改善**才接受。
不是对那个元组求梯度，也不保证找到全局最优起点。

### 14.3 三种多样性不能混用

普通 `prepare-poses` 使用第 3 节的跨刚体距离谱。独立 SO(3) shortlist 工具则使用：

```text
d_SO3(q1,q2)=2 acos(clip(abs(q1·q2)/(||q1||||q2||),0,1))
```

对可行排名前 `pool_size` 的候选贪心选择，要求与所有已选候选角距离≥阈值；多组姿态
需明确指定比较组。`q` 与 `-q` 等价。该距离忽略位置，不是全装配形状距离。

质量—多样性工具先保留 `max(1,ceil(N_feasible*quality_pool_fraction))` 个排名靠前
候选，再按声明形态指标分箱：`[edge_i,edge_{i+1})`，最后一箱包含右端点。
缺指标或超范围的候选跳过，按每箱配额轮流保留，并应用可选 SO(3) 间距和 pose seed
去重。它不自动进入普通 `run`，也不保证扩散后的结构多样性。

## 15. 刚体提案：从模型预测或几何梯度到有限步长

**调用范围：**启用可移动 orbit 且位于其活动窗口；locked 不提交移动。
来源：[motif_mobility.py](../../models/rfd3/src/rfd3/inference/symmetry/motif_mobility.py)、
[scaffold_guidance.py](../../models/rfd3/src/rfd3/inference/symmetry/scaffold_guidance.py)。

### 15.1 提案时机与响应

设采样步数 N、配置间隔 k、希望的更新次数 U：

```text
k_effective=1                         denoiser 每步提案路径
            min(k,max(1,floor(N/U)))   U>0 且非每步路径
            k                         其他
提案步：0,k_effective,2*k_effective,...；p=step/max(N-1,1)
窗口内 u=(p-a)/(b-a)，W_motion=sin²(πu)；p≤a 或 p≥b 时为 0
```

U 是调度目标，不保证实际提交 U 次；还受活动窗口和候选拒绝影响。
窗口内默认 `u<.4` 为 capture、`.4≤u<.8` 为 settle、其余 polish，尺度分别 5/2.5/1：

```text
capture_response=min(1,base_response*5)
phase_response=capture_response*phase_scale/5
实际提案步长还乘 W_motion，并受每步及累计移动上限限制
```

这与第 6 节按接触状态划分的 packing capture/expand/polish 是**两种阶段系统**。
联合路径可进一步依据 packing 阶段限制响应，不能把同名 capture 当成同一触发条件。

### 15.2 模型提案的刚体拟合

将各副本预测逆变换并平均到代表 frame，再用第 11 节 Kabsch 拟合模板。
这里采用以模板质心 c 为中心的参数化：

```text
x'=(R(x-c))+c+t，t=预测质心-模板质心
```

只把拟合后的旋转和平移应用于完整固定组，不能把预测中的内部形变直接写回 seed。
拟合 RMSD 表示模型预测与可实现刚体运动的差异，不是生成质量分数。

### 15.3 几何目标的 SE(3) 提案

允许方向基底用 QR 正交化为列矩阵 Q，向量投影 `P_B(v)=QQᵀv`；空基底返回零。
对旋转向量 ω 和平移 t 的负梯度分别投影、单位化，再乘指定步长。使用
`R'=exp([δω]×)R`、`t'=t+δt`，并限制相对初始姿态的总旋转角和总平移范数。

```text
clip_norm(v,M)=v*min(1,M/max(||v||,epsilon))
relative_angle(R)=acos(clip((tr R-1)/2,-1,1))
```

以主轴 a、组件中心至轴的单位径向 r 为基础：

| 子空间 | 平移 | 旋转 |
|---|---|---|
| radial | span(r) | 禁止 |
| radial_axial | span(r,a) | 禁止 |
| radial_rotation | span(r) | 按声明旋转上限允许 |
| radial_axial_rotation | span(r,a) | 按声明旋转上限允许 |
| tilt_only | 禁止 | 无穷小旋转轴限于 a 的垂直平面 |
| bounded_se3 | 三维 | 三维 |

组件中心在轴上而需要径向基底时拒绝。`tilt_only` 限制的是每步无穷小旋转方向，不能将
有限旋转的非交换组合误解为任意长度轨迹都等价于一个“零 twist”的单次旋转。
姿态先验尺度也跟随允许范围：`translation_scale=max(config_scale,max_translation/3)`；
旋转先验角尺度取 `max(config_scale,max_rotation/3)`。

### 15.4 搜索、近优随机选择与回退

普通 SE(3) 提案默认依次试 `1,.5,.25` 步长，选第一个有限且严格降低目标的候选。
早期 multistart 还枚举允许基轴正负方向的平移、旋转及按索引配对的联合方向；
不是遍历全部方向组合，也不接受升高当前目标的候选。

```text
gain_i=E_before-E_i>0，best_gain=max_i gain_i
near_optimal={i : gain_i≥.75*best_gain}
```

有选择 seed 时，从 near_optimal 中可复现地均匀抽一个；否则取最低分。
这保留局部提案差异，不保证不同设计不会最终收敛。SE(3) 提案通过之后还要经过
第 7 节联合事务的 graph/联合总分及保护条件；只有最终 `committed=true` 才改变运行状态。

## 16. 界面容量、片段选择和梯度移动细节

**调用范围：**编译得到的 generated-interface 拓扑；不把完整 fixed interface 当成待重设计片段。
来源：[graph_interface_guidance.py](../../models/rfd3/src/rfd3/inference/symmetry/graph_interface_guidance.py)。

### 16.1 为什么要求这些接触残基数

自动质量路径取 `n=min(N_left,N_right)`：

```text
coverage=min(n,min(12,max(3,ceil(sqrt(n)))))
continuity=min(coverage,max(2,ceil(.6*coverage)))
```

显式每侧残基数优先；只给接触对数 m 时，覆盖默认来自
`min(N_left,N_right,max(2,ceil(sqrt(m))))`。连续数量显式给定时采用该值，否则还受
两侧实际最长连续序列容量限制。平方根、上限 12 和 .6 比例是工程规则，不是普适界面面积定律。

扩散前检查 `coverage≤min(N_left,N_right)`、连续目标≤两侧最长连续容量、
`contact_count≤N_left*N_right`。把竞争同一残基池的参与方连成重叠分量，要求：

```text
sum_{分量内参与方} requested_coverage ≤ |union(可用 token 集合)|
```

不满足则提前拒绝。**通过该并集容量检查不是全局匹配可解的证明**：它没有遍历所有子集
的容量约束，也没有解决坐标上能否同时接触的问题。

### 16.2 片段是怎么选出来的

枚举左右两侧合法的连续 token 窗口，对每对窗口计算其距离子矩阵。设最近距离的
软占用 `o_i=sigmoid((D-r_i)/softness)`：

```text
deficit(m,occupied)=([m-occupied]+/m)²
coverage_score=.5*(左右各自 top-m 软占用之和的 deficit)
continuity_score=.5*(左右各自最佳连续 m 窗口软占用之和的 deficit)
attraction_score=mean H_1([选中最短接触对距离-D]+)
selection_score=attraction_score+coverage_score+continuity_score
```

取 `argmin(selection_score)` 的窗口对；同分按展开枚举次序取首个。这个离散选择分数
**不是第 6 节完整加权 E_graph**，没有在选择阶段加入方向、shape 等全部目标。
选择使用 detached 距离；选定后从原张量重新取片段，连续损失可求导，离散 argmin 本身不求导。
一轮 line search 不重新挑片段，避免靠换一组残基伪造分数改善；跨步未锁定前可更新身份。

### 16.3 梯度怎样转成坐标变化

每个生成 token 的平移梯度为其原子梯度的**和**，不是平均：

```text
g_token=sum_{该 token 原子 i} ∂E/∂x_i
g_smooth=(1-λ)*g_token+λ*mean(g_token,同链相邻已选择 token 梯度)
```

默认平滑 λ=.5、一次；不越过固定空隙或不同链。非零梯度过小且尚未满足代理目标时，
可用 `boost=desired_step/max_raw_step` 放大，`desired_step=.5*阶段 token 上限`；
只在 `1e-8<max_raw_step<desired_step` 时启用，仍受位移上限约束。

片段刚体拟合将负梯度 d_i 投影到平移加绕质心旋转的速度场：

```text
c=mean(x_i)，r_i=x_i-c，v=mean(d_i)
I_patch=sum_i (||r_i||² I-r_i r_iᵀ)
τ=sum_i r_i×(d_i-v)，ω=pinv(I_patch)*τ
```

点数不足或惯量退化时旋转为零。v、ω 乘窗口和 boost 后，旋转限幅；
片段外同链邻居按序列距离 k 混合：`blend=1-k/(blend_radius+1)`，仅 k≤半径参与。
默认半径 2，片段内部 blend=1。对同一片段用共享缩放约束原子最大位移，最多三轮
缩放；重叠片段对同一 token 的提案取平均，再与局部平移按 `patch_rigid_weight` 混合。
因此“刚体片段提案”不等于整个生成区域最终都是刚体；只有固定 seed 有刚体保真合同。
最后仍按第 6 节投影、回溯及几何 guard 接受，梯度方向本身不构成通过证明。

## 17. 初始化、点/线段距离和末端连续性修正

### 17.1 生成区域从哪里开始

来源：[input_parsing.py](../../models/rfd3/src/rfd3/inference/input_parsing.py)。
编译器指定 `local_fixed_anchor` 时，生成残基用本链固定锚点初始化：双侧锚点之间
按残基编号线性插值，单侧锚点则使用该侧最近锚点；没有固定锚点的链保持原有默认。
这提供每个对称副本自己的局部起点，不是预先生成一条正确折叠的 backbone。
后续仍叠加第 13 节的扩散噪声。

独立的旧式 compactness 引导只有显式权重>0、存在本链固定锚点且处于其窗口时才执行：

```text
Δ_token=clip_norm(weight*(1-p/end_fraction)*(c_anchor-c_token),max_step)
```

它移动生成 token，不移动固定原子；这与第 8 节核心接触损失是不同机制。
仅凭模块存在不能声称当前任务启用了它。

### 17.2 点到线段与线段到线段距离

来源：[scaffold_core_guidance.py](../../models/rfd3/src/rfd3/inference/symmetry/scaffold_core_guidance.py)、
[scaffold_validity.py](../../src/rfd3_mosaic/validation/scaffold_validity.py)。

```text
点 p 到线段 [a,b]：u=clip((p-a)·(b-a)/max(||b-a||²,eps),0,1)
d=||p-(a+u(b-a))||
两线段：p(s)=p0+s*u，q(t)=q0+t*v，w=p0-q0
A=u·u，B=u·v，C=v·v，D=u·w，E=v·w，Δ=AC-B²
s=(BE-CD)/Δ，t=(AE-BD)/Δ
```

仅当 Δ 大于数值阈值且 `0≤s,t≤1` 时加入内部最近点候选；另取四个端点到对侧线段
的距离，最小值为有限线段距离。近乎平行或零长度线段退回端点候选。

运行保护使用完整有限线段距离；最终 scaffold 审计把“近距离”与“内部相交风险”分开：
只有内部候选有效且内部距离小于 `ca_segment_collision_distance`（默认 1 Å）才计入
跨链拓扑碰撞。端点邻近会进入其他距离提示。因此它不是严格的结/链接数检测，也不是
连续时间穿越检测。不同模块的 eps 和阈值应分别读取配置，不能混成一个“无穿插证书”。

### 17.3 最终连续性投影

仅末端启用，先测 `|d_i-d0|`，全部≤容差就原样返回。否则先做最多四轮定向锚点传播，
再将相邻边按奇偶分两组交替修正。设 `u=(x_right-x_left)/d`、`e=d-d0`：

```text
两端可动：Δ_left=relaxation*e*u/2，Δ_right=-relaxation*e*u/2
仅左端可动：Δ_left=relaxation*e*u
仅右端可动：Δ_right=-relaxation*e*u
固定 token 位移为零；一个 token 的所有原子使用同一平移
```

仅超出容差的边修正；分色避免同半轮对同一 token 写入冲突。默认目标 3.8 Å、容差 .5 Å，
迭代上限 64；完成后应用调用方投影和几何 guard。达到上限仍不满足时如实报告，
guard 拒绝则保留修正前坐标。修好 Cα 步长不意味着 N–C 肽键、键角或二面角已正确。

## 18. 用户目标、功能几何和形态指标

**调用范围：**显式目标评估、候选排序、对应输出审计；声明一项几何检查不自动创建
扩散中的力或引导项，必须有 compiler/sampler 对应支持。
来源：[objectives/core.py](../../src/rfd3_mosaic/objectives/core.py)、
[functional_geometry.py](../../src/rfd3_mosaic/functional_geometry.py)、
[assembly_morphology.py](../../src/rfd3_mosaic/validation/assembly_morphology.py)。

### 18.1 通用标量目标

指标 v，正尺度 s、权重 w；不存在指标或非有限值时报错：

| 模式 | 未加权 penalty | 满足条件 |
|---|---|---|
| minimize | v/s | 仅优化，没有 satisfied 布尔结论 |
| maximize | -v/s | 同上 |
| at_most | `([v-threshold]+/s)²` | v≤threshold |
| at_least | `([threshold-v]+/s)²` | v≥threshold |
| target | `([abs(v-target)-tolerance]+/s)²` | 在目标容差内 |
| range | `max(minimum-v,0,v-maximum)²/s²` | minimum≤v≤maximum |

总目标惩罚 `Σ w_j penalty_j`；required 失败数是 required 且不满足的项数。
排名先比较 required 失败数，再比总惩罚。scale 改变相对影响，不能忽略它只比较 weight。

### 18.2 功能几何关系

```text
距离 d=||x1-x2||
角 θ=acos(clip((x1-x2)·(x3-x2)/(||x1-x2||||x3-x2||),-1,1))
二面角：u=normalize(x3-x2)，v=(x1-x2)-((x1-x2)·u)u
        w=(x4-x3)-((x4-x3)·u)u
        φ=atan2((u×v)·w,v·w)
周期角误差=abs((observed-target+180) mod 360-180)
归一违约 v=[error-tolerance]+/max(tolerance,1e-6)
```

角与二面角输出为度，零长度或共线导致未定义时拒绝。普通角误差不做二面角式周期折返。
关系通过条件为其各归一违约最大值≤`1e-12`。

手性用标量三重积 `V=(x1-c)·((x2-c)×(x3-c))`，按期望符号变成 signed V，
要求 signed V≥最小绝对体积；违约除以 `max(minimum_abs_volume,1)`。这里没有除以 6，
不是通常四面体体积数值。

相对姿态用 `T_obs=T_first⁻¹ T_second`，误差变换 `T_target⁻¹ T_obs`，分别检查
平移范数和旋转角。配位几何检查每个配体到中心的距离和每对配体夹角；理想角集合为
线性 180°、三角平面 120°、四面体约 109.471°、方平面/八面体 90°或180°。
每个观测角取到理想集合最近的误差；这种局部角度检查不等于完整配位化学或唯一构型识别。

### 18.3 孔径、外径和轴向跨度

先从声明群作用解共同固定点 `(I-R_g)c=t_g`，识别旋转轴及其阶数。默认取 Cα 坐标：

```text
z_i=(x_i-c)·a，r_i=||(x_i-c)-z_i*a||，ρ_i=||x_i-c||
轴向跨度=max(z)-min(z)
central_pore_diameter=2 min(r)
outer_radial_diameter=2 max(r)
p05 孔径=2 percentile(r,5)，p95 外径=2 percentile(r,95)
spherical_inner/outer_diameter=2 min/max(ρ)
```

只有最高阶轴唯一时才填唯一主轴孔径；D2/T/O/I 等存在多个等价最高阶轴时保留逐轴值，
不任选一个冒充唯一轴。球形内外径要求固定点方程 rank=3；Cn 中心沿轴不唯一，球形项留空。
显式要求一个在当前对称群下不可用的指标会失败。

这些是点坐标的几何包络，没有扣原子范德华半径，也不是溶剂探针可通行孔径。
有用户范围时按包含边界的区间判定；未声明范围时为 measurement-only。
极值指标和 p05/p95 是不同字段，不能用较好看的分位数替代用户声明的极值合同。

## 19. 最终审计的计算对象、对齐方式与阈值

**调用范围：**生成后或显式 post-hoc audit。审计报告中的阈值是该次实际值；以下为
函数默认和计算定义，不代表通用蛋白质量标准。来源为 `src/rfd3_mosaic/validation/`
及 `rfd3_*_audit.py`；最终合同/建议映射见第 9 节。

### 19.1 三种误差不可混称 RMSD

```text
坐标 RMSD=sqrt(mean_i ||x_i-y_i||²)
拟合 RMSD=min_{R,t} sqrt(mean_i ||Rx_i+t-y_i||²)
距离矩阵 RMSD=sqrt(mean_{i,j}(||x_i-x_j||-||y_i-y_j||)²)
```

固定 seed 内部保真可以用拟合/内部距离；绝对位置或对称复制需检查已声明 frame 下的
坐标误差，不能用独立对齐隐藏错误摆放。距离矩阵本身也不能区分镜像。

固定完整 orbit 与参考做整体拟合；移动 orbit 分别检查每个副本内部刚体保真，再检查
移动范围。默认原子完整率 `matched/expected≥.99`、相应接受 RMSD≤.5 Å。
中心固定 motif 还有独立坐标误差门槛，不能仅凭这 .5 Å 就判其绝对位置正确。

**商轨道的特殊分支：**固定 quotient 在完整 runtime 固定目标记录有效时，使用直接
runtime target 误差接受，旧式重建坐标的误差另列为 legacy 诊断；此分支的
`distance_matrix_rmsd=0` 是实现中的占位，不是重新测得的零误差。必须连同
`acceptance_reference` 读取，不能拿这个 0 当额外科学证据。

### 19.2 seed 接触保持及配对

```text
reference_contacts={(i,j):d_ref,ij≤cutoff}
retention=|reference_contacts 中 d_out,ij≤cutoff 的对|/|reference_contacts|
contact_distance_RMSE=sqrt(mean_reference_contacts (d_out-d_ref)²)
```

无参考接触时实现返回 retention=0、距离 RMSE=inf，不能解释成保持率 100%。
原子按链、残基和原子名映射，缺失原子进入完整率。需要跨链匹配的历史 seed 审计用
匈牙利分配最小化 `Σ CA_RMSD(left,assigned_right)`，同链配对成本设无穷大，要求一一配对。
配对搜索发现一个低 RMSD 组合不证明它就是编译声明的连接；声明关系审计需单独核查。

### 19.3 主链、碰撞、紧凑度和对称

- 残基按链内编号/插入码排序；检查 N/CA/C 完整性。只有 CA 的输入不等于已证明全骨架完整。
- 相邻连续编号的 C–N 默认参考区间 [1,2] Å；这是肽键几何提示。相邻 Cα 步长上限默认
  4.5 Å，编号断档等也记录连续性失败。
- `Rg=sqrt(mean_i ||x_i-mean(x)||²)`，默认逐链 Cα Rg≤25 Å 是建议，不能跨所有长度使用。
- 原子碰撞提示用非相邻 Cα 距离<3 Å；此审计排除同链列表索引差≤2，运行核心损失通常
  排除序列差≤1，**两者排除集合不同**。
- 跨链线段检查见第 17 节；没有线段内部近交并不意味着完整拓扑无缠绕。
- 同一实体的对应链比较内部距离矩阵，并按 `T_observed*T_reference⁻¹` 预测对应副本坐标；
  不额外逐链自由拟合。默认 RMSD≤.01 Å 且最大误差≤.03 Å。链对应使用编译布局和实际输出
  顺序，不能将 AA、B 等链名重新按字母排序。

### 19.4 移动、引导执行和日志一致性

位移审计比较每步/累计 `||t||` 和旋转角与各自声明上限，并核对最后轨迹、更新计数和模式。
限制子空间还分解：`t_axial=t·a`、`t_radial=t·r`、
`t_tangential=t-t_axial*a-t_radial*r`。径向模式要求禁止的分量在容差内；缺轴、缺向量或
轴上退化不能默认为通过。

graph/core 审计分别检查：是否声明并启用、配置/界面 ID 是否一致、是否有有限的逐步
诊断、最终指标是否来自最终坐标，以及质量代理是否达到阈值。`measurement_only`
不要求坐标发生更新；所有候选合法被拒绝也不同于“没有执行证据”。
最终结构分项由重新评价生成，不能用最后一次提案的修正前指标冒充最终分数。

### 19.5 批量比较和队列筛选

来源：[backbone_comparison.py](../../src/rfd3_mosaic/backbone_comparison.py)、
[rfd3_batch_screen.py](../../src/rfd3_mosaic/rfd3_batch_screen.py)。
分布摘要仅对有限的已测量值计算 min/p05/median/p95/max/mean；缺失不是零，也不是通过。
比较工具的特定队列筛选使用同队列中链 A 的 carbonyl-C Rg、loop 比例和最长连续 loop，
要求三者**严格小于**各自中位数；缺二级结构指标时，只能报告 Rg-only 分支。
这不是第 9 节单结构的绝对门槛，也不是折叠成功率；不能用 Cα Rg 偷换 carbonyl-C Rg。

报告必须区分 requested、generated、audited、contract-met、proxy-reached 等计数。
若计算比例，应明确分母是请求数、已生成数还是已审计数；排队/失败/缺审计的结构不能
凭缺失记录从请求分母消失。代码中没有独立回折叠证据时，不把骨架筛选率称为 designability。

### 19.6 声明界面与重原子 packing 的独立审计

来源：[rfd3_interface_relation_audit.py](../../src/rfd3_mosaic/rfd3_interface_relation_audit.py)。
`reference_transform` 先分别拟合左右参考原子到输出，得到 `T_L=(R_L,t_L)` 和
`T_R=(R_R,t_R)`；每侧至少需要 3 个匹配原子。使用本文列向量记法：

```text
e_translation=||R_L*c_right,reference+t_L-c_right,output||
e_rotation=acos(clip((tr(R_LᵀR_R)-1)/2,-1,1))，换算为度
contact_count=Σ_ij 1[d_ij<contact_cutoff]
hard_clashes=Σ_ij 1[d_ij<2 Å]
```

该分支接受要求：完整率达标、两个误差分别不超过声明容差、接触数达到声明值、
`hard_clashes=0`。最小接触数默认 0，接触 cutoff 默认 4.5 Å。左右内部拟合 RMSD
另外报告，不作为这两个相对变换门槛的替代；内部刚体保真由对应 seed 审计负责。

`geometric_constraints` 对声明的距离检查 `|两侧质心距离-target|≤tolerance`，
对 contacts 检查 `count≥max(1,declared_minimum)`。至少存在一个有效检查，所有声明
检查通过、完整率达标且无上述硬碰撞才满足该关系。`satisfaction_stage=output` 时，
这一分支使用对应输出链的生成区域重原子，并排除映射到固定 seed 的残基；不能把固定
seed 原本已有的接触充当新生成界面的成绩。最终合同聚合只要求 required 关系满足。

重原子 packing 描述符按**唯一残基对**与**原子对**分别计算。令接触 cutoff 为 c，
`P={(r,s):至少有一对属于这两个残基的重原子距离<c}`，L/R 为参与接触的左右残基集合：

```text
reciprocal_contact_density=|P|/max(|L|,|R|,1)
depth(r)=c-min_{r 内原子 a，对侧原子 b} d_ab，仅统计接触残基
contact_depth_mean=mean(depth)，standard_deviation=总体标准差(depth)
heavy_atom_burial_proxy=Σ_ab clip((c-d_ab)/max(c-2,1),0,1)
P_loose={(r,s)∈L×R:至少一对重原子距离<c+1.5 Å}
local_contact_void_fraction_proxy=1-|P_loose|/max(|L|*|R|,1)
hydrophobic_contact_residue_fraction_proxy=|H∩(L∪R)|/max(|L∪R|,1)
unpaired_hydrophobic_residue_fraction_proxy=|H\(L∪R)|/max(|A|,1)
```

A 是两侧全部可评价残基，H 是其中残基名属于
`ALA/VAL/ILE/LEU/MET/PHE/TRP/TYR/PRO` 的集合。contact islands 是同链连续残基编号的
片段数。无接触时 depth 均值和标准差填 0，void proxy 为 1；不能把 depth 标准差为零
解释成 packing 均匀。burial/void/hydrophobic 都是几何或残基分类代理，没有计算 SASA、
真实空腔体积、溶剂可达性或疏水自由能。

此处报告的代理束为：

```text
|P|≥minimum_coverage
且 reciprocal_contact_density≥1
且 depth_std≤max(2 Å,0.45*c)
```

`controller_proxy_bundle_satisfied` 及其历史别名 `output_packing_quality_satisfied`
**不是此函数的最终接受门槛**。`maximum_contact_islands_per_side=
max(1,minimum_coverage-minimum_contiguous+1)` 也只是报告参考值，未加入这个代理束。
只有用户显式声明的 coverage/contiguous 目标才将对应覆盖检查加入关系接受条件；
自动推导目标标为 `measurement_only` / `controller_reference_only`。

### 19.7 环形均匀度及邻链接触描述符

来源：[rfd3_batch_screen.py](../../src/rfd3_mosaic/rfd3_batch_screen.py)，用于批量诊断。
要求输出链数与预期阶数相符且每链有 Cα。以各链 Cα 质心 `c_i` 的平均 c 为中心，
对 `c_i-c` 做 SVD，前两个右奇异向量张成拟合平面，最后一个作为轴 a。
平面投影的极坐标为 `(r_i,θ_i)`，角度排序后计算含首尾闭合的角间隔 `Δθ_i`：

```text
chain_com_radial_cv=std(r_i)/mean(r_i)
chain_com_axial_rms=sqrt(mean_i ((c_i-c)·a)²)
angular_gap_rms_error=sqrt(mean_i (Δθ_i-2π/n)²)，报告单位为度
angular_gap_max_error=max_i |Δθ_i-2π/n|，报告单位为度
```

对全部 Cα，计算到此轴的距离 ρ 和轴向 z：最小 ρ 是 axis clearance、最大 ρ 是
axis extent；厚度比例 `(maxρ-minρ)/maxρ`，高宽比 `(maxz-minz)/(2maxρ)`。
分母均值/最大半径≤`1e-8` 时相关比值为缺失值。拟合轴是描述用轴，不等于已声明的
对称轴；退化或近退化的质心分布可能使它不稳定，不能作为精确群对称证明。

按角度相邻（含首尾）的链对定义邻链，去掉重复无序对。每对计 Cα 距离<8 Å 的数量，
报告 min/mean/max、`std(count)/mean(count)`、`mean(count)/mean(chain_length)`，
另报非邻链接触总数与最小链间距离。没有接触时 CV 为缺失值；非邻链接触也不能直接
等同于错误连接，必须回到任务声明的图判断。

### 19.8 可视化对齐的共识选择不等于保真审计

来源：[pymol_fixed_orbit_alignment.py](../../scripts/rfd3_mosaic/pymol_fixed_orbit_alignment.py)。
这是 PyMOL 辅助工具，不参与生成接受。结构匹配候选默认内部 RMSD 限制 2 Å；对每个
候选刚体变换，每个片段选择变换后 RMSD 最低的匹配。设片段残差 e_f、原子数 n_f，
默认共识门槛 τ=1.5 Å：

```text
inlier_atoms=Σ_f n_f * 1[e_f≤τ]
robust_error=Σ_f n_f * min(e_f,2τ)²
选择字典序最小的 (-inlier_atoms,robust_error,提案内部 RMSD)
```

再迭代最多 3 次重新选匹配、取共识片段并整体 Kabsch 拟合；最后对共识集合再拟合，
无共识时报错。`mosaic_align_fixed` 则使用固定区域的明确来源映射。显示时可忽略不属于
共识的移动片段，因此“图叠得很好”不能替代检查所有固定原子或所有界面合同。

## 20. 局部邻域推理、跨链 attention 与数值精度

**调用范围：**显式选择局部 backend，或网络进入对应多链稀疏 attention 分支；不是每项
任务都启用。来源：[local_neighbourhood.py](../../models/rfd3/src/rfd3/inference/symmetry/local_neighbourhood.py)、
[RFD3.py](../../models/rfd3/src/rfd3/model/RFD3.py)、
[block_utils.py](../../models/rfd3/src/rfd3/model/layers/block_utils.py)。

### 20.1 局部副本集合及展开

Cn 以主副本 k 和邻域半径 h 取 `(k+δ) mod n`，其中
`δ∈{0,-1,+1,...,-h,+h}`，重复项去重。Dn 可同时取两个陪集相应邻域。
它依据群索引邻接，**不是按当前欧氏距离动态选择最近所有接触方**。

原子/token/配对特征在 TokenInitializer 之前一起裁剪，用双向索引映射保持一致。
局部预测展开到全装配时：

```text
y_bar_local=mean_{g 在已计算邻域中} R_gᵀ(x_pred,g-t_g)
x_full,g=R_g*y_bar_local+t_g
```

未计算副本不进入平均，不会把 3 个副本的更新除以整个群阶。标量 token 预测按对应
轨道映射聚合，不对标量施加坐标旋转。

当前局部路径只支持 Cn/Dn，要求 `orbit_average`、`coupled` 噪声、固定 motif 保留、
`low_memory_mode=True`，并拒绝动态 motif mobility。原子/token 映射不匹配、把一个 token
切成部分原子、固定目标张量形状错误或未覆盖某个全局轨道时会报错。特征裁剪按已知特殊
字段及张量首维/前两维长度推断；未命中规则的张量原样传递，**并非对所有未知特征主动报错**。
因此新增特征布局仍需检查其裁剪语义。局部展开保持对称不等于与全副本网络推理等价：邻域外相互作用
可能没有进入模型输入，质量/速度/显存需要分别评价。

### 20.2 跨链注意力邻居

当该 attention 分支的链数>3 时，实际邻居数为 K：

```text
K_inter=min(max(32,floor(K/4)),max(K-1,1))
K_intra=K-K_inter
eligible(i)={j : chain(j)≠chain(i) 且 base_mask(i,j)}
```

对每个查询原子 i，从 eligible 中取距离最小的 `min(K_inter,|eligible|)` 个。
不足时重复已选合法 key；没有合法跨链 key 时用自身索引回退。
相对上游的修改是逐查询原子取邻居，修复以链编号误索引原子行的问题，并限制小输入的
邻居配额。邻居注意力可以传递跨链接触信息，但不等于有显式防碰撞约束。

### 20.3 数值精度和加权对齐

来源：[engine.py](../../models/rfd3/src/rfd3/engine.py)、
[alignment.py](../../src/foundry/utils/alignment.py)。精确 orbit 运行要求相应的真精度配置，
设备搬运前保存几何状态、之后恢复坐标/旋转/固定目标精度。加权对齐使用：

```text
c_P=Σw_i p_i/Σw_i，c_Q=Σw_i q_i/Σw_i
H=Σw_i (p_i-c_P)(q_i-c_Q)ᵀ
```

再用第 11 节带反射修正的 SVD 求解。对齐阶段禁用外层 autocast，工作精度至少 float32，
任一输入为 float64 则保留 float64，返回时按调用约定转换 dtype。
这是兼容性和精度修正，不是新的结合能或新的训练损失。

## 21. 输入声明、执行和报告中的非能量判定

### 21.1 编译和参数解析

schema/selector 检查链与残基范围、组件/端口 ID、非空原子集合、长度上下界、权重非负、
活动窗口先后顺序、自由度合法组合及冲突声明。不支持的组合应明确报错；字段存在不等于
当前 sampler 已实现它。interface 保留/新建、链内/链间和 required/advisory 的语义
通过编译计划传递，不能仅从某个任务标题猜测。

原子映射使用输入链/残基/插入码/原子名与输出实例身份，生成与固定遮罩、序列身份、
对称 slot 和连接端点必须一致。显式多组件作用还检查旋转后的原子签名和坐标匹配；
不能以“最近的一颗原子”替代用户声明的同源原子。低层 parser 的数据结构合法性判断
服务于这些映射，不另定义一套几何能量。

### 21.2 同任务采样、分片和复现

来源：[sampling_plan.py](../../src/rfd3_mosaic/sampling_plan.py)、
[experiment_worker.py](../../src/rfd3_mosaic/experiment_worker.py)、
[provenance/](../../src/rfd3_mosaic/provenance/)。

```text
design i：diffusion_seed=base_seed+i，i=0..designs-1
同一任务全部 i 使用同一个已实现的 input pose
分片应保留原设计索引/种子及冻结坐标，不重新采样 pose
```

编译重放比较固定输入和映射；`prepare-poses` 冻结输出重新编译的逐坐标最大误差必须
≤`1e-5 Å`。源码、输入、配置保存 SHA-256 与版本信息；hash 一致证明文件字节一致，
不证明数值算法正确，也不保证不同 CUDA/硬件环境严格逐位复现。
worker 复用任务内模型加载，逐 design 重新设置扩散 seed；初始 pose 的随机源和扩散
随机源是不同角色。旧 `replicates_per_pose` 与现语义冲突时拒绝，不能静默换实验定义。

### 21.3 状态机、审计聚合与 benchmark

direct/Slurm 执行器负责提交、排队、运行、进程退出码和产物路径；scheduler 的
COMPLETED 只表示进程层状态，不替代结构审计。存在 CIF 不等于合同通过；审计报错应
保留已生成结构并暴露错误，不通过删除失败样本制造高成功率。

`decision_explanation` 转述现有记录，不重新计算另一套分数。报告/目录工具处理索引、
路径、状态和渲染；这些工程判断不具有独立的科学阈值。协议比较和 benchmark 脚本负责
冻结条件、共享成对 pose、分配不重叠种子与统计产物；一个脚本提交了 50 个 design
不等于已有 50 个完成审计的结果。

软件没有凭空得到全原子 Rosetta 能量、真实结合自由能、独立回折叠成功率、结构聚类多样性
或实验成功率。只有相应工具实际运行且输出可追溯时，才能将这些指标加入结果声明。

## 22. 如何维护这份公式说明

本文解释的是相对 `551a901` 的新增/修改算法在 `e20a04e` 实现中的重要数学与决策，
并不把上游原有神经网络内部每一层重新推导。神经网络、扩散日程等未改变部分仍属于
上游方法；本文重点说明新增控制在何时介入，以及改变了什么。

逐文件对应关系见 [上游差异与规则索引](DECISION_RULES_CODE_INDEX.zh-CN.md)。索引列出
生产 Python 模块和维护脚本的差异，不把测试行数、空兼容文件或文档当成新科学算法。
第 1–10 节适合按用户问题阅读，第 11–21 节给出原来遗漏的实现细节。

维护要求：

1. 新增或改变公式、阈值、归一化、配对集合、随机选择或默认开关时，同时更新本说明及索引。
2. 明确是精确几何关系、数值保护、可行性必要条件还是工程启发式，不能只写“科学合理”。
3. 每个新判定必须有来源、触发条件、失败含义及实际日志入口；没有记录的部分应直接说明。
4. 当前源码与本文版本不同，应重新核对差异；不要用本文的默认参数倒填旧实验。

“公式可查”与“每个候选完整可重放”是两个交付层次。本文补全数学说明，不会让已经生成的
日志自动获得缺失坐标。第 6 节已列明 graph 被拒绝 trial 的证据限制；同样，依赖 runtime
记录的审计不能伪装成对输出结构的完全独立测量。

## 22. Benchmark 整改：链边界、事务和最终几何修正

这些改动修复执行与几何控制问题；**不表示核心紧致度和新界面 packing 的低通过率已经解决**。
审计门槛、原任务 pose 和逐 design 随机种子均不因本次整改而放宽或替换。

### 22.1 片段连续性必须是聚合物连续性

界面窗口中的相邻候选残基 i、j 只有在下面三个条件同时成立时，才算连续：

```text
token_j = token_i + 1
chain_j = chain_i
residue_index_j = residue_index_i + 1
```

全局 token 编号相邻不够：上一条链的最后一个 token 与下一条链的第一个 token
也可能编号相邻。窗口枚举、容量估算和骨架几何项现在共享以上连续性标识；
patch 的实际身份仍记录真实 token ID。这样不会把两条链当成同一个刚体片段。

### 22.2 联合更新只发布实际提交的状态

设候选选择的离散窗口为 P'，比较时必须使用同一个窗口：
`E_graph(X_before; P')` 与 `E_graph(X_candidate; P')`。
不能使用 `E_graph(X_before; P_old)` 作为基线，否则窗口变化也会被算成几何改善。

联合拒绝时，坐标、刚体姿态和 patch 锁定一起回滚。日志顶层 `patch_locked`、
`patch_lock_reason`、`patch_assignments` 反映**已提交状态**；尝试但未提交的状态保存在
`proposal_patch_state`。`applied` 只有在整个事务提交时才能为真。
scaffold-driven 刚体提案及联合 packing 提案也应用第 17 节逐对几何保护。
这项保护检查实际候选完整坐标，不能由总损失下降代替。
SE(3) 的每个回溯候选先经过几何保护；完整步长不安全时继续尝试较小步长。
多轨道合并后的坐标还要再次检查，因为分别安全的移动不保证组合起来仍安全。

### 22.3 最终几何可行性修正

触发条件：启用 generated polymer continuity，且已构造 scaffold topology。
作用阶段：最后一次扩散更新之后、最终 graph packing polish 之前。
只平移生成残基的原子，固定残基步长为零，每次候选再经过原来的精确固定/对称投影。

按现有拓扑定义三个违反量集合，固定–固定对不作为优化对象：

```text
v_CA(i,j)   = max(0, clash_distance - ||CA_i-CA_j||)
v_seg(a,b)  = max(0, 1 Å - distance(segment_a, segment_b))
v_bond(i,j) = max(0, abs(||CA_i-CA_j|| - target_CA_distance) - tolerance)
F(X) = sum_categories sum_pairs v(X)^2
```

CA 默认阈值 3.2 Å；连续性使用该次 sampler 的实际 target/tolerance。
线段这里使用 1 Å 的近交修正目标，**不改变**常规 guidance 的 3.2 Å 线段排斥项或最终审计。
这是几何可行性目标，不是分子力场。不能以 Cα 满足条件推断全原子肽键几何或无缠绕。

将原子梯度按 token 求和，对固定 token 清零；令最大 token 梯度范数为 G，
以 `delta_token = -maximum_token_step * gradient_token/G` 提案。
默认每轮最多 0.2 Å，沿用 5 次回溯和 0.5 收缩；最大轮数使用
`generated_polymer_continuity_iterations`。G 非有限或 ≤1e-12 时停止。
每轮接受必须同时满足：

1. 投影后坐标有限，`F_after < F_before - 1e-10`。
2. 对每一类违反量，平方和不增加超过 1e-10，最大值不增加超过 1e-6 Å。
3. 每一个原本满足约束的 pair（v≤1e-6 Å）在候选中仍满足该容差。

已违反的 pair 可在**同类别总平方和和最大值均不恶化**的条件内变化；
这与常规 guidance 的逐 pair 不恶化规则不同，目的在于允许修复已出现的坏几何。
不允许用新碰撞换取断链改善，也不允许用一类违约恶化换另一类改善。
梯度退化、回溯无可接受步长、全部满足容差或预算用完就停止。
未修好的违反量继续记录，最终审计仍按原门槛判定。

证据位于 `generated_polymer_continuity_diagnostics.geometry_restoration`：
包含每类初始/最终违反数量、最大值、平方和，以及各轮回溯的接受条件。
该记录与后续 continuity 的 `steps` 分开，不能把一次最终修正误报为逐步扩散投影。

### 22.4 全任务复跑与样本对应

维护入口为 `scripts/rfd3_mosaic/replay_benchmark.py`。读取冻结的完整任务清单，
按服务器分配准备所有任务，重编译后校验原 pose 的坐标 SHA256，并校验输入和 checkpoint。
分片 `[a,b)` 内第 j 个 design 使用 `seed_base+a+j`；全局编号为 `a+j`。
同一任务所有分片使用同一冻结 pose。改变分片大小不能改变样本集合或 seed。

每个任务的 design 0 先运行；其余样本以有界分片提交 Slurm 数组。
后续数组依赖该服务器全部首样本作业执行成功。这是**执行门槛**，不是质量通过门槛：
低质量输出仍必须记录并纳入分母。调度拒绝、超时、缺少审计、通过合同和推荐样本分开统计。
每个任务至少 50 个 design 的要求指完整矩阵，不是要求每个小分片也有 50 个。

若调度器限制待提交作业数，可以在一个数组作业内串行执行若干独立分片。
使用上一批实测的 `elapsed_seconds/generated_count` 估计每个 design 耗时；
每片预估 `180 seconds + 1.25 * designs * seconds_per_design`，按耗时降序首次适配分组，
每组默认预算 8 小时，低于本批 12 小时作业时限。25% 和 3 分钟是调度余量，不是科学参数，
也不能保证不同 GPU 上不超时。分组不改变每片配置、种子或独立审计。
明确的 `QOSMaxSubmitJobPerUserLimit` 拒绝可按回执重试后续批次；已经接受或状态不确定的提交禁止盲目重复。
