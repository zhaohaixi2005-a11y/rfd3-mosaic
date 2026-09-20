# 相对上游新增/修改代码的公式与判定索引

本索引配合 [公式与决策说明](DECISION_RULES.zh-CN.md) 使用。对照基线为本仓库保留的
`production@551a90103a51d0bff27e269fe454fdba4aefb274`，实现版本为
`e20a04e0b45fb3224b23248e23e31aa5eb7a815f` 及后续 benchmark、生成区域归属整改。这不是与当前最新上游的比较。

原始索引列出差异中的 **126 个生产 Python 文件、22 个维护脚本**；新增完整骨架实现另列如下。
每行给出主文档章节和该文件的职责；包导出、工程适配和空文件不算新科学公式。
同一公式可能跨编译、采样和审计三个入口，因此多个文件引用同一章节不表示重复算法。

这一清单核对文件覆盖，不声称逐行代码经过形式化证明，也不声称每种模式已经通过 GPU 或实验验证。
独立工具、可选研究路径和默认生成流程的区别，以主文档调用条件和实际运行配置为准。

## 早期方案与当前实现

主文档第 24 节记录早期候选设计。当前完整骨架路径采用第 25 节的 **全装配 seed/扭转角联合求解、
实际生成骨架验收与共同参考移动**；它不依赖未校准的端口打分系数，也没有接入先跑完整 diffusion
再挑共享 pose 的额外 GPU 搜索。普通共享 pose、静态可达性与 core 引导按原章节列出。

| 新增实现 | 主文档章节 | 职责 |
|---|---|---|
| [scaffold_tasks.py](../../src/rfd3_mosaic/scaffold_tasks.py) | 25.1–25.3、25.7 | 精确长度、整个 seed 拟合、独立验收、准备目录完整发布 |
| [scaffold_pose.py](../../src/rfd3_mosaic/scaffold_pose.py) | 25.1 | 模板对称帧和群一致性 |
| [scaffold_builder.py](../../src/rfd3_mosaic/scaffold_builder.py) | 25.2–25.3 | 内部坐标、扭转导数及单个生成区几何工具 |
| [scaffold_assembly.py](../../src/rfd3_mosaic/scaffold_assembly.py) | 25.2–25.3 | seed 与全部生成区同时求解、全装配排斥、显式有限预算 |
| [scaffold_input.py](../../src/rfd3_mosaic/scaffold_input.py) | 25.6–25.7 | artifact、固定身份、群作用、参考移动计划绑定 |
| [scaffold_contract.py](../../src/rfd3_mosaic/validation/scaffold_contract.py) | 25.4–25.5 | CA 形状/支撑/分离，v2 必需真实骨架验收 |
| [generated_backbone.py](../../src/rfd3_mosaic/validation/generated_backbone.py) | 25.4 | 实际 alpha 氢键指派、漏声明 H、独立骨架几何 |
| [reference_transport.py](../../src/rfd3_mosaic/validation/reference_transport.py) | 25.6 | SE(3) 弧长运输、群共轭、边界与最终重放 |
| [reference_scaffold.py](../../models/rfd3/src/rfd3/inference/symmetry/reference_scaffold.py) | 25.5 | 原生特征绑定、CA 和完整骨架可微残差 |
| [scaffold_transport.py](../../models/rfd3/src/rfd3/inference/symmetry/scaffold_transport.py) | 25.6 | 采样中的参考/seed/生成区共同准备与提交 |

既有 `engine.py`、`experiment_worker.py`、`result_auditing.py`、`posthoc_audit.py`、
`run_reporting.py` 新增故障隔离与部分结果语义见 25.8；`inference_sampler.py`、
`trainer/rfd3.py`、`rfd3_scaffold_audit.py` 新增最终骨架诊断和独立重放见 25.6–25.7。
可选 `metrics/hbonds_metrics.py` 和 `testing/testing_utils.py` 的旧工程引用/条件统计修复见 25.9。

开发工作区中的参考坐标分析脚本及 JSON 仅为离线证据，路径与测量定义见主文档
第 24.6–24.7 节；它们不是软件安装后自动提供的评分工具。

## 生产实现

### Mosaic 编译、搜索、接口与审计

| 源码 | 主文档章节 | 公式或判定职责 |
|---|---|---|
| [__init__.py](../../src/rfd3_mosaic/__init__.py) | 21.1 | 包初始化/公开导出；公式见所导出的实现模块 |
| [advisory_screening.py](../../src/rfd3_mosaic/advisory_screening.py) | 9、19、21.3 | 聚合已有审计、required 合同和建议；缺失/报错与通过分开 |
| [assembly_compiler.py](../../src/rfd3_mosaic/assembly_compiler.py) | 11、12、21.1 | 把用户装配声明展开为组件、群作用、端口和可执行连接；检查支持范围 |
| [assembly_frontends.py](../../src/rfd3_mosaic/assembly_frontends.py) | 11、12、21.1 | 把用户装配声明展开为组件、群作用、端口和可执行连接；检查支持范围 |
| [backbone_comparison.py](../../src/rfd3_mosaic/backbone_comparison.py) | 19.3、19.5 | carbonyl-C Rg、二级结构和队列中位数筛选 |
| [capabilities.py](../../src/rfd3_mosaic/capabilities.py) | 2、21.1、21.3 | 入口、环境/能力和参数检查；无独立几何打分公式 |
| [cli.py](../../src/rfd3_mosaic/cli.py) | 2、21.1、21.3 | 入口、环境/能力和参数检查；无独立几何打分公式 |
| [compile.py](../../src/rfd3_mosaic/compile.py) | 3、11–14、16.1、21.1 | 实例化群变换、pose、约束、原子映射及界面参数 |
| [constraint_plan.py](../../src/rfd3_mosaic/constraint_plan.py) | 4、13、21.1 | 固定目标/运动自由度声明、冲突检测及编译计划 |
| [decision_explanation.py](../../src/rfd3_mosaic/decision_explanation.py) | 2、6、21.3 | 解释已有记录中的公式参数及接受/拒绝原因，不另算隐含分数 |
| [design_compiler.py](../../src/rfd3_mosaic/design_compiler.py) | 11、12、21.1 | 把用户装配声明展开为组件、群作用、端口和可执行连接；检查支持范围 |
| [design_preferences.py](../../src/rfd3_mosaic/design_preferences.py) | 2、3、6、14.2 | 预设到具体权重、半径与控制参数的映射；数值不是普适定律 |
| [doctor.py](../../src/rfd3_mosaic/doctor.py) | 2、21.1、21.3 | 入口、环境/能力和参数检查；无独立几何打分公式 |
| [execution.py](../../src/rfd3_mosaic/execution.py) | 21.3 | 本地/调度器执行状态和退出码，不替代科学审计 |
| [experiment.py](../../src/rfd3_mosaic/experiment.py) | 21.2、21.3 | 固定任务 pose、逐 design 种子、分片与执行；不重复采样起点 |
| [experiment_worker.py](../../src/rfd3_mosaic/experiment_worker.py) | 21.2、21.3 | 固定任务 pose、逐 design 种子、分片与执行；不重复采样起点 |
| [feasibility_restoration.py](../../src/rfd3_mosaic/feasibility_restoration.py) | 3、12.4 | linker 范围交集、轮廓下界和长度绑定；不可满足时拒绝 |
| [functional_geometry.py](../../src/rfd3_mosaic/functional_geometry.py) | 18.2 | 距离/角度/二面角、相对 frame、有向体积及配位几何评价 |
| [geometry/__init__.py](../../src/rfd3_mosaic/geometry/__init__.py) | 21.1 | 包初始化/公开导出；公式见所导出的实现模块 |
| [geometry/frames.py](../../src/rfd3_mosaic/geometry/frames.py) | 11.1、13 | 坐标参考系、刚体变换和检查 |
| [geometry/se3.py](../../src/rfd3_mosaic/geometry/se3.py) | 11.1 | SE(3) 构造、组合、逆和旋转表示 |
| [geometry/symmetry_registry.py](../../src/rfd3_mosaic/geometry/symmetry_registry.py) | 11.2、11.3 | Cn/Dn/T/O/I 群变换与群表 |
| [graph_search.py](../../src/rfd3_mosaic/graph_search.py) | 12.3、12.5 | 候选图枚举、预算/完整性标记、兼容性与字典序排名 |
| [installation.py](../../src/rfd3_mosaic/installation.py) | 2、21.1、21.3 | 入口、环境/能力和参数检查；无独立几何打分公式 |
| [interface_semantics.py](../../src/rfd3_mosaic/interface_semantics.py) | 9、12、16、19.6 | 输入界面保留与输出新建、端口和质量目标的不同语义 |
| [objectives/__init__.py](../../src/rfd3_mosaic/objectives/__init__.py) | 21.1 | 包初始化/公开导出；公式见所导出的实现模块 |
| [objectives/core.py](../../src/rfd3_mosaic/objectives/core.py) | 18.1 | 标量目标归一化、阈值和范围违反量 |
| [objectives/static_metrics.py](../../src/rfd3_mosaic/objectives/static_metrics.py) | 3、18.1 | 静态几何指标及目标适配 |
| [onboarding.py](../../src/rfd3_mosaic/onboarding.py) | 2、21.1、21.3 | 入口、环境/能力和参数检查；无独立几何打分公式 |
| [output/__init__.py](../../src/rfd3_mosaic/output/__init__.py) | 21.1 | 包初始化/公开导出；公式见所导出的实现模块 |
| [output/rfd3_adapter.py](../../src/rfd3_mosaic/output/rfd3_adapter.py) | 21.1、21.3 | 输出适配与产物序列化；不定义新的科学分数 |
| [output/standalone.py](../../src/rfd3_mosaic/output/standalone.py) | 21.1、21.3 | 输出适配与产物序列化；不定义新的科学分数 |
| [pose_ensemble.py](../../src/rfd3_mosaic/pose_ensemble.py) | 3、14.1、14.3 | 随机 pose 候选和可行性信息；与优化及去重分开 |
| [pose_optimizer.py](../../src/rfd3_mosaic/pose_optimizer.py) | 3、14.2 | 确定全局候选、有界直接搜索及严格字典序改进 |
| [pose_qd.py](../../src/rfd3_mosaic/pose_qd.py) | 14.3 | 质量池内选取不同姿态；显式随机种子与重复处理 |
| [pose_select.py](../../src/rfd3_mosaic/pose_select.py) | 3、14.3 | 质量候选池与距离谱分离选择 |
| [pose_stratify.py](../../src/rfd3_mosaic/pose_stratify.py) | 14.3 | 描述符分箱、边界及配额；不等于最终结构多样性 |
| [pose_tasks.py](../../src/rfd3_mosaic/pose_tasks.py) | 14、21.2 | 将各 pose 固化为独立任务、重编译一致性检查 |
| [posthoc_audit.py](../../src/rfd3_mosaic/posthoc_audit.py) | 9、19、21.3 | 聚合已有审计、required 合同和建议；缺失/报错与通过分开 |
| [provenance/__init__.py](../../src/rfd3_mosaic/provenance/__init__.py) | 21.1 | 包初始化/公开导出；公式见所导出的实现模块 |
| [provenance/mapping_registry.py](../../src/rfd3_mosaic/provenance/mapping_registry.py) | 21.1、21.2 | 输入到实例/输出原子的身份映射与一致性 |
| [provenance/software.py](../../src/rfd3_mosaic/provenance/software.py) | 21.2 | 版本和软件来源记录 |
| [provenance/source_snapshot.py](../../src/rfd3_mosaic/provenance/source_snapshot.py) | 21.2 | 源码摘要与来源快照；hash 不证明科学正确 |
| [relation_compatibility.py](../../src/rfd3_mosaic/relation_compatibility.py) | 12.5 | 有限阶关系闭合、旋转阶数/平移残差与兼容性排名 |
| [result_auditing.py](../../src/rfd3_mosaic/result_auditing.py) | 9、19、21.3 | 聚合已有审计、required 合同和建议；缺失/报错与通过分开 |
| [rfd3_adapter.py](../../src/rfd3_mosaic/rfd3_adapter.py) | 13、20、21.1 | 将编译计划接入 RFD3，传递约束与 backend 条件 |
| [rfd3_audit_gate.py](../../src/rfd3_mosaic/rfd3_audit_gate.py) | 9、19、21.3 | 聚合已有审计、required 合同和建议；缺失/报错与通过分开 |
| [rfd3_batch_screen.py](../../src/rfd3_mosaic/rfd3_batch_screen.py) | 9、19.3、19.5、19.7 | 批量骨架/环形/链间接触描述符与审计汇总 |
| [rfd3_central_motif_audit.py](../../src/rfd3_mosaic/rfd3_central_motif_audit.py) | 4、13、19.1 | 中心固定 motif 坐标/对称/来源检查；probe 属独立检查工具 |
| [rfd3_central_motif_probe.py](../../src/rfd3_mosaic/rfd3_central_motif_probe.py) | 4、13、19.1 | 中心固定 motif 坐标/对称/来源检查；probe 属独立检查工具 |
| [rfd3_constraint_orbit_audit.py](../../src/rfd3_mosaic/rfd3_constraint_orbit_audit.py) | 4、13、19.1 | 固定/移动/商轨道保真、原子完整率与 runtime 目标证据 |
| [rfd3_cylindrical_audit.py](../../src/rfd3_mosaic/rfd3_cylindrical_audit.py) | 13.3、19.4 | 声明柱坐标自由度与运行/输出误差检查 |
| [rfd3_graph_interface_guidance_audit.py](../../src/rfd3_mosaic/rfd3_graph_interface_guidance_audit.py) | 6、9、16、19.4 | 引导执行证据、最终代理指标、标识和配置一致性 |
| [rfd3_interface_relation_audit.py](../../src/rfd3_mosaic/rfd3_interface_relation_audit.py) | 9、19.6 | 声明界面的相对变换、接触/覆盖和重原子 packing 代理 |
| [rfd3_mobility_audit.py](../../src/rfd3_mosaic/rfd3_mobility_audit.py) | 7、15、19.4 | 累计/逐步刚体移动、限制子空间与轨迹证据 |
| [rfd3_prevalidate.py](../../src/rfd3_mosaic/rfd3_prevalidate.py) | 3、12、13、21.1 | 采样前输入、映射、约束和支持能力检查 |
| [rfd3_scaffold_audit.py](../../src/rfd3_mosaic/rfd3_scaffold_audit.py) | 17、19.3、23.3 | 骨架主链/对称/穿插检查，以及最终空间归属独立审计入口 |
| [rfd3_scaffold_core_audit.py](../../src/rfd3_mosaic/rfd3_scaffold_core_audit.py) | 8、9、19.4 | 核心引导执行与 required/advisory 代理目标检查 |
| [rfd3_seed_audit.py](../../src/rfd3_mosaic/rfd3_seed_audit.py) | 19.1、19.2 | 链配对、接触保持与 seed 误差；区分拟合和内部距离 |
| [run_artifacts.py](../../src/rfd3_mosaic/run_artifacts.py) | 21.3 | 运行产物、路径、状态和报告；转述已有指标 |
| [run_catalog.py](../../src/rfd3_mosaic/run_catalog.py) | 21.3 | 运行产物、路径、状态和报告；转述已有指标 |
| [run_index.py](../../src/rfd3_mosaic/run_index.py) | 21.3 | 运行产物、路径、状态和报告；转述已有指标 |
| [run_layout.py](../../src/rfd3_mosaic/run_layout.py) | 21.3 | 运行产物、路径、状态和报告；转述已有指标 |
| [run_reorganization.py](../../src/rfd3_mosaic/run_reorganization.py) | 21.3 | 运行产物、路径、状态和报告；转述已有指标 |
| [run_reporting.py](../../src/rfd3_mosaic/run_reporting.py) | 21.3 | 运行产物、路径、状态和报告；转述已有指标 |
| [sampling_plan.py](../../src/rfd3_mosaic/sampling_plan.py) | 21.2、21.3 | 固定任务 pose、逐 design 种子、分片与执行；不重复采样起点 |
| [schema/__init__.py](../../src/rfd3_mosaic/schema/__init__.py) | 21.1 | 包初始化/公开导出；公式见所导出的实现模块 |
| [schema/design.py](../../src/rfd3_mosaic/schema/design.py) | 2、11–13、18、21.1 | 声明类型、默认参数、范围及组合校验；实际公式见消费字段的模块 |
| [schema/functional_geometry.py](../../src/rfd3_mosaic/schema/functional_geometry.py) | 2、11–13、18、21.1 | 声明类型、默认参数、范围及组合校验；实际公式见消费字段的模块 |
| [schema/instances.py](../../src/rfd3_mosaic/schema/instances.py) | 2、11–13、18、21.1 | 声明类型、默认参数、范围及组合校验；实际公式见消费字段的模块 |
| [schema/simple_intent.py](../../src/rfd3_mosaic/schema/simple_intent.py) | 2、11–13、18、21.1 | 声明类型、默认参数、范围及组合校验；实际公式见消费字段的模块 |
| [schema/specs.py](../../src/rfd3_mosaic/schema/specs.py) | 2、11–13、18、21.1 | 声明类型、默认参数、范围及组合校验；实际公式见消费字段的模块 |
| [schema/states.py](../../src/rfd3_mosaic/schema/states.py) | — | 空占位文件，无可执行公式或判定 |
| [schema/symmetry_names.py](../../src/rfd3_mosaic/schema/symmetry_names.py) | 2、11–13、18、21.1 | 声明类型、默认参数、范围及组合校验；实际公式见消费字段的模块 |
| [seed_library.py](../../src/rfd3_mosaic/seed_library.py) | 11.3、21.1、21.2 | seed/结构读取、选择和来源元数据；几何判断追溯对应编译/审计 |
| [seed_stabilizer.py](../../src/rfd3_mosaic/seed_stabilizer.py) | 11.3 | seed 内部变换拟合、闭合/公共中心、稳定子与群表匹配 |
| [simple_architecture.py](../../src/rfd3_mosaic/simple_architecture.py) | 11、12、21.1 | 把用户装配声明展开为组件、群作用、端口和可执行连接；检查支持范围 |
| [simple_resolver.py](../../src/rfd3_mosaic/simple_resolver.py) | 11、12、21.1 | 把用户装配声明展开为组件、群作用、端口和可执行连接；检查支持范围 |
| [standalone.py](../../src/rfd3_mosaic/standalone.py) | 2、21.1、21.3 | 入口、环境/能力和参数检查；无独立几何打分公式 |
| [structure/__init__.py](../../src/rfd3_mosaic/structure/__init__.py) | 21.1 | 包初始化/公开导出；公式见所导出的实现模块 |
| [structure/pdb.py](../../src/rfd3_mosaic/structure/pdb.py) | 21.1 | 结构原子记录、残基身份及读取 |
| [structure/selection.py](../../src/rfd3_mosaic/structure/selection.py) | 21.1 | 链/残基/原子选择器和空集合检查 |
| [structure_archive.py](../../src/rfd3_mosaic/structure_archive.py) | 11.3、21.1、21.2 | seed/结构读取、选择和来源元数据；几何判断追溯对应编译/审计 |
| [structure_inspection.py](../../src/rfd3_mosaic/structure_inspection.py) | 11.3、21.1、21.2 | seed/结构读取、选择和来源元数据；几何判断追溯对应编译/审计 |
| [topology/__init__.py](../../src/rfd3_mosaic/topology/__init__.py) | 21.1 | 包初始化/公开导出；公式见所导出的实现模块 |
| [topology/component_incidence.py](../../src/rfd3_mosaic/topology/component_incidence.py) | 11.4、12.1 | 组件与界面关联、度数和连通性 |
| [topology/interface_seed_graph.py](../../src/rfd3_mosaic/topology/interface_seed_graph.py) | 12.1–12.3 | seed 端口和跨 seed 连接约束 |
| [topology/polymer_path_solver.py](../../src/rfd3_mosaic/topology/polymer_path_solver.py) | 12.1–12.3 | 有向聚合物路径、匹配与搜索预算 |
| [topology/pose_graph.py](../../src/rfd3_mosaic/topology/pose_graph.py) | — | 空占位文件，无可执行公式或判定 |
| [topology/scaffold_graph.py](../../src/rfd3_mosaic/topology/scaffold_graph.py) | 12.1、12.4 | scaffold 连接、端点及 linker 结构 |
| [topology/stabilizer_cosets.py](../../src/rfd3_mosaic/topology/stabilizer_cosets.py) | 11.3、11.4 | 稳定子子群、陪集分区与商轨道 |
| [topology/symmetry_connectivity.py](../../src/rfd3_mosaic/topology/symmetry_connectivity.py) | 12.2 | 群生成关系闭包与全装配连通性 |
| [validation/__init__.py](../../src/rfd3_mosaic/validation/__init__.py) | 21.1 | 包初始化/公开导出；公式见所导出的实现模块 |
| [validation/assembly_morphology.py](../../src/rfd3_mosaic/validation/assembly_morphology.py) | 18.3 | 用户形态目标、轴可辨识性及孔径/外径/高度 |
| [validation/generated_route_ownership.py](../../src/rfd3_mosaic/validation/generated_route_ownership.py) | 23 | 最终 Cα/中点空间归属独立审计、覆盖范围和参考路线余量冲突 |
| [validation/scaffold_validity.py](../../src/rfd3_mosaic/validation/scaffold_validity.py) | 17.2、19.3 | 主链完整、键长、碰撞、线段接近及对称误差 |
| [validation/schema.py](../../src/rfd3_mosaic/validation/schema.py) | — | 空占位文件，无可执行公式或判定 |
| [validation/seed_integrity.py](../../src/rfd3_mosaic/validation/seed_integrity.py) | 19.1、19.2 | 刚体拟合、内部距离与界面接触保持 |

### RFD3 内部接入及采样控制

| 源码 | 主文档章节 | 公式或判定职责 |
|---|---|---|
| [rfd3/engine.py](../../models/rfd3/src/rfd3/engine.py) | 13、20.3、21.3 | 精度/设备搬运、运行状态及几何上下文接入 |
| [rfd3/inference/datasets.py](../../models/rfd3/src/rfd3/inference/datasets.py) | 13、21.1 | 输入实例、局部/对称元数据和数据集装载 |
| [rfd3/inference/input_parsing.py](../../models/rfd3/src/rfd3/inference/input_parsing.py) | 12、13、17.1、21.1 | 输入映射、固定片段和生成区初始化 |
| [rfd3/inference/parsing.py](../../models/rfd3/src/rfd3/inference/parsing.py) | 13、21.1 | 声明字段到推理配置的解析和校验 |
| [rfd3/inference/symmetry/atom_array.py](../../models/rfd3/src/rfd3/inference/symmetry/atom_array.py) | 11、13、21.1 | 对称复制与原子数组身份/坐标映射 |
| [rfd3/inference/symmetry/checks.py](../../models/rfd3/src/rfd3/inference/symmetry/checks.py) | 4、13 | 对称几何及约束一致性检查 |
| [rfd3/inference/symmetry/constraint_orbit.py](../../models/rfd3/src/rfd3/inference/symmetry/constraint_orbit.py) | 4、11、13 | 固定约束轨道及来源/副本对应 |
| [rfd3/inference/symmetry/constraint_runtime.py](../../models/rfd3/src/rfd3/inference/symmetry/constraint_runtime.py) | 4、13、15 | 固定目标生命周期、更新及恢复 |
| [rfd3/inference/symmetry/cylindrical_projector.py](../../models/rfd3/src/rfd3/inference/symmetry/cylindrical_projector.py) | 13.3 | 柱坐标投影、轴退化及容差 |
| [rfd3/inference/symmetry/frames.py](../../models/rfd3/src/rfd3/inference/symmetry/frames.py) | 11.1、13 | 数值 frame 正交化和合法性 |
| [rfd3/inference/symmetry/generated_routes.py](../../models/rfd3/src/rfd3/inference/symmetry/generated_routes.py) | 23 | 正余量、全部竞争路线、CA/中点违反量及逐 clean prediction 有界修正 |
| [rfd3/inference/symmetry/graph_interface_guidance.py](../../models/rfd3/src/rfd3/inference/symmetry/graph_interface_guidance.py) | 5、6、16、23.4 | 接触势、全部 graph 损失、选片段、梯度和接受规则；联合更新共享同一有效目标 |
| [rfd3/inference/symmetry/geometry_restoration.py](../../models/rfd3/src/rfd3/inference/symmetry/geometry_restoration.py) | 22、23.3 | 最终生成区域的有界可行性修正；记录残留违反，不保证任意固定 pose 可修复 |
| [rfd3/inference/symmetry/interface_constraint_orbit.py](../../models/rfd3/src/rfd3/inference/symmetry/interface_constraint_orbit.py) | 11、13、21.1 | 界面组约束到轨道及运行原子身份的转换 |
| [rfd3/inference/symmetry/joint_projector.py](../../models/rfd3/src/rfd3/inference/symmetry/joint_projector.py) | 4、13 | 对称与固定约束联合投影、恢复及验证 |
| [rfd3/inference/symmetry/local_neighbourhood.py](../../models/rfd3/src/rfd3/inference/symmetry/local_neighbourhood.py) | 20.1 | 局部群索引邻域、特征裁剪和全轨道展开 |
| [rfd3/inference/symmetry/motif_mobility.py](../../models/rfd3/src/rfd3/inference/symmetry/motif_mobility.py) | 7、15、23.4 | 模型/目标刚体提案、自由度、先验和搜索；同一 context 的事务比较 |
| [rfd3/inference/symmetry/scaffold_core_guidance.py](../../models/rfd3/src/rfd3/inference/symmetry/scaffold_core_guidance.py) | 8、17、23 | 核心/路径/排他/碰撞引导、连续性与路线逐对保护 |
| [rfd3/inference/symmetry/scaffold_guidance.py](../../models/rfd3/src/rfd3/inference/symmetry/scaffold_guidance.py) | 6–8、15 | 组合各引导、捕获/绑定和联合接受 |
| [rfd3/inference/symmetry/symmetry_utils.py](../../models/rfd3/src/rfd3/inference/symmetry/symmetry_utils.py) | 4、11、13 | 群作用、轨道投影及耦合随机噪声 |
| [rfd3/model/RFD3.py](../../models/rfd3/src/rfd3/model/RFD3.py) | 20.1 | 局部输入进入 TokenInitializer 之前的特征裁剪接入 |
| [rfd3/model/inference_sampler.py](../../models/rfd3/src/rfd3/model/inference_sampler.py) | 4–8、13、15、17、20、23 | 采样各阶段调用顺序、固定恢复、引导和最终修正；逐 clean prediction 路线修正 |
| [rfd3/model/layers/block_utils.py](../../models/rfd3/src/rfd3/model/layers/block_utils.py) | 20.2 | 逐查询原子合法跨链邻居及小输入配额 |
| [rfd3/trainer/rfd3.py](../../models/rfd3/src/rfd3/trainer/rfd3.py) | 21.1、21.3 | 训练/推理接口兼容接入；不新增本文意义下的人工能量 |
| [rfd3/transforms/conditioning_base.py](../../models/rfd3/src/rfd3/transforms/conditioning_base.py) | 13、21.1 | 条件/固定遮罩与对称输入一致性 |
| [rfd3/transforms/symmetry.py](../../models/rfd3/src/rfd3/transforms/symmetry.py) | 11、13 | 对称条件变换和轨道元数据 |
| [rfd3/utils/inference.py](../../models/rfd3/src/rfd3/utils/inference.py) | 13、21.1、21.3 | 推理条件、输出及元数据传递 |

### 共享几何工具

| 源码 | 主文档章节 | 公式或判定职责 |
|---|---|---|
| [utils/alignment.py](../../src/foundry/utils/alignment.py) | 11.3、20.3 | 加权 Kabsch、反射修正及 autocast/工作精度保护 |

## 维护脚本

这些脚本多数负责参数构造、提交、收集或可视化，不能仅凭脚本名认定参与了本次采样。

| 源码 | 主文档章节 | 职责 |
|---|---|---|
| [activate_local_dev.sh](../../scripts/rfd3_mosaic/activate_local_dev.sh) | 21.1 | 激活本地开发环境；无科学公式 |
| [check_public_surface.py](../../scripts/rfd3_mosaic/check_public_surface.py) | 22 | 公共文档链接/边界检查；不检查公式科学有效性 |
| [collect_packing_campaign.py](../../scripts/rfd3_mosaic/collect_packing_campaign.py) | 9、19、21.3 | 收集已有 packing 运行与审计，保留测量/合同语义 |
| [collect_scientific_breadth_campaign.py](../../scripts/rfd3_mosaic/collect_scientific_breadth_campaign.py) | 9、19、21.3 | 收集跨任务结果及完成/审计状态 |
| [compare_hoyeung_lhd101_backbones.py](../../scripts/rfd3_mosaic/compare_hoyeung_lhd101_backbones.py) | 19.5 | 骨架比较入口；计算来自 backbone_comparison |
| [hoyeung_lhd101_1000_array.sbatch](../../scripts/rfd3_mosaic/hoyeung_lhd101_1000_array.sbatch) | 21.2、21.3 | 基线数组任务、种子及调度执行 |
| [lhd101_c3_pose_qd.sh](../../scripts/rfd3_mosaic/lhd101_c3_pose_qd.sh) | 14.3 | C3 pose 多样性工具包装，不定义另一套评分 |
| [lhd101_cn_pose_qd.sh](../../scripts/rfd3_mosaic/lhd101_cn_pose_qd.sh) | 14.3 | Cn pose 多样性工具包装，不定义另一套评分 |
| [load_cif_ensemble.py](../../scripts/rfd3_mosaic/load_cif_ensemble.py) | 21.3 | PyMOL CIF 集合展示；状态按 ((state-1) mod N)+1 循环 |
| [prepare_lhd101_benchmark.py](../../scripts/rfd3_mosaic/prepare_lhd101_benchmark.py) | 21.1–21.3 | 准备 benchmark 输入、共享来源及参数 |
| [pymol_fixed_orbit_alignment.py](../../scripts/rfd3_mosaic/pymol_fixed_orbit_alignment.py) | 19.8 | 显示用匹配/共识选择和固定来源对齐 |
| [release_smoke.sh](../../scripts/rfd3_mosaic/release_smoke.sh) | 21.3 | 发布流程检查编排；成功退出不等于科学有效 |
| [screen_extracted_cn_structures.sh](../../scripts/rfd3_mosaic/screen_extracted_cn_structures.sh) | 9、19.5、19.7 | 结构批量筛选入口 |
| [setup_local_cpu_dev.sh](../../scripts/rfd3_mosaic/setup_local_cpu_dev.sh) | 21.1 | 开发依赖安装；无科学公式 |
| [submit_c3_packing_gpu_matrix.sh](../../scripts/rfd3_mosaic/submit_c3_packing_gpu_matrix.sh) | 6、21.2、21.3 | packing 参数组合实验与提交；具体参数以运行输入为准 |
| [submit_c3_patch_capture_canaries.sh](../../scripts/rfd3_mosaic/submit_c3_patch_capture_canaries.sh) | 6、21.2、21.3 | 片段捕获小规模实验配置与提交 |
| [submit_gpu_release_gates.py](../../scripts/rfd3_mosaic/submit_gpu_release_gates.py) | 9、21.2、21.3 | 发布门槛任务构造与提交，保留实际审计结果 |
| [submit_hoyeung_lhd101_1000.sh](../../scripts/rfd3_mosaic/submit_hoyeung_lhd101_1000.sh) | 21.2、21.3 | 基线任务提交与输入/种子记录 |
| [submit_mosaic_lhd101_c3_1000.py](../../scripts/rfd3_mosaic/submit_mosaic_lhd101_c3_1000.py) | 14、21.2、21.3 | Mosaic 大批量任务准备，区分任务 pose 与 design 随机种子 |
| [submit_packing_replicates.py](../../scripts/rfd3_mosaic/submit_packing_replicates.py) | 6、21.2、21.3 | packing 成对条件/重复实验配置与提交 |
| [submit_scientific_breadth_campaign.py](../../scripts/rfd3_mosaic/submit_scientific_breadth_campaign.py) | 11–20、21.2、21.3 | 跨模式验证任务矩阵；定义实验不代表验证已经成功 |

## 参数、测试与非算法文件的边界

声明默认值主要在 `schema/`、推理 parser、控制器构造及 `design_preferences.py`；这些文件已列在上表。
YAML/JSON 示例和任务生成脚本可能覆盖默认参数，运行时应以解析后的输入及报告为准，不能从示例推断通用阈值。
第 2 节说明参数层级，第 6 节列权重/接受规则及复算，第 21 节解释来源和复现边界。
依赖清单、安装/CI、文档、测试及输入结构不是额外的人工几何能量；不计入生产公式文件数。
测试期望值也不自动构成科学依据。输入结构提供坐标，具体判定仍追溯到上面的实现。

## 如何检查索引是否过期

新增维护入口：[replay_benchmark.py](../../scripts/rfd3_mosaic/replay_benchmark.py)，
对应主文档第 22.4 节：全任务分片、原 pose 和种子不变、首样本依赖及调度回执。

重新比较同一基线与目标实现，检查下面集合中的每个文件均有一行；删除或改名同样需要同步。

```bash
git diff --name-only 551a901 HEAD -- src models/rfd3/src scripts/rfd3_mosaic
```

生产集合为 `src/` 与 `models/rfd3/src/` 下变更的 `.py`；维护脚本集合为
`scripts/rfd3_mosaic/` 下变更的 `.py`、`.sh` 和 `.sbatch`。运行此命令只能检查文件清单，
新增函数、公式或门槛仍须读差异并更新主文档，不能以“文件已有索引”代替内容核查。
