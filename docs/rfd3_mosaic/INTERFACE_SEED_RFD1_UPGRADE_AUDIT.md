# RFdiffusion1 Interface-Seed 全量升级审计

## 1. 审计范围与版本

本报告审计公开仓库：

```text
/home/haixi/Documents/HYC_repeat/RFdiffusion_interfaceseed
```

比较范围：

| 角色 | Git commit | 说明 |
|---|---|---|
| 原始 RFdiffusion 基线 | `2d0c003df46b9db41d119321f15403dec3716cd9` | Interface-Seed 功能提交的直接父提交 |
| Interface-Seed 功能 | `fc0df9d532abdade5a0c02725a5976155382a1ac` | `Add interface seed modifications` |
| 当前公开分支 HEAD | `a81ed1930941e95b3c3c5dbbc9e790bb5e80791b` | 在功能提交基础上补充 README 和图片 |

功能提交的精确规模：

```text
6 files changed, 754 insertions(+), 3 deletions(-)
```

其中 500 行是示例 PDB，算法代码实际集中在两个 Python 文件：

```text
rfdiffusion/inference/model_runners.py  +135 / -3
scripts/run_inference.py                 +63 / -0
```

> 注意：服务器上的旧版 `RFdiffusion_asy` 和 `run_inference_asy.py` 可能不是这个公开提交。要汇报服务器实际使用版本，还需要对服务器普通版和 asy 版做一次目录级 diff。本报告描述的是当前本地公开 fork 的可验证改动。

## 2. 全部变更文件

### 2.1 功能提交 `fc0df9d`

| 文件 | 类型 | 实际作用 |
|---|---|---|
| `.gitignore` | 修改 | 忽略所有目录下的 `.DS_Store` |
| `config/inference/interface_seed.yaml` | 新增 | 定义 Interface-Seed 配置和默认参数 |
| `examples/design_interfaceseed_oligos.sh` | 新增 | LHD101 C3 示例命令 |
| `examples/input_pdbs/7mwr_interface.pdb` | 新增 | LHD101 双链界面 seed 输入 |
| `rfdiffusion/inference/model_runners.py` | 修改 | 非对称 seed 初始化、随机刚体 pose、对称复制和跨界面 contig 构建 |
| `scripts/run_inference.py` | 修改 | denoising 过程中计算接触并执行 motif dragging |

### 2.2 文档提交 `a81ed19`

| 文件 | 类型 | 实际作用 |
|---|---|---|
| `README.md` | 修改 | 增加 fork 警告、Interface-Seed 使用说明、参数解释和兼容性警告 |
| `img/interfaceseed.png` | 新增 | 方法示意图 |

## 3. 没有改动的部分

以下模块没有被 Interface-Seed 功能提交修改：

- RFdiffusion 神经网络结构；
- RFdiffusion checkpoint 和模型权重；
- 训练代码和训练数据；
- `rfdiffusion/inference/symmetry.py`；
- `rfdiffusion/contigs.py`；
- `rfdiffusion/potentials/`；
- `rfdiffusion/inference/utils.py` 中的 sampler selector；
- SE(3)-Transformer；
- diffuser 和 denoiser 的基础实现。

因此，Interface-Seed 不是重新训练的模型，而是一个 **inference-time sampling extension**：在已有 RFdiffusion 权重和 sampler 外围加入新的输入初始化、几何展开、contig 重写和逐步坐标引导。

## 4. 新增配置

新增 `config/inference/interface_seed.yaml`，继承原始 `base` 配置。

### 4.1 新参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `inference.asy_motif` | `False` | 启用非对称双链 motif 的自动对称展开 |
| `inference.motif_drag` | `True` | 在每个 reverse-diffusion step 前执行 motif dragging |
| `inference.asy_motif_weight` | `1` | dragging 基础强度 |
| `inference.asy_motif_rot_range` | `[180,180,180]` | 三个 Euler 角的整数采样范围，分别在 `[-range,+range]` 内采样 |
| `inference.asy_motif_dist` | `25` | 初始平移参数中心值 |
| `inference.asy_motif_dist_range` | `5` | 初始平移参数的整数采样半径 |
| `inference.random_drag` | `True` | 每一步为 dragging 乘以随机系数 |
| `inference.final_rot` | `[]` | 运行时写入最终采样的三个旋转角 |
| `inference.final_dist` | `0` | 运行时写入最终采样的距离参数 |

### 4.2 同时设置但并非新算法的参数

| 参数 | 默认值 | 来源 |
|---|---:|---|
| `inference.symmetry` | `c3` | 使用原生 RFdiffusion symmetry 功能 |
| `inference.model_only_neighbors` | `False` | 原生 symmetry sampler 能力 |
| `inference.output_prefix` | `samples/interface_seed` | 普通输出配置 |
| `contigmap.contigs` | `['100']` | 普通 contig 配置占位值 |

## 5. 输入格式扩展

原始 RFdiffusion symmetric motif scaffolding 通常要求用户预先准备完整对称 motif。Interface-Seed 增加了一个双链非对称输入约定：

```text
输入 motif：最多使用 A、B 两个 chain ID
用户 contig：使用 X、Y 作为占位符
```

示例：

```text
Y211-241/70-100/X165-194
```

其中：

- `X` 代表每个对称 copy 中第一个 motif fragment；
- `Y` 代表第二个 motif fragment；
- `70-100` 是生成的连接区域；
- X/Y 在初始化时会被替换为实际展开后的 A、B、C、D……链号。

新增示例 PDB `7mwr_interface.pdb` 包含：

```text
chain A: residues 165-194, 30 residues
chain B: residues 211-241, 31 residues
total:   61 residues, 496 ATOM records
HETATM:  0
```

## 6. `model_runners.py` 的全部算法改动

### 6.1 新增依赖

```python
from random import randint
import copy
```

- `randint` 用于角度、距离和 linker 长度的离散整数采样；
- `copy.deepcopy` 用于复制 contig 配置，避免直接覆盖原始 Hydra 配置节点。

### 6.2 根据 `asy_motif` 分流输入处理

原版总是直接执行：

```python
self.target_feats = iu.process_target(...)
```

新版增加：

```text
asy_motif=False -> 保持原始 RFdiffusion 路径
asy_motif=True  -> 执行 Interface-Seed 初始化路径
```

这使常规功能理论上仍可使用原始处理方式。

### 6.3 随机采样 seed 旋转

从配置读取三个角度范围：

```python
x_range, y_range, z_range = asy_motif_rot_range
ran_x = randint(-x_range, x_range)
ran_y = randint(-y_range, y_range)
ran_z = randint(-z_range, z_range)
```

特点：

- 使用整数角度；
- 上下界均包含；
- 角度从 degree 转换为 radian；
- 手工构建 3×3 Euler rotation matrix；
- 对 `xyz_27` 的全部 27 个 atom slots 应用同一个旋转矩阵；
- A/B 两个 fragment 作为一个整体旋转，二者相对几何不变。

变换形式：

```text
x_rot = R(rx, ry, rz) · x
```

### 6.4 随机采样初始位置

```python
dist = randint(dist_init - dist_range, dist_init + dist_range)
self.target_feats['xyz_27'] = self.target_feats['xyz_27'] + dist
```

代码行为是给 x、y、z 三个坐标同时加同一个标量：

```text
t = [dist, dist, dist]
|t| = sqrt(3) * dist
```

因此配置名虽然写作 radius/distance，但当前公开实现不是沿单一径向轴平移 `dist` Å；实际平移向量长度为 `sqrt(3) × dist`。源码对此留有：

```python
### TODO: fix radius initialize
```

### 6.5 保存 pose metadata

运行时写回：

```python
inference.final_rot = [ran_x, ran_y, ran_z]
inference.final_dist = dist
```

因为最终输出 `.trb` 会序列化完整 config，这两个值会进入 `.trb` metadata。

README 声称输出 JSON 记录这些值，但当前 `scripts/run_inference.py` 实际写出的是：

```text
PDB + TRB (pickle)
```

公开代码中没有新增 JSON writer。

### 6.6 自动对称展开 A/B seed

对于 symmetry order `n`：

```text
copy 0: A, B
copy 1: C, D
copy 2: E, F
...
copy i: alphabet[2i], alphabet[2i+1]
```

每个 copy 使用已有：

```python
self.symmetry.sym_rots[i]
```

进行旋转。展开内容包括：

- `xyz_27`；
- `mask_27`；
- `seq`；
- `pdb_idx` 的链 ID 重命名。

这一步调用了原生 `symmetry.py` 提供的旋转矩阵；Interface-Seed 新增的是“用这些矩阵展开双链 seed 和重新命名链”的逻辑。

### 6.7 HETATM 处理行为

代码创建了 `xyz_het` 和 `info_het` 的空容器，但分支中使用：

```python
np.cat(...)
```

NumPy 没有 `np.cat` API，通常应为 `np.concatenate`。同时判断条件为：

```python
if not isinstance(xyz_het, np.ndarray)
```

该逻辑可疑。LHD101 示例不含 HETATM，因此不会覆盖这条路径。当前公开实现不能视为已经支持 ligand/heteroatom 的 Interface-Seed。

### 6.8 自动识别 motif fragment 长度

内部函数 `cal_inter_A_len` 从 `pdb_idx` 开头遍历，遇到第一个 chain ID 变化时返回第一条链长度。

隐含假设：

- 输入记录按 chain 排序；
- 第一段是 A；
- 第二段是 B；
- 恰好存在一次有效的 A→B 边界；
- 不支持 interleaved chain records。

### 6.9 识别相邻 symmetry copy

展开后，代码比较：

```text
第一 copy 的 fragment A COM
第二 copy 的 fragment B COM
最后一个 copy 的 fragment B COM
```

通过两组欧氏距离判断 A 应连接“顺时针相邻”还是“逆时针相邻”的 B fragment。

COM 使用 atom slot `0`：

```python
xyz_27[:, 0]
```

在 RFdiffusion atom ordering 中通常对应 N，而不是 CA 或全 backbone COM。因此这里实际比较的是 N 原子坐标平均值。

### 6.10 提前采样 linker 长度

用户输入：

```text
Y211-241/70-100/X165-194
```

Interface-Seed 在交给 `ContigMap` 之前先识别以数字开头并包含 `-` 的片段：

```python
length_inpaint = randint(70, 100)
```

然后将范围替换成固定整数，例如：

```text
Y211-241/77/X165-194
```

该长度只采样一次，因此同一个对称设计的所有亚基使用相同 linker 长度，保证长度对称。

### 6.11 X/Y 占位符展开为跨界面 contig

以 C3 为例，代码会产生两种可能的邻接映射之一。

方向一：

```text
F.../linker/A...
B.../linker/C...
D.../linker/E...
```

方向二：

```text
B.../linker/E...
D.../linker/A...
F.../linker/C...
```

不同 output-chain block 用空格分隔，之后交给原生 `ContigMap`。

意义：每个最终亚基连接相邻两个 interface copies 的 motif fragments，而不是连接同一个原始 A/B copy。

### 6.12 替换 sampler 的 contig 配置

代码：

```python
contigmap = copy.deepcopy(self._conf.contigmap)
contigmap.contigs = [expanded_contig]
self.contig_conf = contigmap
```

随后恢复原始 RFdiffusion 流程：

```python
self.contig_map = self.construct_contig(self.target_feats)
```

也就是说，`contigs.py` 未被修改；Interface-Seed 通过在调用它之前把特殊 X/Y 语法转换成标准多链 contig 来复用它。

## 7. `run_inference.py` 的全部算法改动

### 7.1 新增 `string` 导入

用于按照：

```text
A/B, C/D, E/F, ...
```

选择每个 symmetry copy 的两个 motif chain IDs。

### 7.2 新增 motif 接触计数函数

```python
count_motif_inter(x_t, sampler)
```

计算过程：

1. 只取第一个 symmetry unit；
2. 每个残基取 atom index `1`，即 CA；
3. 计算完整 CA–CA 距离矩阵；
4. 距离 ≤ 8 Å 记为接触；
5. 对每个残基统计接触数；
6. 使用 `contig_map.mask_1d` 选择 motif residues；
7. 对 motif residues 的接触数求和。

注意：

- 距离矩阵包含 residue 与自身的距离，因此每个残基至少贡献一个 self-contact；
- residue pair 会从矩阵两个方向出现；
- 这是启发式 contact count，不是标准 interface contact/buried-SASA loss。

### 7.3 记录初始 motif contact count

在 `sample_init()` 后、reverse diffusion 前：

```python
motif_inter_init = count_motif_inter(x_init, sampler)
```

只在 `motif_drag=True` 时执行。

### 7.4 在每个 denoising step 前执行 dragging

除第一个 timestep 外，每一步都在调用原始：

```python
sampler.sample_step(...)
```

之前执行 motif dragging。

因此完整循环变成：

```text
当前 x_t
  -> Interface-Seed 坐标调整
  -> 原始 RFdiffusion sample_step
  -> 下一 timestep
```

### 7.5 提取每个亚基的 generated region

代码反转 `mask_1d`，取得非 motif residues：

```python
un_mask_1d = [not elem for elem in mask_1d]
x_t_unmasked = x_t[un_mask_1d]
```

随后假定所有亚基 generated-region 长度相同：

```text
unmasked_subunit_len = total_unmasked / symmetry_order
```

### 7.6 计算相邻亚基的目标中心

对每个 generated subunit：

```python
subunit_com = x_t_unmasked[subunit_slice].mean(dim=0)
```

由于 `x_t` 仍含 atom 维度，这会得到一个 `[n_atoms, 3]` 张量，不是单一 `[3]` 几何质心。

然后对前一个和当前 subunit 的结果取平均，并将列表循环移位一次，作为每个 interface seed 的目标位置。

### 7.7 dragging 权重

基础权重：

```text
weight = asy_motif_weight / total_timesteps
```

这是每一步相同的常数，不随 timestep 衰减。

若 `random_drag=True`，每一步额外采样：

```text
ran_drag ∈ {0.5, 0.6, ..., 1.5}
```

接触自适应比例：

```text
inter_drag = motif_inter_init / motif_inter_current
```

generated fraction：

```text
generated_fraction = generated_length_per_subunit / total_length_per_subunit
```

最终单步位移近似为：

```text
Δx = COM_difference
     × random_drag
     × contact_ratio
     × (asy_motif_weight / T)
     × generated_fraction
```

### 7.8 移动每个 symmetry interface motif

对第 `i` 个 copy，选择：

```text
chain[2i] 和 chain[2i+1]
```

对应的所有 motif residue，然后计算：

```text
com_diff = target_subunit_pair_com - interface_com
```

并将位移加到所有相关 motif residue。

由于 `com_diff` 形状是 `[n_atoms,3]`，不同 atom slot 可能获得不同平移向量。这不是严格的 rigid-body translation；N、CA、C、O 等 atom slots 之间可能发生不同的整体位移。

最后调用：

```python
x_t = x_t.nan_to_num()
```

将 NaN/Inf 转换成数值后再进入原始 denoising step。

## 8. 示例任务新增内容

新增 `examples/design_interfaceseed_oligos.sh`，配置：

```text
checkpoint:       Base_ckpt.pt（由原始 checkpoint selector 决定）
timesteps:        50
symmetry:         C3
input:            7mwr_interface.pdb
motif:            Y211-241 + X165-194
generated linker: 70-100 aa
num_designs:      10
rotation range:   ±180° on x/y/z
distance:         25 ± 5
motif dragging:   enabled
```

同时启用原生 oligomer contact potential：

```text
type: olig_contacts
weight_intra: 1
weight_inter: 0.1
olig_intra_all: True
olig_inter_all: False
guide_scale: 2.0
guide_decay: quadratic
```

这些 potential 参数不是 Interface-Seed 新实现，而是把已有 RFdiffusion potential 与新初始化/dragging 组合使用。

## 9. 输出行为

沿用原始 RFdiffusion 输出：

- 最终 `.pdb`；
- `.trb` pickle metadata；
- 可选 trajectory PDB；
- pLDDT stack；
- contig mappings；
- 完整 resolved config。

Interface-Seed 通过 resolved config 间接新增：

- `final_rot`；
- `final_dist`；
- 展开后的 contig；
- 所有 Interface-Seed 参数。

公开实现没有新增独立 JSON 输出文件。

## 10. 已识别的限制与兼容性风险

### 10.1 新版 chain-handling 回归

当前上游链输出逻辑要求一个 designed chain 中的固定 motif 只来自一个 source chain ID：

```python
assert len(chain_ids) == 1
```

Interface-Seed 每条输出链故意连接两个内部 source chain IDs，因此公开 example 会发生：

```text
Error: Multiple chain IDs in chain
```

这是公开 fork 与其上游基线的兼容性问题，不是用户双链 PDB 错误。

### 10.2 没有 Interface-Seed 单元测试

- 功能提交未增加测试；
- 未增加 reference output；
- example 测试解析器只识别以 `python` 或 `../` 开头的命令；
- 新 example 以 `./scripts/run_inference.py` 开头，因此没有形成可靠的回归覆盖。

### 10.3 平移参数不等于真实 radius

`xyz + dist` 实际沿 `[1,1,1]` 平移，模长为 `sqrt(3)×dist`。

### 10.4 dragging 不是严格刚体变换

atom-wise COM 导致不同 atom slots 可能获得不同位移。

### 10.5 除零风险

```text
motif_inter_init / motif_inter_current
```

没有 epsilon 或显式零值保护。

### 10.6 `nan_to_num` 可能掩盖数值异常

NaN/Inf 被静默替换，可能使任务继续运行但产生异常结构。

### 10.7 HETATM 路径未经验证

存在 `np.cat` 和可疑类型判断，LHD101 示例不覆盖该分支。

### 10.8 输入结构假设较强

- 只显式处理 A/B；
- chain records 必须连续排序；
- 使用英文字母生成内部 chain IDs；
- 两个 fragment 被视为整体刚体；
- 实际上只适合 cyclic symmetry；
- 未充分验证 hotspot/inpainting 等组合。

### 10.9 代码中的可疑拼写

```python
string.ascii_uppercase[22 * order + 1]
```

该表达式只出现在 `order == 0` 分支，因此当前等价于 index 1，没有在 C3 示例中直接造成越界，但明显像 `2 * order + 1` 的笔误。

## 11. 方法创新与原生能力边界

### 11.1 可以归因于 Interface-Seed 的创新

1. 接受 A/B 双链非对称 interface seed；
2. 在 inference 时随机采样 seed 的整体旋转；
3. 在 inference 时随机采样 seed 初始位置；
4. 自动沿 cyclic symmetry 展开 seed；
5. 自动判断顺/逆时针相邻 interface pairing；
6. 将 X/Y 特殊 contig 转换为标准多链 contig；
7. 保证所有 symmetry copies 使用相同 sampled linker length；
8. 在 denoising 过程中根据亚基中心和 motif contact 动态移动 seed；
9. 在 metadata 中保存 sampled pose 参数。

### 11.2 不能归因于 Interface-Seed 的能力

1. RFdiffusion 模型本身；
2. motif scaffolding 基础能力；
3. cyclic symmetry rotations；
4. `ContigMap` 解析器；
5. `olig_contacts` potential；
6. quadratic guide decay；
7. PDB/TRB/trajectory writer；
8. checkpoint 选择和 Hydra 配置系统。

## 12. 迁移到 RFD3 时必须重实现的模块

| RFD1 Interface-Seed 行为 | RFD3 迁移要求 |
|---|---|
| A/B 双链 seed 输入约定 | 新的 wrapper schema 或扩展 RFD3 InputSpecification |
| Euler rotation sampling | 独立、可测试的 SE(3) geometry 模块 |
| 初始 distance/radius sampling | 明确定义径向轴和真实 Å 单位的 translation 模块 |
| cyclic copy expansion | 使用 RFD3 symmetry conventions 生成预对称输入，或接入 symmetry sampler |
| 相邻 fragment pairing | 基于 transform IDs/chain mapping 的显式邻接图 |
| X/Y contig rewrite | 转换成 RFD3 `contig`/`unindex`/`length` specification |
| motif contact count | RFD3 atom/token 表示上的 contact objective |
| step-wise motif dragging | RFD3 sampler hook 或外部 pose-search；不能直接复制 `x_t` indexing |
| final_rot/final_dist metadata | RFD3 output JSON/metadata extension |
| chain output mapping | 使用 RFD3/AtomWorks transform-aware chain mapping |

## 13. 建议作为“升级”的修正

如果你的项目称为 Interface-Seed for RFD3，可以把以下内容明确列为工程升级，而非仅迁移：

1. 将几何逻辑从 sampler 巨型函数拆成独立模块；
2. 使用 quaternion 或标准 SE(3) 库，避免手写 Euler matrix；
3. 使用真正的 radial translation；
4. 用刚体 pose 更新代替 atom-slot-wise displacement；
5. 对 contact ratio 增加 epsilon、clamp 和日志；
6. 不用 `nan_to_num` 静默吞掉异常；
7. 支持明确的 transform/chain mapping；
8. 支持 PDB/CIF 与 HETATM/ligand；
9. 增加输入 validation 和 `prevalidate`；
10. 增加单元测试、integration test 和 deterministic smoke test；
11. 将 sampled pose、每步 loss、contact、clash 写入结构化 JSON；
12. 将旧版行为保留为 `legacy` 模式，方便严格对照。

## 14. 建议作为“新功能”的方向

### 14.1 Fragment-wise SE(3) sampling

允许 A/B fragments 在保持局部结构的同时进行独立、小幅相对 pose 优化，而不是只能整体移动。

### 14.2 Time-dependent guidance schedule

早期强调 distance/orientation search，后期强调 motif fidelity、contacts 和 clash avoidance。

### 14.3 Multi-interface / non-equivalent seeds

支持两个以上非等价 interface seeds，用于 cage 或多功能寡聚体。

### 14.4 Adaptive radius and orientation search

根据当前 contact、clash 和生成骨架 COM 在线调整 pose，不再使用固定随机 dragging 系数。

### 14.5 Explicit geometry objectives

引入可解释的：

```text
interface distance loss
relative orientation loss
motif RMSD loss
contact loss
clash loss
symmetry consistency loss
```

### 14.6 Sidechain-aware interface conditioning

利用 RFD3 all-atom 表示，对关键 sidechain atoms、氢键供受体或 ligand contacts 进行约束。

## 15. 汇报用一句话总结

> 原始 Interface-Seed 没有重新训练 RFdiffusion，而是在 RFD1 inference pipeline 中加入了双链非对称界面 seed 的随机 SE(3) 初始化、cyclic copy 展开、跨相邻界面 contig 重建，以及基于亚基中心和 motif contact 的逐步 dragging；我们的升级将这些启发式、与 RFD1 数据结构耦合的逻辑重构为适用于 RFD3 all-atom sampler 的模块化几何 conditioning，并进一步支持 fragment-wise pose optimization、time-dependent guidance 和多界面设计。

## 16. 可复现审计命令

```bash
REPO=/home/haixi/Documents/HYC_repeat/RFdiffusion_interfaceseed

git -C "$REPO" diff --name-status 2d0c003..fc0df9d
git -C "$REPO" diff --stat 2d0c003..fc0df9d
git -C "$REPO" diff 2d0c003..fc0df9d -- \
  config/inference/interface_seed.yaml \
  rfdiffusion/inference/model_runners.py \
  scripts/run_inference.py \
  examples/design_interfaceseed_oligos.sh

git -C "$REPO" diff fc0df9d..a81ed19 -- README.md
```
