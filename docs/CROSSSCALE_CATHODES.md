# 0.0.9：微结构—DFN退化端口、掺杂与三类正极

研究主线是几何/相/成分→守恒输运和能量端口→DFN状态演化→局部场恢复，以及退化状态对这些端口的反馈。宏观平均SOC不唯一决定真实微结构；低维浓度和退化库存是随时间积分的独立状态。本文档的接口范围不等于所有材料机理均已通过实验验证。

## 微结构到DFN的实际变量

对每个电极厚度积分点，输入为电解液浓度、电势差和温度；微结构状态为区域固相浓度、SEI/析出锂/死锂库存及SVE接触损伤。几何离线给出体积权重、反应面积、扩散图、电子/离子有效输运和变化界面能量核。区域浓度替换原DFN径向状态，不重复计锂。

由同一能量核给出宏观应变、弹性能、机械化学势、温度导数、损伤驱动力及一致切线。主/副反应电流共同回到DFN电荷/电解液方程；孔隙、膜电阻、可达面积和电子输运随退化变化。场恢复使用相同核与当前状态，能够输出浓度、应力和界面位移跳量。

本版明确取消旧原型中固相图扩散对`1−d`的统一乘子：CBD接触损失并不自动降低完整活性宿主内部的本征扩散。D只由声明的浓度/温度/掺杂定律改变；接触对可达面积/电子输运的现象学关系单独保留。该模型修正后的轨迹必须重算，不能沿用旧数值。

## NMC、LFP与LCO参数入口

```python
from morqex.electrochemistry.cathodes import cathode_parameters
p, audit = cathode_parameters('NMC')  # 或 LFP / LCO
```

| 入口 | 原始电化学参数来源 | 当前限定 |
|---|---|---|
| NMC | PyBaMM固定OKane2022，含Chen2020 NMC参数 | 明示的凸化/平滑OCP；不是目标CT样品校准 |
| LFP | 固定Prada2013包及其中Afshar2017有效OCP | 有效单相自由能；未解析两相前沿、马赛克转变或滞回；原包缺失的热/力学项逐项列为借用补充 |
| LCO | 固定Ai2020，独立LCO OCP/扩散/反应与容量 | 为现有能量原函数显式关闭化学OCP熵变，不能当作原文全部热行为复现 |

全部入口保留原始包名/源hash、OCP投影改动、压缩误差、补充参数与理想化列表。公开参数包与经过本模型投影的参数必须区分；真实NMC CT形貌换用LFP/LCO物性是受控的假想几何实验，不能改称真实LFP/LCO CT。三类材料分别选择，并不意味着已实现三种宿主混合在同一颗粒内的多物种模型。

## 掺杂位置与物性映射

`DopantLaw`声明掺杂物种、替位位点、来源，以及OCP偏移/斜率、扩散/主反应/副反应倍率、固定弹性和化学应变倍率。掺杂率是明确定义的原子或位点分数，不能直接当质量分数。当前为固定掺杂场，且同一电极共用宿主最大浓度；掺杂改变容量或形成新宿主时，必须使用新的多物种/容量闭合，不允许套入本假设。

```python
from morqex.doping import (
    DopantLaw, RegionalChemistry, CompositionBasis, DopedOperator,
    composition_materials,
)

law = DopantLaw(
    species='X', site='declared transition-metal substitution site',
    source='hypothetical sensitivity example; not measured',
    offset_V_per_fraction=.5, slope_V_per_fraction=-.2,
    log_diffusion_per_fraction=-10., log_kinetics_per_fraction=-5.,
    log_side_reaction_per_fraction=-10.,
    log_young_per_fraction=-5., log_chemical_strain_per_fraction=-3.,
)
# active_fractions和原活性体素一一对应；base_catalog来自材料数据。
derived_geometry, catalog = composition_materials(
    source_geometry, active_fractions, law, base_catalog,
)
# 构建MicrostructureElectrode时传materials=catalog。
# 浓度区域不能无声明地混合不同掺杂量：
basis = CompositionBasis(base=base_basis, fractions=active_fractions)
operator = DopedOperator(fractions=active_fractions, law=law)
```

`composition_materials`生成有出处的派生相标签与分相C/化学应变，保留原几何指纹和源hash。`CompositionBasis`在成分边界处分区，必要时增加区域，必须计入维数和成本；不暗中把不同掺杂区平均为一种材料。微结构局部的OCP、扩散、动力学及机械参数确实进入DFN和场恢复。Profile随算子缓存保存并在独立加载/运行时恢复。

当前OCP修正为温度无关的仿射形式，并限制额外负斜率以保持该有效化学自由能的凸性。其解析原函数纳入独立能量审计。不同成分间采用正曲率迁移率，避免将成分导致的化学势差除以浓度差而产生负扩散。更一般OCP可替换`ocp`及粒子子模型，但必须同时提供相容自由能与热项。

示例系数只用于灵敏度和接口验证，不代表Mg、Al、Ti等真实掺杂元素的性能。元素名称本身不能确定物性。当前运行脚本支持uniform/gradient/core-shell三种演示布局；core-shell按选定几何域的距离定义，不宣称已识别所有真实颗粒的独立包覆层。

## 保持结构的界面算子压缩

完整核为`H(d)=D−Fᵀ A(d)⁻¹ F`，`A=I+s(d)diag(ρ)>0`。选择正交Q，令`T=Qᵀdiag(ρ)Q`、`B=QᵀF`，得到`Hr=D−Bᵀ(I+sT)⁻¹B`。再对T对角化，保留原有正极点/非负残数结构；局部场使用同一Q投影。训练只使用算子和预定损伤采样，不读取验证轨迹。

设`z=(QᵀAQ)⁻¹QᵀFq`，残差`r=Fq−AQz`，`α=1+s maxρ>0`。Galerkin正交给出

`0 ≤ Wr−W = 1/2 rᵀA⁻¹r ≤ ||r||²/(2α)`。

相同q,d下，广义力第i分量误差不超过`||F[:,i]|| ||r||/α`。完整应力响应列的Frobenius范数给出体积L2和逐点保守界；Weyl不等式可将逐点界传至主应力与体积加权分位数。残差通过离线QR根矩阵计算，在线不遍历原谱/网格。代码加入浮点误差余量，但不声称区间算术意义的机器验证。

这些是成熟Galerkin/RB残差思想的本问题推导和实现；不单独列为原创。证书仅覆盖相同离散q、d下的界面压缩，排除浓度截断、DFN轨迹、时间/空间、材料及SVE代表性误差。训练网格达到目标不构成整个参数域的统一误差证明；每次查询仍需计算对应界。

## 退化状态富集与控制器

`AgingRuntime.enrich(new_paths)`支持同几何、同成分和同界面定律的嵌套细化。浓度按体积投影；SEI/金属锂/死锂按真实反应面积分配；宏观微分状态与损伤保留。检查锂库存、材料/成分相容性和弹性能跳变，超过预算拒绝且原实例不变。新增代数状态是初猜，后续推进重新求一致初值；不能把此投影当作已认证的非线性自适应策略。

共享控制器用一个电流代数未知量和离散输入切换CC/rest/CV方程。只有命名的协议停止事件允许继续，浓度/孔隙/电压等适用域事件不允许越过。该方法避免为每个控制阶段重复编译，需与原生Experiment按相同协议对照。紧凑线性端口和CasADi相同样条表复用都保持原系数和插值数据，不能把减少插值精度当作加速。

所有数值、场图、误差对照与资源结论以外部本版报告及源码hash为准。顶刊创新、实验准确性、全部微结构物理和普遍加速不由上述接口自动成立。

## 变化界面的瞬时反应闭合界

`error_estimation.aging_port.compile_reaction_certificate(kernel_path)`在离线读取局部应力图，针对增广输入`[q, Fq/(1+sρ)]`生成区域反应面积加权QR根矩阵。`AgingRuntime.reaction_bound(domain)`只读取这些紧凑数组；未编译时显式拒绝。它比较同一浓度/应变/温度/损伤及局部核下，区域机械化学势与局部机械化学势引起的反应闭合差，不包含浓度、算子压缩或DFN全过程误差。

隐式膜电流满足`z=jmain(δ−Rz−U+µ/F)+jside(δ−Rz)`。对被动单调副反应，令主、副反应对其电势的导数为`a,b≥0`，则膜方程导数为`1+R(a+b)≥1`。隐函数求导给出`|dz/dµ|≤a/F`，主反应自身亦满足该上界。BV的区间导数上界结合面积Cauchy–Schwarz不等式给出电流误差界；实现另计机械化学势与膜代数方程数值残差。自定义副反应必须另行证明单调性，不能继承默认资格。

## 多微结构与标准DFN对照

`scripts/run_generated_suite.py`绑定集成GEMsim的固定种子生成器，三组3³SVE分别改变相关长度方向和CBD比例。每例的完整固相与8区候选使用相同几何、宏观网格、容差、能量及退化定律；所有几何筛选原因记录，不按输出好坏挑种子。

`scripts/run_geometry_dfn_controls.py`构建标准球形DFN，匹配同一体积分数、几何电/离子输运和初始反应面积，采用`R=3εactive/a`。原生SEI/析锂与原生集总热不同于MORQEX端口定律，这一组是模型形式和成本对照；降阶误差必须另与MORQEX完整固相参照比较。低倍率不产生析锂时如实报告，另用从零金属库存开始的低温薄电极脉冲验证析锂/死锂。

失败分三类处理：实现错误修复后回归；截断误差通过模式富集重新对照；真实适用域事件保留端点并拒绝继续，不裁剪浓度或放宽守恒门禁。原CT5点0.1C充电在3.9888V处达到负极局部0.999上限，收紧100倍积分容差仍触发；它不是完整循环或网格收敛证据。LFP8区场误差未过门槛时同样不能计作达标。完整循环、电压切换最大误差、时间积分误差和输出成本分别报告。

该CT的5点0.06C、3.5–4.0V三循环已通过原物理和数值门禁，属于明确改变倍率后的可行工况，不把它冒充0.1C成功。另一个0.05C尝试的CC电流恰等于固定C/20的CV停止阈值，CV初始事件不可行；这是协议配置错误，须保证CC电流大于CV停止电流，不能强行积分零长度的CV阶段。方向性合成SVE在1e-9/1e-10容差下的三循环作业触及CPU120s上限，默认1e-7/1e-8同物理配对重跑，并用更紧容差的首循环单独核对，不提高资源预算。
