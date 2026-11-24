# 体系结构与接口契约

当前是可演进研究架构，不预先指定最终压缩算法。已有源码在src内实际集成；未来物理模块只在通过阶段门禁后添加，不用空实现冒充能力。

```text
MORQEX/
  src/morqex/
    geometry.py             # 已实现：显式相语义/SI/不可变标签/几何指纹
    workspace.py            # 已实现：外部路径约束
    gemsim.py               # 已实现：内部微结构与统一数据契约转换
    microstructure/         # 已集成：原GEMsim全部src/gem源码
      core/                 # 颗粒图、体素场、分布、光栅/Cython/EDT
      generate/             # v0、条件扩散、混合/空间/重构/选择
      mesh/                 # 共节点体素Hex/Tet、曲面、TetGen、ACVD
      simulate/             # NGSolve输运/扩散/均匀化、PyBaMM、echemAMR源码
      resources/            # 小文本物理输入模板
      volume.py, image.py, analysis.py, pipeline.py, reporting.py ...
  configs/                  # 默认单CPU资源与研究种子
  literature/               # 参考与能力元数据
  provenance/               # 上游提交、逐文件迁移hash与原构建说明
  benchmarks/, tests/, scripts/, docs/
```

```mermaid
flowchart LR
  CT[CT或生成结构] --> G[集成微结构与显式相/尺度契约]
  G --> M[共节点网格/真实界面]
  M --> R[待验证3D参考后端]
  R --> C[守恒浓度基候选]
  R --> H[同源U/S/H能量算子]
  C --> H
  H --> A[宏观应力/能量/机械mu]
  H --> B[加权应力统计/尾部]
  H --> L[按需ROI或全场]
  A <--> D[未来DFN状态替换/宏观力学]
  E[分源误差/适用域] --> D
  E --> R
```

## 已实现的几何入口

```python
from pathlib import Path
from morqex import Workspace
from morqex.gemsim import GEMsim
from morqex.microstructure import Volume

ws = Workspace(Path('/home/zephyrain/lw/morph_workspace'))
provider = GEMsim(ws)  # 仅导入当前src，无Git/外部checkout依赖
data = provider.random_field(
    shape=(8, 8, 8), fractions=(.3, .6, .1), length_scale=(1.,1.,1.),
    spacing=(1e-6,)*3, seed=7,
    phase_kinds={0:'pore', 1:'active', 2:'cbd'},
)
mesh = provider.mesh(data, cell_type='tetra')
provider.to_volume(data).save(ws.path('outputs', 'example.npz'))
```

`MicrostructureData`为面向物理的契约，`morqex.microstructure.Microstructure`为原GEMsim流式facade，两者含义不同。标签ZYX、spacing/origin也ZYX；点坐标XYZ。相标签、颗粒ID和晶粒ID不得混为一个字段。当前后两项未由契约自动推断；未来以独立可选数组及出处加入。

## 后续模块的输入/输出（尚未实现）

| 模块 | 输入 | 输出及不变量 |
|---|---|---|
| 材料 | 相/晶粒取向、c/T、出处/单位、参考态 | Mandel C、eps_ch及导数；SPD/对称；未经校准标记 |
| 参考力学 | 网格、材料、BC、宏观应变/压力、本征场 | 位移/应变/应力/反力/能量/真实残差/收敛状态 |
| 浓度状态 | 真实扩散域/界面、质量M、反应摩尔通量、初态 | a、库存、可行性、界面trace；不重复DFN锂状态 |
| 能量响应 | 相同离散约束、材料/模式/参考态 | H/h/w0、U/S、解析梯度/切线、适用域与hash |
| 均匀化A | q、能量算子、宏观BC | 应力/能量/模式mu；成本不遍历微网格 |
| 统计B | q、分相矩/采样、体积权重、风险定义 | 均值/协方差/尾部+误差标签；非线性风险不伪装二次型 |
| 局部化C | q、区域/相/颗粒选择、U/S分块 | 实际请求区域的场；不默认每时步输出全部 |
| 误差/适应 | 分源残差、目标、适用域、资源预算 | 保持状态/质量的富集或回退，含滞回与失败原因 |
| 电化学耦合 | DFN非固扩散状态、替换后的a、机械mu/trace | 同一反应源与电荷、能量功率；OCV机械项只计一次 |
| 宏观力学 | 各位置算子、几何/层合、载荷 | 电极/电芯应变应力、外功、代表性误差 |

算子manifest至少记录schema、几何/材料/基/BC/求积hash、代码提交、单位、张量顺序、参考态、有效范围、精度、离线残差及源数据。禁止仅以文件名作为缓存身份。路径/状态恢复失败必须显式报错。

## 构建与可选依赖

纯Python基础wheel已支持；微结构图像/网格依赖在`geometry` extra，AI与仿真在独立extra。默认安装不编译Cython或AMReX，不触发训练/下载数据。`setup.py`通过显式 `MORQEX_BUILD_GEOMETRY=1` 启用六个Cython扩展；需先在外部构建环境安装Cython>=3.1、NumPy和C++编译器，再用`--no-build-isolation`构建。生成C++、对象、metadata、wheel均外置。此可选全原生构建尚未数值认证。

echemAMR保留独立原生构建入口和原模型/许可证；默认CPU。安装它不代表参考方程与容限已通过MORQEX资格门禁。未来若后端成为默认，须为原始量输出、收敛和物理导数增加统一适配并单独验证。
