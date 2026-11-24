# GEMsim

二维 CT → 条件化三维微结构 → 体素/有限元/COMSOL 网格 → 微观输运 → 连续尺度电池计算。
所有新代码从 `gem` 导入；源码放在 `GEMsim/src/gem`，数据、权重、图像和实验放在外部工作区。
原始 `PCG` 与 `data_process` 不修改。当前版本是科研框架，性能和质量以实验报告为准。

## 安装

Windows 使用 Python 3.11/3.12 和 Visual Studio 2022 C++ Build Tools；Linux 使用 GCC/G++。
在项目目录执行：

```powershell
python -m venv ../ResearchWorkspaces/.venv
../ResearchWorkspaces/.venv/Scripts/python -m pip install -e ".[ai,simulation,view,test]"
```

Linux 对应解释器为 `../ResearchWorkspaces/.venv/bin/python`。
`pip` 自动编译原 Cython 离散化、形态学、几何算法、TetGen 和 ACVD 五个扩展及新增的 C++ 体素网格内核。
EDT 使用发布的 C++ `edt` 包，避免复制旧 Python 3.10 的 `.pyd`。
根据显卡在 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)安装 CUDA wheel；普通 PyPI 安装可能得到 CPU 版本。
GPU 训练/推断使用 PyTorch CUDA 算子；此项不表示 NGSolve 串行 FEM 已支持 CUDA。

可选原生 echemAMR 插件使用独立的 pip 构建：

```powershell
python -m pip install ./src/gem/simulate/native
```

此插件在构建期间下载固定版本 AMReX，编译第三方 C++ 电化学求解器与 Python 绑定。
CPU 为默认配置。CUDA 构建需 CUDA Toolkit 和兼容 C++ 编译器：

```bash
python -m pip install ./src/gem/simulate/native -Ccmake.define.AMReX_GPU_BACKEND=CUDA
```

可使用 CMake 选项 `GEM_ECHEM_CASE=regression_tests/CEA` 切换原模型。
各 upstream 问题的 `Prob.H/Chemistry.H` 定义物理模型，不能任意混用其输入参数。
本仓库记录实际 Windows、Linux、CUDA 验证状态，未验证的平台不作保证。

## 一个贯通的数据结构

```python
from gem import Workspace, Volume, Microstructure
from gem.mesh import voxel_to_tetra
from gem.simulate.transport import conduction
from gem import view

ws = Workspace('../ResearchWorkspaces').initialize()
volume = Volume.load(ws.path('outputs', 'microstructure.npz'))
print(volume.fractions)
fig = view.slices(volume)
fig.savefig(ws.path('outputs', 'slices.png'))
mesh = voxel_to_tetra(volume)
mesh.save(ws.path('outputs', 'electrode.vtu'))
mesh.save_comsol(ws.path('outputs', 'electrode.mphtxt'))
result = conduction(mesh, coefficients={0: 1., 128: .01, 255: .001}, axis='x', order=2)
print(result.effective_conductivity, result.relative_residual)
```

数组维度固定为 `(z,y,x)`，`spacing`、`origin` 使用相同顺序；物理单位为米。
网格顶点为 `(x,y,z)`，转换时明确调换轴序。原始相位 ID 保留在单元属性中。
COMSOL 不允许零域号，导出时生成正整数域号，并保存 `*.phases.json` 显式映射。
`.npz` 使用非 pickle 格式保存数据、尺度与元数据；TIFF 使用 JSON 旁车保存物理尺度。

也可以使用流式入口：

```python
structure = Microstructure.load(ws.path('outputs', 'microstructure.npz'))
print(structure.describe())
structure.mesh().save(ws.path('outputs', 'mesh.vtu'))
```

## v1：条件扩散（重点）

训练仅接受一张代表性二维分割图，或按 `(xy,xz,yz)` 顺序给出三张图。
**完整三维 CT 只能进入独立评估进程**。二维未分割灰度 CT 请先使用 `gem.image.segment`，并人工确认相位语义。
真实灰度、RGB 图和相位标签不是同一种数据，接口会拒绝把三维 TIFF 或 RGB 当作二维相位图。

```python
from gem.generate import DiffusionConfig, train, sample

config = DiffusionConfig(
    image_size=64, phases=3, channels=(96,192,256,384),
    layers=2, attention=True, batch_size=32,
    steps=100000, time_budget_seconds=3300,
    morphology_variants=12, device='cuda',
)
checkpoint = train(
    [ws.path('data', 'training', 'nmc_xy.tif')],
    ws.path('runs', 'nmc_v1'), config,
)
volume = sample(checkpoint, fractions=(.5,.38,.12), seed=7, steps=50)
volume.save(ws.path('outputs', 'nmc_generated.npz'))
```

实现基于成熟 Diffusers residual UNet、低分辨率注意力、DDPM 训练和 DDIM 采样。
二维模型接收相位分数、平面方向、独立参考图 patch；三维生成在每一步融合三个方向的扩散分数。
训练包含 BF16、梯度裁剪、Min-SNR 权重、EMA、设备端随机裁剪、条件 dropout、检查点及优化器/RNG 恢复。
`train(..., resume=checkpoint)` 从保存步骤继续；日志逐步记录实际 wall time 和损失。
`time_budget_seconds` 是训练循环的软停止预算，初始化、显存探测、保存、评估另外计时。

参数化条件的支持范围由二维图中观测到的结构和相位比例决定。
`morphology_variants` 从输入二维图的有符号距离场生成膨胀/侵蚀变体，扩展组成条件；这些是合成增强，不是新的实测结构。
不能据此宣称任意组成下的形态都物理正确。应报告训练范围内、范围外的误差。
`exact_fractions=True` 是显式整数配额后处理，精确比例不能当作模型本身学到了精确组成。

生成更大的体积可使用 `sample(..., size=128, overlap=16)`。
模型在二维 patch 上通过重叠加权进行分数预测，所有方向始终更新同一个三维场。
大体积的长程相关与接缝需单独评估，不能直接外推 64³ 的结果。
`prior=gan_volume, strength=.5` 可使用同相位/同尺寸 GAN 体积作为扩散修复初值。

RTX PRO 6000 示例：

```bash
python examples/train_rtx6000.py --image /workspace/data/training/nmc_xy.tif --run /workspace/runs/nmc_v1 --seconds 3300
```

自动显存探测测量真实前向/反向峰值，选择留有余量的批量，不使用无意义的显存占位。
单卡一小时的质量与速度必须由目标卡上的日志、多个随机种子和三维指标证实。

## v1 混合生成：扩散教师 + 条件三维对抗学生

```python
from gem.generate.hybrid import HybridConfig, distill, sample_student

student = distill(checkpoint, [ws.path('data','training','nmc_xy.tif')],
                 ws.path('runs','nmc_student'), HybridConfig(seconds=1200))
volume = sample_student(student, fractions=(.5,.38,.12), size=128)
```

学生使用 FiLM 组成/参考图条件、三维插值卷积主干、二维谱归一化判别器。
损失结合扩散分数蒸馏、三个方向的 hinge GAN 和体积分数约束。
全流程不需要三维训练标签，学生一次前向即可生成体积。
该方法是本框架新增的实验性组合，需与纯扩散、v0 做消融；不能因为使用扩散教师就默认优于 v0。

## v0：保留原始算法，新统一入口

原 MGENERATE/EMR 的网络、训练、生成、图像处理实现保留在 gem.generate.v0；Python 模块重新命名，已有模型参数可追溯。

`python
from gem.generate import GANConfig, train_gan, sample_gan
cfg = GANConfig(images=(ws.path('data','training','nmc_xy.tif'),),
                output=str(ws.path('runs','v0','nmc')))
# train_gan(cfg)
# image, raw, net = sample_gan(cfg)
`

按最新需求，框架不包含 BatteryGan 包；其原始源码仍保留在 PCG，迁移副本已移到外部归档。
核心粒子、体素、网格历史算法改名保留；新代码优先使用本文的统一接口。
迁移 notebook 位于外部 ResearchWorkspaces/notebooks，历史 pickle 通过 gem.io.load_legacy 转换。

## 图像、网格和物理

`gem.image` 提供分割、TIFF 序列载入、连通域清理；`gem.design` 提供参数化随机场基线。
`gem.analysis` 提供相位比例、连通组分、贯通性、非周期且重叠归一化的方向两点相关和比较。
`gem.view.slices` 返回 matplotlib Figure，`gem.view.mesh` 使用 PyVista。
`gem.mesh.voxel_to_hex` 与 `voxel_to_tetra` 共用界面顶点，六分四面体具有全局一致的面剖分。
曲面等值面使用 `gem.mesh.surface`；原自适应点/颗粒/图结构、TetGen、ACVD、Voronoi 算法仍由 gem 命名空间提供。
避免直接将 256³ 体素展开为约一亿个四面体：先裁剪 RVE、降采样或使用原自适应网格。

`gem.simulate.transport` 使用 NGSolve 的原生 C++ 高阶 FEM，支持多相导热/导电/扩散的稳态同构问题。
默认两个对置面施加单位势差，侧面绝热；输出等效系数、能量和自由自由度残差。
此边界条件不是周期均匀化，不应把对角方向结果当作完整各向异性张量。
`gem.simulate.homogenization.homogenize` 批量计算三方向；`tortuosity` 用贯通孔道计算有效扩散和曲折度。
残差小不等于离散误差小；需要解析解、网格/阶次收敛和独立方法验证。

`gem.simulate.electrochemistry` 使用 upstream echemAMR 的电势、物种、Butler–Volmer、力学和自适应 AMR 源码。
它读取笛卡尔体素，不读取非结构四面体；`export_geometry` 显式映射标签及纳米孔隙参数。
各项物理的可用性取决于选择的 upstream 模型，不能以一个导通测试代表全耦合电化学验证。

`gem.simulate.cell.discharge` 对接 PyBaMM 的 DFN/SPMe/SPM 与 IDAKLU 编译求解器。
`microstructure_parameters` 显式映射孔隙率及已计算曲折度：

```python
from gem.simulate.cell import discharge, microstructure_parameters
updates = microstructure_parameters(volume, pore_phase=0, electrode='Positive', tortuosity=2.)
solution = discharge(model='DFN', updates=updates, duration=100.)
print(solution['Terminal voltage [V]'].entries)
```

## 评估与复现

```bash
python -m pytest -q
python examples/evaluate_reconstruction.py --checkpoint /workspace/runs/nmc_v1/checkpoint.pt --ct /workspace/data/evaluation/nmc_ct.tif --output /workspace/reports/nmc --seeds 3
```

二维到三维是欠定的统计重构问题，不能要求同一个体素位置复现唯一真实三维结构。
评估比较相位比例、三方向 S2、最大连通组分、贯通性、物理输运及多个随机种子的误差分布。
对比体积必须统一物理分辨率、相位含义和裁剪协议；标签 127/128 的映射应写入元数据。
旧版生成体不能混入真实 CT 训练或真实 CT 测试集。

训练权重、缓存和输出禁止加入源码仓库。`.gitignore` 排除二进制、构建目录、数组、TIFF 和模型权重。
原 `MGENERATE` 数据已复制到外部工作区；原始数据未删除，便于核对。
源代码迁移不等于所有旧 notebook 已成功执行，执行覆盖情况必须单独列出。

## 方法来源

- [SliceGAN](https://arxiv.org/abs/2102.07708)：二维判别、三维生成的原始路线。
- [条件二维扩散到三维微结构](https://arxiv.org/abs/2312.07832)：多正交平面联合反向过程的研究依据。
- [Diffusers UNet](https://huggingface.co/docs/diffusers/api/models/unet2d)、[DDIM](https://huggingface.co/docs/diffusers/api/schedulers/ddim)。
- [NGSolve](https://docu.ngsolve.org/latest/)、[PyBaMM](https://github.com/pybamm-team/PyBaMM)。
- [echemAMR](https://github.com/NatLabRockies/echemAMR)，固定源码版本与许可证见 `src/gem/simulate/native/upstream`。

第三方代码保留各自许可证与署名；请在论文中引用实际使用的方法和求解器。
