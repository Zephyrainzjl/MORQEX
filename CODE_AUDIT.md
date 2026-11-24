# Phase 0：代码、资源与物理审计

日期：2026-11-24。当前 ZynJoule 提交 `2ef98200e90a52e7a5e0c78ae1f420838b0b0cf6`；GEMsim 提交 `6fd789c78d58d49ef645c539414567fc2a2bc171`。审计前 ZynJoule 工作树干净；本任务不改动该仓库。

## 范围与证据级别

全量读取/指纹清单覆盖当前 ZynJoule 的 ZynSim、ZynMorph、physics、根级文档和配置，共 354 个文件；GEMsim 固定提交全部 336 个跟踪文件，约 3.73 MB。另建立当前 ZynJoule README/RESUME/日志/配置发现清单。机器清单位于外部 `evidence/source_audit.json`，包含 SHA256、源码定义和行号。

**清单扫描不等于每一行都已经经过物理认证。** 人工语义审读重点为下述实际计算路径、README、pyproject、根级研究说明、GEMsim 迁移清单、API 清单、测试和原生构建/关键物理源码。lw 中既有训练运行、历史副本和原子模型数据不属于当前连续体审计对象，没有递归运行它们。当前相关源目录未发现 RESUME 或专用研究日志，不能以历史训练日志代替本项目证据。

证据等级：S=源码检查，P=本地小规模独立探测，T=原测试执行，U=未执行/未验证。所有输出在 `morph_workspace`；源码清单与数值检查分开记录。

## 十二项核查

| 项目 | 已确认事实 | 复用决定与未解决项 |
|---|---|---|
| 微结构生成 | GEMsim `gem.design.random_field` 为周期滤波随机场基线；`generate/v0` 保留 GAN，v1/混合生成需要 Torch/Diffusers；ZynMorph 另有球/椭球颗粒、CBD 与堆积生成 | 直接接入 GEMsim；本阶段运行随机场，不训练网络；二维到三维重构的统计非唯一性必须保留 |
| 格式 | GEMsim Volume 数组/spacing/origin 为 ZYX，网格点为 XYZ，SI 米；NPZ 禁用 pickle；VTU/COMSOL 各自转换 | 已测试非等距、非零原点与正体积；相 ID 不可猜测 |
| 相定义 | GEMsim 保留任意整数相 ID；echemAMR 使用 1..7 专用相；ZynSim 全电芯使用另一套 1..5 定义 | 必须显式 phase map；127/128/255 不自带物理含义；CBD 与孔隙分开 |
| 边界 | GEMsim transport：对置面势差+侧面绝热；periodic.py 是独立周期输运实现；elasticity.py 是 KUBC | 三者不得混称周期均匀化；力学 PBC、SUBC 待实现验证 |
| 参数来源 | GEMsim cell 指向 PyBaMM `Chen2020`；ZynSim OCP/几何参数为文献量级参考；NMC 力学默认明确未校准 | 来源链必须追至原文表格/版本；默认参数不得得出真实材料机理结论 |
| 有效输运 | NGSolve FEM 能量积分与方向导通；孔道贯通分量曲折度；独立完整 3×3 周期输运张量源码 | 非贯通分量、区域面积及体积归一化需基准；NGSolve 当前缺失，数值测试跳过 |
| 三维浓度 | GEMsim evolution 为隐式标量扩散；echemAMR 为笛卡尔 AMR 物种/电势/BV；ZynSim FFT 有占位率演化；3D battery 保留每宏观单元球形径向状态 | 不能把后者认作任意真实颗粒浓度；真实界面通量闭合尚需新实现 |
| FEM/FV/FFT | ZynSim Tet4 稀疏 FEM、FV P2D、FFT 力学已存在；GEMsim NGSolve 与 echemAMR C++/AMReX 源码存在 | Python Tet4 可作交叉检查；保留独立 MORPH 求解器候选，不能将既有混合模型整体当真值 |
| 硬件 | Ryzen 7 9800X3D，16 逻辑 CPU；约30 GiB RAM；RTX5080 16303 MiB，审计时已用7749 MiB | 单 CPU、GPU禁用；硬件测量是时点快照，运行前重新检测 |
| 可直接复用 | GEMsim Volume/网格/随机场与原生几何导出经本地桥接测试；ZynJoule 的分层命名、物性 provider、诊断结构可借鉴 | 无 ZynJoule 运行依赖；其源码仅只读审计，不复制整个大框架 |
| 需重构 | 张量约定、周期约束、非均匀本征应变、浓度质量闭合、统一自由能导数和残差诊断 | 新增独立接口；原稳定代码保持不动 |
| 正确性/性能 | 只确认小规模探测，未进行真实三维网格收敛和同精度性能比较 | 不报告算法加速倍数，不用成功导入或残差小代替正确性 |

## 关键源码与实测结果

路径均相对于 `lw/ZynJoule/src/zynjoule` 或外部 `upstream/GEMsim/src/gem`。

1. `zynsim/fem/assembly.py`、`elements.py`、`problems.py`：Tet4 小变形各向同性，含本征应变载荷和反力。Python 弹性装配逐单元循环，每单元 144 个 COO 项；每次 `solve()` 重装配并求解，无默认多 RHS 分解复用。独立 2³ 体素自由本征应变探测：最大应力约 `2.60e-17`（E=100 的合成单位算例），位移误差 `1.08e-19`。不据此认证复杂界面或各向异性。
2. `zynsim/battery/p2d/{model,numerics,parameters}.py`：FV 球形固相扩散、势方程非线性求解及外层 Picard，守恒诊断存在。2/2/2 宏观网格、3 个径向单元、0.1 A、0.01 s 的独立单步得到 3.616350055 V，势残差 `1.38e-11`、库存误差约 `1e-16`。没有执行全倍率/全循环/网格收敛或与 PyBaMM 对照，不能当 DFN 已完全验证。
3. `zynsim/battery/threed/model.py:_mechanics_step`：由每单元球形浓度平均构造各向同性本征应变，指定边界固定，返回位移；电化学求解路径未见由此弹性能变分构造的多模式机械化学势。该实现并非本需求的完整微结构双向解。
4. `zynsim/battery/mechanics/multiscale.py:synchronize_theta`：均值平移后逐点裁剪。构造 27 个活性体素（13 个 0.01、14 个0.99），请求均值0.95，实测0.7312619509，误差0.2187380491。**确认该同步在裁剪激活时不守恒**；方法名/注释的 conservative 不能当证据。MORPH 后续应使用带盒约束和质量等式的投影，投影失败显式回退。
5. `mechanics/spectral.py:_chemical_potential` 含 `-sigma:deps_ch/dtheta/cmax`；`materials.py` 刚度随浓度变化。固定总应变、均匀theta=0.5、h=1e-6 的局部能量导数探测：机械化学势有限差分 `331.3041007 J/mol`，实现值 `-197.5216778 J/mol`，差 `528.8257785 J/mol`。梯度项为零。**该组合不满足当前弹性能的完整浓度导数**；后续须加入 C'，并跨步长/状态验证。本探测不等同于验证其全部自由能和耗散律。
6. `spectral.py:_mechanical_equilibrium`：非活性区使用有限弱刚度，不是真空；自由宏观变形通过平均本征应变调整，并非求解零平均应力方程；外压同时影响平均应变和显式应力，需要独立外功/压力基准排查重复作用。循环中应变更新后返回此前应力，迭代超限仍返回结果，调用方不能只依赖返回对象判断收敛。默认 `fft_workers=-1` 可能用满 CPU，禁止沿用。本任务探测显式设为1。
7. `gem/simulate/elasticity.py`：仅 (E,nu) 各向同性 KUBC 六载荷，无浓度本征应变参数、周期力学/流体压力/刚体模态专项验证。`transport.to_ngsolve(periodic=True)` 的节点识别属于另一条输运路径，不能当作已有周期力学证据。
8. 原生 `simulate/native/upstream/Source/echemAMR_MechanicsSolve.cpp` 使用 AMReX MLTensorOp；机械 `mlmg.solve(...,0.005,0.0)` 有固定容限。Microstructure/Mechanics.H 使用 Lamé 参数及化学膨胀源。该默认容限不可直接作为1e-6能量导数参考。CMake 固定 AMReX、CPU默认、MPI/OMP关闭；尚未编译，也未核实所有各向异性/双向项，不宣布其具备本项目全部高保真能力。
9. **分量冲突**：ZynSim FEM 是 `[xx,yy,zz,xy,yz,xz]`，GEMsim 弹性是 `[xx,yy,zz,yz,xz,xy]`。工程剪切与张量剪切还相差2。MORPH 内部拟用 Mandel，任何后端转换需能量不变量测试。ZynSim 体素工具按 XYZ 解释轴，亦不能直接接收 GEMsim ZYX。

## 原测试执行与缺失依赖

执行 GEMsim 的 pipeline、physics、periodic、cell_contract 四个原测试文件：**6通过、9跳过、1失败**。失败为 COMSOL 导出链导入 `edt` 缺失；跳过来自 NGSolve、PyBaMM、Diffusers 缺失。保留原失败，不静默删除测试或自动安装大型 AI/原生依赖。MORPH 自身桥接5项通过。原始生成器/Cython/TetGen/ACVD/AMReX 未编译验证；其历史 Windows/Linux 成功说明不是当前机器证据。

当前基础解释器 Python3.13.9；NumPy2.3.5、SciPy1.16.3、pytest8.4.2、scikit-image0.25.2、tifffile2025.10.4。缺 NGSolve/PyBaMM/PETSc/AMG/Torch/meshio/edt。未来原生依赖优先外部 Python3.11/3.12 环境，与上游CI覆盖版本对齐；不修改现有 ZynJoule 环境。

## 结论和门禁

几何桥接可用，已有 FEM/P2D 有可供小规模交叉核对的基础；现有多尺度物理链不能原样作为参考。Phase0 的静态清单和关键路径审计已有可复现交付；未执行的原生与长程验证明确保留为 PhaseIII 的前置资格检查。下一阶段仅启动小规模独立参考解，不把未验证损伤、经验反馈或AI生成质量作为论文证据。

