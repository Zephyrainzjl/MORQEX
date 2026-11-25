# 完整DFN与热—力闭合：0.0.6研究范围

2026-11-25，分支`research/dfn-coupled-cycling`。完整放电和可逆CC-CV循环已接入；本版本不是论文验收或真实电芯力学认证。数值结果以外部`reports/MORQEX_DFN_report.html`及`results/dfn/*/result.json`为准。0.0.5二次模型报告另存`reports/MORQEX_0.0.5_report.html`，旧速度倍率不能推广至DFN。

## 求解器与源码来源

PyBaMM **26.3.1** 的355个Python/文本资源文件迁入`src/pybamm`，保留BSD-3-Clause许可证和上游版权。GEMsim仍为原274个迁移文件。逐文件上游与本地SHA见[迁移清单](../provenance/pybamm_migration.json)。仅修改PyBaMM的版本标记及`convert_to_casadi.py`；自定义物理通过公开submodel接口实现。CasADi3.7.2、pybammsolvers0.6.0/IDAKLU二进制只装外部venv。使用成熟DFN求解器是A类工作，接入能量闭合属于待验证扩展，均不宣布原创。

本地复现了固定版本的导数转换问题：原转换忽略1D Interpolant的`_num_derivatives`，将U'(c)编译成U(c)。Ai2020初始负极化学曲率Python为+0.064921604，原编译值为−0.338748；这也影响上游混合热。补丁保留1D cubic/PCHIP的导数阶次，其他导数类型明确拒绝。测试包含一至三阶、标量/向量、插值节点、外推及自动微分。结论只针对固定版本，不推断其他版本。

曾测试PCHIP的重节点B样条压缩；节点处输出错误，向量回归捕获后已撤销，当前不保留该优化。4001节点条件分支PCHIP构建曾超过RSS限制，仍保留失败日志。在线采用经过凸性审查的平滑cubic，不能为了速度忽略导数错误。

## 完整电化学状态与力学假设

DFN保留正/负极径向固相扩散、电解液质量守恒、固相/电解液电荷守恒、Butler–Volmer及CC/CV控制；没有增加第二套锂状态。c为mol/m³，T为K，应变无量纲，应力Pa，拉应力为正，正放电电流单位A，正压力表示压缩。

第一条耦合路径采用**可审查的加性闭合**，没有将球径向浓度复制进任意不规则颗粒：

\[
F_{el}=\sum_j L_j\langle W_j(\epsilon_x,\bar c-c_{ref},T-T_{ref})
+\tfrac12\epsilon_{s,j}\kappa_j\langle(c-\bar c)^2\rangle_r\rangle_x
+\tfrac12 L_s\sigma_x^2/C_s,
\quad \kappa_j=2E_j\Omega_j^2/[9(1-\nu_j)].
\]

W来自GEMsim固定随机种子小SVE的各向同性线弹性、本征应变ΩΔc/3与线性热膨胀。KUBC，不冒充随机切割面的PBC。它只接收活性相平均浓度；第二项是独立自由球零均值扩散能。多粒子实际非均匀浓度与非均匀微结构应力相互作用尚未由这条DFN闭合解析。上层为电极/隔膜共轴向应力、侧向平面应变的串联层合，可给定压力或总厚度。

相体积分数的体素舍入通过对整个SVE能量乘ε_s/φ_active校正；局部应力及宏观刚度同步乘该因子。此为显式均匀缩放假设，不能称为精确相形貌匹配。局部输出标为“平均浓度SVE”，不能与另算的球径向应力叠加成唯一真实场。

\[
\mu_{el}=\epsilon_s^{-1}W_{,\bar c}+\kappa(c-\bar c),
\quad U_{eff}=U-\lambda\mu_{el,surf}/F.
\]

所有应力、广义化学力及混合导数从同一二次能量得到。反应OCP只修正一次；λ=0精确回到相同参数、相同热模型的原DFN。自由球表面径向应力为0，切向应力ΩE(\bar c−c_surf)/[3(1−ν)]。均匀嵌锂自由球应力为0，但侧向约束SVE应力可不为0，二者边界不同。

给定非零压力时，应对含外部边界功的势能做驻值/Legendre变换再取浓度导数；输出的内部弹性能不包含该外功。不能把沿压力平衡路径的内部能量总导数直接当作固定应变化学势。压力为0和固定厚度时上述数值导数测试可直接使用凝聚内部能量；非零压力全局功账本仍须单列。

## 化学曲率、参数化与失败处理

以f'_chem=−FU，f''_chem=−FU_{,c}，采用M=D/f''_chem以及

\[
J=-M\nabla\mu=-D(1+\lambda\kappa/f''_{chem})\nabla c.
\]

这要求f''_chem>0。原始Ai2020和OKane2022的部分拟合分支不满足该条件；原始耦合分别在约2424s和28s触发适用域事件，**不是完成放电**。原始DFN的经验Fick扩散仍可正常求解。不能删掉事件或裁剪负曲率冒充热力学一致。

非凸曲率可能与相变物理或拟合细节有关，不能仅凭这个事件断言材料或原始DFN错误。真正非凸自由能需要相应迁移率与梯度能/相分离动力学；当前正化学扩散系数与M=D/f''闭合无法覆盖该范围。凸化只定义本轮可验证的固溶支路模型。

因此研究区分两条证据：

1. **不改参数的参照轨**：Marquis2019对公开COMSOL；Ai2020对公开Enertech电压/温升。常规经验DFN，不以这条轨认证新增力学。
2. **修改且显式标注的验证轨**：OKane2022零熵OCP，经等距4001点isotonic投影、8采样点高斯平滑、0.01V/sto斜率底线及致密导数检查后cubic插值。不是原论文参数复现。最初仅投影最大负极改动2.66mV；最终平滑版本全区间最大改动约21.43mV，主要靠近极端sto，必须报告曲线偏差，不能沿用2.66mV作为最终上限。

验证轨默认Ω乘0.1，以减小线性几何假设的风险；这是公开声明的敏感性设定，**不是材料校准值**。另保留Ω乘1的强反馈案例。E/ν/Ω来源为OKane2022参数记录，其上游自身也说明部分退化参数非唯一真实材料参数；本版本没有启用SEI、析锂、裂纹或容量衰减。

## 热耦合与可验证收支

集总温度反馈至DFN温度依赖扩散、电导率与反应。包括欧姆热、不可逆反应热、可逆反应热、混合热与Newton冷却。修正OCP的熵导数为U_{,T}−λμ_{el,T}/F；统一凝聚能量给出

\[
C_{eff}=C_p-\lambda T\chi F_{el,TT},\quad \chi=N_{parallel}A_{pair}/V_{thermal},\qquad
Q_{mix,s}=\epsilon_s\langle (D/f''_{chem})(f''_{chem}+\lambda\kappa)^2|\nabla c|^2\rangle_r.
\]

固定C、Ω、热膨胀系数和线性层合约束下，凝聚后的F_TT及μ_T为常数，独立有限差分检查其TT和cT导数。λ=0调用上游混合热，保证零反馈极限。非正有效热容明确终止。

χ与原生DFN热源一样，以整个热电芯体积归一化并计入并联极对；不能用1/名义层厚替代任意几何电芯的面积/体积比。独立测试检查该系数。

对本轮恒定Cp参数组，引入热源/冷却积分辅助状态，检查

\[
C_p(T-T_0)-\lambda\chi F_{el,TT}(T^2-T_0^2)/2
=\int(Q_{total}+Q_{contact}+Q_{cool})dt.
\]

这是**集总温度ODE的积分一致性检查**。它并不证明完整DFN化学能、电功、弹性能和非等温电解液熵通量全部闭合；变Cp时该简化账本拒绝适用，完整非等温能量证书仍是后续门槛。Richardson–Korotkin已有严格DFN能量守恒理论，必须以其为基线，不能把热账本重新命名为原创能量定理。

## 验证协议与复现

独立参照使用[PyBaMM官方COMSOL示例](https://docs.pybamm.org/en/v23.5_a/source/examples/notebooks/models/compare-comsol-discharge-curve.html)的Marquis2019、1m²、24A及126/16.6S/m。数据固定为[pybamm-data v1.0.1](https://github.com/pybamm-team/pybamm-data/releases/tag/v1.0.1)，逐文件SHA在外部`reports/dfn_reference_data.json`。

Ai2020的0.5/1/2C公开电压/温升按相同物理时间比较，不拟合时间轴或电压偏置，只在重叠窗口算RMSE并报告覆盖率和截止时刻差；不是完整Ai论文的应力/厚度算法复现。DFN验证轨检查0.5/1/2C至2.5V，三循环每次“放电至2.5V—静置600s—1C充至4.2V—4.2V恒压至C/20—静置600s”，必须全15步完成。只返回部分Solution不能验收。

同时记录锂库存、正负极积分反应电流、步间c/T连续性、热账本、温度极值、径向球应力、宏观约束应力和按需恢复的平均浓度SVE局部场。不同网格同条件比较，IDAKLU与CasADi交叉只能支持数值一致性，不能替代独立连续体或实验验证。

独立反例`casadi_loose_ic`保留代数初值求解的默认步长停止阈值1e-4：起点积分电荷残差约5.86e-4，未通过5e-5门槛，虽然协议完成且求解器返回成功。只收紧root_tol、关闭随机初值扰动均未解决；显式设置`CasadiAlgebraicSolver(tol=1e-10, step_tol=1e-12)`后通过。默认交叉还关闭随机扰动以固定数值条件。19个工况中18个通过限定数值检查，该反例明确保留为失败，而非调宽验收阈值。

原生PyBaMM本地Solution保存/加载后接续求解已通过短时放电—静置交叉，逐项核对浓度、温度和热积分状态。OCP参数函数使用可序列化的顶层callable。该pickle只用于自行生成且信任的本地检查点；不是跨版本可移植状态格式，后续仍需模型/hash绑定的长期归档接口。

```bash
# 以下从源码根目录运行；主venv预先具备DFN、test和geometry依赖。
python scripts/run_local.py --label dfn_reference_data -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/scripts/fetch_dfn_data.py
# 每个数值子进程自动经过CPU/RSS/时间限制，串行执行
/home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python -B scripts/run_dfn_suite.py
python scripts/run_local.py --label dfn_full_tests -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python -m pytest /home/zephyrain/lw/MORQEX/tests -q --junitxml=/home/zephyrain/lw/morph_workspace/reports/dfn_full_junit.xml --basetemp=/home/zephyrain/lw/morph_workspace/tmp/dfn_full_tests
python scripts/run_local.py --label dfn_integrity -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/scripts/check_integrity.py
python scripts/run_local.py --label dfn_wheel -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python -m pip wheel /home/zephyrain/lw/MORQEX --no-deps --no-build-isolation --wheel-dir /home/zephyrain/lw/morph_workspace/wheels
python scripts/run_local.py --label dfn_clean_venv -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python -m venv --system-site-packages /home/zephyrain/lw/morph_workspace/venvs/morqex_dfn
python scripts/run_local.py --label dfn_clean_install --dependency-install -- /home/zephyrain/lw/morph_workspace/venvs/morqex_dfn/bin/python -m pip install '/home/zephyrain/lw/morph_workspace/wheels/morqex_research-0.0.6-py3-none-any.whl[dfn,geometry]'
python scripts/run_local.py --label dfn_installed -- /home/zephyrain/lw/morph_workspace/venvs/morqex_dfn/bin/python /home/zephyrain/lw/MORQEX/scripts/probe_dfn_installation.py
python scripts/run_local.py --label dfn_html -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/scripts/build_dfn_report.py
python scripts/run_local.py --label dfn_html_check -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/scripts/check_dfn_report.py
```

若已有同版本wheel而源码发生变化，需先重新构建，并对独立venv执行`pip install --force-reinstall --no-deps <wheel>`再运行安装检查；依赖首次安装和同版本源码更新是两种不同操作。HTML依赖上述公开数据清单、数值结果、JUnit和安装验证记录，不能只生成空页面。

源码不含算例数组、图片、HTML、wheel或二进制。依赖安装使用受限`--dependency-install`磁盘配置，仍单核/GPU禁用/RSS1.5GiB；不能把该开关用于任意命令。先不引入GPU，避免与本地训练争抢资源；本轮没有GPU加速结论。

## 尚未达到顶刊验收的部分

真实CT/材料参数校准、任意不规则颗粒非均匀浓度—力学闭合、独立完整三维双向热参照、全局热力学证书、DFN降阶误差界和自适应、先进NTFA/EHM/FE²公平基线、近DFN成本及三项独立原创均未完成。原始非凸分支与强反馈参数是明确的模型适用性问题，不能被当前测试数或HTML掩盖。

主要理论来源：[PyBaMM](https://doi.org/10.5334/jors.309)、[Ai2020](https://doi.org/10.1149/2.0122001JES)、[Richardson–Korotkin2021](https://arxiv.org/abs/2104.01590)、[O’Kane2022](https://pubs.rsc.org/en/content/articlehtml/2022/cp/d2cp00417h)。
