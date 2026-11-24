# MORQEX恢复入口

更新：2026-11-24。软件0.0.5，分支 `research/sparse-coupled-scaling`。当前包括稀疏二次耦合规模路径、独立在线核/统计/场和离线HTML；**尚未完成论文框架、完整DFN或三项原创性验证**。本轮见 [SPARSE_SCALING](docs/SPARSE_SCALING.md)，旧推导/实测见 [ENERGY_ADAPTIVITY](docs/ENERGY_ADAPTIVITY.md)，旧Fick证据见 [CERTIFIED_TRANSPORT](docs/CERTIFIED_TRANSPORT.md)，旧线性证据见 [LINEAR_CORE](docs/LINEAR_CORE.md)。

## 路径与限制

- 源码 `/home/zephyrain/lw/MORQEX`；全部数据/运行/测试/缓存/构建 `/home/zephyrain/lw/morph_workspace`。
- 外部环境 `morph_workspace/venvs/morqex/bin/python`，Python3.13.9，继承本机基础科学包；未改全局环境。
- 所有数值任务使用 `python scripts/run_local.py --label <label> -- <command>`。单CPU/线程、nice15、空闲I/O、GPU禁用、AS3GiB/组RSS1.5GiB、180s/CPU120s、单任务锁、主机可用内存保留4GiB。
- 最新证据 `reports/<label>_latest.json`，历史在外部 `runs/<UTC>_<label>`。失败日志保留。
- 本任务对原ZynJoule保持只读，审计基线提交 `2ef9820`；最终检查发现其有外部并行未提交改动，原样保留，不检查或干预预训练。MORQEX的Phase0/I稳定回退提交 `dd72515`。

## 已实现且有证据

1. Phase0/I审计：ZynJoule相关354文件/GEMsim336文件清单、物理缺陷、硬件预算、多候选比较、理论/门禁文件。45文献、28方法×25字段；新增抛物认证/Krylov先例，R08等复现仍缺，D类创新为空。
2. GEMsim提交 `6fd789c78d58d49ef645c539414567fc2a2bc171` 的全部274源码迁入 `src/morqex/microstructure`；本轮未改这些迁移文件。运行不依赖外部checkout；可选AI/原生源码保留不等于数值认证。
3. 独立FP64 Tet4：完整各向异性Mandel、非均匀本征应变、真孔隙、KUBC/PBC/Neumann、压力功/反力/残差、多RHS复用LU。处理每个孤岛及周期切割孤岛的相容刚体模态，拒绝不平衡载荷。
4. 元素矩阵自由CG/Jacobi与同Tet装配交叉；4³/8³、刚度对比1/100应力差<1e-9，小规模更慢，不替代默认LU；尚无AMG/FFT/Hex8比较。
5. 真活性域守恒FV（XYZ对角D、颗粒ID隔离、表面源）；几何/谱/质量POD/联合POD，每个扩散分量常数必须保留；不可行状态拒绝，不裁剪冒充守恒。
6. 同源U/S/H/h/w0、平均应力/模式mu/一致切线、精确分相矩、P99/CVaR离散包络、ROI/全场、机械残差Gram与当前输入富集；算子校验/hash外置缓存。
7. 二次化学能微结构双向扩散/力学及储库BV：封闭耗散、外部化学/机械功、库存/电流/BE数值耗散。默认在线不恢复体场；表面反应成本仍随表面数增长。
8. 一维层合压力/厚度约束；checkpoint核对模型指纹/能量/库存，重启轨迹一致。不是完整电芯。
9. 自身111测试通过，当前外部JUnit逐项记录。原GEMsim历史7通过/9可选依赖跳过，迁移代码未改故未重复，可选后端仍未认证。
10. 0.0.5 wheel外部构建/安装、清空PYTHONPATH验证几何/FV/Tet/耦合/缓存/重启，并验证新Fick认证与P99树入口，不借源码/外部gem。

## 0.0.5规模升级与恢复顺序

新模块 `sparse_coupled.py`/`streamed_rom.py`：稀疏混合块全阶、Schur-GMRES交叉、按模式H作用、质量Riesz/宏观提升、完整耦合有理输入空间、非扩张+局部夹持能量上界、真正独立在线核及外置检查点、精确小矩阵应力统计、按需应力恢复。新路径没有N≤512限制，但仍受稀疏LU填充预算约束；不构造N×N H或identity应力算子。

正式配置sparse_scaling.toml：8³/14³/18³×104/205/306，每例40步，Fick/完整耦合响应×16/32/48初始非恒定模式，共54条候选。Riesz/宏观提升方向另加且计入总阶。原2 mol/m³等价H容限不放宽，按证书选阶，低阶/Fick失败不算合格加速。最大活性N=4082、最大机械自由度及性能以最终JSON为准；不是固定几何网格收敛或真实CT。

18³三种子合并任务在第三结构触发1.5GiB RSS限制；失败保留，不提高限额。当前18³必须--seed分进程（标签sparse_scaling_18_104/_205/_306），旧LU更换前释放；8/14各一个三种子任务。汇总脚本校验五任务源码hash/退出码，不能悄悄使用旧成功。所有数据、核、图、NPZ/轮包均外置。

恢复：先看最新manifest；运行全部测试、五个规模任务和assemble_scaling_report.py，再生成/检查HTML。旧linear_study/controlled_study/energy_adaptivity哈希包含所有根模块，新模块改变后须重跑；参考/后端仅mechanics哈希未变，可沿用已限定范围证据，不重复近内存限额球细化。所有数值任务仍用run_local，单核/GPU禁用，原ZynJoule/训练不动。

## 0.0.4新增耦合适应与误差动力学

推导、已有方法归属、失败候选及限制见 [ENERGY_ADAPTIVITY](docs/ENERGY_ADAPTIVITY.md)。新模块gradient_transport/adaptive_transport/error_dynamics在固定二次耦合系统上实现质量Riesz方向、能量投影、图界/辅助误差动力学、嵌套富集、包含当前场的非嵌套重启和校验检查点。全生成元与既有CoupledTransport独立交叉；不把Fick证书直接套到耦合系统。

3结构×2工况×8候选，80步；16维误差动力学候选六例均完成，纠正14–15次、浓度阶24–28，应力轨迹误差约0.12%–0.16%；外部最终JSON为核定值。相对保守图界减少查询，但仍慢于N=80复用LU全阶。r=16额外状态、离线稠密谱空间和全部重建/纠正计费。完整/压缩响应块失败、24维重启更贵均保留。失败前缀不报完整时域精度/加速。

新增28项测试覆盖非零初始误差、谱余项/非零跨块泄漏、转移/预算拒绝、重启、物理界和安装入口；总数以JUnit为准。硬限制N≤512，未认证非线性BV/完整DFN。R26–R29为直接先例，D仍为空。

## 本轮HTML与新证据

- 外部 `reports/MORQEX_test_report.html` 为单文件离线报告；含111项测试、实际场切片、各结构/候选/工况筛选、原始JSON下载、失败门槛和复现。`test_report_manifest.json`保存hash。
- `scripts/check_test_report.py`检查DOM/导航/PNG/证据hash，并以QuickJS DOM夹具运行筛选/导出；`reports/test_report_validation.json`留证。没有本地浏览器，未声称完整浏览器渲染验证。QuickJS只装外部venv，不是核心依赖。
- `boundary_resolvent`构造已知表面通量响应空间；`ControlledDiffusion`预投影输入通道；`DiffusionCertificate`按每分量常数/零均值分解，使用图生成树Poincaré下界和离线QR残差，在线不恢复体场。
- `CertifiedControlledDiffusion`超误差容限或物理误差包络拒绝步进，不返回不合格状态，旧状态只读/模型绑定。独立反例：平均模式为正、完整FV表面层为负，新增误差包络能拒绝。
- 三GEMsim结构、四候选、40步训练/留出协议；原空间对新热点失准且证书超2mol/m³容限。观测新热点通量方向后重建，20非恒定模式应力轨迹误差约0.10%–0.27%，不使用验证浓度/应力；这不是零样本或同信息预算对照。
- 已知4通量方向内POD是强基线，未证实响应基普遍胜出；原任意初态协议重新执行，失败仍在。所有本轮固定Fick离散误差界覆盖轨迹，但不适用于完整双向/DFN。
- P99树最终区间宽约0%–4.1%，却比小网格向量化统计慢数十倍；宽度通过、速度未过，不作为小规模默认。全部内部节点与成本保留。

## 实测与未通过项

- 球独立解析当前可重复默认n24：4368节点/25967Tet，应力全域L2 **11.01%**、位移0.0650%、能量1.331%、残差1.07e-13，采样RSS约732MiB。不能把小残差称连续体准确。
- 历史n28球应力9.60%、能量0.954%，采样RSS约1476MiB接近门槛；不扩大直接分解，CLI去掉32。历史路径 `runs/20261124T080254.956384Z_reference_convergence`；当前默认24有源码hash。
- 三个独立GEMsim随机场（104/205/306，每例80活性体素）持出载荷：7/8模式应力误差48%–77%；联合度量POD未稳定优于POD；谱64/65阶仍6.43%–22.61%，扫描仅全80阶过5%。**C1未通过。**
- P99包络覆盖但64点区间宽约真值18倍；全代表集精确却失去采样收益。**C2未通过。**
- 同输出/同精度/同输入模式/复用LU小基准，完整场恢复约2.2–2.4倍；模式精度未过，不算论文合格加速。无FE²/近DFN整体倍率。
- 微结构闭系统能量平衡约1.4e-11 J/m³，储库约1.5e-14；尚无端电压、电解液/电荷方程，不等于完整电池认证。
- 原ZynJoule审计反例保留：裁剪同步目标0.95得到0.7312619509；C(c)机械势缺项受控差528.8257785J/mol。原项目未改。
- 参数为合成验证值；无CT/材料设计结论、SVE代表性或NTFA/EHM等完整先进基线复现。

## 复现

从MORQEX目录启动，实际cwd自动切到外部：

```bash
python scripts/run_local.py --label final_tests -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python -m pytest /home/zephyrain/lw/MORQEX/tests -q --junitxml=test_results.xml --basetemp=/home/zephyrain/lw/morph_workspace/tmp/final_certified_tests
python scripts/run_local.py --label reference_convergence -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/qualify_reference.py --fitted-refinement 24
python scripts/run_local.py --label linear_study -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/run_linear_study.py --config /home/zephyrain/lw/MORQEX/configs/linear_study.toml
python scripts/run_local.py --label mechanics_backends -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/compare_mechanics_backends.py
python scripts/run_local.py --label controlled_study -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/study_controlled_transport.py --config /home/zephyrain/lw/MORQEX/configs/controlled_transport.toml
python scripts/run_local.py --label energy_adaptivity -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/study_energy_adaptivity.py --config /home/zephyrain/lw/MORQEX/configs/energy_adaptivity.toml
python scripts/run_local.py --label test_html -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/scripts/build_test_report.py
python scripts/run_local.py --label html_validation -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/scripts/check_test_report.py
```

实验存配置、关键源码hash、几何/响应指纹、指标及真实图；最终汇总 `reports/verification_summary.json`。安装探测用 `env PYTHONPATH=` 清除源码注入。

## 下一步顺序

1. 固定二次系统状态转移/误差动力学已验证；下一步低成本大N耦合H/余项强制性构建、局部空间与目标误差，保留任意高频失败工况。
2. 机械mu/尾部目标误差与profile；保留小网格向量化默认，树性能不佳不隐藏，研究局部谱/目标受限空间并保留高频失败工况。
3. 拟合/高阶离散、AMG/FFT/Hex8资格，降低界面误差与分解内存，不无预算扩网格。
4. 原始公式核对、独立短时三维全耦合及非凸化学门禁后，DFN替换固扩散状态，补电解液/电荷/端电压与OCV不重复计入。
5. 多SVE/CT/真实参数、先进基线、设计案例、完整消融和Pareto后，才能决定论文方法与D类贡献。

继续前必读RESEARCH_PLAN、LINEAR_CORE、THEORY、NOVELTY_MATRIX、RESOURCE_BUDGET；查看Git和最新manifest，不用旧日志替代当前验证。未push或公开发布私有GEMsim。

0.0.4最终交付：97测试、0.0.4外部wheel/清空PYTHONPATH安装探测、45文献/28方法完整性与资源探测均通过。HTML约5.14MB，7图/48耦合候选/96旧通量记录，DOM/证据hash/QuickJS筛选导出通过；仍未做完整浏览器渲染。所有研究失败门槛继续显示，源代码不含运行大文件。

最终冻结规模结果：最大N4082、力学自由度13623–13821、56总浓度模式；三独立种子包含误差证书的在线加速119.6–126.7倍（相对两种复用全阶中最快者），末时刻应力场L2误差0.0216%–0.0255%，末时刻P99误差0.0093%–0.0768%。完整离线库计费后盈亏平衡14.7–18.4条40步轨迹；完整场恢复另测，不混入在线倍率。当前最高成功任务采样组RSS1262.5MiB，GPU未使用；失败合并任务及控制器终止原因仍保留。全套111测试通过，包括周期全各向异性/剪切膨胀、稀疏双参考、独立紧凑核/重启和余项界。数字只适用于此固定离散二次协议，不代表连续体、真实电芯或期刊验收。

0.0.5报告将包含111项测试、54条规模候选、48条旧耦合适应、96条旧通量记录与9张PNG；HTML/数组/检查点/轮包全部外置。读旧0.0.4段落时注意其密集H/80体素限制，不适用于新稀疏路径。
