# MORQEX恢复入口

更新：2026-11-24。软件0.0.3，分支 `research/certified-transport-report`。当前是三维线性核心+固定Fick浓度认证+离线测试HTML；**尚未完成论文框架、完整DFN或三项原创性验证**。本轮推导/实测见 [docs/CERTIFIED_TRANSPORT.md](docs/CERTIFIED_TRANSPORT.md)，旧线性证据见 [LINEAR_CORE](docs/LINEAR_CORE.md)。

## 路径与限制

- 源码 `/home/zephyrain/lw/MORQEX`；全部数据/运行/测试/缓存/构建 `/home/zephyrain/lw/morph_workspace`。
- 外部环境 `morph_workspace/venvs/morqex/bin/python`，Python3.13.9，继承本机基础科学包；未改全局环境。
- 所有数值任务使用 `python scripts/run_local.py --label <label> -- <command>`。单CPU/线程、nice15、空闲I/O、GPU禁用、AS3GiB/组RSS1.5GiB、180s/CPU120s、单任务锁、主机可用内存保留4GiB。
- 最新证据 `reports/<label>_latest.json`，历史在外部 `runs/<UTC>_<label>`。失败日志保留。
- 本任务对原ZynJoule保持只读，审计基线提交 `2ef9820`；最终检查发现其有外部并行未提交改动，原样保留，不检查或干预预训练。MORQEX的Phase0/I稳定回退提交 `dd72515`。

## 已实现且有证据

1. Phase0/I审计：ZynJoule相关354文件/GEMsim336文件清单、物理缺陷、硬件预算、多候选比较、理论/门禁文件。41文献、24方法×25字段；新增抛物认证/Krylov先例，R08等复现仍缺，D类创新为空。
2. GEMsim提交 `6fd789c78d58d49ef645c539414567fc2a2bc171` 的全部274源码迁入 `src/morqex/microstructure`；本轮未改这些迁移文件。运行不依赖外部checkout；可选AI/原生源码保留不等于数值认证。
3. 独立FP64 Tet4：完整各向异性Mandel、非均匀本征应变、真孔隙、KUBC/PBC/Neumann、压力功/反力/残差、多RHS复用LU。处理每个孤岛及周期切割孤岛的相容刚体模态，拒绝不平衡载荷。
4. 元素矩阵自由CG/Jacobi与同Tet装配交叉；4³/8³、刚度对比1/100应力差<1e-9，小规模更慢，不替代默认LU；尚无AMG/FFT/Hex8比较。
5. 真活性域守恒FV（XYZ对角D、颗粒ID隔离、表面源）；几何/谱/质量POD/联合POD，每个扩散分量常数必须保留；不可行状态拒绝，不裁剪冒充守恒。
6. 同源U/S/H/h/w0、平均应力/模式mu/一致切线、精确分相矩、P99/CVaR离散包络、ROI/全场、机械残差Gram与当前输入富集；算子校验/hash外置缓存。
7. 二次化学能微结构双向扩散/力学及储库BV：封闭耗散、外部化学/机械功、库存/电流/BE数值耗散。默认在线不恢复体场；表面反应成本仍随表面数增长。
8. 一维层合压力/厚度约束；checkpoint核对模型指纹/能量/库存，重启轨迹一致。不是完整电芯。
9. 自身69测试通过，当前外部JUnit逐项记录。原GEMsim历史7通过/9可选依赖跳过，迁移代码未改故未重复，可选后端仍未认证。
10. 0.0.3 wheel外部构建/安装、清空PYTHONPATH验证几何/FV/Tet/耦合/缓存/重启，并验证新Fick认证与P99树入口，不借源码/外部gem。

## 本轮HTML与新证据

- 外部 `reports/MORQEX_test_report.html` 为约2.8MB单文件离线报告；含69项测试、实际场切片、各结构/候选/工况筛选、原始JSON下载、失败门槛和复现。`test_report_manifest.json`保存hash。
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
python scripts/run_local.py --label test_html -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/scripts/build_test_report.py
python scripts/run_local.py --label html_validation -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/scripts/check_test_report.py
```

实验存配置、关键源码hash、几何/响应指纹、指标及真实图；最终汇总 `reports/verification_summary.json`。安装探测用 `env PYTHONPATH=` 清除源码注入。

## 下一步顺序

1. 新通量方向更新后的嵌套富集、累计浓度界传递、质量/能量状态转移；目前只做拒绝/显式重建，不声称已自动热力学切换。
2. 机械mu/尾部目标误差与profile；保留小网格向量化默认，树性能不佳不隐藏，研究局部谱/目标受限空间并保留高频失败工况。
3. 拟合/高阶离散、AMG/FFT/Hex8资格，降低界面误差与分解内存，不无预算扩网格。
4. 原始公式核对、独立短时三维全耦合及非凸化学门禁后，DFN替换固扩散状态，补电解液/电荷/端电压与OCV不重复计入。
5. 多SVE/CT/真实参数、先进基线、设计案例、完整消融和Pareto后，才能决定论文方法与D类贡献。

继续前必读RESEARCH_PLAN、LINEAR_CORE、THEORY、NOVELTY_MATRIX、RESOURCE_BUDGET；查看Git和最新manifest，不用旧日志替代当前验证。未push或公开发布私有GEMsim。
