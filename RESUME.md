# MORQEX恢复入口

更新：2026-11-24。软件0.0.2，分支 `research/verified-linear-core`。当前是已验证子集明确的三维线性研究原型，**尚未完成论文框架、完整DFN或三项原创性验证**。主要实测与失败见 [docs/LINEAR_CORE.md](docs/LINEAR_CORE.md)。

## 路径与限制

- 源码 `/home/zephyrain/lw/MORQEX`；全部数据/运行/测试/缓存/构建 `/home/zephyrain/lw/morph_workspace`。
- 外部环境 `morph_workspace/venvs/morqex/bin/python`，Python3.13.9，继承本机基础科学包；未改全局环境。
- 所有数值任务使用 `python scripts/run_local.py --label <label> -- <command>`。单CPU/线程、nice15、空闲I/O、GPU禁用、AS3GiB/组RSS1.5GiB、180s/CPU120s、单任务锁、主机可用内存保留4GiB。
- 最新证据 `reports/<label>_latest.json`，历史在外部 `runs/<UTC>_<label>`。失败日志保留。
- 本任务对原ZynJoule保持只读，审计基线提交 `2ef9820`；最终检查发现其有外部并行未提交改动，原样保留，不检查或干预预训练。MORQEX的Phase0/I稳定回退提交 `dd72515`。

## 已实现且有证据

1. Phase0/I审计：ZynJoule相关354文件/GEMsim336文件清单、物理缺陷、硬件预算、多候选比较、理论/门禁文件。39文献、22方法×25字段；R06作者全文方法段已核对，R08及关键公式复现仍缺，D类创新为空。
2. GEMsim提交 `6fd789c78d58d49ef645c539414567fc2a2bc171` 的全部274源码迁入 `src/morqex/microstructure`；本轮未改这些迁移文件。运行不依赖外部checkout；可选AI/原生源码保留不等于数值认证。
3. 独立FP64 Tet4：完整各向异性Mandel、非均匀本征应变、真孔隙、KUBC/PBC/Neumann、压力功/反力/残差、多RHS复用LU。处理每个孤岛及周期切割孤岛的相容刚体模态，拒绝不平衡载荷。
4. 元素矩阵自由CG/Jacobi与同Tet装配交叉；4³/8³、刚度对比1/100应力差<1e-9，小规模更慢，不替代默认LU；尚无AMG/FFT/Hex8比较。
5. 真活性域守恒FV（XYZ对角D、颗粒ID隔离、表面源）；几何/谱/质量POD/联合POD，每个扩散分量常数必须保留；不可行状态拒绝，不裁剪冒充守恒。
6. 同源U/S/H/h/w0、平均应力/模式mu/一致切线、精确分相矩、P99/CVaR离散包络、ROI/全场、机械残差Gram与当前输入富集；算子校验/hash外置缓存。
7. 二次化学能微结构双向扩散/力学及储库BV：封闭耗散、外部化学/机械功、库存/电流/BE数值耗散。默认在线不恢复体场；表面反应成本仍随表面数增长。
8. 一维层合压力/厚度约束；checkpoint核对模型指纹/能量/库存，重启轨迹一致。不是完整电芯。
9. 自身53测试通过。原GEMsim历史7通过/9可选依赖跳过，本轮未改迁移代码故未重复，可选后端仍未认证。
10. 0.0.2 wheel构建/外部安装，清空PYTHONPATH从site-packages验证几何/格式、FV→Tet能量→耦合→缓存→重启，无源码/外部gem依赖。

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
python scripts/run_local.py --label final_tests -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python -m pytest /home/zephyrain/lw/MORQEX/tests -q --basetemp=/home/zephyrain/lw/morph_workspace/tmp/final_linear_tests
python scripts/run_local.py --label reference_convergence -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/qualify_reference.py --fitted-refinement 24
python scripts/run_local.py --label linear_study -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/run_linear_study.py --config /home/zephyrain/lw/MORQEX/configs/linear_study.toml
python scripts/run_local.py --label mechanics_backends -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/compare_mechanics_backends.py
```

实验存配置、关键源码hash、几何/响应指纹、指标及真实图；最终汇总 `reports/verification_summary.json`。安装探测用 `env PYTHONPATH=` 清除源码注入。

## 下一步顺序

1. 区分任意初态与反应通量可达载荷，比较边界驱动空间、局部谱/图分解与约束式机械目标空间；保留失败高频工况。
2. 浓度截断/机械mu/尾部目标的可计算误差与富集；机械残差不当浓度误差。尾部不可靠回退ROI/全场。
3. 拟合/高阶离散、AMG/FFT/Hex8资格，降低界面误差与分解内存，不无预算扩网格。
4. 原始公式核对、独立短时三维全耦合及非凸化学门禁后，DFN替换固扩散状态，补电解液/电荷/端电压与OCV不重复计入。
5. 多SVE/CT/真实参数、先进基线、设计案例、完整消融和Pareto后，才能决定论文方法与D类贡献。

继续前必读RESEARCH_PLAN、LINEAR_CORE、THEORY、NOVELTY_MATRIX、RESOURCE_BUDGET；查看Git和最新manifest，不用旧日志替代当前验证。未push或公开发布私有GEMsim。
