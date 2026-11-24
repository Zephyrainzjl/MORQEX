# MORPH

**Multiscale Operator-based Reconstruction and Physics-consistent Homogenization**

复杂电极微结构电化学—力学跨尺度方法的独立研究仓库。当前版本为 **0.0.1：Phase 0/I 研究基础**，不是已验证的 MORPH-0，也没有宣称达到论文性能目标。最终数值架构由基准与消融决定。

源码：`/home/zephyrain/lw/MORPH`；全部测试、依赖源码、数据、日志、缓存和结果：`/home/zephyrain/lw/morph_workspace`。不依赖、不修改 ZynJoule。保留 MORPH 作为工作名；已检索到 [LANL 同名 PDE 项目](https://github.com/lanl/MORPH)及 [MORLab/MORpH](https://github.com/MORLab/MORpH)，因此不声称名称独占或“无人使用”，投稿名称另行审查。

## 当前可运行内容

- 固定提交的 GEMsim 直接源码集成：生成随机场微结构、读写 Volume、四面体/六面体网格；保留原生成器和原生求解器源码。
- 显式相语义、SI 尺度、ZYX/XYZ 转换、几何内容指纹与来源追踪。
- 本地受限运行器：单逻辑 CPU、低优先级、进程组内存监控、超时、单任务锁、GPU 隐藏。
- 源码清单审计、既有 FEM/P2D/FFT 小规模独立探测、能量静态凝聚代数验证。
- 文献登记、候选方法比较、理论、参考求解器设计、阶段门禁及恢复记录。

NGSolve、PyBaMM、echemAMR 原生插件及 AI 生成器的数值验证尚未完成；存在源码不等于功能已经认证。GEMsim 顶层未声明许可证，当前保持用户授权的本地私有集成，不将其重授权或发布。

## 本地运行

当前已使用本机 Anaconda Python 3.13.9、NumPy 2.3.5、SciPy 1.16.3、pytest 8.4.2 完成基础检查。无需安装 AI/GPU 依赖。

```bash
cd /home/zephyrain/lw/MORPH
python scripts/run_local.py --label bootstrap -- python /home/zephyrain/lw/MORPH/scripts/bootstrap_gemsim.py
python scripts/run_local.py --label bridge_tests -- python -m pytest /home/zephyrain/lw/MORPH/tests -q --basetemp=/home/zephyrain/lw/morph_workspace/tmp/bridge_tests
python scripts/run_local.py --label energy_algebra -- python /home/zephyrain/lw/MORPH/benchmarks/verify_energy_algebra.py
python scripts/run_local.py --label source_audit -- python /home/zephyrain/lw/MORPH/scripts/audit_sources.py
python scripts/run_local.py --label existing_probes -- python /home/zephyrain/lw/MORPH/scripts/probe_existing.py
```

运行器将工作目录切换到外部 `runs/<UTC>_<label>`，保存 `manifest.json`、`stdout.txt`、`stderr.txt`；最近运行位置见外部 `reports/*_latest.json`。重复运行保留历史日志；pytest 临时目录仅供可重建测试使用。

## 阅读顺序

1. [RESUME.md](RESUME.md)：已完成、失败证据和下一步。
2. [CODE_AUDIT.md](CODE_AUDIT.md)：复用边界与现有物理问题。
3. [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)、[NOVELTY_MATRIX.md](NOVELTY_MATRIX.md)：证据与创新审查。
4. [THEORY.md](THEORY.md)、[ALGORITHM_COMPARISON.md](ALGORITHM_COMPARISON.md)：数学基础与候选比较。
5. [REFERENCE_SOLVER_PLAN.md](REFERENCE_SOLVER_PLAN.md)、[BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)：参考解与最短验证闭环。
6. [RESEARCH_PLAN.md](RESEARCH_PLAN.md)、[RESOURCE_BUDGET.md](RESOURCE_BUDGET.md)、[docs/INTERFACES.md](docs/INTERFACES.md)：推进计划、资源与可演进接口。

