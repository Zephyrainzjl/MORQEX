# MORQEX

面向复杂微结构的能量一致跨尺度电化学—力学研究框架。原工作名MORPH，经用户授权更名为 **MORQEX**；公开检索暂未发现同名计算框架，不作名称独占保证。

当前 **0.0.6：完整DFN、集总热反馈与能量导出的力学闭合**。PyBaMM26.3.1纯Python源码已迁入src并修复固定版本的插值导数转换；接入完整放电与三次CC-CV可逆循环。微结构采用GEMsim平均浓度SVE闭合，球内非均匀应力单独表示，尚非任意微结构全三维热电化学参照。参数凸化、膨胀缩放、失败分支和未过门槛均显式记录，见 [DFN热—力说明](docs/DFN_THERMOMECHANICS.md)。旧稀疏二次模型见 [规模升级说明](docs/SPARSE_SCALING.md)，其加速结果不能推广到DFN。

当前测试HTML在外部 `morph_workspace/reports/MORQEX_DFN_report.html`，总入口为`MORQEX_test_report.html`，单文件离线打开，包含真实曲线、逐步终止条件、JUnit、负结果和内嵌证据下载。旧0.0.5归档为`MORQEX_0.0.5_report.html`；源码只保留生成/检查脚本。

源码：`/home/zephyrain/lw/MORQEX`。所有数据、测试输出、编译产物、安装环境、权重和缓存：`/home/zephyrain/lw/morph_workspace`。独立Git仓库，不修改、不依赖ZynJoule。

## 已实际集成到src

原GEMsim固定提交 `6fd789c78d58d49ef645c539414567fc2a2bc171` 的全部274个源码文件已迁入 [src/morqex/microstructure](src/morqex/microstructure)，统一为 `morqex.microstructure` 命名空间。包含体素/颗粒图、图像处理、v0/条件扩散/混合生成、重构、Tet/Hex/曲面网格、TetGen/ACVD/Cython、NGSolve输运/扩散/均匀化、PyBaMM和echemAMR原生源码。

**运行时不读取外部GEMsim checkout，也不需要导入`gem`。** 外部原始checkout仅作迁移/审计依据；逐文件SHA256和转换记录见 [provenance/gemsim_migration.json](provenance/gemsim_migration.json)。数据和二进制未迁入。

PyBaMM的355个Python/文本文件另迁入[src/pybamm](src/pybamm)，BSD许可证保留；二进制求解依赖留外部环境。迁移hash见[清单](provenance/pybamm_migration.json)。默认core安装不要求DFN依赖，使用DFN需`morqex-research[dfn]`。该发行包提供`pybamm`命名空间，应在独立venv安装，避免与另一个PyBaMM发行包共同覆盖同名文件。

保留已有算法不等于所有算法已验证。当前随机场、体素/网格、数据格式及输入模板已测试；DFN验证范围见专门说明，AI训练、NGSolve与AMReX可选后端仍缺独立资格验证。

## 使用

```python
from pathlib import Path
from morqex import Workspace
from morqex.gemsim import GEMsim

ws = Workspace(Path('/home/zephyrain/lw/morph_workspace'))
geometry = GEMsim(ws)
structure = geometry.random_field(
    shape=(8, 8, 8), fractions=(.3, .6, .1),
    length_scale=(1., 1., 1.), spacing=(1e-6,)*3, seed=7,
    phase_kinds={0:'pore', 1:'active', 2:'cbd'},
)
mesh = geometry.mesh(structure, cell_type='tetra')
geometry.to_volume(structure).save(ws.path('outputs', 'example.npz'))
```

也可直接使用集成入口 `from morqex.microstructure import Volume, Microstructure`。体素与尺度采用ZYX，网格点XYZ，SI米；相语义必须显式指定。统一数据对象为`morqex.MicrostructureData`。

## 本地受限运行

```bash
cd /home/zephyrain/lw/MORQEX
python scripts/run_local.py --label final_tests -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python -m pytest /home/zephyrain/lw/MORQEX/tests -q --junitxml=test_results.xml --basetemp=/home/zephyrain/lw/morph_workspace/tmp/final_certified_tests
python scripts/run_local.py --label energy_algebra -- python /home/zephyrain/lw/MORQEX/benchmarks/verify_energy_algebra.py
python scripts/run_local.py --label source_audit -- python /home/zephyrain/lw/MORQEX/scripts/audit_sources.py
python scripts/run_local.py --label budget -- python /home/zephyrain/lw/MORQEX/scripts/probe_budget.py
python scripts/run_local.py --label linear_study -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/run_linear_study.py --config /home/zephyrain/lw/MORQEX/configs/linear_study.toml
python scripts/run_local.py --label reference_convergence -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/qualify_reference.py --fitted-refinement 24
python scripts/run_local.py --label controlled_study -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/study_controlled_transport.py --config /home/zephyrain/lw/MORQEX/configs/controlled_transport.toml
python scripts/run_local.py --label energy_adaptivity -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/study_energy_adaptivity.py --config /home/zephyrain/lw/MORQEX/configs/energy_adaptivity.toml
python scripts/run_local.py --label sparse_scaling_18_104 -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/study_sparse_scaling.py --config /home/zephyrain/lw/MORQEX/configs/sparse_scaling.toml --size 18 --seed 104
# 汇总前需以同样方式运行8/14（各三种子）和18的205/306两个单种子任务。
python scripts/run_local.py --label sparse_scaling -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/scripts/assemble_scaling_report.py
python scripts/run_local.py --label test_html -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/scripts/build_test_report.py
```

默认单逻辑CPU、nice+15、GPU禁用；虚拟内存3GiB、进程组RSS1.5GiB、单任务180s。工作目录、缓存和日志全部外置，具体限制见 [RESOURCE_BUDGET.md](RESOURCE_BUDGET.md)。外部`reports/*_latest.json`指向各次运行日志，历史失败不会覆盖。

纯Pythonwheel可用`pip wheel . --no-deps --no-build-isolation --wheel-dir /home/zephyrain/lw/morph_workspace/wheels`构建；请通过运行器执行。`geometry`、`ai`、`simulation`、`view`分别管理额外依赖。Cython仅在显式`MORQEX_BUILD_GEOMETRY=1`时构建；默认不编译AMReX、不下载训练数据、不启动GPU。`setup.py`将所有生成文件放到外部workspace。

## 研究文档

- [RESUME.md](RESUME.md)、[RESEARCH_LOG.md](RESEARCH_LOG.md)：恢复入口、实测结果、失败与下一步。
- [CODE_AUDIT.md](CODE_AUDIT.md)：已有物理、可复用能力及实测问题。
- [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)、[NOVELTY_MATRIX.md](NOVELTY_MATRIX.md)：45条文献、28方法矩阵及创新门禁。
- [docs/SPARSE_SCALING.md](docs/SPARSE_SCALING.md)：稀疏耦合、全生成元响应空间、分离在线核/统计/场和规模验证。
- [docs/ENERGY_ADAPTIVITY.md](docs/ENERGY_ADAPTIVITY.md)：同能量耦合、状态转移、辅助误差动力学及八候选消融。
- [docs/CERTIFIED_TRANSPORT.md](docs/CERTIFIED_TRANSPORT.md)：后验界推导、通量/新热点试验与HTML说明。
- [docs/LINEAR_CORE.md](docs/LINEAR_CORE.md)：实际物理接口、解析/守恒验证、失败结果及性能范围。
- [THEORY.md](THEORY.md)、[ALGORITHM_COMPARISON.md](ALGORITHM_COMPARISON.md)：数学基础和多候选比较。
- [REFERENCE_SOLVER_PLAN.md](REFERENCE_SOLVER_PLAN.md)、[BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)：最短正确性闭环与公平实验。
- [RESEARCH_PLAN.md](RESEARCH_PLAN.md)、[docs/INTERFACES.md](docs/INTERFACES.md)：逐阶段验收和体系接口。

第三方来源与未验证项见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本地私有研究集成，不重授权或公开发布用户私有GEMsim源码。

