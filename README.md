# MORQEX

面向复杂微结构的能量一致跨尺度电化学—力学研究框架。原工作名MORPH，经用户授权更名为 **MORQEX**；公开检索暂未发现同名计算框架，不作名称独占保证。

当前 **0.0.2：经独立检查的三维线性研究原型**。已实现Tet4/矩阵自由力学、守恒扩散/浓度模式、统一能量响应、三级输出、残差富集和储库反应双向构件。自身53测试通过；三结构对照同时发现低秩精度和尾部包络未达标。完整DFN/CT/先进基线与论文创新尚未完成，见 [线性核心报告](docs/LINEAR_CORE.md)。

源码：`/home/zephyrain/lw/MORQEX`。所有数据、测试输出、编译产物、安装环境、权重和缓存：`/home/zephyrain/lw/morph_workspace`。独立Git仓库，不修改、不依赖ZynJoule。

## 已实际集成到src

原GEMsim固定提交 `6fd789c78d58d49ef645c539414567fc2a2bc171` 的全部274个源码文件已迁入 [src/morqex/microstructure](src/morqex/microstructure)，统一为 `morqex.microstructure` 命名空间。包含体素/颗粒图、图像处理、v0/条件扩散/混合生成、重构、Tet/Hex/曲面网格、TetGen/ACVD/Cython、NGSolve输运/扩散/均匀化、PyBaMM和echemAMR原生源码。

**运行时不读取外部GEMsim checkout，也不需要导入`gem`。** 外部原始checkout仅作迁移/审计依据；逐文件SHA256和转换记录见 [provenance/gemsim_migration.json](provenance/gemsim_migration.json)。数据和二进制未迁入。

保留已有算法不等于所有算法已验证。当前随机场、体素/网格、数据格式及输入模板已测试；AI训练、NGSolve、PyBaMM与AMReX尚缺独立数值资格验证，均为可选依赖。

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
python scripts/run_local.py --label tests -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python -m pytest /home/zephyrain/lw/MORQEX/tests -q --basetemp=/home/zephyrain/lw/morph_workspace/tmp/morqex_tests
python scripts/run_local.py --label energy_algebra -- python /home/zephyrain/lw/MORQEX/benchmarks/verify_energy_algebra.py
python scripts/run_local.py --label source_audit -- python /home/zephyrain/lw/MORQEX/scripts/audit_sources.py
python scripts/run_local.py --label budget -- python /home/zephyrain/lw/MORQEX/scripts/probe_budget.py
python scripts/run_local.py --label linear_study -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/run_linear_study.py --config /home/zephyrain/lw/MORQEX/configs/linear_study.toml
python scripts/run_local.py --label reference_convergence -- /home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python /home/zephyrain/lw/MORQEX/benchmarks/qualify_reference.py --fitted-refinement 24
```

默认单逻辑CPU、nice+15、GPU禁用；虚拟内存3GiB、进程组RSS1.5GiB、单任务180s。工作目录、缓存和日志全部外置，具体限制见 [RESOURCE_BUDGET.md](RESOURCE_BUDGET.md)。外部`reports/*_latest.json`指向各次运行日志，历史失败不会覆盖。

纯Pythonwheel可用`pip wheel . --no-deps --no-build-isolation --wheel-dir /home/zephyrain/lw/morph_workspace/wheels`构建；请通过运行器执行。`geometry`、`ai`、`simulation`、`view`分别管理额外依赖。Cython仅在显式`MORQEX_BUILD_GEOMETRY=1`时构建；默认不编译AMReX、不下载训练数据、不启动GPU。`setup.py`将所有生成文件放到外部workspace。

## 研究文档

- [RESUME.md](RESUME.md)、[RESEARCH_LOG.md](RESEARCH_LOG.md)：恢复入口、实测结果、失败与下一步。
- [CODE_AUDIT.md](CODE_AUDIT.md)：已有物理、可复用能力及实测问题。
- [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)、[NOVELTY_MATRIX.md](NOVELTY_MATRIX.md)：39条文献、22方法矩阵及创新门禁。
- [docs/LINEAR_CORE.md](docs/LINEAR_CORE.md)：实际物理接口、解析/守恒验证、失败结果及性能范围。
- [THEORY.md](THEORY.md)、[ALGORITHM_COMPARISON.md](ALGORITHM_COMPARISON.md)：数学基础和多候选比较。
- [REFERENCE_SOLVER_PLAN.md](REFERENCE_SOLVER_PLAN.md)、[BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)：最短正确性闭环与公平实验。
- [RESEARCH_PLAN.md](RESEARCH_PLAN.md)、[docs/INTERFACES.md](docs/INTERFACES.md)：逐阶段验收和体系接口。

第三方来源与未验证项见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本地私有研究集成，不重授权或公开发布用户私有GEMsim源码。

