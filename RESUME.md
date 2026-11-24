# MORQEX恢复入口

更新：2026-11-24。当前版本0.0.1，分支`research/phase0-i`；原MORPH已按用户授权更名。**当前交付是Phase0/I研究基础与完整微结构源码整合，尚不是完成的跨尺度论文方法。**

## 目录与运行

- 源码：`/home/zephyrain/lw/MORQEX`，Python包`morqex`。
- 全部运行/数据/构建/缓存：`/home/zephyrain/lw/morph_workspace`。
- 外部环境：`morph_workspace/venvs/morqex/bin/python`，Python3.13.9，继承本机基础科学包（不是完全隔离/可移植环境）；另安装edt3.1.2、meshio5.3.5。未改全局环境。
- 所有数值/测试/构建通过`python scripts/run_local.py --label <label> -- <command>`。默认1逻辑CPU、nice15、GPU0可见、AS3GiB/组RSS1.5GiB、180s。
- 运行清单：外部`reports/<label>_latest.json`，历史日志在`runs/<UTC>_<label>`；最终复核为`final_tests`、`integrated_upstream_tests`、`installed_package`、`resource_limits`、`integrity`等标签。

## 已完成且有证据

1. ZynJoule相关354文件与GEMsim336文件清单、关键物理源码/边界/参数审计；ZynJoule保持不变。重点风险详见CODE_AUDIT。
2. 用户GEMsim提交`6fd789c78d58d49ef645c539414567fc2a2bc171`的全部274源码文件在`src/morqex/microstructure`；运行不依赖外部`gem`包或Git checkout。完整生成/网格/原生源保留，不含权重/CT/二进制。
3. 显式相语义、SI几何、ZYX/XYZ及内容指纹；内部namespace和资源模板迁移；来源hash可核查。
4. 当前自身11项测试通过，覆盖生成/网格/轴序/格式/模板/命名空间、旧序列化和启动器轻量导入；原测试迁移导入后7通过、9因可选依赖跳过。原先edt缺失失败的日志保留。
5. 可安装wheel与独立site-packages运行验证：720正体积Tet、NPZ/VTU/COMSOL及原生输入准备，无源码/外部GEMsim路径。
6. Tet4与小P2D独立探测；现有浓度同步不守恒和C(c)机械mu导数不一致的可复现反例。
7. Schur补/响应/能量梯度/残差Gram/模式质量投影合成FP64验证，响应3.51e-16、能量4.19e-16、梯度1.93e-11。
8. 37文献元数据、22方法×25字段矩阵；用户重点题名/DOI核实，三类关键算法以上详细比较。最终创新D类为空。
9. 统一理论草案、四类参考后端设计、球形解析/网格最短闭环、十组实验、全部Phase门禁/预算/回退计划。

## 已确认的失败/限制

- `synchronize_theta`目标均值0.95得到0.7312619509，裁剪同步不守恒。
- 当前NMC变刚度机械mu探测与能量浓度导数差528.8257785J/mol；不作为热力学参考。
- 原Tet/FFT与GEMsim弹性Voigt顺序不同；弱材料/真空、周期/KUBC、外压、孤岛模态必须分别处理。
- NGSolve、PyBaMM、Torch/Diffusers、echemAMR/Cython原生数值资格未验证；跳过不算通过。
- R06/R08等竞争方法全文/公式尚未完全取得/复现，支持材料下载403；原始状态保留，最终创新认定仍关闭。
- 没有真实CT研究结果、网格收敛、完整三维双向真值、算法加速倍数或已确立原创贡献。

## 下一步（保持顺序）

1. 补齐关键全文与勘误差异核对，冻结共同物理假设/符号/独立参照；不要把所有相关领域能力都当首篇强制范围。
2. 实现独立Mandel各向异性Tet4+非均匀本征应变，先仿射/自由/约束/旋转patch。原有ZynJoule仅作只读交叉核对。
3. 周期映射、刚体/孤岛、真孔隙与外压功，球/层合解析及2³→16³收敛；再比较Hex/FFT/matrix-free。
4. 通过参考资格后才构造精确U/S/H与模式mu，随后比较谱/质量POD/目标基及三级输出。
5. 双向DFN与误差驱动富集须在独立模块认证后开展，不提前添加损伤/AI或大规模训练。

## 继续工作前必读

RESEARCH_PLAN、CODE_AUDIT、LITERATURE_REVIEW、NOVELTY_MATRIX、THEORY、ALGORITHM_COMPARISON、REFERENCE_SOLVER_PLAN、BENCHMARK_PROTOCOL、RESOURCE_BUDGET和docs/INTERFACES。遵守当前资源配置；不重命名外部`morph_workspace`，避免破坏已留存证据。代码和环境若改变，重新核对算子/实验hash。

Git历史初始提交`6e7b040`记录最初MORPH基础，后继提交包含重命名和src实际集成；最终提交可用`git log -1`读取。没有push、部署或公开发布。
