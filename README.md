# MORQEX

面向复杂电极微结构与 DFN 的电化学、力学、热及退化跨尺度研究框架。独立于 ZynJoule；GEMsim 全部 274 个源码文件与 PyBaMM 26.3.1 的 355 个源码文件已集成到 `src`，不依赖外部 GEMsim checkout。迁移记录和许可证保留。

当前分支 `research/open-voxel-gpu` 正在升级全体积 CT、受限 GPU 后端及开放材料接口；上一稳定版本为 0.0.9。开发中的接口、数值通过、材料标定和论文验收分别记录，不宣称已达到顶刊验收或支持任意物理组合。

## 使用

完整接口、单位、操作步骤与适用范围见 [API.md](API.md)。源码仅保留 README 和 API 两份 Markdown；历史理论、计划、日志和旧接口说明完整归档到源码外 `morph_workspace/research_archive`。文献表保留在 [literature](literature)，第三方说明见 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)。

```text
/home/zephyrain/lw/MORQEX                 # 轻量源码、配置、测试、复现脚本
/home/zephyrain/lw/morph_workspace        # 数据、环境、编译、算子、结果、HTML
```

数值步骤均经运行器执行。默认单逻辑 CPU、单线程、nice +15、RSS 1.5 GiB、任务 180 秒、禁用 GPU。显式 `--gpu` 才允许 GPU：进程显存预算 3 GiB，保留空闲显存 4 GiB，协作占空比 20%，资源超限中止。占空比是主动节流策略，不是硬件配额；不修改全局 GPU 设置或其他作业。

```bash
cd /home/zephyrain/lw/MORQEX
PY=/home/zephyrain/lw/morph_workspace/venvs/morqex/bin/python
$PY scripts/run_local.py --label tests -- $PY -m pytest /home/zephyrain/lw/MORQEX/tests -q --junitxml=test_results.xml
$PY scripts/run_local.py --label gpu_compile --gpu -- $PY /home/zephyrain/lw/MORQEX/benchmarks/probe_voxel_backend.py
$PY scripts/run_local.py --label ct_fetch -- $PY /home/zephyrain/lw/MORQEX/scripts/fetch_full_electrodes.py
$PY scripts/prepare_full_suite.py
$PY scripts/run_full_transport_suite.py
```

## 新接口与验收范围

- 五个公开 NMC532 CT 使用完整提供体积，含原作者重构 CBD。它们不是五种化学材料，也不是整个制造电极片。
- 编译 C++/CUDA 兼容差分力学和 CUDA 体素输运采用矩阵自由计算；数据、库和缓存全部外置。新力学离散与原 Tet4 FEM 分开验证。
- 正浓度基按连通活性区域保留独立锂库存。固定八个状态不能普遍解析任意拓扑、反应热点和相界面。
- NMC532、NMC811、LFP、LCO 有独立参数入口；支持自定义字典、原始非单调 OCP、线性温度熵、梯度能和正迁移率。缺项补充与模型变更显式审计。
- 分相材料、掺杂、cohesive 界面损伤、SEI、析锂/剥离/死锂和自定义子模型保留。新 POU 与旧非均匀掺杂迁移率组合有独立门禁，不静默忽略不支持的物理。
- 小尺度路线的同能量反馈、统计、局部场恢复已有验证。完整 CT 多模式双向力学的误差及性能需要单独通过，不能由全体积输运通过自动推出。

```bash
# 完整 CT 力学；未收敛则保存断点并明确标注，不能作为合格参考解
$PY scripts/run_local.py --label full_mechanics --gpu -- $PY /home/zephyrain/lw/MORQEX/benchmarks/study_full_voxel_response.py --dataset nmc1cal --iterations 1500 --preconditioner coarse
```

报告在外部 `morph_workspace/reports`。0.0.9 稳定报告为 `MORQEX_crossscale_cathodes_report.html`；新结果依据自己的来源、配置、残差与误差验收，不借用旧 SVE 的通过状态。

开发版本 `0.1.0.dev0` 的新报告：`morph_workspace/reports/MORQEX_full_volume_report.html`。五个完整 CT 共 80,308,348 体素；10 项完整体积输运已通过 1e-7 相对残差，43 个原生 DFN 放电/循环基线已通过协议检查。NMC1CAL 的 DFN (20,40) 对 (40,80) 网格电压 RMSE 为 0.1C 0.00531 mV、3C 0.13285 mV、5C 0.24834 mV；它们是同物理原生 DFN 离散对照，不是微结构跨尺度误差。完整 CT 多模态双向力学及退化尚需独立验收。
