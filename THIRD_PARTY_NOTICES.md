# 来源与集成说明

GEMsim来自用户指定私有仓库 `https://github.com/Zephyrainzjl/GEMsim.git`，提交 `6fd789c78d58d49ef645c539414567fc2a2bc171`。已获用户明确指示，将微结构代码实际纳入src，而非仅作外部适配。原全部`src/gem`文本源码迁入`src/morqex/microstructure`；改动主要是绝对导入、动态导入、资源包路径、默认外部工作区、CLI CPU默认及原构建目录外置。

逐文件原始SHA256、命名空间迁移后SHA256见`provenance/gemsim_migration.json`；最终本地集成hash及有意改动见`provenance/integration_state.json`。原README、pyproject、setup、API与迁移清单另存于`provenance/gemsim_upstream`，其中原包名为历史记录，不应作为当前安装命令。

GEMsim顶层未声明统一许可证；不据此推断可自由公开再分发。当前仅在用户本机私有项目中集成，不添加将上游全部重授权的LICENSE。原代码、署名和已有许可证保留。ZynJoule未作为运行依赖，未复制其整个包。

原生echemAMR源码来源及许可证保留在`src/morqex/microstructure/simulate/native/upstream/{provenance.json,LICENSE}`，提交 `29fb99f450481c8e5f2e3f132b44ab14ee054871`。其CMake引用AMReX固定提交 `9ed35ab0e58462f951528462376cc379ba834826`；该依赖尚未构建。TetGen/ACVD/EDT等继承上游相应实现；部分历史源码提到的独立许可证文件未随GEMsim提供，发布前需补齐来源/授权，不假定统一许可。

AI生成路线、NGSolve、PyBaMM和echemAMR的科学出处仍应在实际使用时引用原方法。源码存在、可导入、可编译和物理数值验证是四种不同状态。没有进行模型训练或宣称复现上游全部平台性能。

