# PAIBox 编译产物可视化

PAIBox Visualizer 用于检查后端编译产物中的芯片布局、配置帧、I/O 映射和离线核参数。
它采用 frame-first 解析路径：优先从最终配置帧还原芯片状态，再用 `config.pb` 中的
metadata 做补充和交叉校验。这样可以直接验证工具链最终写出的帧是否符合预期，而不是只查看
后端内部对象。

当前 V1 面向 backendv2 单芯片产物，visualizer backend 名称为 `v2`；UI 和 viewer schema 已预留后续多芯片 / 多 backend
适配空间。

## 安装

可视化服务依赖 FastAPI 和 Uvicorn，作为可选依赖安装：

```bash
pip install "paibox[visualizer]"
```

安装后用户可直接使用 `paiviz` 启动页面。Node.js、npm 和 Vite 只用于开发或重新打包前端，
普通安装用户不需要安装它们。

## 命令

启动 Web UI：

```bash
paiviz --artifact /path/to/config.pb
```

默认命令是 `serve`，显式写出也等价。默认监听 `127.0.0.1`，`--port 0` 会自动选择可用端口：

```bash
paiviz serve --artifact /path/to/config.pb --host 127.0.0.1 --port 0
```

只做解析和校验，不启动 UI：

```bash
paiviz validate --artifact /path/to/config.pb
paiviz validate --artifact /path/to/config.pb --json
```

`validate` 遇到非法帧会失败，并输出 artifact 路径、帧索引、raw frame、header、核坐标、
package 范围、SRAM 地址等溯源信息。当前策略是“发现即失败”，不把非法帧静默降级成 warning。

## 支持的输入

`--artifact` 可以指向以下对象：

- `proto/config.pb`
- `proto/config.json`
- `cfg_frames.npy`
- 包含上述文件的 artifact 目录

当同时存在 `config.pb` 和帧文件时，核心配置、LUT、神经元、权重和 raw frame 视图以最终帧解析结果
为准；`config.pb` 主要提供 I/O 映射、shape、runtime tick 和节点名等上下文。

## Chip 视图

Chip 视图展示 9 x 9 核阵列：

- `(0, 0)` 是 RISC-V CPU 核，使用特殊颜色标识。
- 下两行除 CPU 外的 17 个核是 online core。
- 其余核是 offline core。
- 颜色区分核角色、使用状态、tick start 和 I/O 热度。
- 可选 overlay 包括 global send/receive 方向、选中核的神经元目的地箭头、tick 标签和 I/O Map。

左侧 Tick Compute 面板按 `tick / SOPS / core` 展示计算压力，并支持按列升序或降序排序。
SOPS 是基于解码后的神经元和权重存储信息估计的突触运算量，可选择是否把 CSC padding 计入。

## Core Inspector

在 Chip 模式下，右侧 inspector 面向当前选中核，包含：

- `Core`：core config 的语义值或 raw 值、LCN/Axon、timing、signal bits、SOPS。
- `LUT`：存在配置帧 2 时显示，提供 2D 函数图和可滚动表格。
- `Neurons`：离线核整数路径的神经元参数摘要和分页详情，folded neuron 默认按实际神经元数聚合。
- `Weights`：dense / CSC 权重存储摘要、数据类型、存储大小、nonzero、padding 和已解码存储值。

标题栏提供 Raw 调试入口，用于按 frame index、SRAM address 或 package type 搜索底层帧和 package。
Raw 是增强调试工具，不作为默认信息入口。

## IO Map 视图

IO Map 是独立主视图，不放在 Core Inspector 的 tab 内。它用于查看：

- 原始 tensor 区域如何映射到目标 core。
- 每个 core 的 256 x 512-bit input buffer 占用。
- 输入 / 输出方向切换。
- 对 rank >= 2 的 tensor，默认用最后两个维度作为 2D plane，外层维度作为 slice。
- 对一维输出，使用条带或小矩阵展示，不强行画 2D 平面。

选中某个 core 后，右侧 IO inspector 会显示该 core 相关的 source plane、buffer map、tensor strip
和 entry detail。mask 支持 hover / click 查看：

- tensor 名称、slice、dim range
- bbox 范围和 bbox 尺寸
- active element count、bbox area、coverage ratio
- 目标 core、buffer row/bit range、dtype、target LCN

没有显式 layout metadata 时，UI 不使用 NCHW/NHWC 术语，只显示 `dim0/dim1/...`。

## 开发与打包边界

Python 应用层位于 `paibox/visualizer/`，v2 适配器位于
`paibox/visualizer/backends/v2/`。前端源码位于 `tools/visualizer_ui/`。

发布到 Python wheel 时，只打包 `paibox/visualizer/static/` 中的构建后静态资源和 Python 服务代码；
`tools/visualizer_ui/` 中的 Node/npm 工程不作为安装后的运行依赖。
