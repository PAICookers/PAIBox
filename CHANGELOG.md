## v1.0.0a3

- 添加了示例程序，MNIST双输入端口两层全连接网络
- 修复工作流版本错误
- 修复神经元膜电位溢出处理错误
- 修复神经元分组的计数错误

## v1.0.0a4

- 重命名突触连接类型 `ConnType` 为 `SynConnType` ，现在通过 `pb.SynCnnType.x` 调用。例如，`pb.SynCnnType.All2All`

## v1.0.0a5

- 支持无限嵌套深度的网络
- 支持全展开2D卷积算子构建与部署（`padding` 不支持）
- 修复当 `keep_shape=True` 时，，神经元状态变量在运行时尺寸错误
