# 独立包验证

所有命令从本仓库根目录运行。安装与构建可能访问包索引；核心契约检查和离线示例不调用真实模型、不连接数据库。

## 核心契约

```bash
uv sync --frozen
uv run --frozen python scripts/check_core.py
```

验收器使用 `unittest` 加载 [tests/agent_core_tests](../tests/agent_core_tests/) 中的全部测试，禁止业务宿主、CrewAI、FastAPI、PostgreSQL/图数据库客户端导入，并拒绝网络连接/DNS、SQLite 连接与子进程启动。测试使用 Fake 模型、普通工具、固定图和替代端口；包括流式取消、资源清理、工具预算、结构化阶段、官方记忆组件以及 checkpoint 恢复。

## 打包后仓库外导入

```bash
uv build --wheel --out-dir dist
uv run --frozen python scripts/check_core.py --wheel dist/stategrid_agent_core-0.2.0-py3-none-any.whl
```

wheel 模式解包到临时仓库外目录，核验真实导入位置，拒绝 wheel 包含宿主或测试源码，再运行同一契约集合。检查不会安装依赖，所选 Python 环境须已准备运行依赖。

## 示例与文档

```bash
uv run --frozen python examples/offline_react.py
uv run --frozen python scripts/check_docs.py
git diff --check
```

离线演示以确定性 Fake 模型调用加法工具并输出事件，展示公共接入方式。文档检查只验证仓库内 Markdown 路径及锚点，不能自动证明文档语义符合实现。

## 证据边界

以上检查不能证明真实模型的规划/记忆质量、供应商请求费用、PostgreSQL 初始化/事务/并发与跨进程恢复，或任意工具副作用恰好一次。第三方 middleware 全兼容、认证与审批 UI、业务队列和部署由宿主另行验收。通过数量只引用相应实际运行记录，CI 结果以 GitHub Actions 运行记录为准。
