# Agent Core：架构与运行契约

本页从既有核心设计整理出独立包的通用部分，以当前源码为准。它描述实现与接入契约，验证范围见 [测试说明](testing.md)。

## 宿主与内核边界

内核负责一次 Run 的执行、恢复、预算、取消与资源清理。宿主负责身份、模型配置与凭据、工具发布和授权、业务队列、用户确认、持久化资源以及公开事件投影。

内核不导入业务宿主、HTTP 框架或领域数据库客户端，不读取环境变量。`AgentConfig` 和模型连接配置由调用方显式传入。导入模块不连接服务、不构造模型客户端。

| 位置 | 当前职责 |
| --- | --- |
| [contracts](../src/agent_core/contracts/) | Run、上下文、事件、产物、结果与错误 |
| [runtime](../src/agent_core/runtime/) | 单次执行、流转、预算、取消及同步桥 |
| [tools](../src/agent_core/tools/) | 注册、参数验证、授权、治理、重试与按需发现 |
| [planning](../src/agent_core/planning/) | ReAct 与 Plan-and-Execute |
| [workflows](../src/agent_core/workflows/) | 显式工作流工厂与注册 |
| [context](../src/agent_core/context/) | 历史选择、归档、完整输入预算与摘要 |
| [memory](../src/agent_core/memory/) | 作用域、记忆端口、官方组件组合与 Task 投影 |
| [models](../src/agent_core/models/) | 显式模型配置、流式适配与请求边界 |
| [observability](../src/agent_core/observability/) | 单次模型调用的 usage 出口 |

## 公共 API 与调用链

公共导出由 [__init__.py](../src/agent_core/__init__.py) 维护。`build_agent` 装配 `AgentService`；`WorkflowRegistry` 解析显式注册的 `WorkflowDefinition`，工厂接收 `RunRequest`、`RunContext` 和 `RunResources`，返回 `WorkflowExecution`。

`AgentService.astream(request, context)` 是唯一执行实现。图、模型、工具、授权和历史端口逐层 await。同步宿主通过 `stream` 驱动同一异步生成器，使用有界队列传递事件；已有事件循环的宿主应使用 `astream`。直接调用编译图时使用 `ainvoke/astream`。

模型工厂可以是同步或异步函数。`model_ownership="owned"` 在成功、失败、取消与提前关闭流时关闭模型；`borrowed` 保留宿主共享模型。直接注入的 Saver/Store 是借用资源；`persistence_factory` 返回宿主异步上下文管理器与 `PersistenceResources`，在当前事件循环内创建和释放资源。清理或观测错误不能替代原始运行错误。

## 工具治理

普通工具通过 `ToolSpec`、`ToolRegistry` 和 `ToolExecutor` 接入。执行器统一处理可信 schema 验证、授权、调用计次、工具/分类预算、截止时间、取消与结果提交。普通工具可以返回字符串或 `ToolResult`，无需依赖某个业务领域的证据对象。

社区原生工具保留上游实现，通过 `NativeToolGovernance` 共享治理边界。宿主原生工具须明确注入 `NativeAuthorizer`；未提供可信授权不能默认执行。工具选择、授权、计次和结果政策分别保留职责。

按需发现默认关闭。开启后，`ToolCatalog` 只提供安全元数据投影，`search_tools` 在已授权、可用、阶段允许的候选中搜索，接受后才在后续模型轮次绑定 deferred 工具。隐藏工具不进入 schema 或搜索结果。可信意图簇由工具所有者贡献，内核不硬编码领域词表。schema 预算之外仍执行完整模型输入预算。

安全只读重试也默认关闭。工具维护者必须明确声明 `effect=read` 和 `safe_read`；只有宿主分类为 transient 且可重试的 handler 失败才能重试。尝试共享原截止时间，每次重新授权；结果构造与提交失败不能导致 handler 重放。逻辑调用、额外重试预留与已知执行次数分别计量。

## checkpoint 与恢复

官方 Saver 是执行状态权威，由宿主注入；图配置中的 thread ID 使用 `RunRequest.run_id`。持久化启用时使用 `durability="sync"`，后继节点等待前一步提交；这不表示数据库驱动为同步实现。

同 Run 恢复要求已有 checkpoint，核验状态版本、Run/workflow 身份、可信绑定指纹以及可选精确 checkpoint ID。缺失或不兼容时明确失败，禁止静默从头执行。通过校验后以 `graph.astream(None, ...)` 继续已保存状态，恢复工具预算、已完成调用、引用与观察账本，不能获得新额度。

恢复适配通过公开 `get_state/update_state` 压实已成功的 pending writes，再派发仍未回答的调用。重试协议下已开始但没有已提交终态的调用返回结果未知回执，保留消耗，不直接重放原 ID。外部副作用与 checkpoint 提交之间仍存在崩溃窗口；写入工具须自行提供幂等键、结果核验或补偿，checkpoint 不能证明通用 exactly-once。

官方 interrupt 通过 `run.suspended` 暴露 JSON 安全的暂停 ID 与 checkpoint。`ResumeDecision` 必须匹配保存的 checkpoint 和全部暂停 ID；等待审批不伪造工具完成。宿主实现审批界面、可信决策转换与字段过滤。

同 Task 新 Run 和同 Run 恢复不同。前者仅继承明确允许的 Notes/Todo/source 与计划状态；不复制旧 Run 的消息、scratch 或私密工具调用状态。Task 投影通过官方图 API 物化文件 DeltaChannel，不能猜测 Saver 原始字段。计划实例的累计预算与确认规则见 [计划执行](planning.md)。

## 事件、结果与用量

`RunEvent` 关联 Run、attempt、步骤、活动和工具调用；宿主消费公共事件，无需依赖 LangGraph 私有状态。`RunResult` 的产物由工作流明确提供，不从任意私有图字段推断。宿主仍负责公开 payload 白名单、身份过滤和持久事件序号。

每次模型调用终止后只通知一次合并 usage。缺失用量保持未知，真实零值保持零，失败/取消保留状态。observer 异常不得改变运行结果。估算 token、供应商实际 usage、观察字节和字符长度分别命名，不把单次输入上限写成累计 token 硬上限。

## 扩展与当前限制

ReAct 复用 LangChain `create_agent`，官方 middleware 拥有模型/工具路由。Plan-and-Execute 使用薄 `StateGraph` 编排，Memory/Skills/Filesystem/Todo 复用公开组件。新固定流程注册 `WorkflowDefinition`，无需向运行器增加业务分支。

目前不提供通用并行 subagent、任意 Workflow 自动发布执行、完整业务认证/审批 UI 或第三方 middleware 全兼容承诺。长期记忆、摘要与自动学习的作用域和证据边界见 [记忆与上下文](memory-context.md)。异步 Store 适配包含固定 Deep Agents 版本的内部 helper；升级依赖须回归文件、作用域与异步 Store 语义，不能仅凭可导入判断兼容。
