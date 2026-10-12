# 记忆与工作上下文

本页整理既有记忆设计中内核拥有的契约。入口包括 [memory/components.py](../src/agent_core/memory/components.py)、[memory/learning.py](../src/agent_core/memory/learning.py)、[context/summary.py](../src/agent_core/context/summary.py) 与 [context/builder.py](../src/agent_core/context/builder.py)。数据库、后台作业与发布事务由宿主实现。

## 官方组件与状态归属

内核组合 Deep Agents 的 Memory/Skills/Filesystem middleware 与 State/Store backend，以及 LangChain 的 Todo middleware。官方组件负责通用文件和待办机制；内核补充可信作用域、发布版本失效、路径权限与预算；宿主提供身份、持久化和审核政策。

| 路径/材料 | 生命周期与权限 |
| --- | --- |
| `/memory/user/AGENTS.md`、`/memory/project/AGENTS.md` | 正式发布视图，只读 |
| `/skills/user/`、`/skills/project/` | 已批准 Skill，只读；正文按需读取 |
| `/notes/TASK.md` | Task 笔记，写入官方图文件状态并保存于 checkpoint |
| `/scratch/` | 当前 Run 临时材料，随该 Run checkpoint 保存 |
| `/drafts/` | 宿主绑定的 Task 草稿，不能自动升级为正式记忆 |

tenant/user/project/session/task/run 与 namespace 由可信宿主绑定，模型路径或参数不能扩大作用域。正式发布目录拒绝写入，候选与正式内容分别维护；历史、笔记或工具结果不能成为权限指令。

## 来源与发布

`candidate` 不参与正式注入；`confirmed` 表示宿主核实用户原述或经用户审核的内容；`learned` 保留自动推断来源，不能冒充用户确认；`revoked` 不再生效。用户确认内容与当前请求优先，跨项目偏好须有明确用户来源。

记忆工具的保存成功依赖真实操作回执。Skill 是模型参考材料，不能增加权限；Workflow 提案确认只保存候选，不注册、编译或执行新工作流。

`memory/learning.py` 提供结构化经历提炼与整理：区分用户要求、工具观察、建议、完成、失败和未验证结论，允许没有适合沉淀的内容。来源有效性、作业租约、事务、CAS 发布、幂等及撤回依赖由宿主实现。格式与来源检查不能证明自然语言内容真实。

## 历史与完整输入预算

最近问答通过 `HistoryReader` 构造；安全归档、检索与回读通过 `HistoryArchive`；长期记忆通过 `MemoryServices` 注入。保留的是允许返回给模型的安全材料，不能把私密工具 payload 自动归档为公开历史。

完整模型输入预算包含最终系统提示、用户消息、工具与结构化响应 schema、正式记忆、Notes/Todo 和工具观察。字符、字节、估算 token 与供应商 usage 分别计量。归档成功后才允许移除工作视图中的旧消息；保留当前请求与未完成工具组，归档/选择失败不得假装压缩成功。

大工具结果可由官方文件组件 offload 到当前 Run 的 scratch。新 Run 不能猜测旧文件路径，而应按稳定 History 来源 ID 回读原始安全记录。Task 新 Run 只继承允许的文档与来源，不复制旧 Run 私密调用状态。

## 摘要与验证边界

Run 内摘要使用公开 Summarization middleware，仅替换工作视图；原始来源、执行状态与官方 checkpoint 保留。摘要调用有独立的有界额度，并进入模型账本。来源撤回或版本变化不能通过旧摘要重新注入失效内容。

`context_summary_enabled`、`session_summary_enabled`、工具发现与安全重试在通用 `AgentConfig` 中默认关闭。跨会话缓存与后台发布需要宿主实现相应端口；单独安装内核不等于后台服务已经部署。

Fake 和内存 Store/Saver 可以验证作用域、预算、失效和恢复契约，不能证明真实记忆准确率、总成本下降、延迟收益或 PostgreSQL 并发与跨进程恢复。对照实验需固定模型、任务和来源，同时计入学习、摘要与失败重试费用。
