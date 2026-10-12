"""Public community memory components exercised with fake models and stores."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

from deepagents.backends import StateBackend, StoreBackend
from deepagents.backends.protocol import WriteResult
from deepagents.backends.utils import create_file_data, file_data_to_string
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, TodoListMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from agent_core.memory import (
    HistoryEntry, MemoryBinding, MemoryLimits, MemoryMutationResult, MemoryOperationError, MemoryRecord,
    MemoryRelease, MemoryScope, MemorySnapshot, TASK_NOTES_PATH, TASK_NOTES_TEMPLATE,
    build_memory_components, inherit_task_memory, read_task_checkpoint, restore_memory_binding, task_memory_snapshot,
    validate_task_notes,
)
from agent_core.config import AgentConfig
from agent_core.contracts.runs import RunRequest
from agent_core.contracts.errors import AgentContextError
from agent_core.models.streaming import stream_chat_message
from agent_core.planning.react.builder import compile_react_graph
from agent_core.planning.react.policy import ReActContext, ReActPolicy
from agent_core.runtime.budget import ExecutionBudget
from agent_core.runtime.cancellation import CancellationToken
from agent_core.runtime.streaming import AsyncStreamRunner
from agent_core.tools import ToolExecutor, ToolRegistry, ToolSpec
from agent_core.tools.langchain import build_langchain_tools
from agent_core.memory.tools import history_payload
from agent_core_tests.test_runtime import MinimalToolRuntime, ScriptedModel


class ScriptModel(FakeMessagesListChatModel):
    seen: list[Any] = Field(default_factory=list)
    on_call: Any = None

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        index = len(self.seen)
        self.seen.append(list(messages))
        if self.on_call is not None:
            self.on_call(index, messages)
        if index >= len(self.responses):
            raise AssertionError("unexpected model call")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class ScopedService:
    def __init__(self, store):
        self.store = store
        self.revision = "1"
        self.calls = []
        self.receipts = {}
        self.fail_snapshot = False

    def snapshot(self, scope):
        if self.fail_snapshot:
            raise RuntimeError("snapshot unavailable")
        return MemorySnapshot(MemoryRelease(
            self.revision, (scope.tenant_id, scope.user_id, self.revision, "memory"),
            (scope.tenant_id, scope.user_id, self.revision, "skills"),
        ))

    def publish(self, scope, content, *, skill=None):
        release = self.snapshot(scope).user
        StoreBackend(store=self.store, namespace=lambda _: release.memory_namespace).write("/AGENTS.md", content)
        if skill:
            StoreBackend(store=self.store, namespace=lambda _: release.skills_namespace).write("/lookup/SKILL.md", skill)

    def list_records(self, scope, **kwargs):
        return []

    def read_record(self, scope, record_id, *, version=None):
        return None

    def mutate(self, scope, change, *, operation_id):
        if operation_id not in self.receipts:
            self.calls.append((scope, change, operation_id))
            # A model request alone creates only a candidate.
            record = MemoryRecord("candidate-1", "memory", change.target, change.title,
                                  change.content, 1, "candidate", change.source_ids)
            self.receipts[operation_id] = MemoryMutationResult(record)
        return self.receipts[operation_id]


class SafeHistory:
    def __init__(self):
        self.scope = None

    def append(self, scope, entries):
        self.scope = scope

    def search(self, scope, query, *, limit=10, task_id=None):
        self.scope = scope
        return [HistoryEntry("history-1", "previous-run", 1, "tool", "evidence " * 100,
                             call_id="source-call", tool_name="read_data")]

    def read(self, scope, source_ids, *, max_characters=16_000, offset_characters=0):
        return self.search(scope, "")


class EmptyInput(BaseModel):
    pass


class TestToolPolicy(AgentMiddleware):
    """The production native executor applies this same exported decision."""

    def __init__(self, components):
        self.components = components

    def wrap_tool_call(self, request, handler):
        decision = self.components.authorize_tool(request.tool_call["name"], request.tool_call["args"])
        if not decision.allowed:
            return ToolMessage(content=decision.message, status="error",
                               tool_call_id=request.tool_call["id"])
        return handler(request)


def call(name, arguments, call_id="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": call_id, "type": "tool_call"}])


class MemoryComponentTest(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.scope = MemoryScope("tenant", "user-a", None, "session", "task", "run")
        self.services = ScopedService(self.store)
        self.binding = MemoryBinding(self.scope, self.store, self.services)

    def run_graph(self, responses, *, on_call=None, binding=None):
        binding = binding or self.binding
        components = build_memory_components(binding)
        model = ScriptModel(responses=responses, on_call=on_call)
        saver = InMemorySaver()
        graph = create_agent(model, tools=list(components.tools),
                             middleware=[TestToolPolicy(components), *components.middleware],
                             checkpointer=saver)
        context = SimpleNamespace(memory=binding, cancellation=CancellationToken())
        config = {"configurable": {"thread_id": binding.scope.run_id}}
        result = graph.invoke({"messages": [HumanMessage("完成任务")]}, config=config, context=context)
        return result, graph.get_state(config).values, model, components

    def core_graph(self, responses, *, binding=None, saver=None, middleware=(), tools=()):
        binding = binding or self.binding
        model = ScriptedModel(responses=[[response] for response in responses])
        runner = AsyncStreamRunner()
        self.addCleanup(runner.close)
        settings = AgentConfig()
        cancellation = CancellationToken()
        tool_runtime = MinimalToolRuntime(cancellation,
            execution=ExecutionBudget(max_tool_calls=settings.max_tool_calls), run_awaitable=runner.run)
        policy = ReActPolicy()
        context = ReActContext(RunRequest(binding.scope.run_id, "memory-test", "完成任务"),
            cancellation, settings, runner, tool_runtime.clock, tool_runtime, policy, memory=binding)

        def collect(model, messages, cancellation, **kwargs):
            kwargs.pop("phase", None)
            kwargs.pop("step_id", None)
            return stream_chat_message(model, messages, cancellation, **kwargs)

        graph = compile_react_graph(model, tools, stream_chat_message=collect, policy=policy,
                                    memory=binding, checkpointer=saver, middleware=middleware)
        config = {"configurable": {"thread_id": binding.scope.run_id}, "recursion_limit": settings.recursion_limit}
        return graph, context, config, model

    def test_official_components_and_notes_refresh_after_file_tool(self):
        notes = TASK_NOTES_TEMPLATE.replace("待记录。", "保留当前目标 SOURCE-123", 1)
        result, state, model, components = self.run_graph([
            call("write_file", {"file_path": TASK_NOTES_PATH, "content": notes}),
            call("write_todos", {"todos": [{"content": "核对结果", "status": "completed"}]}, "todo-call"),
            AIMessage(content="完成结果"),
        ])
        self.assertTrue(any(isinstance(m, MemoryMiddleware) for m in components.middleware))
        self.assertTrue(any(isinstance(m, FilesystemMiddleware) for m in components.middleware))
        self.assertTrue(any(isinstance(m, SkillsMiddleware) for m in components.middleware))
        self.assertTrue(any(isinstance(m, TodoListMiddleware) for m in components.middleware))
        self.assertIn("SOURCE-123", model.seen[1][0].text)
        projected = task_memory_snapshot(state)
        self.assertEqual(projected["notesMarkdown"], notes)
        self.assertEqual(projected["todos"], [{"content": "核对结果", "status": "completed"}])
        self.assertEqual(result["messages"][-1].content, "完成结果")

    def test_published_files_are_read_only_and_other_users_are_not_loaded(self):
        self.services.publish(self.scope, "USER_A_CONFIRMED")
        other = replace(self.scope, user_id="user-b")
        self.services.publish(other, "USER_B_SECRET")
        _, state, model, _ = self.run_graph([
            call("write_file", {"file_path": "/memory/user/AGENTS.md", "content": "OVERRIDE"}),
            AIMessage(content="完成"),
        ])
        self.assertIn("USER_A_CONFIRMED", model.seen[0][0].text)
        self.assertNotIn("USER_B_SECRET", repr(model.seen))
        self.assertNotIn("OVERRIDE", state["memory_contents"]["/memory/user/AGENTS.md"])
        self.assertEqual(next(m for m in state["messages"] if isinstance(m, ToolMessage)).status, "error")

    def test_layered_budget_combines_user_and_project_without_splitting_records(self):
        import json
        from langchain_core.messages import SystemMessage
        from langchain_core.messages.utils import count_tokens_approximately

        class BothLayers(ScopedService):
            def snapshot(self, scope):
                user = replace(super().snapshot(scope).user, layered=True)
                return MemorySnapshot(user, replace(user,
                    memory_namespace=(*user.memory_namespace, 'project'),
                    skills_namespace=(*user.skills_namespace, 'project')))

        scope = replace(self.scope, project_id='project')
        services = BothLayers(self.store)
        binding = MemoryBinding(scope, self.store, services)
        snapshot = services.snapshot(scope)
        body = '完整条件：只有证据充分时采用结论。' * 45
        for target, release in (('user', snapshot.user), ('project', snapshot.project)):
            backend = StoreBackend(store=self.store, namespace=lambda _, ns=release.memory_namespace: ns)
            for path in ('/AGENTS.md', '/EXPERIENCE.md'):
                entries = ['MEMORY_ITEM ' + json.dumps({'id': f'{target}-{index}', 'version': 1,
                    'content': body}, ensure_ascii=False) for index in range(14)]
                backend.write(path, '# 分层记忆\n' + '\n'.join(entries) + '\n此视图展开 14/14 条；用 memory_list 读取。')
        _, state, model, _ = self.run_graph([AIMessage(content='完成')], binding=binding)
        for suffix, budget in (('/AGENTS.md', 2048), ('/EXPERIENCE.md', 1024)):
            values = [value for path, value in state['memory_contents'].items()
                      if path.startswith('/memory/') and path.endswith(suffix)]
            self.assertEqual(len(values), 2)
            self.assertLessEqual(count_tokens_approximately([SystemMessage('\n'.join(values))]), budget)
            for value in values:
                for line in value.splitlines():
                    if line.startswith('MEMORY_ITEM '):
                        self.assertEqual(json.loads(line.split(' ', 1)[1])['content'], body)
            self.assertIn('未展开记录', '\n'.join(values))
        self.assertIn('自动经验', model.seen[0][0].text)

    def test_release_change_refreshes_official_memory_and_skill_cache(self):
        skill1 = "---\nname: lookup\ndescription: OLD_SKILL_DESCRIPTION\n---\nold body"
        self.services.publish(self.scope, "OLD_CONSTRAINT", skill=skill1)

        def publish_change(index, _messages):
            if index == 0:
                self.services.revision = "2"
                skill2 = skill1.replace("OLD_SKILL_DESCRIPTION", "NEW_SKILL_DESCRIPTION")
                self.services.publish(self.scope, "NEW_CONSTRAINT", skill=skill2)

        _, state, model, _ = self.run_graph([
            call("ls", {"path": "/scratch/"}), AIMessage(content="完成"),
        ], on_call=publish_change)
        self.assertIn("OLD_CONSTRAINT", model.seen[0][0].text)
        self.assertNotIn("OLD_CONSTRAINT", model.seen[1][0].text)
        self.assertIn("NEW_CONSTRAINT", model.seen[1][0].text)
        self.assertIn("NEW_SKILL_DESCRIPTION", model.seen[1][0].text)
        self.assertEqual(state["loaded_memory_releases"], {"user": "2"})

    def test_refresh_failure_stops_before_next_model_call(self):
        def fail(index, _messages):
            if index == 0:
                self.services.fail_snapshot = True

        with self.assertRaisesRegex(RuntimeError, "snapshot unavailable"):
            self.run_graph([call("ls", {"path": "/scratch/"}), AIMessage(content="must not run")], on_call=fail)

    def test_invalid_note_edit_is_rejected_before_checkpoint_file_changes(self):
        _, state, _, _ = self.run_graph([
            call("edit_file", {"file_path": TASK_NOTES_PATH, "old_string": "## 目标", "new_string": "removed"}),
            AIMessage(content="需要保留笔记结构"),
        ])
        self.assertEqual(task_memory_snapshot(state)["notesMarkdown"], TASK_NOTES_TEMPLATE)
        self.assertEqual(next(m for m in state["messages"] if isinstance(m, ToolMessage)).status, "error")

    def test_semantic_write_keeps_scope_call_identity_and_candidate_status(self):
        _, state, _, components = self.run_graph([
            call("memory_remember", {"target": "user", "content": "remembered candidate", "source_ids": ["history-1"]}),
            AIMessage(content="已提出候选"),
        ])
        scope, change, operation = self.services.calls[0]
        self.assertEqual(scope, self.scope)
        self.assertEqual(operation, "run:call-1")
        self.assertEqual(change.source_ids, ("history-1",))
        tool = next(t for t in components.tools if t.name == "memory_remember")
        schema = tool.tool_call_schema.model_json_schema()
        self.assertNotIn("runtime", schema["properties"])
        self.assertNotIn("user_id", schema["properties"])
        self.assertNotIn("confirm", schema["properties"])
        message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
        self.assertIn('"status":"candidate"', message.content)
        self.assertIn("memory:candidate-1:v1", state["task_source_ids"])

    def test_history_result_is_bounded_and_preserves_source_and_scope(self):
        history = SafeHistory()
        binding = MemoryBinding(self.scope, self.store, self.services, history,
                                limits=MemoryLimits(max_history_characters=20))
        _, state, _, _ = self.run_graph([
            call("history_search", {"query": "evidence"}), AIMessage(content="完成"),
        ], binding=binding)
        message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
        self.assertIn('"truncated":true', message.content)
        self.assertIn('"sourceId":"history-1"', message.content)
        self.assertEqual(history.scope, self.scope)
        self.assertEqual(state["task_source_ids"], ["history-1"])

    def test_inherit_copies_only_task_memory_and_rejects_traversal(self):
        _, state, _, components = self.run_graph([AIMessage(content="完成")])
        state["files"]["/scratch/private"] = {"content": "not inherited", "encoding": "utf-8"}
        inherited = inherit_task_memory(state)
        self.assertEqual(set(inherited["files"]), {TASK_NOTES_PATH})
        self.assertNotIn("memory_contents", inherited)
        self.assertNotIn("messages", inherited)
        self.assertFalse(components.authorize_tool("write_file", {
            "file_path": "/scratch/../memory/user/AGENTS.md", "content": "attack",
        }).allowed)

    def test_drafts_persist_across_runs_of_same_task_and_remain_user_scoped(self):
        self.run_graph([
            call("write_file", {"file_path": "/drafts/proposal.md", "content": "DRAFT_EVIDENCE"}),
            AIMessage(content="已保存候选草稿"),
        ])
        next_binding = MemoryBinding(replace(self.scope, run_id="run-2"), self.store, self.services)
        _, _, model, _ = self.run_graph([
            call("read_file", {"file_path": "/drafts/proposal.md"}), AIMessage(content="已读取"),
        ], binding=next_binding)
        self.assertIn("DRAFT_EVIDENCE", repr(model.seen[1]))
        other_binding = MemoryBinding(replace(self.scope, user_id="user-b", run_id="other-run"),
                                      self.store, self.services)
        _, _, other, _ = self.run_graph([
            call("read_file", {"file_path": "/drafts/proposal.md"}), AIMessage(content="没有草稿"),
        ], binding=other_binding)
        self.assertNotIn("DRAFT_EVIDENCE", repr(other.seen))

    def test_model_sees_scratch_lifetime_and_original_history_lookup_guidance(self):
        binding = MemoryBinding(self.scope, self.store, self.services, history=SafeHistory())
        graph, context, config, model = self.core_graph([AIMessage(content="已核对读取边界")], binding=binding)
        advertised = {}
        bind_tools = model.bind_tools

        def capture(tools, **kwargs):
            advertised.update({tool.name: tool.description for tool in tools})
            return bind_tools(tools, **kwargs)

        model.bind_tools = capture
        graph.invoke({"messages": [HumanMessage("回查上一轮证据")]}, config=config, context=context)
        self.assertIn("not inherited by a new Run", advertised["read_file"])
        self.assertIn("zero-based source-line", advertised["read_file"])
        self.assertIn("literal substring", advertised["history_search"])
        self.assertIn("Tool-name metadata is not searched", advertised["history_search"])
        self.assertIn("retrieval receipts", advertised["history_read"])
        self.assertIn("shared tool and observation budgets", advertised["history_read"])
        prompt = model.calls[0][1][0].text
        self.assertIn("新 Run 不继承", prompt)
        self.assertIn("搜索空结果不代表业务数据不存在或没有异常", prompt)
        self.assertIn("不能通过重复读取或新步骤重置额度", prompt)
        self.assertEqual(context.tool_runtime.execution.tool_call_count, 0)

    def test_history_no_match_and_unknown_ids_return_successful_empty_results(self):
        class EmptyHistory(SafeHistory):
            def search(self, scope, query, *, limit=10, task_id=None):
                self.search_query = query
                return ()

            def read(self, scope, source_ids, *, max_characters=16000, offset_characters=0):
                self.requested_ids = source_ids
                return ()

        history = EmptyHistory()
        binding = MemoryBinding(self.scope, self.store, self.services, history=history)
        graph, context, config, model = self.core_graph([
            call("history_search", {"query": "literal excerpt with spaces"}, "no-match"),
            call("history_read", {"source_ids": ["unknown-or-unavailable"]}, "unknown-source"),
            AIMessage(content="未查到可用历史证据，无法据此判断业务数据不存在"),
        ], binding=binding)
        result = graph.invoke({"messages": [HumanMessage("回查已有证据")]}, config=config, context=context)
        observations = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(len(observations), 2)
        for message in observations:
            self.assertEqual(message.status, "success")
            self.assertEqual(json.loads(message.content), {"available": True, "entries": []})
        self.assertEqual(history.search_query, "literal excerpt with spaces")
        self.assertEqual(history.requested_ids, ["unknown-or-unavailable"])
        self.assertEqual(context.tool_runtime.execution.tool_call_count, 2)
        self.assertEqual(context.tool_runtime.execution.calls_by_category, {"memory": 2})

    def test_core_react_multiple_todos_rejected_without_blocking_later_read(self):
        batch = AIMessage(content="", tool_calls=[
            {"name": "write_todos", "args": {"todos": [{"content": "first", "status": "in_progress"}]},
             "id": "todo-a", "type": "tool_call"},
            {"name": "write_todos", "args": {"todos": [{"content": "second", "status": "pending"}]},
             "id": "todo-b", "type": "tool_call"},
            {"name": "read_file", "args": {"file_path": TASK_NOTES_PATH}, "id": "read-c", "type": "tool_call"},
        ])
        saver = InMemorySaver()
        graph, context, config, model = self.core_graph([batch, AIMessage(content="已核对任务笔记")], saver=saver)
        result = graph.invoke({"messages": [HumanMessage("完成任务")]}, config=config, context=context)
        state = graph.get_state(config).values
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(result["final_answer"], "已核对任务笔记")
        messages = {m.tool_call_id: m for m in state["messages"] if isinstance(m, ToolMessage)}
        self.assertEqual(set(messages), {"todo-a", "todo-b", "read-c"})
        self.assertIn("任务笔记", messages["read-c"].content)
        self.assertEqual(state["execution_ledger"]["calls_by_category"]["memory"], 3)
        self.assertEqual(state.get("todos", []), [])

    def test_core_react_resume_tools_with_new_binding_restores_selected_release(self):
        self.services.publish(self.scope, "PERSISTED_RELEASE")

        class CrashBeforeFileRead(AgentMiddleware):
            def wrap_tool_call(self, request, handler):
                raise RuntimeError("simulated process loss before tool I/O")

        saver = InMemorySaver()
        graph, context, config, model = self.core_graph([
            call("read_file", {"file_path": "/memory/user/AGENTS.md"}),
        ], saver=saver, middleware=[CrashBeforeFileRead()])
        with self.assertRaisesRegex(RuntimeError, "simulated process loss"):
            graph.invoke({"messages": [HumanMessage("读取已有记忆")]}, config=config, context=context)
        state = graph.get_state(config).values
        self.assertEqual(len(model.calls), 1)
        self.assertIn("memory_release_snapshot", state)
        new_binding = MemoryBinding(self.scope, self.store, self.services)
        self.assertIsNone(new_binding.snapshot)
        self.assertTrue(restore_memory_binding(new_binding, state))
        resumed, resumed_context, _, resumed_model = self.core_graph([
            AIMessage(content="已读取保存的发布视图"),
        ], binding=new_binding, saver=saver)
        result = resumed.invoke(None, config=config, context=resumed_context)
        self.assertEqual(result["final_answer"], "已读取保存的发布视图")
        self.assertEqual(len(resumed_model.calls), 1)
        self.assertTrue(any(isinstance(m, ToolMessage) and "PERSISTED_RELEASE" in m.content
                            for m in resumed_model.calls[0][1]))
        self.assertEqual(resumed.get_state(config).values["execution_ledger"]["tool_call_count"], 1)

    def test_initial_oversized_published_memory_is_rejected(self):
        self.services.publish(self.scope, "x" * 100)
        binding = MemoryBinding(self.scope, self.store, self.services,
                                limits=MemoryLimits(max_published_characters=20))
        graph, context, config, model = self.core_graph([AIMessage(content="must not run")], binding=binding)
        with self.assertRaisesRegex(ValueError, "published memory exceeds"):
            graph.invoke({"messages": [HumanMessage("任务")]}, config=config, context=context)
        self.assertEqual(model.calls, [])

    def test_core_offloads_safe_domain_tool_and_reads_it_with_official_file_tool(self):
        content = "\n".join(f"SAFE-EVIDENCE-{number:04d}" for number in range(2_000))
        executor = ToolExecutor(registry=ToolRegistry([ToolSpec(
            "large_evidence", "Return a safe evidence sample", EmptyInput, lambda _args, _runtime: content,
        )]))
        saver = InMemorySaver()
        graph, context, config, model = self.core_graph([
            call("large_evidence", {}, "large-call"),
            call("read_file", {"file_path": "/scratch/large_tool_results/large-call", "offset": 899, "limit": 3}, "read-large"),
            AIMessage(content="已读取第900行附近的证据"),
        ], saver=saver, tools=build_langchain_tools(executor.registry.definitions, executor=executor))
        graph.invoke({"messages": [HumanMessage("读取证据")]}, config=config, context=context)
        second_messages = model.calls[1][1]
        observation = next(m for m in second_messages if isinstance(m, ToolMessage))
        self.assertIn("/scratch/large_tool_results/large-call", observation.content)
        self.assertLess(len(observation.content), len(content))
        self.assertTrue(any(isinstance(m, ToolMessage) and "SAFE-EVIDENCE-0899" in m.content
                            for m in model.calls[2][1]))
        state = graph.get_state(config).values
        self.assertEqual(file_data_to_string(state["files"]["/scratch/large_tool_results/large-call"]), content)
        self.assertEqual(state["execution_ledger"]["calls_by_category"], {"external": 1, "memory": 1})

    def test_small_read_file_pages_are_valid_and_file_errors_keep_actionable_safe_reasons(self):
        graph, context, config, model = self.core_graph([
            call("read_file", {"file_path": "/scratch/evidence", "offset": 1, "limit": 10}, "past-end"),
            call("read_file", {"file_path": "/scratch/evidence.json", "limit": 10}, "missing-file"),
            call("read_file", {"file_path": "/scratch/evidence", "offset": 0, "limit": 1}, "small-page"),
            AIMessage(content="Read the single source line"),
        ])
        graph.invoke({"messages": [HumanMessage("Read evidence")],
                      "files": {"/scratch/evidence": create_file_data("one source line")}},
                     config=config, context=context)
        observations = {message.tool_call_id: message for message in model.calls[-1][1] if isinstance(message, ToolMessage)}
        self.assertEqual(json.loads(observations["past-end"].content)["errorCode"], "file_read_offset_out_of_range")
        self.assertEqual(json.loads(observations["past-end"].content)["totalLines"], 1)
        self.assertEqual(json.loads(observations["missing-file"].content)["errorCode"], "file_not_found")
        self.assertEqual(observations["small-page"].status, "success")
        self.assertIn("one source line", observations["small-page"].content)

    def test_large_read_file_page_fits_real_model_view_and_history_recovers_its_tail(self):
        class StoredHistory(SafeHistory):
            def __init__(self):
                self.entries = {}

            def append(self, scope, entries):
                for entry in entries:
                    old = self.entries.get(entry.source_id)
                    if old is not None and old.content != entry.content:
                        raise AssertionError("full archived evidence was overwritten")
                    self.entries[entry.source_id] = entry

            def read(self, scope, source_ids, *, max_characters=16000, offset_characters=0):
                self.last_page_size = max_characters
                return [replace(self.entries[key], content=self.entries[key].content[offset_characters:offset_characters + max_characters],
                                metadata={"offset_characters": offset_characters,
                                          "original_characters": len(self.entries[key].content)}) for key in source_ids]

        class RequiredInstructions(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                return handler(request.override(system_message=SystemMessage(request.system_message.text + "mandatory " * 3000)))

        history = StoredHistory()
        binding = MemoryBinding(self.scope, self.store, self.services, history=history)
        saver = InMemorySaver()
        graph, context, config, model = self.core_graph([
            call("read_file", {"file_path": "/scratch/evidence", "limit": 100}, "wide-read"),
            AIMessage(content="placeholder"), AIMessage(content="Read the saved tail without repeating the original query"),
        ], binding=binding, saver=saver, middleware=[RequiredInstructions()])
        original_stream = model.stream

        def scripted(messages, **kwargs):
            if len(model.calls) == 1:
                window = next(message for message in messages if isinstance(message, ToolMessage))
                page = json.loads(window.content.splitlines()[1])
                model.window_page = page
                model.responses[1] = [call("history_read", {"source_ids": [page["sourceId"]],
                                                           "offset_characters": page["nextOffset"], "max_characters": 1000}, "read-tail")]
            yield from original_stream(messages, **kwargs)

        model.stream = scripted
        original = "0123456789" * 2400
        graph.invoke({"messages": [HumanMessage("Read evidence")],
                      "files": {"/scratch/evidence": create_file_data(original)}}, config=config, context=context)
        state = graph.get_state(config).values
        wide = next(message for message in state["messages"] if isinstance(message, ToolMessage) and message.tool_call_id == "wide-read")
        self.assertEqual(len(wide.content), 16000)
        self.assertEqual(history.entries[model.window_page["sourceId"]].content, wide.content)
        self.assertEqual(file_data_to_string(state["files"]["/scratch/evidence"]), original)
        recalled = next(message for message in model.calls[-1][1] if isinstance(message, ToolMessage) and message.tool_call_id == "read-tail")
        page = json.loads(recalled.content)["entries"][0]
        self.assertEqual(page["offset"], model.window_page["nextOffset"])
        self.assertEqual(page["content"], wide.content[page["offset"]:page["nextOffset"]])
        self.assertEqual(history.last_page_size, 1000)
        self.assertEqual(context.tool_runtime.execution.tool_call_count, 2)
        self.assertEqual(len(model.calls), 3)

    def test_core_failed_offload_keeps_original_and_refuses_over_budget_input(self):
        content = "UNSHORTENED_EVIDENCE\n" * 4_000
        executor = ToolExecutor(registry=ToolRegistry([ToolSpec(
            "large_evidence", "Return a safe evidence sample", EmptyInput, lambda _args, _runtime: content,
        )]))
        original_write = StateBackend.write

        def fail_offload(backend, path, value):
            if path.startswith("/scratch/large_tool_results/"):
                return WriteResult(error="simulated storage failure")
            return original_write(backend, path, value)

        saver = InMemorySaver()
        graph, context, config, model = self.core_graph([
            call("large_evidence", {}, "failed-offload"), AIMessage(content="must not run"),
        ], saver=saver, tools=build_langchain_tools(executor.registry.definitions, executor=executor))
        with patch.object(StateBackend, "write", autospec=True, side_effect=fail_offload):
            with self.assertRaisesRegex(AgentContextError, "Input budget exceeded"):
                graph.invoke({"messages": [HumanMessage("读取证据")]}, config=config, context=context)
        state = graph.get_state(config).values
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(next(m for m in state["messages"] if isinstance(m, ToolMessage)).content, content)
        self.assertNotIn("/scratch/large_tool_results/failed-offload", state["files"])

    def test_core_memory_conflict_is_a_recoverable_tool_observation(self):
        def conflict(scope, change, *, operation_id):
            raise MemoryOperationError("版本已变更，请先读取当前版本", code="memory_version_conflict", retryable=True)

        self.services.mutate = conflict
        saver = InMemorySaver()
        graph, context, config, model = self.core_graph([
            call("memory_correct", {"target": "user", "record_id": "known", "expected_version": 1,
                                     "content": "corrected"}),
            AIMessage(content="当前版本已变化，需要重新核对"),
        ], saver=saver)
        result = graph.invoke({"messages": [HumanMessage("纠正记忆")]}, config=config, context=context)
        state = graph.get_state(config).values
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(result["final_answer"], "当前版本已变化，需要重新核对")
        message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
        self.assertEqual(message.status, "error")
        self.assertIn("memory_version_conflict", message.content)

    def test_core_resume_reads_file_from_completed_sibling_tool_pending_writes(self):
        from threading import Event

        for path in (TASK_NOTES_PATH, "/scratch/completed.md"):
            with self.subTest(path=path):
                content = (TASK_NOTES_TEMPLATE.replace("待记录。", "PENDING_WRITE_EVIDENCE", 1)
                           if path == TASK_NOTES_PATH else "PENDING_WRITE_EVIDENCE")
                sibling_saved = Event()

                class PendingWriteSaver(InMemorySaver):
                    def put_writes(self, config, pending, task_id, task_path=""):
                        super().put_writes(config, pending, task_id, task_path)
                        for channel, value in pending:
                            if channel == "messages" and isinstance(value, list) and any(
                                isinstance(message, ToolMessage) and message.tool_call_id == "completed-write"
                                for message in value
                            ):
                                sibling_saved.set()

                class CrashSecondTool(AgentMiddleware):
                    def wrap_tool_call(self, request, handler):
                        if request.tool_call["name"] == "read_file":
                            # The serial I/O gate can release before the official
                            # Send callback persists its result. This fixture
                            # specifically exercises a committed pending sibling.
                            if not sibling_saved.wait(timeout=5):
                                raise AssertionError("The sibling write did not reach the official pending-write checkpoint")
                            raise RuntimeError("process loss after sibling write")
                        return handler(request)

                batch = AIMessage(content="", tool_calls=[
                    {"name": "write_file", "args": {"file_path": path, "content": content},
                     "id": "completed-write", "type": "tool_call"},
                    {"name": "read_file", "args": {"file_path": path},
                     "id": "pending-read", "type": "tool_call"},
                ])
                saver = PendingWriteSaver()
                first_binding = MemoryBinding(self.scope, self.store, self.services)
                graph, context, config, _ = self.core_graph([batch], saver=saver, binding=first_binding,
                                                           middleware=[CrashSecondTool()])
                with self.assertRaisesRegex(RuntimeError, "process loss after sibling write"):
                    graph.invoke({"messages": [HumanMessage("保存并读取任务文件")]}, config=config, context=context)
                self.assertTrue(sibling_saved.is_set())
                state = graph.get_state(config).values
                self.assertEqual(file_data_to_string(state["files"][path]), content)
                new_binding = MemoryBinding(self.scope, self.store, self.services)
                self.assertTrue(restore_memory_binding(new_binding, state))
                resumed, resumed_context, _, model = self.core_graph([
                    AIMessage(content="已读取中断前保存的文件"),
                ], binding=new_binding, saver=saver)
                # The production runner hydrates both resources from get_state.
                resumed_context.tool_runtime.execution.restore(state["execution_ledger"])
                from agent_core.planning.react.recovery import prepare_react_resume
                resumed_config = prepare_react_resume(resumed, config, graph.get_state(config))
                result = resumed.invoke(None, config=resumed_config, context=resumed_context)
                self.assertEqual(result["final_answer"], "已读取中断前保存的文件")
                self.assertTrue(any(isinstance(m, ToolMessage) and m.tool_call_id == "pending-read"
                                    and "PENDING_WRITE_EVIDENCE" in m.content for m in model.calls[0][1]))
                completed = resumed.get_state(config).values
                self.assertEqual(completed["execution_ledger"]["tool_call_count"], 2)
                self.assertEqual(sum(isinstance(m, ToolMessage) and m.tool_call_id == "completed-write"
                                     for m in completed["messages"]), 1)

    def test_history_preserves_host_truncation_and_bounds_tool_metadata(self):
        entries = [HistoryEntry("source", "run", 0, "assistant", "short excerpt",
                                metadata={"truncated": True, "original_characters": 50_000,
                                          "tool_calls": [{"args": "x" * 50_000}]})]
        payload = history_payload(entries, max_characters=100)
        self.assertTrue(payload[0]["truncated"])
        self.assertTrue(payload[0]["metadataTruncated"])
        self.assertLessEqual(len(payload[0]["metadata"]["preview"]), 50)
        self.assertEqual(payload[0]["sourceId"], "source")

    def test_task_checkpoint_reader_materializes_official_file_deltas_at_exact_pointer(self):
        notes = TASK_NOTES_TEMPLATE.replace("待记录。", "EXACT_NOTE_VERSION", 1)
        saver = InMemorySaver()
        graph, context, config, model = self.core_graph([
            call("write_file", {"file_path": TASK_NOTES_PATH, "content": notes}),
            AIMessage(content="任务笔记已保存"),
        ], saver=saver)
        graph.invoke({"messages": [HumanMessage("保存笔记")]}, config=config, context=context)
        saved = graph.get_state(config)
        pointer = saved.config["configurable"]
        before_count = len(list(saver.list(config)))
        restored = read_task_checkpoint(saver, pointer)
        self.assertEqual(task_memory_snapshot(restored)["notesMarkdown"], notes)
        self.assertEqual(inherit_task_memory(restored)["files"][TASK_NOTES_PATH], saved.values["files"][TASK_NOTES_PATH])
        self.assertEqual(len(list(saver.list(config))), before_count)
        self.assertNotIn("messages", restored)

        changed = notes.replace("EXACT_NOTE_VERSION", "LATEST_NOTE_VERSION")
        graph.update_state(config, {"files": {TASK_NOTES_PATH: create_file_data(changed)},
                                    "todos": [{"content": "检查", "status": "completed"}],
                                    "task_source_ids": ["run:message-id"]})
        self.assertEqual(task_memory_snapshot(read_task_checkpoint(saver, pointer))["notesMarkdown"], notes)
        latest = task_memory_snapshot(read_task_checkpoint(saver, {"thread_id": self.scope.run_id}))
        self.assertEqual(latest["notesMarkdown"], changed)
        self.assertEqual(latest["todos"], [{"content": "检查", "status": "completed"}])
        self.assertEqual(latest["sourceIds"], ["run:message-id"])
        self.assertEqual(len(model.calls), 2)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            read_task_checkpoint(saver, {"thread_id": self.scope.run_id, "checkpoint_id": "missing"})
        self.assertEqual(read_task_checkpoint(saver, {"thread_id": "no-run"}), {})

    def test_conclusion_notes_require_a_reference_location_but_not_for_goal_only_edits(self):
        validate_task_notes(TASK_NOTES_TEMPLATE.replace("## 目标\n待记录。", "## 目标\n新的目标"))
        conclusion = TASK_NOTES_TEMPLATE.replace("## 关键结论\n待记录。", "## 关键结论\n已核对的结论")
        with self.assertRaisesRegex(ValueError, "关键结论需要证据位置"):
            validate_task_notes(conclusion)
        for reference in ("run-id:message-id", "memory:record-id:v2", "/scratch/evidence.md"):
            with self.subTest(reference=reference):
                validate_task_notes(conclusion.replace("## 证据位置\n待记录。", f"## 证据位置\n{reference}"))
        with self.assertRaisesRegex(ValueError, "关键结论需要证据位置"):
            validate_task_notes(conclusion.replace("## 证据位置\n待记录。", "## 证据位置\n/scratch/../../private"))

    def test_new_run_history_tool_recovers_original_large_result_before_offload(self):
        class OriginalHistory(SafeHistory):
            def __init__(self):
                self.entries = {}

            def append(self, scope, entries):
                for entry in entries:
                    previous = self.entries.get(entry.source_id)
                    if previous is not None and previous.content != entry.content:
                        raise AssertionError("an original history source was replaced by a preview")
                    self.entries[entry.source_id] = entry

            def read(self, scope, source_ids, *, max_characters=16_000, offset_characters=0):
                self.scope = scope
                self.read_offset = offset_characters
                return [replace(self.entries[source_id],
                        content=self.entries[source_id].content[offset_characters:offset_characters + max_characters],
                        metadata={"offset_characters": offset_characters,
                                  "original_characters": len(self.entries[source_id].content),
                                  "truncated": offset_characters > 0 or len(self.entries[source_id].content) > max_characters})
                        for source_id in source_ids if source_id in self.entries]

        history = OriginalHistory()
        content = "\n".join(f"ORIGINAL-EVIDENCE-{number:04d}" for number in range(2_000))
        executor = ToolExecutor(registry=ToolRegistry([ToolSpec(
            "large_evidence", "Return a safe evidence sample", EmptyInput, lambda _args, _runtime: content,
        )]))
        first_binding = MemoryBinding(self.scope, self.store, self.services, history=history)
        saver = InMemorySaver()
        graph, context, config, _ = self.core_graph([
            call("large_evidence", {}, "original-large"), AIMessage(content="原始证据已读取"),
        ], binding=first_binding, saver=saver,
            tools=build_langchain_tools(executor.registry.definitions, executor=executor))
        graph.invoke({"messages": [HumanMessage("保存原始证据")]}, config=config, context=context)
        source = next(entry for entry in history.entries.values()
                      if entry.role == "tool" and entry.call_id == "original-large")
        self.assertEqual(source.content, content)

        next_scope = replace(self.scope, run_id="next-run")
        next_binding = MemoryBinding(next_scope, self.store, self.services, history=history)
        next_saver = InMemorySaver()
        offset = content.index("ORIGINAL-EVIDENCE-1999")
        resumed, next_context, next_config, model = self.core_graph([
            call("read_file", {"file_path": "/scratch/large_tool_results/original-large", "limit": 50}, "expired-scratch"),
            call("history_read", {"source_ids": [source.source_id], "offset_characters": offset}, "history-recall"),
            AIMessage(content="已按来源回查原始证据"),
        ], binding=next_binding, saver=next_saver)
        resumed.invoke({"messages": [HumanMessage("查找上一轮证据尾部")]}, config=next_config, context=next_context)
        state = resumed.get_state(next_config).values
        # New Runs do not inherit old scratch files. The default bounded read
        # can still retrieve the original tail through the explicit offset.
        self.assertNotIn("/scratch/large_tool_results/original-large", state["files"])
        expired = next(m for m in state["messages"]
                       if isinstance(m, ToolMessage) and m.tool_call_id == "expired-scratch")
        self.assertEqual(expired.status, "error")
        self.assertEqual(json.loads(expired.content)["errorCode"], "file_not_found")
        recalled = next(m.content for m in model.calls[2][1]
                        if isinstance(m, ToolMessage) and m.tool_call_id == "history-recall")
        self.assertIn("ORIGINAL-EVIDENCE-1999", recalled)
        self.assertIn(source.source_id, recalled)
        self.assertEqual(history.read_offset, offset)
        self.assertEqual(history.scope, next_scope)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(next_context.tool_runtime.execution.calls_by_category, {"memory": 2})

    def test_history_page_metadata_tracks_local_clipping_and_last_page(self):
        entry = HistoryEntry("run:source", "run", 0, "tool", "tail-text",
                             metadata={"offset_characters": 91, "original_characters": 100})
        page = history_payload([entry], max_characters=6)[0]
        self.assertEqual(page["offset"], 91)
        self.assertEqual(page["nextOffset"], 91 + len(page["content"]))
        self.assertEqual(page["original_characters"], 100)
        self.assertTrue(page["truncated"])
        last = history_payload([entry], max_characters=1_000)[0]
        self.assertIsNone(last["nextOffset"])
        self.assertEqual(last["content"], "tail-text")

    def test_core_project_skill_overrides_same_user_name_and_excludes_drafts(self):
        class ProjectService(ScopedService):
            def snapshot(self, scope):
                user = super().snapshot(scope).user
                return MemorySnapshot(user, MemoryRelease("1",
                    (scope.tenant_id, scope.user_id, scope.project_id, "memory"),
                    (scope.tenant_id, scope.user_id, scope.project_id, "skills")))

        scope = replace(self.scope, project_id="project-a")
        services = ProjectService(self.store)
        snapshot = services.snapshot(scope)

        def skill(description, body):
            return f"---\nname: lookup\ndescription: {description}\n---\n{body}"

        StoreBackend(store=self.store, namespace=lambda _: snapshot.user.skills_namespace).write(
            "/lookup/SKILL.md", skill("USER_APPROVED_DESCRIPTION", "USER_APPROVED_BODY"))
        StoreBackend(store=self.store, namespace=lambda _: snapshot.project.skills_namespace).write(
            "/lookup/SKILL.md", skill("PROJECT_APPROVED_DESCRIPTION", "PROJECT_APPROVED_BODY"))
        StoreBackend(store=self.store, namespace=lambda _: ("candidate-release", scope.user_id, scope.project_id)).write(
            "/lookup/SKILL.md", skill("UNAPPROVED_DESCRIPTION", "UNAPPROVED_BODY"))
        binding = MemoryBinding(scope, self.store, services)
        saver = InMemorySaver()
        graph, context, config, model = self.core_graph([
            call("write_file", {"file_path": "/drafts/lookup/SKILL.md",
                               "content": skill("DRAFT_DESCRIPTION", "DRAFT_BODY")}, "write-draft"),
            call("read_file", {"file_path": "/skills/project/lookup/SKILL.md"}, "read-approved"),
            AIMessage(content="按已批准的项目方法继续"),
        ], binding=binding, saver=saver)
        graph.invoke({"messages": [HumanMessage("查找当前项目的复用方法")]}, config=config, context=context)
        for _, messages in model.calls:
            prompt = messages[0].text
            self.assertIn("PROJECT_APPROVED_DESCRIPTION", prompt)
            self.assertNotIn("USER_APPROVED_DESCRIPTION", prompt)
            self.assertNotIn("UNAPPROVED_DESCRIPTION", prompt)
            self.assertNotIn("DRAFT_DESCRIPTION", prompt)
        self.assertTrue(any(isinstance(message, ToolMessage) and message.tool_call_id == "read-approved"
                            and "PROJECT_APPROVED_BODY" in message.content for message in model.calls[2][1]))
        selected = [item for item in graph.get_state(config).values["skills_metadata"] if item["name"] == "lookup"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["description"], "PROJECT_APPROVED_DESCRIPTION")

    def test_history_offset_rejects_negative_and_ambiguous_multisource_reads(self):
        history = SafeHistory()
        binding = MemoryBinding(self.scope, self.store, self.services, history=history)
        _, state, _, _ = self.run_graph([
            call("history_read", {"source_ids": ["run:source"], "offset_characters": -1}, "negative-offset"),
            call("history_read", {"source_ids": ["run:a", "run:b"], "offset_characters": 4}, "mixed-offset"),
            AIMessage(content="需要单个来源与非负偏移"),
        ], binding=binding)
        observations = [message for message in state["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual([message.status for message in observations], ["error", "error"])
        self.assertTrue(all("memory_invalid_input" in message.content for message in observations))
        self.assertIsNone(history.scope)

    def test_core_revision_change_warns_about_visible_old_sources_without_rewriting_notes(self):
        class InvalidatingService(ScopedService):
            def snapshot(self, scope):
                value = super().snapshot(scope)
                revoked = ("memory:old-note:v1", "memory:old-history:v1", "memory:tracked:v1",
                           "memory:invisible:v1", "memory:prefix:v1") if self.revision == "2" else ()
                return replace(value, revoked_source_ids=revoked)

        services = InvalidatingService(self.store)
        services.publish(self.scope, "OLD_CURRENT_CONSTRAINT")
        notes = TASK_NOTES_TEMPLATE.replace("## 关键结论\n待记录。", "## 关键结论\nOLD_NOTE_CONCLUSION")
        notes = notes.replace("## 证据位置\n待记录。", "## 证据位置\nmemory:old-note:v1")

        class CorrectAfterTool(AgentMiddleware):
            def wrap_tool_call(self, request, handler):
                result = handler(request)
                services.revision = "2"
                services.publish(self_scope, "CORRECTED_CURRENT_CONSTRAINT")
                return result

        self_scope = self.scope
        binding = MemoryBinding(self.scope, self.store, services)
        saver = InMemorySaver()
        graph, context, config, model = self.core_graph([
            call("ls", {"path": "/scratch/"}), AIMessage(content="旧引用需要按当前版本核对"),
        ], binding=binding, saver=saver, middleware=[CorrectAfterTool()])
        graph.invoke({"messages": [HumanMessage("核对 memory:old-history:v1；memory:prefix:v10 仍是不同版本")],
                      "files": {TASK_NOTES_PATH: create_file_data(notes)},
                      "task_source_ids": ["memory:tracked:v1"]}, config=config, context=context)
        self.assertNotIn("已失效的记忆来源", model.calls[0][1][0].text)
        prompt = model.calls[1][1][0].text
        self.assertIn("已失效的记忆来源", prompt)
        self.assertIn("CORRECTED_CURRENT_CONSTRAINT", prompt)
        self.assertNotIn("OLD_CURRENT_CONSTRAINT", prompt)
        warning = prompt[prompt.index("已失效的记忆来源"):prompt.index("原 Notes/History 正文保留")]
        self.assertIn("不能继续作为现行约束", warning)
        for source_id in ("memory:old-note:v1", "memory:old-history:v1", "memory:tracked:v1"):
            self.assertIn(source_id, prompt)
        self.assertNotIn("memory:invisible:v1", prompt)
        self.assertNotIn('"memory:prefix:v1"', prompt)
        self.assertIn("OLD_NOTE_CONCLUSION", prompt)
        self.assertEqual(task_memory_snapshot(graph.get_state(config).values)["notesMarkdown"], notes)

    def test_core_invalidated_source_warning_overflow_stops_before_model_call(self):
        original_snapshot = self.services.snapshot
        ids = tuple(f"memory:old-{number}:v1" for number in range(65))
        self.services.snapshot = lambda scope: replace(original_snapshot(scope), revoked_source_ids=ids)
        graph, context, config, model = self.core_graph([AIMessage(content="must not run")])
        with self.assertRaisesRegex(AgentContextError, "invalidated memory sources exceed"):
            graph.invoke({"messages": [HumanMessage("核对旧来源")], "task_source_ids": list(ids)},
                         config=config, context=context)
        self.assertEqual(model.calls, [])

    def test_core_all_six_file_entry_points_reject_traversal_before_io(self):
        path = "/memory/project/../../other-user/AGENTS.md"
        arguments = {
            "ls": {"path": path}, "read_file": {"file_path": path},
            "write_file": {"file_path": path, "content": "override"},
            "edit_file": {"file_path": path, "old_string": "old", "new_string": "new"},
            "glob": {"path": path, "pattern": "**/*"},
            "grep": {"path": path, "pattern": "private"},
        }
        batch = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name, "type": "tool_call"}
                                                   for name, args in arguments.items()])
        saver = InMemorySaver()
        graph, context, config, _ = self.core_graph([batch, AIMessage(content="这些路径不在允许范围")], saver=saver)
        graph.invoke({"messages": [HumanMessage("验证文件范围")]}, config=config, context=context)
        observations = [message for message in graph.get_state(config).values["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(len(observations), 6)
        self.assertTrue(all(message.status == "error" and "memory_path_denied" in message.content for message in observations))

    def test_core_public_file_searches_remain_scoped_and_formal_write_edit_are_denied(self):
        class ProjectService(ScopedService):
            def snapshot(self, scope):
                return MemorySnapshot(super().snapshot(scope).user, MemoryRelease("1",
                    (scope.tenant_id, scope.user_id, scope.project_id, "memory"),
                    (scope.tenant_id, scope.user_id, scope.project_id, "skills")))

        scope = replace(self.scope, project_id="project-a")
        services = ProjectService(self.store)
        for target, content in ((scope, "CURRENT_PROJECT_BODY"),
                                (replace(scope, user_id="foreign-user"), "FOREIGN_USER_BODY"),
                                (replace(scope, project_id="foreign-project"), "FOREIGN_PROJECT_BODY")):
            namespace = services.snapshot(target).project.memory_namespace
            StoreBackend(store=self.store, namespace=lambda _, ns=namespace: ns).write("/AGENTS.md", content)
        inputs = (
            ("ls", {"path": "/memory/project/"}),
            ("read_file", {"file_path": "/memory/project/AGENTS.md"}),
            ("glob", {"path": "/", "pattern": "/memory/project/*.md"}),
            ("glob", {"path": "/memory/project/", "pattern": "../../**/*.md"}),
            ("grep", {"path": "/", "pattern": "BODY", "output_mode": "content"}),
            ("write_file", {"file_path": "/memory/project/AGENTS.md", "content": "OVERRIDE"}),
            ("edit_file", {"file_path": "/memory/project/AGENTS.md", "old_string": "CURRENT", "new_string": "OVERRIDE"}),
        )
        batch = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"file-{index}", "type": "tool_call"}
                                                   for index, (name, args) in enumerate(inputs)])
        binding = MemoryBinding(scope, self.store, services)
        saver = InMemorySaver()
        graph, context, config, model = self.core_graph([batch, AIMessage(content="只读取当前项目发布内容")], binding=binding, saver=saver)
        graph.invoke({"messages": [HumanMessage("检查当前作用域文件")]}, config=config, context=context)
        observations = [message for message in graph.get_state(config).values["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(len(observations), len(inputs))
        text = "\n".join(str(message.content) for message in observations)
        self.assertIn("CURRENT_PROJECT_BODY", text)
        self.assertNotIn("FOREIGN_USER_BODY", text)
        self.assertNotIn("FOREIGN_PROJECT_BODY", text)
        self.assertIn("/memory/project/AGENTS.md", observations[2].content)
        self.assertNotIn("AGENTS.md", observations[3].content)
        self.assertEqual(observations[3].status, "error")
        self.assertIn("invalid_tool_call", observations[3].content)
        self.assertTrue(all(message.status == "error" and "memory_read_only" in message.content for message in observations[-2:]))
        self.assertIn("CURRENT_PROJECT_BODY", model.calls[1][1][0].text)


if __name__ == "__main__":
    unittest.main()
