"""One reusable async loop for task-owned provider streams."""
from __future__ import annotations

import asyncio
import inspect
from functools import wraps
from concurrent.futures import CancelledError as ConcurrentCancelledError
from threading import Event, Lock, Thread
from typing import Any

from agent_core.contracts.errors import AgentCancelled


async def call_async(operation: Any, *args: Any, **kwargs: Any) -> Any:
    """Await native async ports; isolate legacy blocking ports in a worker.

    This is a compatibility boundary for individual host operations, never a
    wrapper around an agent, graph or async tool. Context variables propagate
    into the worker. A blocking driver still needs its own bounded timeout.
    """
    operation = getattr(operation, "__async__", operation)
    if inspect.iscoroutinefunction(operation):
        return await operation(*args, **kwargs)
    result = await asyncio.to_thread(operation, *args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def run_sync(coroutine: Any) -> Any:
    """Synchronous facade for an async operation; never nest event loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        close = getattr(coroutine, "close", None)
        if close:
            close()
        raise RuntimeError("Use the async API from an active event loop")
    runner = AsyncStreamRunner()
    try:
        return runner.run(coroutine)
    finally:
        runner.close()


class AsyncStreamRunner:
    """Reuse one event loop for every async model stream in a run.

    ``ChatOpenAI`` owns an async HTTP connection pool. Recreating a loop for
    every planning round can strand that pool on a closed loop, so a small
    dedicated loop thread lives for exactly one run.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = Event()
        self._start_lock = Lock()
        self._started = False
        self._closed = False
        self._active_futures: set[Any] = set()
        self._thread = Thread(
            target=self._run_loop,
            name="agent-model-stream",
            daemon=True,
        )

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.set_task_factory(self._tracked_task_factory)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    @staticmethod
    def _tracked_task_factory(
        loop: asyncio.AbstractEventLoop,
        coroutine: Any,
        context: Any = None,
    ) -> asyncio.Task[Any]:
        """Keep cancelled cross-thread streams from becoming orphan Tasks.

        ``run_coroutine_threadsafe`` may have its concurrent Future cancelled
        before the loop Task translates provider cancellation into the public
        ``AgentCancelled`` exception. Retrieving the Task result here keeps
        that expected terminal path from being reported later as an unhandled
        background exception; callers still receive the same result through
        the concurrent Future during normal execution.
        """
        task = asyncio.Task(coroutine, loop=loop, context=context)
        task.add_done_callback(AsyncStreamRunner._consume_task_result)
        return task

    @staticmethod
    def _consume_task_result(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except BaseException:
            # Retrieval is cleanup only; the awaiting Future owns propagation.
            pass

    @staticmethod
    async def _cancel_loop_tasks() -> None:
        current = asyncio.current_task()
        tasks = [
            task
            for task in asyncio.all_tasks()
            if task is not current and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=0.75)
            for task in done:
                if not task.cancelled():
                    try:
                        task.exception()
                    except Exception:
                        pass
            for task in pending:
                task.cancel()

    def run(self, coroutine: Any) -> Any:
        with self._start_lock:
            if self._closed:
                raise AgentCancelled("Agent任务已取消")
            if not self._started:
                self._thread.start()
                self._started = True
        self._ready.wait()
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        with self._start_lock:
            if self._closed:
                future.cancel()
            else:
                self._active_futures.add(future)
        try:
            return future.result()
        except ConcurrentCancelledError as error:
            raise AgentCancelled("Agent任务已取消") from error
        finally:
            with self._start_lock:
                self._active_futures.discard(future)

    def close(self) -> None:
        with self._start_lock:
            if self._closed:
                return
            self._closed = True
            started = self._started
            active_futures = tuple(self._active_futures)
        if not started:
            self._loop.close()
            return
        for future in active_futures:
            future.cancel()
        for future in active_futures:
            try:
                future.result(timeout=1.0)
            except Exception:
                pass
        try:
            drain = asyncio.run_coroutine_threadsafe(
                self._cancel_loop_tasks(),
                self._loop,
            )
            drain.result(timeout=1.0)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)


async def call_port(owner: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """Prefer an async host port while retaining explicit legacy sync support."""
    operation = getattr(owner, "a" + name, None) or getattr(owner, name)
    return await call_async(operation, *args, **kwargs)


def sync_facade(operation: Any) -> Any:
    @wraps(operation)
    def invoke(*args: Any, **kwargs: Any) -> Any:
        return run_sync(operation(*args, **kwargs))
    invoke.__async__ = operation
    return invoke


def iter_sync(events):
    """Bridge one async producer task to a synchronous, backpressured iterator."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Use the async API from an active event loop")
    runner = AsyncStreamRunner()
    closing = False

    async def start():
        queue = asyncio.Queue(maxsize=1)
        async def produce():
            try:
                async for event in events:
                    await queue.put((False, event))
            except BaseException as error:
                if not closing:
                    await queue.put((True, error))
            else:
                await queue.put((True, None))
            finally:
                await events.aclose()
        return queue, asyncio.create_task(produce())

    queue, producer = runner.run(start())
    try:
        while True:
            done, value = runner.run(queue.get())
            if done:
                if value is not None:
                    raise value
                return
            yield value
    finally:
        closing = True
        async def stop():
            if not producer.done():
                producer.cancel()
            try:
                await producer
            except (Exception, asyncio.CancelledError):
                pass
        try:
            runner.run(stop())
        finally:
            runner.close()
