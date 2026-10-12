"""Order governed tool calls without restricting the graph's event worker."""

from collections.abc import Callable, Sequence
import asyncio
from agent_core.runtime.streaming import call_async, run_sync
from threading import Condition
from typing import Any

from agent_core.runtime.cancellation import CancellationToken


class SerialToolCalls:
    """A per-Run tool hook gate; scheduling still belongs to create_agent.

    The executor allocates evidence in commit order and a failed call must
    prevent later calls from reaching I/O. Waiters also observe cancellation.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._pending: list[str] = []
        self._error: BaseException | None = None
        self._batch: tuple[str, ...] = ()

    def begin(self, call_ids: Sequence[str]) -> None:
        with self._condition:
            self._pending = list(call_ids)
            self._error = None
            self._batch = tuple(call_ids)

    def prepare(self, call_ids: Sequence[str], completed: Sequence[str]) -> None:
        """Hydrate a fresh gate and skip official middleware's early rejections."""
        with self._condition:
            batch = tuple(call_ids)
            if self._batch != batch:
                self._batch = batch
                self._pending = list(batch)
                self._error = None
            self._pending = [item for item in self._pending if item not in completed]
            self._condition.notify_all()

    async def aexecute(self, call_id: str, cancellation: CancellationToken, operation: Callable[[], Any]) -> Any:
        while True:
            with self._condition:
                if call_id not in self._pending:
                    raise RuntimeError("tool call was not accepted by the Run")
                if self._error is not None:
                    raise self._error
                ready = self._pending[0] == call_id
            cancellation.raise_if_cancelled()
            if ready:
                break
            await asyncio.sleep(0.01)
        try:
            return await call_async(operation)
        except BaseException as error:
            with self._condition:
                self._error = error
            raise
        finally:
            with self._condition:
                if call_id in self._pending:
                    self._pending.remove(call_id)
                self._condition.notify_all()

    def execute(self, call_id, cancellation, operation):
        return run_sync(self.aexecute(call_id, cancellation, operation))
