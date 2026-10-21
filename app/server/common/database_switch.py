from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")

###############################################################################
class DatabaseRequestGate:
    """Drains API requests before a connection facade is changed."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._active_requests = 0
        self._switching = False

    # -------------------------------------------------------------------------
    async def enter(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: not self._switching)
            self._active_requests += 1

    # -------------------------------------------------------------------------
    async def leave(self) -> None:
        async with self._condition:
            self._active_requests = max(0, self._active_requests - 1)
            self._condition.notify_all()

    # -------------------------------------------------------------------------
    async def pause_and_drain(self) -> None:
        async with self._condition:
            if self._switching:
                raise RuntimeError("A database switch is already in progress.")
            self._switching = True
            try:
                await self._condition.wait_for(lambda: self._active_requests == 0)
            except BaseException:
                self._switching = False
                self._condition.notify_all()
                raise

    # -------------------------------------------------------------------------
    async def resume(self) -> None:
        async with self._condition:
            self._switching = False
            self._condition.notify_all()

    # -------------------------------------------------------------------------
    async def run_request(
        self,
        request_handler: Callable[[], Awaitable[T]],
    ) -> T:
        await self.enter()
        try:
            return await request_handler()
        finally:
            await self.leave()


__all__ = ["DatabaseRequestGate"]
