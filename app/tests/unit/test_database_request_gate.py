from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from server.common.database_switch import DatabaseRequestGate

###############################################################################
def test_database_switch_gate_drains_active_requests_and_holds_new_requests() -> None:
    async def scenario() -> None:
        gate = DatabaseRequestGate()
        await gate.enter()

        draining = asyncio.create_task(gate.pause_and_drain())
        await asyncio.sleep(0)
        waiting_request = asyncio.create_task(gate.enter())
        await asyncio.sleep(0)

        assert not draining.done()
        assert not waiting_request.done()

        await gate.leave()
        await draining
        assert not waiting_request.done()

        await gate.resume()
        await waiting_request
        await gate.leave()

    # Playwright's sync API in earlier tests can leave a loop running on this
    # thread, so the scenario runs on a clean thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(asyncio.run, scenario()).result()
