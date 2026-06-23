"""De-risk: verify BalatroBot save/load endpoints work over the API.

Connects to one game, saves the current run to a .jkr, then loads it back.
If both return success, state-injection curriculum (plan A step 2) is feasible.
Run with the trainer OFF the port (so RPCs don't cross).
"""
import asyncio
import os
from environment.game_state import GameStateManager

SEED_DIR = "C:/Users/jarms/repos/balatron/seeds"
PATH = SEED_DIR + "/derisk_test.jkr"


async def main():
    os.makedirs(SEED_DIR, exist_ok=True)
    g = GameStateManager(port=12346)
    await g.connect()
    try:
        st = await g.fetch_gamestate()
        print(f"[derisk] connected; game state = {st.get('state')!r} ante={st.get('ante_num')}")

        print("[derisk] calling save...")
        r1 = await g.execute_action("save", {"path": PATH})
        print(f"[derisk] save -> {r1}")
        exists = os.path.exists(PATH)
        print(f"[derisk] save file exists: {exists} ({os.path.getsize(PATH) if exists else 0} bytes)")

        print("[derisk] calling load...")
        r2 = await g.execute_action("load", {"path": PATH})
        print(f"[derisk] load -> {r2}")

        st2 = await g.fetch_gamestate()
        print(f"[derisk] post-load state = {st2.get('state')!r} ante={st2.get('ante_num')}")

        ok = bool(r1 and r1.get("success")) and bool(r2 and r2.get("success"))
        print(f"[derisk] VERDICT: save/load {'WORK' if ok else 'FAILED'} -> curriculum {'FEASIBLE' if ok else 'BLOCKED'}")
    finally:
        await g.disconnect()


asyncio.run(main())
