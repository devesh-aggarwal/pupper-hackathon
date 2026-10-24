"""Hackathon, part E: learning on the fly. "Learn a new trick called happy dance: ..."

The lab's tricks are trained policies: a new one takes a GPU and an afternoon.
On the fly, Pupper learns *routines*: a named sequence of things it can already
do. Teaching is by doing: learn_routine starts recording, every motion tool
Gemini calls after that is done for real (so the person sees it) and written
down, finish_learning saves it under the name, and do_routine replays it, each
step going through the same tool and the same checks as the first time. Routines
live in the memory file (hackathon_memory.py) and survive a restart.

How the recording works: tool_registry.call runs CALL_HOOKS after every tool;
record() below is one of them. Only motion tools are recorded, and only the ones
that were accepted: a refused move is not part of the trick.
"""

from __future__ import annotations

import re
import time

import hackathon_memory as memory
import tool_registry
from robot_api import Result, Robot
from tool_registry import Param, tool

RECORDABLE = {"move", "turn", "arc", "play_trick", "interactive_action", "stand_up", "lie_down"}
MAX_STEPS = 12
LEARNING: dict | None = None  # {"name", "steps": [{"tool", "args"}], "started"} while teaching


# ============================================================ what the robot does
def routine_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")[:40]


def record(name: str, args: dict, res: Result) -> None:
    """CALL_HOOK: write down an accepted motion while a routine is being taught."""
    if LEARNING is not None and name in RECORDABLE and res.ok and len(LEARNING["steps"]) < MAX_STEPS:
        LEARNING["steps"].append({"tool": name, "args": dict(args)})


tool_registry.CALL_HOOKS.append(record)


def start_learning(name: str) -> Result:
    global LEARNING
    rname = routine_name(name)
    if not rname:
        return Result(False, "the routine needs a name, e.g. 'happy dance'")
    if LEARNING is not None:
        return Result(False, f"already learning '{LEARNING['name']}' ({len(LEARNING['steps'])} steps so far);"
                             " call finish_learning first (save=false to throw it away)")
    if rname in tool_registry.TOOLS:
        return Result(False, f"'{rname}' is a tool's name; pick another")
    LEARNING = {"name": rname, "steps": [], "started": time.time()}
    known = memory.load()["routines"]
    replacing = " (replacing the old one)" if rname in known else ""
    return Result(True, f"learning '{rname}'{replacing}: do the moves now, one tool call each (move, turn, arc,"
                        f" play_trick, interactive_action, stand_up, lie_down), up to {MAX_STEPS}; then call"
                        " finish_learning")


def finish_learning(save: bool = True) -> Result:
    global LEARNING
    if LEARNING is None:
        return Result(False, "not learning anything; call learn_routine first")
    name, steps = LEARNING["name"], LEARNING["steps"]
    LEARNING = None
    if not save:
        return Result(True, f"threw away '{name}' ({len(steps)} steps not saved)")
    if not steps:
        return Result(False, f"nothing recorded for '{name}': no motion tool was accepted while learning; try again"
                             " and do the moves after learn_routine")
    data = memory.load()
    data["routines"][name] = {"steps": steps, "taught_by": data["active"], "time": time.time()}
    memory.save(data)
    return Result(True, f"learned '{name}': {len(steps)} steps ({summary(steps)}). Say do_routine to play it")


def summary(steps: list[dict]) -> str:
    def one(s: dict) -> str:
        args = ", ".join(f"{k}={v}" for k, v in s["args"].items())
        return f"{s['tool']}({args})" if args else s["tool"]

    return " then ".join(one(s) for s in steps)


def list_routines() -> Result:
    routines = memory.load()["routines"]
    if not routines:
        return Result(True, "no routines learned yet")
    return Result(True, "routines: " + "; ".join(f"{n} ({len(r['steps'])} steps: {summary(r['steps'])})" for n, r in routines.items()))


def forget_routine(name: str) -> Result:
    rname = routine_name(name)
    data = memory.load()
    if rname not in data["routines"]:
        return Result(False, f"unknown routine '{rname}'; known: {list(data['routines'])}")
    del data["routines"][rname]
    memory.save(data)
    return Result(True, f"forgot routine '{rname}'")


async def do_routine(robot: Robot, name: str) -> Result:
    """Replay a learned routine: each step is the same tool call it was taught with, with the same checks."""
    rname = routine_name(name)
    routines = memory.load()["routines"]
    if rname not in routines:
        return Result(False, f"unknown routine '{rname}'; known: {list(routines) or 'none yet'}")
    steps = routines[rname]["steps"]
    done = []
    for i, s in enumerate(steps, 1):
        res = await tool_registry.call(robot, s["tool"], s["args"])
        if not res.ok:
            return Result(False, f"{rname} stopped at step {i} ({s['tool']}): {res.message}"
                                 + (f". Steps 1-{i - 1} are queued" if i > 1 else ""))
        done.append(res.message)
    return Result(True, f"doing {rname}: {len(steps)} steps queued ({summary(steps)}). {done[-1]}")


# ============================================================ what Gemini reads
@tool(
    "learn_routine",
    "Start teaching Pupper a new trick made of things it can already do (a routine). After this,"
    " every motion you call (move, turn, arc, play_trick, interactive_action, stand_up, lie_down)"
    " is done AND recorded as a step, until finish_learning. Example: 'learn a happy dance: a push-up,"
    " spin around, then sneeze' -> learn_routine(name='happy dance'), play_trick(push_up), turn(left,"
    " 360), play_trick(sneeze), finish_learning(). Ask for the steps if the person didn't say them.",
    name=Param("string", "The new routine's name, e.g. 'happy dance'."),
)
async def learn_routine_tool(robot: Robot, name: str) -> Result:
    return start_learning(name)


@tool(
    "finish_learning",
    "Stop recording and save the routine being taught (or throw it away with save=false). Then it"
    " can be played with do_routine, and recall lists it.",
    save=Param("boolean", "true to save it (default), false to discard it.", required=False),
)
async def finish_learning_tool(robot: Robot, save: bool = True) -> Result:
    return finish_learning(save)


@tool(
    "do_routine",
    "Play a routine Pupper learned (recall or list_routines names them). Each step runs through its"
    " normal tool and checks, in order. If follow mode is on, this ends it: say so first.",
    name=Param("string", "The routine's name, as learned."),
)
async def do_routine_tool(robot: Robot, name: str) -> Result:
    return await do_routine(robot, name)


@tool("list_routines", "List the routines Pupper has learned, with their steps.")
async def list_routines_tool(robot: Robot) -> Result:
    return list_routines()


@tool("forget_routine", "Forget a learned routine by name.", name=Param("string", "The routine's name."))
async def forget_routine_tool(robot: Robot, name: str) -> Result:
    return forget_routine(name)
