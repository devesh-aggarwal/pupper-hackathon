"""The tool registry: turns the @tool functions in lab/ into what Gemini reads,
and checks every call against it before running it (provided).

You don't change this file. In a lab/ file:

    @tool("my_tool", "What Gemini reads about the tool.",
          speed=Param("number", "What Gemini reads about this parameter.", required=False))
    async def my_tool_tool(robot, speed=0.4) -> Result:
        ...

The first word is the tool's name, the one Gemini calls; the Python function
can be called anything (in lab/ it is the name plus _tool, so it doesn't clash
with the behavior of the same name).

A description may contain {tricks} or {actions}: the installed tricks, or the
installed interactive actions, one per line. A parameter's enum (the values Gemini
may pass) may be TRICK_NAMES or ACTION_IDS for the same lists.

The web app asks the robot server for the declarations (GET /tools) when it
connects, and sends every call to POST /tools/<name>, which runs call() below.
"""

from __future__ import annotations

import importlib
import logging
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from robot_api import CALL_STOPS, Result, Robot

log = logging.getLogger("tools")


@dataclass
class Param:
    """One parameter of a tool, as Gemini sees it."""

    type: str  # "string", "number", "integer" or "boolean"
    description: str
    enum: list | Callable[[Robot], list] | None = None  # allowed values
    required: bool = True


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, Param]
    run: Callable[..., Awaitable[Result]]
    module: str = ""


TOOLS: dict[str, Tool] = {}

# The lab's files, in the order of the lab. Gemini gets the tools in this order.
LAB_DIR = Path(__file__).resolve().parent.parent / "lab"
LAB_MODULES = [
    "basics", "part2_walk", "part3_follow", "part4_tricks", "part5_interactive_action",
    # Hackathon additions (README, "Hackathon features"): memory, the screen and hearing,
    # reminders, and seeing things.
    "hackathon_memory", "hackathon_alerts", "hackathon_vision",
    "reminders",  # set_reminder, list_reminders, dismiss_reminder, cancel_reminder (robot/reminder_store.py keeps them)
    "hackathon_eyes",  # go_to, look, what_is_around (the eyes service, vision/eyes.py)
    "hackathon_explore",  # explore: roam to find a thing
    "hackathon_log",  # the session transcript and frames (no tools; the server records into it)
]
# Run after every tool call with (name, args, result); hackathon_vision's follow guard watches this way.
CALL_HOOKS: list[Callable[[str, dict, Result], None]] = []


def tool(tool_name: str, description: str, /, **params: Param):
    """Register the function below as the tool Gemini calls ``tool_name``. (The
    ``/`` lets a tool have a parameter called name, like play_trick.)"""

    def register(fn):
        TOOLS[tool_name] = Tool(tool_name, description, params, fn, fn.__module__)
        return fn

    return register


def load_lab() -> types.SimpleNamespace:
    """Import every lab/ file (which registers its tools), and return everything
    they define in one namespace: lab.walk, lab.check_move, lab.play_trick, ..."""
    if str(LAB_DIR) not in sys.path:
        sys.path.append(str(LAB_DIR))  # last, so a lab/ file never hides a provided module
    names: dict[str, Any] = {}
    for module in LAB_MODULES:
        m = importlib.import_module(module)
        names.update({k: v for k, v in vars(m).items() if not k.startswith("_")})
    return types.SimpleNamespace(**names)


def _in_lab_order(tools: list[Tool]) -> list[Tool]:
    def key(t: Tool) -> int:
        return LAB_MODULES.index(t.module) if t.module in LAB_MODULES else len(LAB_MODULES)

    return sorted(tools, key=key)  # stable: within a file, in the order they are written


def TRICK_NAMES(robot: Robot) -> list:
    """enum: the installed tricks."""
    return list(robot.tricks)


def ACTION_IDS(robot: Robot) -> list:
    """enum: the installed interactive actions."""
    return list(robot.action_ids())


def _trick_list(robot: Robot) -> str:
    return "\n".join(f'- "{name}": {t.description}' for name, t in robot.tricks.items())


def _action_list(robot: Robot) -> str:
    lines = [f'- "{aid}" (during "{t.name}"): {t.actions[clip]}' for aid, (t, clip) in robot.action_ids().items()]
    return "\n".join(lines) or "(none installed yet)"


def _describe(text: str, robot: Robot) -> str:
    if "{tricks}" in text:
        text = text.replace("{tricks}", _trick_list(robot))
    if "{actions}" in text:
        text = text.replace("{actions}", _action_list(robot))
    return text


def _resolve(value, robot: Robot):
    return value(robot) if callable(value) else value


def declarations(robot: Robot) -> list[dict]:
    """Every tool in the format Gemini expects."""
    out = []
    for t in _in_lab_order(list(TOOLS.values())):
        props, required = {}, []
        for name, p in t.params.items():
            prop: dict[str, Any] = {"type": p.type.upper(), "description": p.description}
            enum = _resolve(p.enum, robot)
            if enum:  # an empty list (nothing installed yet) is left out: Gemini rejects it
                prop["enum"] = [str(v) for v in enum]
            props[name] = prop
            if p.required:
                required.append(name)
        decl: dict[str, Any] = {"name": t.name, "description": _describe(t.description, robot)}
        if props:
            decl["parameters"] = {"type": "OBJECT", "properties": props, "required": required}
        out.append(decl)
    return out


_TYPES = {"string": (str,), "number": (int, float), "integer": (int,), "boolean": (bool,)}


async def call(robot: Robot, name: str, args: dict) -> Result:
    """Check a tool call against its declaration, then run it."""
    t = TOOLS.get(name)
    if t is None:
        return Result(False, f"unknown tool {name}; the tools are {sorted(TOOLS)}")
    if not isinstance(args, dict):
        return Result(False, f"{name}: arguments must be an object")
    args = dict(args)
    extra = sorted(set(args) - set(t.params))
    if extra:
        return Result(False, f"{name} has no parameter {extra}; it takes {sorted(t.params)}")
    for pname, p in t.params.items():
        if pname not in args:
            if p.required:
                return Result(False, f"{name} needs {pname}: {p.description}")
            continue
        v = args[pname]
        if isinstance(v, bool) and p.type != "boolean" or not isinstance(v, _TYPES[p.type]):
            return Result(False, f"{name}: {pname} must be a {p.type}, not {v!r}")
        if isinstance(v, float) and not math.isfinite(v):
            return Result(False, f"{name}: {pname} must be finite, not {v!r}")
        enum = _resolve(p.enum, robot)
        if enum is not None and not enum:
            return Result(False, f"{name}: nothing is installed for {pname} yet, so it can't be used")
        if enum is not None and v not in enum:
            return Result(False, f"{name}: {pname} must be one of {list(enum)}, not {v!r}")
    token = CALL_STOPS.set(robot.stops)  # a stop after this point cancels what this call would queue
    try:
        res = await t.run(robot, **args)
    except Exception as e:  # a bug in a tool must not crash the server
        log.exception("tool %s failed", name)
        res = Result(False, f"{name} failed on the robot: {type(e).__name__}: {e}")
    finally:
        CALL_STOPS.reset(token)
    for hook in CALL_HOOKS:
        try:
            hook(name, args, res)
        except Exception:  # a hook must not break the tool it watches
            log.exception("tool hook %s failed", getattr(hook, "__name__", hook))
    return res
