#!/usr/bin/env python3
"""Talk to Gemini in text, and watch every tool call it makes.

Gemini Live in about a hundred lines: no browser, no microphone. You type, Gemini
answers and calls tools, and this script runs each call and sends the reply
back, the same loop the web app runs when you talk to Pupper.

    python3 lab/chat.py            # YOUR tools, on a fake robot (a laptop is enough)
    python3 lab/chat.py --robot    # your tools on the real Pupper (it moves!)

(default)  Your lab/prompt.md, and your tools from lab/, exactly as the web app
           gets them, run on the fake robot from robot/test_tools.py. Time is fake, so
           a queued job finishes at once: you see the reply Gemini hears
           (accepted or refused), then what the job did (the outcome) and what
           the fake robot was told. Nothing moves; no Pupper needed.
--robot    The same, but every call goes to the robot server
           (http://localhost:8000, on the Pupper or through your ssh tunnel),
           the exact path a voice command takes. Pupper on the ground, your hand
           near the e-stop.

The Gemini key comes from GEMINI_API_KEY, or ~/.config/gemini-pupper/api_key
(robot/setup.sh writes it). Needs the google-genai package:
python3 -m pip install --user --break-system-packages google-genai

Based on Peng Xu's (Google DeepMind) Gemini Live robot-tool example. Thank you, Peng!
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY_FILE = Path.home() / ".config" / "gemini-pupper" / "api_key"
MODEL = "gemini-3.1-flash-live-preview"  # the web app's default (constants.ts)


# ============================================================ your tools, on the fake robot
def prompt_text(path: Path = HERE.parent / "lab" / "prompt.md") -> str:
    """prompt.md without its <!-- comments -->, as the robot server sends it."""
    return re.sub(r"<!--.*?-->", "", path.read_text(), flags=re.S).strip()


class DeskBackend:
    """Your lab/ tools and behaviors on the fake robot from test_tools.py."""

    name = "your tools on a fake robot (nothing moves)"

    def setup(self) -> tuple[str, list[dict]]:
        import test_tools  # installs a fake ROS before robot_api is imported
        import tricks
        import tool_registry

        import hackathon_alerts, hackathon_memory  # the hackathon's events and memory

        self.tt, self.registry = test_tools, tool_registry
        self.alerts = hackathon_alerts
        self.rig = test_tools.Rig(tricks=tricks.load_tricks)  # the real tricks/ folder
        self.seen_pub = self.seen_switch = self.seen_log = self.seen_follow = 0
        self.seen_event = hackathon_alerts.EVENTS.last_id
        self.what_happened()  # forget the start-up
        print(f"memory: {hackathon_memory.MEMORY_FILE}  (python3 lab/hackathon_memory.py show / reset)")
        return prompt_text() + "\n\n" + hackathon_memory.memory_prompt(), tool_registry.declarations(self.rig.ts)

    def events(self) -> list[dict]:
        """Events since the last turn: reminders that came due (nobody runs the server's
        reminder loop here, so fire them now), sound alerts, search results."""
        evs = self.alerts.EVENTS.since(self.seen_event)
        self.seen_event = self.alerts.EVENTS.last_id
        return evs

    async def call(self, name: str, args: dict) -> dict:
        res = await self.registry.call(self.rig.ts, name, args)
        await self.rig.settle()  # fake time: every queued job runs to the end now
        return res.as_dict()

    def after(self) -> None:
        """Print what the robot did: this happens later, after Gemini has its reply."""
        lines = self.what_happened()
        if lines:
            print("     ...later, when the queue gets to it:")
        for line in lines:
            print(f"     [fake robot] {line}")

    def what_happened(self) -> list[str]:
        robot = self.rig.robot
        lines = []
        for t, off, on in robot.switches[self.seen_switch:]:
            lines.append(f"legs: {' + '.join(on) or 'nobody (motors relaxed)'} took over" if on else
                         f"legs: relaxed {', '.join(off)}")
        twists = [(t, m) for t, topic, m in robot.published[self.seen_pub:] if topic == "/llm_cmd_vel"]
        moving = [(t, m) for t, m in twists if abs(m.linear.x) + abs(m.linear.y) + abs(m.angular.z) > 1e-9]
        if moving:
            speeds = sorted({(round(m.linear.x, 2), round(m.linear.y, 2), round(m.angular.z * 57.2958))
                             for _, m in moving})
            span = moving[-1][0] - moving[0][0]
            lines.append(f"speed: {len(moving)} commands over {span:.1f} s, (vx, vy, wz_deg) = "
                         + ", ".join(map(str, speeds)) + ("; then a zero (stand still)" if twists[-1][1].linear.x == 0
                                                         and twists[-1][1].angular.z == 0 else ""))
        for _, topic, m in robot.published[self.seen_pub:]:
            if topic.endswith("/clip"):
                lines.append(f"clip: {topic.split('/')[1]} plays {m.data}")
            elif topic.endswith("/reference_trick"):
                lines.append("reference: sent to the locomotion policy")
        for call in robot.follow_calls[self.seen_follow:]:
            lines.append(f"person follower: {call}")
        for rec in self.rig.records[self.seen_log:]:
            lines.append(f"outcome: {rec.getMessage()}")
        self.seen_pub, self.seen_switch = len(robot.published), len(robot.switches)
        self.seen_log, self.seen_follow = len(self.rig.records), len(robot.follow_calls)
        return lines


# ============================================================ your tools, on the real robot
class RobotBackend:
    """The robot server, exactly as the web app uses it: GET /tools, POST /tools/<name>."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.name = f"the real robot at {self.url} (Pupper moves!)"

    def _get(self, path: str):
        with urllib.request.urlopen(self.url + path, timeout=10) as r:
            return json.loads(r.read())

    def setup(self) -> tuple[str, list[dict]]:
        spec = self._get("/tools")
        self.seen_event = self._get("/events?after=-1")["last"]
        return spec["prompt"], spec["tools"]

    def events(self) -> list[dict]:
        """Events the robot server collected since the last turn (GET /events)."""
        res = self._get(f"/events?after={self.seen_event}")
        self.seen_event = res["last"]
        return res["events"]

    async def call(self, name: str, args: dict) -> dict:
        req = urllib.request.Request(f"{self.url}/tools/{name}", data=json.dumps(args).encode(),
                                     headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())


# ============================================================ the loop
def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        sys.exit(f"No Gemini key: set GEMINI_API_KEY, or put it in {KEY_FILE} (robot/setup.sh does).")
    return key


def show(prompt: str, tools: list[dict], full: bool) -> None:
    if full:
        print("=== What Gemini reads: the prompt ===\n" + prompt + "\n")
        print("=== ... and the tools ===\n" + json.dumps(tools, indent=2) + "\n")
    else:
        print("Gemini's tools: " + ", ".join(t["name"] for t in tools) + "   (--show prints everything Gemini reads)")


async def chat(backend, model: str, full: bool) -> None:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("Needs the google-genai package: python3 -m pip install --user --break-system-packages google-genai")

    prompt, tools = backend.setup()
    show(prompt, tools, full)
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],  # Live models answer in audio...
        output_audio_transcription=types.AudioTranscriptionConfig(),  # ...and we print what they say
        tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**t) for t in tools])],
        system_instruction=types.Content(parts=[types.Part(text=prompt)]),
    )
    client = genai.Client(api_key=api_key())
    async with client.aio.live.connect(model=model, config=config) as session:
        print(f"Connected to {model}, with {backend.name}. Type 'quit' to exit.\n")
        while True:
            user = (await asyncio.to_thread(input, "You > ")).strip()
            if user.lower() in ("quit", "exit"):
                break
            # Events (a reminder came due, a search finished): the web app hands them to
            # Gemini as a turn, so Pupper reacts on its own. Here they go in with your line.
            events = backend.events()
            for ev in events:
                print(f"  [event {ev['kind']}] {ev['text']}" + (f"  (screen: {ev['display']['text']})" if ev.get("display") else ""))
            text = "\n".join([ev["for_gemini"] for ev in events if ev.get("for_gemini")] + ([user] if user else []))
            if not text:
                continue
            await session.send_client_content(
                turns=[types.Content(role="user", parts=[types.Part(text=text)])], turn_complete=True
            )
            said = False
            async for msg in session.receive():
                content = msg.server_content
                if content and content.output_transcription and content.output_transcription.text:
                    if not said:
                        print("Pupper > ", end="")
                        said = True
                    print(content.output_transcription.text, end="", flush=True)
                if msg.tool_call:  # Gemini wants a tool: run it, and send back what happened
                    replies = []
                    for fc in msg.tool_call.function_calls:
                        args = dict(fc.args or {})
                        print(("\n" if said else "") + f"  -> Gemini calls {fc.name}({json.dumps(args)})")
                        said = False
                        try:
                            result = await backend.call(fc.name, args)
                        except Exception as e:  # e.g. the robot server is not running
                            result = {"success": False, "message": f"{type(e).__name__}: {e}"}
                        print(f"  <- reply Gemini hears now: {json.dumps(result)}")
                        if hasattr(backend, "after"):
                            backend.after()
                        replies.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"result": result}))
                    await session.send_tool_response(function_responses=replies)
                if content and content.turn_complete:
                    print("\n" if said else "")
                    break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--robot", nargs="?", const="http://localhost:8000", metavar="URL",
                      help="your tools on the real robot server (default http://localhost:8000)")
    ap.add_argument("--show", action="store_true", help="print the whole prompt and every tool declaration first")
    ap.add_argument("--model", default=MODEL, help=f"Gemini Live model (default {MODEL})")
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    backend = RobotBackend(args.robot) if args.robot else DeskBackend()
    try:
        asyncio.run(chat(backend, args.model, args.show))
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
