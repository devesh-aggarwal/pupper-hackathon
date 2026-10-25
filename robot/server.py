"""Gemini Pupper robot server: HTTP + WebSocket API on port 8000, and the web app.

Run on the Pupper after the robot stack is up (see start.sh):
    python3 robot/server.py

The web app gets Gemini's instructions and tools from GET /tools (lab/prompt.md
and the @tool functions in lab/) and sends every tool call to POST /tools/<name>. Every call
returns {"success": bool, "message": str}, which goes straight back to Gemini,
so Gemini hears about refused speeds, unknown tricks and failed controller
switches. The other endpoints are for the debug panel and for testing by hand.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import reminder_store
import tool_registry as gemini_tools
from robot_api import LIMITS, Robot

behaviors = gemini_tools.load_lab()  # lab/: every tool, and the behaviors they call

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("server")

REPO = Path(__file__).resolve().parent.parent
WEB_DIST = REPO / "web" / "dist"
PROMPT_FILE = REPO / "lab" / "prompt.md"
BATTERY_SCRIPT = Path("/home/pi/pupperv3-monorepo/robot/utils/check_batt_voltage.py")

tools: Robot | None = None


class Detections:
    """Forwards person detections (ZMQ on localhost:5556) to browser WebSockets.

    The follower publishes them while it follows (it keeps the Hailo off
    otherwise); the monorepo detector, when used instead, publishes them always.
    """

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def run(self) -> None:
        try:
            import zmq
            import zmq.asyncio
        except ImportError:
            log.warning("pyzmq missing; eye tracking from robot detections disabled")
            return
        sock = zmq.asyncio.Context().socket(zmq.SUB)
        sock.connect("tcp://127.0.0.1:5556")
        sock.subscribe(b"")
        while True:
            payload = (await sock.recv_multipart())[-1].decode("utf-8", errors="ignore")
            if payload.startswith("{"):
                for ws in list(self.clients):
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        self.clients.discard(ws)


detections = Detections()

# Reminders are told by the web app (through Gemini) within seconds of being due: it
# asks every 5 s. One still unclaimed this long after it was due means no web app is
# open, so the server says it through the robot's speaker itself.
REMINDER_FALLBACK_AFTER_S = 90.0
rlog = logging.getLogger("reminders")


async def speak_due_reminders() -> None:
    """The fallback: say overdue reminders with a local text-to-speech program."""
    speaker = next((p for p in ("espeak-ng", "espeak", "spd-say") if shutil.which(p)), None)
    if speaker is None:
        rlog.warning("no espeak-ng/espeak/spd-say: if no web app is open, reminders wait until one is")
    warned: set[int] = set()
    while True:
        await asyncio.sleep(10)
        try:
            store = reminder_store.get_store()
            if speaker is None:  # nothing to say it with: leave it due for the web app, and say so once
                for r in store.list():
                    late = (store.now() - r.next_dt).total_seconds()
                    if late >= REMINDER_FALLBACK_AFTER_S and r.id not in warned:
                        warned.add(r.id)
                        rlog.warning('#%d "%s" is overdue and no web app took it; it waits for one', r.id, r.message)
                continue
            for item in store.claim(min_late_s=REMINDER_FALLBACK_AFTER_S):
                rlog.info('no web app took #%d; saying it with %s', item["id"], speaker)
                cmd = [speaker, item["speech"]] if speaker == "spd-say" else [speaker, "-s", "140", item["speech"]]
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if await proc.wait() == 0:
                    store.delivered(item["id"], how=speaker)
                # else: not confirmed, so it is due again after the claim runs out
        except Exception:  # the fallback must never take the server down
            rlog.exception("reminder fallback failed")


def make_hearing(robot: Robot):
    """The calibrated loud-noise detector (hearing/listener.py). It starts off, and only the
    sound_alerts tool turns it on (opt in); what it hears goes to lab/hackathon_alerts.py."""
    try:
        sys.path.insert(0, str(REPO / "hearing"))
        from listener import Hearing
    except Exception as e:  # numpy/scipy/sounddevice missing: the browser's measurement is used
        log.warning("calibrated hearing unavailable (%s); sound alerts use the browser's measurement", e)
        return None
    return Hearing(on_event=lambda ev: behaviors.hearing_detected(robot, ev))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tools
    tools = Robot()
    tools.hearing = make_hearing(tools)
    reminder_store.get_store()  # load the saved reminders now (and log how many)
    tasks = [asyncio.create_task(detections.run()),
             asyncio.create_task(speak_due_reminders()),  # reminders: robot/reminder_store.py, lab/reminders.py
             asyncio.create_task(behaviors.follow_guard_loop(tools))]  # hackathon: pauses follow mode for things in the way
    behaviors.record("server", text="robot server started")
    yield
    for task in tasks:
        task.cancel()
    if tools.hearing is not None and tools.hearing.enabled:
        await tools.hearing.stop()


app = FastAPI(title="Gemini Pupper robot server", lifespan=lifespan)
# The web app is served from this same origin; also allow a dev server on the laptop.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


def T() -> Robot:
    if tools is None:
        raise HTTPException(503, "tool server not ready")
    return tools


# ------------------------------------------------------------------ models
class MoveVelocity(BaseModel):
    forward_backward_velocity: float = Field(..., description="m/s, positive = forward")
    right_left_velocity: float = Field(..., description="m/s, positive = right")
    turning_velocity: float = Field(..., description="deg/s, positive = turn right")
    duration: float = Field(..., description="seconds")


class MoveDirection(BaseModel):
    heading: float = Field(..., description="degrees, 0 = forward, 90 = right")
    speed: float
    duration: float


class Trick(BaseModel):
    animation_name: str


class TrickAction(BaseModel):
    action: str


class Wait(BaseModel):
    duration: float


class Volume(BaseModel):
    volume: int = Field(..., ge=0, le=150)


# ------------------------------------------------------------------ Gemini's tools
def system_prompt() -> str:
    """lab/prompt.md without its <!-- comments -->."""
    text = PROMPT_FILE.read_text() if PROMPT_FILE.exists() else ""
    return re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()


@app.get("/tools")
def tool_list():
    """What Gemini is told: the system prompt and the tool declarations."""
    # Hackathon: what Pupper remembers about the person is part of the prompt (lab/hackathon_memory.py).
    return {"prompt": system_prompt() + "\n\n" + behaviors.memory_prompt(), "tools": gemini_tools.declarations(T())}


@app.post("/tools/{name}")
async def tool_call(name: str, args: dict = Body(default={})):
    res = await gemini_tools.call(T(), name, args)
    log.info("tool %s(%s) -> %s: %s", name, args, "ok" if res.ok else "FAILED", res.message)
    behaviors.record("tool", name=name, args=args, ok=res.ok, message=res.message, frame=behaviors.save_frame(latest_jpeg(), name))
    return res.as_dict()


# ------------------------------------------------------------------ commands
@app.post("/walking/activate")
async def walking_activate():
    return (await behaviors.stand_up(T())).as_dict()


@app.post("/walking/deactivate")
async def walking_deactivate():
    return (await behaviors.lie_down(T())).as_dict()


@app.post("/move/velocity")
async def move_velocity(req: MoveVelocity):
    # The browser uses right/clockwise-positive; ROS uses left/counter-clockwise-positive.
    res = await behaviors.walk(T(), req.forward_backward_velocity, -req.right_left_velocity, -req.turning_velocity, req.duration)
    return res.as_dict()


@app.post("/move/direction")
async def move_direction(req: MoveDirection):
    turn = await behaviors.walk(T(), 0.0, 0.0, -90.0 if req.heading > 0 else 90.0, abs(req.heading) / 90.0) if req.heading else None
    if turn and not turn.ok:
        return turn.as_dict()
    return (await behaviors.walk(T(), req.speed, 0.0, 0.0, req.duration)).as_dict()


@app.post("/animation")
async def animation(req: Trick):
    return (await behaviors.play_trick(T(), req.animation_name)).as_dict()


@app.post("/interactive_action")
async def interactive_action(req: TrickAction):
    return (await behaviors.interactive_action(T(), req.action)).as_dict()


@app.get("/interactive_actions")
def interactive_actions():
    """Actions of interactive tricks (e.g. shake a paw while sitting): id -> trick and description."""
    return {aid: {"trick": t.name, "description": t.actions[clip]} for aid, (t, clip) in T().action_ids().items()}


@app.get("/animations")
def animations():
    return available_tricks()


def available_tricks() -> dict:
    """The tricks in <repo>/tricks, every one played by a policy."""
    return {
        name: {"description": t.description, "duration_s": round(t.total_time, 1), "kind": t.kind}
        for name, t in T().tricks.items()
    }


@app.post("/queue/wait")
async def queue_wait(req: Wait):
    return (await T().wait(req.duration)).as_dict()


@app.post("/queue/reset")
async def queue_reset():
    return (await T().clear_queue()).as_dict()


@app.post("/queue/stop")
@app.post("/stop/immediate")
async def stop_immediate():
    return (await T().immediate_stop(relax_playing_trick=True)).as_dict()


@app.get("/queue/status")
def queue_status():
    return T().status()


@app.post("/following/activate")
async def following_on():
    return (await T().set_following(True)).as_dict()


@app.post("/following/deactivate")
async def following_off():
    return (await T().set_following(False)).as_dict()


@app.get("/system/mode")
async def system_mode():
    return (await T().mode()).as_dict()


@app.get("/limits")
def limits():
    return LIMITS.__dict__


# ------------------------------------------------------------------ reminders
# Gemini sets them with the set_reminder tool (lab/reminders.py). The web app asks for
# due ones every few seconds, has Gemini say each, then confirms (robot/reminder_store.py).
@app.get("/reminders")
async def reminders_list():
    store = reminder_store.get_store()
    now = store.now()
    return {"now": now.isoformat(), "reminders": [
        {**r.__dict__, "text": reminder_store.describe(r, now)} for r in store.list()]}


@app.post("/reminders/claim")
async def reminders_claim(min_late_s: float = 0.0):
    """Due reminders for the caller to tell the person (hidden from others for a minute), and
    the ids to take off the screen (dismissed by voice, dismiss_reminder, or another screen)."""
    store = reminder_store.get_store()
    due = store.claim(min_late_s=min_late_s)
    for item in due:
        rlog.info('#%d "%s" is due: handed to the web app', item["id"], item["message"])
    return {"due": due, "dismissed": store.dismissed_ids()}


@app.post("/reminders/dismiss")
async def reminders_dismiss(ids: list[int] = Body(default=[], embed=True)):
    """A tap on the screen dismissed these, so dismiss_reminder knows they're gone."""
    store = reminder_store.get_store()
    for rid in ids:
        store.dismiss(rid)
    return {"success": True, "message": f"dismissed {ids}"}


@app.post("/reminders/{rid}/delivered")
async def reminder_delivered(rid: int, how: str = Body("", embed=True)):
    r = reminder_store.get_store().delivered(rid, how=how)
    return {"success": r is not None, "message": "delivered" if r else f"no reminder #{rid} (cancelled?)"}


# ------------------------------------------------------------------ sensing
def latest_jpeg() -> bytes | None:
    """The newest camera frame, left in place for the next reader."""
    if tools is None:
        return None
    try:
        msg = tools.latest_image.get_nowait()
    except queue.Empty:
        return None
    tools.latest_image.put_nowait(msg)
    return bytes(msg.data)


@app.get("/camera/image")
def camera_image():
    try:
        msg = T().latest_image.get_nowait()
    except Exception:
        return Response(status_code=503, content="no camera frame yet")
    T().latest_image.put_nowait(msg)  # keep it for the next poll
    return Response(content=bytes(msg.data), media_type="image/jpeg")


@app.websocket("/ws/detections")
async def ws_detections(ws: WebSocket):
    await ws.accept()
    detections.clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        detections.clients.discard(ws)


@app.get("/system/status")
def system_status():
    pct = volts = None
    if BATTERY_SCRIPT.exists():
        try:
            out = subprocess.run(["python3", str(BATTERY_SCRIPT)], capture_output=True, text=True, timeout=2).stdout
            m = re.search(r"(\d+(?:\.\d+)?)V", out)
            volts = float(m.group(1)) if m else None
            out = subprocess.run(["python3", str(BATTERY_SCRIPT), "--percentage_only"], capture_output=True, text=True, timeout=2).stdout
            pct = float(out.strip())
        except Exception:
            pass
    temps = psutil.sensors_temperatures()
    cpu_temp = temps["cpu_thermal"][0].current if "cpu_thermal" in temps else None
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=1).close()
        internet = "online"
    except OSError:
        internet = "offline"
    return {
        "battery": {"percentage": pct, "is_charging": False, "voltage": volts},
        "cpu": {"usage": psutil.cpu_percent(), "temperature": cpu_temp},
        "services": {"robot": "active" if tools else "inactive", "internet": internet},
    }


@app.post("/system/volume")
def system_volume(req: Volume):
    level = max(0.0, min(req.volume / 100.0, 1.5))
    r = subprocess.run(["wpctl", "set-volume", "@DEFAULT_SINK@", str(level)], capture_output=True, text=True)
    return {"success": r.returncode == 0, "message": f"volume set to {req.volume}%" if r.returncode == 0 else r.stderr}


# ------------------------------------------------------------------ hackathon: events, the screen, hearing, vision, memory
class SoundAlert(BaseModel):
    above_db: float = Field(..., description="dB above the room's background level")
    level_db: float = 0.0
    duration_s: float = 0.0


class DetectionsIn(BaseModel):
    detections: list[dict]  # [{label, score, box: [x, y, w, h] as fractions of the frame}]
    width: int = 0
    height: int = 0


class FreeSpaceIn(BaseModel):
    strips: list[dict]  # [{name, closest_m | null, near_frac | null}] left to right (lab/hackathon_vision.py)
    mode: str = "relative"  # "metric" when closest_m is in metres
    eyes_url: str = ""  # where the eyes service answers /look (so the robot can find it)
    circle: list | None = None  # the fisheye's disc [cx, cy, r] as fractions of the frame
    frame_age_s: float = 0.0  # how old the frame behind this summary is (a slow poster says so)
    source: str = ""  # which eyes service (hostname): several may run, and a slow one stands down for a fast one


class TargetIn(BaseModel):
    name: str
    box: list | None = None  # [x, y, w, h] as fractions of the frame; null = looked, not there
    score: float = 1.0
    source: str = "eyes"
    bottom_known: bool = True  # the box's bottom edge is where the thing meets the floor (Lab 5's distance sensor)


class SceneIn(BaseModel):
    text: str


class SpeechIn(BaseModel):
    who: str  # "person" or "pupper"
    text: str


class BrainStateIn(BaseModel):
    listening: bool = True  # the mic is open (it closes while Pupper talks)
    speaking: bool = False
    caption: str | None = None  # what Pupper is saying now


class AnnounceIn(BaseModel):
    text: str


class MemoryScope(BaseModel):
    scope: str = "profile"  # or "all"


class ProfileIn(BaseModel):
    name: str


@app.get("/events")
def events(after: int = -1):
    """Things that happened on the robot (reminders, sounds, searches): the app polls this
    once a second, shows each event's display part and tells Gemini. after=-1 only asks
    for the last id, so a fresh session doesn't replay history."""
    bus = behaviors.EVENTS
    return {"events": bus.since(after) if after >= 0 else [], "last": bus.last_id}


@app.get("/config/api_key")
def api_key():
    """The Gemini key saved on the Pupper (~/.config/gemini-pupper/api_key), so the app on the
    robot's network needn't type it in. Anyone who can reach port 8000 can read it."""
    f = Path("~/.config/gemini-pupper/api_key").expanduser()
    return {"api_key": f.read_text().strip() if f.is_file() else ""}


@app.get("/display")
def display():
    """What is on Pupper's screen right now (show_text, an alert, a reminder), or null."""
    return {"display": behaviors.display_state()}


@app.get("/hearing")
def hearing_status():
    """The calibrated detector: on or off, thresholds, how many sounds it has heard."""
    h = getattr(T(), "hearing", None)
    return {"available": h is not None, **(h.status() if h else {"enabled": False})}


@app.get("/hearing/now")
def hearing_now():
    """What Pupper hears right now (YAMNet's top label, or null for silence), for the screen's corner."""
    h = getattr(T(), "hearing", None)
    return h.now() if h is not None else {"label": None, "confidence": 0.0}


@app.post("/alerts/sound")
def alerts_sound(req: SoundAlert):
    """The browser's sound monitor heard something loud; the robot decides (lab/hackathon_alerts.py)."""
    return behaviors.sound_detected(T(), req.above_db, req.level_db, req.duration_s).as_dict()


@app.post("/vision/detections")
def vision_detections(req: DetectionsIn):
    """The browser's on-device object detector: what is in the camera frame (lab/hackathon_vision.py)."""
    n = behaviors.update_detections(req.detections, req.width, req.height)
    return {"success": True, "message": f"{n} detections"}


@app.post("/vision/freespace")
def vision_freespace(req: FreeSpaceIn):
    """The eyes service's depth sense: how close the nearest surface is in each strip of the lower frame."""
    n = behaviors.update_freespace(req.strips, req.mode, req.eyes_url, req.circle, req.frame_age_s, req.source)
    return {"success": True, "message": f"{n} strips", "wanted": behaviors.WANTED["name"] or None,
            "cam_height_m": behaviors.camera_height_m(T()), "attention": behaviors.attention_now(T()),
            "others_posting": behaviors.others_fresh(req.source)}


@app.get("/vision/wanted")
def vision_wanted():
    """What go_to wants the eyes service to find in the frame right now (null: nothing)."""
    return {"name": behaviors.WANTED["name"] or None, "since": behaviors.WANTED["since"]}


@app.post("/vision/target")
def vision_target(req: TargetIn):
    """The eyes service's latest fix on the wanted thing (lab/hackathon_eyes.py steers on it)."""
    seen = behaviors.update_target(req.name, req.box, req.score, req.source, req.bottom_known)
    return {"success": True, "message": "box" if seen else "not seen", "wanted": behaviors.WANTED["name"] or None}


@app.post("/vision/scene")
def vision_scene(req: SceneIn):
    """Gemma's description of what is around (what_is_around reads it)."""
    n = behaviors.update_scene(req.text)
    return {"success": True, "message": f"{n} chars"}


@app.get("/vision")
def vision(source: str = ""):
    """Everything the eyes posted, plus what an eyes service called ``source`` should do now: run at
    full rate (attention) or stand down (another service is posting)."""
    return {"detections": behaviors.VISION, "freespace": behaviors.FREESPACE, "target": behaviors.TARGET,
            "wanted": behaviors.WANTED, "scene": behaviors.SCENE, "attention": behaviors.attention_now(T()),
            "others_posting": behaviors.others_fresh(source) if source else False}


@app.post("/transcript")
def transcript_post(req: SpeechIn):
    """The web app posts what was said (Gemini's transcriptions of the person and of Pupper)."""
    text = " ".join(req.text.split())[:1000]
    if text:
        behaviors.record("speech", who=req.who[:16], text=text, frame=behaviors.save_frame(latest_jpeg(), req.who) if req.who == "person" else None)
    return {"success": True}


@app.get("/transcript")
def transcript_get(n: int = 50):
    """The last n entries of today's session log (lab/hackathon_log.py)."""
    return {"entries": behaviors.tail(n), "file": str(behaviors.transcript_file())}


# The local brain (robot/gemma_brain.py) has no browser of its own: it posts what it is doing, and
# the face (the web app) shows it.
BRAIN = {"listening": False, "speaking": False, "caption": None, "time": 0.0,
         "local": os.environ.get("PUPPER_BRAIN") == "gemma"}  # run_pupper.sh started the local brain: no Gemini


@app.post("/brain/state")
def brain_state_post(req: BrainStateIn):
    """The local brain: listening, speaking and the caption, now (posted on each change, and every second)."""
    BRAIN.update(listening=req.listening, speaking=req.speaking, caption=req.caption, time=time.time())
    behaviors.brain_speaking(req.speaking)  # no sound alerts from Pupper's own voice
    return {"success": True}


@app.get("/brain/state")
def brain_state_get():
    """What the face shows for the local brain; age_s says how long since it last spoke up (null: never)."""
    return {**BRAIN, "age_s": round(time.time() - BRAIN["time"], 1) if BRAIN["time"] else None,
            "text": behaviors.needs_text()}  # text on the screen: only for a deaf or hard-of-hearing person


@app.post("/brain/announce")
def brain_announce(req: AnnounceIn):
    """A due reminder the face claimed, for the local brain to say: it arrives as an event, like a sound alert."""
    behaviors.EVENTS.emit("reminder", req.text[:300], for_gemini=req.text)
    return {"success": True, "message": "told the brain"}


@app.get("/memory")
def memory_get():
    """The whole memory (profiles, reminders) and the prompt section made from it."""
    return {"memory": behaviors.load(), "prompt": behaviors.memory_prompt(), "file": str(behaviors.MEMORY_FILE)}


@app.post("/memory/reset")
def memory_reset(req: MemoryScope):
    return behaviors.reset_memory(req.scope).as_dict()


@app.post("/memory/profile")
def memory_profile(req: ProfileIn):
    return behaviors.switch_profile(req.name).as_dict()


@app.get("/health")
def health():
    return {"status": "running", "tools": tools is not None}


# The built web app (npm run build in web/ -> web/dist/) is served at /, so the robot's own
# browser, or a laptop through an SSH tunnel, opens everything from one address.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
