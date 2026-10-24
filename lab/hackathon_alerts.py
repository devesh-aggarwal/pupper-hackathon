"""Hackathon, part B: events, the screen, and hearing for people who can't.

Three things the lab's Pupper couldn't do, all built on one idea, an *event*:
something happened on the robot that the brain didn't ask for (a reminder came due,
the microphone heard a bang, a search found the keys). The web app polls
GET /events?after=<id> once a second, shows each event's ``display`` part on
Pupper's screen, and hands its ``for_gemini`` text to Gemini as a turn, so Pupper
reacts out loud without anyone speaking first.

  EVENTS      the bus (emit / since). Reminders, sound alerts and find use it.
  show()      put big text and a background color on the screen (the show_text tool,
              and every alert). The screen is how a deaf or hard-of-hearing person
              gets what Pupper would otherwise only say.
  hearing_detected()  the calibrated detector (hearing/listener.py, run by the robot server
              as robot.hearing) heard an URGENT sound (siren, alarm, doorbell, bang, crash,
              knocking, a baby crying; not shouting). The robot decides: Pupper's own motors, or one alert already showing,
              and nothing happens; otherwise the screen turns red with what it was. Gemini is
              not told: Pupper shows sounds, it doesn't talk about them.
  sound_detected()  the fallback, when the robot has no calibrated detector: the browser
              measures the microphone (services/soundMonitor.ts) and posts loud moments to
              POST /alerts/sound, and the same rules apply, plus a loudness threshold.

Sound alerts are OPT IN: off every time the robot starts, and only the sound_alerts tool (a
person asking) turns them on. Only then does the robot open the mic to listen for them.

Rules are enforced here, not in the browser: the browser only measures.
"""

from __future__ import annotations

import collections
import logging
import time

import hackathon_memory as memory
from robot_api import Result, Robot
from tool_registry import Param, tool

log = logging.getLogger("events")

COLORS = {
    "red": "#ef4444", "orange": "#f97316", "yellow": "#facc15", "green": "#22c55e",
    "blue": "#3b82f6", "purple": "#a855f7", "white": "#f9fafb", "black": "#111111",
}
SENSITIVITY_DB = {"low": 25.0, "medium": 18.0, "high": 12.0}  # dB above the room's background
DEFAULT_SENSITIVITY = "medium"
MIN_SOUND_GAP_S = 4.0  # one alert at a time
MOTION_QUIET_S = 1.5  # after a motion job: the motors are loud, and so is a landing


# ============================================================ what the robot does: the bus
class EventBus:
    def __init__(self, keep: int = 200) -> None:
        self.events: collections.deque[dict] = collections.deque(maxlen=keep)
        self.last_id = 0

    def emit(self, kind: str, text: str, *, for_gemini: str | None = None, display: dict | None = None,
             data: dict | None = None) -> dict:
        self.last_id += 1
        ev = {"id": self.last_id, "time": time.time(), "clock": time.strftime("%H:%M"), "kind": kind,
              # for_gemini "" means: don't tell Gemini (only None falls back to the text).
              "text": text, "for_gemini": text if for_gemini is None else for_gemini, "display": display,
              "data": data or {}}
        self.events.append(ev)
        log.info("event %s: %s", kind, text)
        try:
            import hackathon_log  # here: hackathon_log never imports this file, but keep the import local
            hackathon_log.record("event", event=kind, text=text, for_gemini=ev["for_gemini"])
        except Exception:  # the log must never break an alert
            log.exception("transcript record failed")
        return ev

    def since(self, after: int) -> list[dict]:
        return [e for e in self.events if e["id"] > after]


EVENTS = EventBus()

# ============================================================ what the robot does: the screen
DISPLAY: dict = {"text": "", "detail": "", "color": COLORS["blue"], "until": 0.0, "kind": ""}


def color_hex(color: str) -> str:
    c = (color or "").strip().lower()
    if c.startswith("#") and len(c) in (4, 7):
        return c
    return COLORS.get(c, COLORS["blue"])


def show(text: str, detail: str = "", color: str = "blue", seconds: float = 15.0, *, kind: str = "display",
         tell_gemini: str | None = None) -> dict:
    """Big text on the screen for ``seconds``, and an event so the app shows it now."""
    seconds = max(1.0, min(float(seconds), 600.0))
    DISPLAY.update(text=text.strip()[:80], detail=detail.strip()[:160], color=color_hex(color),
                   until=time.time() + seconds, kind=kind)
    payload = {"text": DISPLAY["text"], "detail": DISPLAY["detail"], "color": DISPLAY["color"], "seconds": seconds,
               "kind": kind}
    return EVENTS.emit(kind, f"screen: {DISPLAY['text']}" + (f" ({DISPLAY['detail']})" if DISPLAY["detail"] else ""),
                       for_gemini=tell_gemini or "", display=payload)


def display_state() -> dict | None:
    """What is on the screen right now (the app asks when it (re)connects)."""
    if time.time() >= DISPLAY["until"] or not DISPLAY["text"]:
        return None
    return {"text": DISPLAY["text"], "detail": DISPLAY["detail"], "color": DISPLAY["color"],
            "seconds": round(DISPLAY["until"] - time.time(), 1), "kind": DISPLAY.get("kind", "display")}


def clear_display() -> None:
    DISPLAY.update(text="", detail="", until=0.0)


# ============================================================ what the robot does: hearing
_last_sound_alert = 0.0
SOUND_ALERTS_ON = False  # opt in: never on at start, and not kept in memory


def sound_settings() -> tuple[bool, str]:
    s = memory.load().get("settings", {})
    return SOUND_ALERTS_ON, str(s.get("sound_sensitivity", DEFAULT_SENSITIVITY))


def set_sound_alerts(enabled: bool, sensitivity: str | None = None) -> Result:
    global SOUND_ALERTS_ON
    if sensitivity and sensitivity not in SENSITIVITY_DB:
        return Result(False, f"unknown sensitivity '{sensitivity}'; choose from {list(SENSITIVITY_DB)}")
    SOUND_ALERTS_ON = bool(enabled)
    data = memory.load()
    if sensitivity:
        data.setdefault("settings", {})["sound_sensitivity"] = sensitivity
        memory.save(data)
    sens = data.get("settings", {}).get("sound_sensitivity", DEFAULT_SENSITIVITY)
    if not enabled:
        return Result(True, "sound alerts off: loud sounds are ignored")
    return Result(True, f"sound alerts on ({sens} sensitivity: a sound {SENSITIVITY_DB[sens]:.0f} dB above the room"
                        " flashes the screen and I tell you)")


def _moving(robot: Robot) -> bool:
    return bool(robot.following or robot.current is not None or robot.now() < robot.busy_until + MOTION_QUIET_S)


# URGENT sounds turn the whole screen red with what it was, for URGENT_SECONDS. Gemini is NOT told:
# Pupper doesn't talk about sounds, it shows them. Everyday sounds (speech, music, typing) only
# get a small label in a corner of the screen (GET /hearing/now), and silence shows nothing.
URGENT = {  # class -> (big words on the screen, the line under them)
    "siren": ("SIREN", "an ambulance, police car or fire truck"),
    "alarm": ("ALARM", "a smoke alarm, fire alarm or timer"),
    "doorbell": ("DOORBELL", "someone may be at the door"),
    "bang": ("LOUD BANG", "something dropped, slammed or fell"),
    "crash": ("CRASH", "something broke"),
    "knock": ("KNOCKING", "someone may be at the door"),
    "baby": ("BABY CRYING", "a baby is crying"),
}
# A hard knock or a crash also trips the meter's bang detector, often first: the more specific
# name replaces LOUD BANG on the screen instead of being dropped as "already alerting".
MORE_SPECIFIC_THAN_BANG = {"knock", "doorbell", "crash"}
_last_urgent_cls = ""
URGENT_COLOR = "red"
URGENT_SECONDS = 10.0


def hearing_detected(robot: Robot, event: dict) -> Result:
    """The calibrated detector (robot.hearing) heard an urgent sound: show it, red, on the screen."""
    global _last_sound_alert, _last_urgent_cls
    if not SOUND_ALERTS_ON:
        return Result(True, "ignored: sound alerts are off")
    if _moving(robot):
        return Result(True, "ignored: that was probably my own motors (I was moving)")
    cls = event.get("cls", "bang")
    if cls not in URGENT:
        return Result(True, f"ignored: {cls} is not an urgent sound")
    now = time.time()
    upgrade = _last_urgent_cls == "bang" and cls in MORE_SPECIFIC_THAN_BANG
    if now - _last_sound_alert < MIN_SOUND_GAP_S and not upgrade:
        return Result(True, "ignored: already alerting")
    _last_sound_alert, _last_urgent_cls = now, cls
    label, what = URGENT[cls]
    info = event.get("info", {})
    how = (f"{info['label']}, {100 * info.get('confidence', 0):.0f}% sure" if info.get("label")   # YAMNet
           else f"{event.get('score', 0):.0f} dB above the room")                                  # the meter
    show(label, f"{time.strftime('%H:%M:%S')}  {what}", URGENT_COLOR, URGENT_SECONDS, kind="sound", tell_gemini="")
    log.info("urgent sound: %s (%s)", label, how)
    return Result(True, f"{label}: shown on the screen ({how})")


def sound_detected(robot: Robot, above_db: float, level_db: float = 0.0, duration_s: float = 0.0) -> Result:
    """The browser heard something loud. Decide whether it is an alert, and alert."""
    global _last_sound_alert
    enabled, sens = sound_settings()
    needed = SENSITIVITY_DB.get(sens, SENSITIVITY_DB[DEFAULT_SENSITIVITY])
    if not enabled:
        return Result(True, "ignored: sound alerts are off")
    hearing = getattr(robot, "hearing", None)
    if hearing is not None and hearing.enabled:
        return Result(True, "ignored: the calibrated detector is listening instead")
    if _moving(robot):
        return Result(True, "ignored: that was probably my own motors (I was moving)")
    if above_db < needed:
        return Result(True, f"ignored: +{above_db:.0f} dB is under the {sens} threshold (+{needed:.0f} dB)")
    now = time.time()
    if now - _last_sound_alert < MIN_SOUND_GAP_S:
        return Result(True, "ignored: already alerting")
    _last_sound_alert = now
    very = above_db >= needed + 10
    label = "VERY LOUD SOUND" if very else "LOUD SOUND"
    clock = time.strftime("%H:%M:%S")
    detail = f"{clock}  +{above_db:.0f} dB for {duration_s:.1f} s"
    show(label, detail, URGENT_COLOR, URGENT_SECONDS, kind="sound", tell_gemini="")
    return Result(True, f"{label} (+{above_db:.0f} dB): shown on the screen")


# ============================================================ what Gemini reads
@tool(
    "show_text",
    "Put big words on Pupper's screen with a background color, for people who can't hear you or"
    " want it written down: what a sound was ('DOORBELL'), a reminder, where a thing is, a short"
    " answer when asked to show it, or 'STOP' when something is wrong. Keep text to a few words;"
    " detail is one short line under it. The screen goes back to Pupper's face after seconds.",
    text=Param("string", "The big words, up to about 6."),
    detail=Param("string", "One short line under them, e.g. the time or a hint. Optional.", required=False),
    color=Param("string", "Background color.", enum=list(COLORS), required=False),
    seconds=Param("number", "How long to show it, 1-600 s. Default 10.", required=False),
)
async def show_text_tool(robot: Robot, text: str, detail: str = "", color: str = "blue", seconds: float = 10.0) -> Result:
    if not text.strip():
        return Result(False, "nothing to show: give a few words")
    show(text, detail, color, seconds)
    return Result(True, f"showing '{text.strip()[:80]}' on the screen in {color} for {max(1.0, min(seconds, 600.0)):g} s")


@tool(
    "sound_alerts",
    "Turn hearing alerts on or off. They are off until a person asks: only turn them on when the"
    " person asks you, in so many words, to listen or watch out for loud sounds (bangs, falls, alarms, sirens, the doorbell, knocking, a baby crying), never"
    " on your own, and never because the words 'sound' or 'alert' came up: 'no sound alerts right now' means leave them off; if unsure, ask first."
    " When on, urgent sounds (siren, alarm, doorbell, bang, crash, knocking, a baby crying; not shouting) turn the screen"
    " red with what they were, and a corner of the screen shows what Pupper hears; you are not told"
    " about them. Sensitivity: low only reacts to very loud"
    " sounds, high to smaller ones. Default medium.",
    enabled=Param("boolean", "true to alert on loud sounds, false to ignore them."),
    sensitivity=Param("string", "How loud a sound must be to count.", enum=list(SENSITIVITY_DB), required=False),
)
async def sound_alerts_tool(robot: Robot, enabled: bool, sensitivity: str = "") -> Result:
    res = set_sound_alerts(enabled, sensitivity or None)
    hearing = getattr(robot, "hearing", None)
    if not res.ok or hearing is None:
        return res  # no calibrated detector: the browser's measurement is used
    ok, how = await hearing.set_enabled(enabled, sound_settings()[1])
    if not ok:
        set_sound_alerts(False)
        return Result(False, f"sound alerts stay off: {how}")
    return Result(True, f"{res.message}; {how}")
