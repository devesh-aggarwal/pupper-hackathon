"""Reminders: Pupper reminds people of things at a day and time, once or repeating.

Added after the lab (not a TODO). Same two halves as every lab/ file, but nothing
here moves the robot, so nothing goes through the queue and follow mode stays on
(talking and questions keep following).

How a reminder reaches the person (robot/reminder_store.py has the details):

    "remind me every weekday at 8 to take my pills"
      -> Gemini calls set_reminder(message, at="08:00", repeat="weekdays")
      -> the store works out the next time from the robot's clock, saves it to
         ~/.config/gemini-pupper/reminders.json, and the reply tells Gemini when
      ... at 8:00 the web app (it asks every 5 s) claims the due reminder and
         sends Gemini a "[REMINDER #n] ..." message, and Pupper says it out loud;
         the web app also shows it on screen, then confirms delivery
      -> once: done. Repeating: moved to the next weekday at 8:00.

If no web app takes it, the robot server says it itself (espeak-ng); if
nothing confirms delivery within a minute, it is due again, so it is never lost.
"""

from __future__ import annotations

from reminder_store import DAY_FORMS, REPEATS, TIME_FORMS, ReminderError, describe, get_store, now_words, rule_words, when
from robot_api import Result, Robot
from tool_registry import Param, tool


# ============================================================ what the robot does
def set_reminder(message: str, at: str | None = None, day: str | None = None, repeat: str = "once",
                 days: str | None = None, in_minutes: float | None = None) -> Result:
    store = get_store()
    try:
        r, note = store.add(message, at=at, day=day, repeat=repeat, days=days, in_minutes=in_minutes)
    except ReminderError as e:
        return Result(False, f"No reminder set: {e}")
    now = store.now()
    how_often = "" if r.repeat == "once" else f", {rule_words(r)}"
    text = (f'Reminder #{r.id} set: "{r.message}"{how_often}; {"next " if how_often else ""}{when(r.next_dt, now)}.'
            f" {now_words(now)} Pupper will say it out loud then.")
    return Result(True, f"{note} {text}".strip())


def list_reminders() -> Result:
    store = get_store()
    now = store.now()
    items = store.list()
    if not items:
        return Result(True, f"No reminders are set. {now_words(now)}")
    return Result(True, f"{now_words(now)} {len(items)} reminder{'s' if len(items) > 1 else ''}: "
                        + "; ".join(describe(r, now) for r in items) + ".")


def dismiss_reminder(rid: int | None = None) -> Result:
    """Take a reminder Pupper just said off the screen (the person said "got it"). Never
    cancels it: a repeating one still goes off next time."""
    store = get_store()
    gone = store.dismiss(rid)
    if gone:
        parts = []
        for e in gone:
            after = f"It still repeats {e['repeats']}." if e["repeats"] else "It was a one-time reminder, so it's done."
            parts.append(f'Dismissed reminder #{e["id"]} "{e["message"]}" from the screen. {after}')
        return Result(True, " ".join(parts))
    if rid is None:
        return Result(True, "No reminder is on the screen right now, so there is nothing to dismiss.")
    if store.recently_told(rid):
        return Result(True, f"Reminder #{rid} is already off the screen.")
    if rid in store.reminders:
        return Result(False, f"Reminder #{rid} hasn't gone off yet, so there is nothing to dismiss."
                             " To stop it going off, use cancel_reminder.")
    showing = ", ".join(f'#{e["id"]}' for e in store.on_screen()) or "none"
    return Result(False, f"There is no reminder #{rid} on the screen (showing: {showing}).")


def cancel_reminder(rid: int) -> Result:
    store = get_store()
    try:
        r = store.cancel(rid)
    except ReminderError as e:
        return Result(False, f"Nothing cancelled: {e}")
    return Result(True, f'Cancelled reminder #{r.id} "{r.message}" ({rule_words(r)}).')


# ============================================================ what Gemini reads
@tool(
    "set_reminder",
    "Set a reminder: at that time Pupper says the message out loud to the person. One-time (a day and a"
    " time, or in_minutes from now) or repeating (every day, weekdays, weekends, every week, or chosen"
    " weekdays). Pass the day in words; the robot works out the date from its own clock, so you don't"
    " need today's date. If the person didn't say when, or the time could be AM or PM and you can't"
    " tell, ask first. The reply says exactly when it will go off: tell the person. Examples:"
    " 'remind me to call mom tomorrow at 5pm' -> set_reminder(message='call mom', at='17:00', day='tomorrow');"
    " 'in 20 minutes remind me to take the pizza out' -> set_reminder(message='take the pizza out', in_minutes=20);"
    " 'every weekday at 8 am remind me to take my pills' -> set_reminder(message='take your pills', at='08:00',"
    " repeat='weekdays'); 'trash goes out Monday and Thursday nights at 7' -> set_reminder(message='take the"
    " trash out', at='19:00', repeat='custom', days='mon,thu'); 'remind me on October 30 at 9 to pay rent' ->"
    " set_reminder(message='pay the rent', at='09:00', day='october 30').",
    message=Param("string", "What to remind them about, as Pupper will say it, e.g. 'take your pills'."),
    at=Param("string", f"The time of day: {TIME_FORMS}. Needed unless in_minutes is given.", required=False),
    day=Param("string", f"The day of a one-time reminder (or the first day of a repeating one): {DAY_FORMS}."
              " Left out: today, or tomorrow if that time already passed.", required=False),
    repeat=Param("string", "How often: once (default), daily, weekdays (Mon-Fri), weekends, weekly (the weekday"
                 " of its first time, or the one in days), or custom (the weekdays in days).",
                 enum=REPEATS, required=False),
    days=Param("string", "Only for repeat='custom' (or 'weekly'): the weekdays, e.g. 'mon,wed,fri'.", required=False),
    in_minutes=Param("number", "For a one-time reminder in a while: minutes from now, e.g. 20 (instead of at"
                     " and day). At most a week.", required=False),
)
async def set_reminder_tool(robot: Robot, message: str, at: str | None = None, day: str | None = None,
                            repeat: str = "once", days: str | None = None, in_minutes: float | None = None) -> Result:
    return set_reminder(message, at=at, day=day, repeat=repeat, days=days, in_minutes=in_minutes)


@tool(
    "list_reminders",
    "List every reminder that is set, with its number, what it says, how often, and when it next goes off;"
    " the reply also says the robot's current date and time. Use it when asked what reminders there are,"
    " before cancelling one (to find its number), or to check today's date.",
)
async def list_reminders_tool(robot: Robot) -> Result:
    return list_reminders()


@tool(
    "dismiss_reminder",
    "Clear a reminder you just said from the screen when the person acknowledges it: 'got it', 'okay',"
    " 'done', 'thanks', 'I took them', 'dismiss that'. It does NOT cancel the reminder: a repeating one still"
    " goes off next time (only cancel_reminder stops it for good, when they say they don't want it any more)."
    " Leave id out to clear every reminder on the screen. Example: after [REMINDER #3], 'okay, got it' ->"
    " dismiss_reminder(id=3).",
    id=Param("number", "The reminder's number from the [REMINDER #n] message. Left out: all on the screen.",
             required=False),
)
async def dismiss_reminder_tool(robot: Robot, id: float | None = None) -> Result:  # noqa: A002 (Gemini's word)
    if id is not None and id != int(id):
        return Result(False, f"Nothing dismissed: reminder numbers are whole numbers, not {id:g}")
    return dismiss_reminder(None if id is None else int(id))


@tool(
    "cancel_reminder",
    "Cancel one reminder by its number (from set_reminder's or list_reminders' reply). A repeating reminder"
    " is cancelled for good. To change a reminder, cancel it and set a new one.",
    id=Param("number", "The reminder's number, e.g. 3 for reminder #3."),
)
async def cancel_reminder_tool(robot: Robot, id: float) -> Result:  # noqa: A002 (Gemini's word)
    # "number", not "integer": a model may send 3.0, which the registry would refuse as an integer
    if id != int(id):
        return Result(False, f"Nothing cancelled: reminder numbers are whole numbers, not {id:g}")
    return cancel_reminder(int(id))
