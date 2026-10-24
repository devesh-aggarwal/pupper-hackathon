"""Desk tests for reminders (robot/reminder_store.py and lab/reminders.py).

Run with the others:  python3 lab/test.py reminders   (or python3 lab/test.py: everything)

A fake clock stands in for the robot's: the tests start on Thursday 2026-10-22 at
10:00 and move time forward by hand. Reminders go to a temporary file.
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import reminder_store
from reminder_store import ReminderError, ReminderStore, parse_day, parse_days, parse_time

THURSDAY_10AM = datetime(2026, 10, 22, 10, 0, 0)


class FakeClock:
    def __init__(self, start: datetime = THURSDAY_10AM):
        self.t = start

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kw) -> None:
        self.t += timedelta(**kw)


def fresh(clock: FakeClock | None = None) -> tuple[ReminderStore, FakeClock, Path]:
    clock = clock or FakeClock()
    path = Path(tempfile.mkdtemp()) / "reminders.json"
    return ReminderStore(path, now=clock), clock, path


def refused(fn, *args, **kw) -> str:
    """The ReminderError message, or "" if it was accepted."""
    try:
        fn(*args, **kw)
    except ReminderError as e:
        return str(e)
    return ""


async def run(check, Rig, tools) -> None:
    logging.getLogger("reminders").addHandler(logging.NullHandler())  # the broken-file test warns on purpose
    logging.getLogger("reminders").propagate = False
    print("Reminders: understanding the time and day")
    for text, want in [("17:00", (17, 0)), ("5pm", (17, 0)), ("5:30 pm", (17, 30)), ("5.30 P.M.", (17, 30)),
                       ("08:05", (8, 5)), ("0730", (7, 30)), ("12am", (0, 0)), ("12 pm", (12, 0)),
                       ("noon", (12, 0)), ("midnight", (0, 0))]:
        got = parse_time(text)
        check(f"time {text!r} is {want[0]:02d}:{want[1]:02d}", got == want, f"got {got}")
    for bad in ("25:00", "13pm", "7:75", "soon", ""):
        msg = refused(parse_time, bad)
        check(f"time {bad!r} is refused, saying what works", "17:30" in msg, f"got {msg!r}")
    now = THURSDAY_10AM
    for text, want in [("today", "2026-10-22"), ("tomorrow", "2026-10-23"), ("friday", "2026-10-23"),
                       ("Monday", "2026-10-26"), ("next tue", "2026-10-27"), ("october 30", "2026-10-30"),
                       ("Oct 30th", "2026-10-30"), ("30 october", "2026-10-30"), ("10-30", "2026-10-30"),
                       ("2026-12-25", "2026-12-25"), ("january 5", "2027-01-05"), ("10/01", "2027-10-01")]:
        got = parse_day(text, 9, 0, now).isoformat()
        check(f"day {text!r} (from Thu Oct 22) is {want}", got == want, f"got {got}")
    got = parse_day("thursday", 17, 0, now).isoformat()
    check("'thursday' on a Thursday, at a time still ahead, is today", got == "2026-10-22", f"got {got}")
    got = parse_day("thursday", 9, 0, now).isoformat()
    check("... at a time already past, is next week", got == "2026-10-29", f"got {got}")
    msg = refused(parse_day, "someday", 9, 0, now)
    check("an unreadable day is refused, listing the forms that work", "tomorrow" in msg and "friday" in msg, msg)
    check("days 'mon,thu' and 'Monday and Thursday' and 'mondays, thursdays'",
          parse_days("mon,thu") == parse_days("Monday and Thursday") == parse_days("mondays, thursdays") == [0, 3])
    print()

    print("Reminders: when they go off")
    store, clock, _ = fresh()
    r, note = store.add("call mom", at="17:00", day="tomorrow")
    check("one-time, tomorrow 17:00 -> Fri Oct 23 17:00", r.next_at == "2026-10-23T17:00:00", r.next_at)
    r, note = store.add("stretch", at="09:00")
    check("no day and the time already passed today -> tomorrow", r.next_at == "2026-10-23T09:00:00", r.next_at)
    check("... and the note says so", "tomorrow" in note, repr(note))
    msg = refused(store.add, "x", at="09:00", day="today")
    check("explicitly today at a time already past is refused, with the current time", "passed" in msg and "It is now" in msg, msg)
    r, _ = store.add("pizza", in_minutes=20)
    check("in 20 minutes -> 10:20", r.next_at == "2026-10-22T10:20:00", r.next_at)
    r, _ = store.add("pills", at="08:00", repeat="weekdays")
    check("weekdays at 8:00 from Thursday 10:00 -> Friday 8:00", r.next_at == "2026-10-23T08:00:00", r.next_at)
    r, _ = store.add("trash", at="19:00", repeat="custom", days="mon,thu")
    check("custom mon,thu at 19:00 from Thursday 10:00 -> today 19:00", r.next_at == "2026-10-22T19:00:00", r.next_at)
    r, _ = store.add("plants", at="18:00", repeat="weekly", day="sunday")
    check("weekly from Sunday -> Sun Oct 25 18:00, repeating on Sundays",
          r.next_at == "2026-10-25T18:00:00" and r.days == [6], f"{r.next_at} {r.days}")
    r, _ = store.add("water", at="11:00", repeat="daily", day="2026-11-01")
    check("daily starting Nov 1 -> Nov 1 11:00 (not today)", r.next_at == "2026-11-01T11:00:00", r.next_at)
    r2, note = store.add("water", at="11:00", repeat="daily", day="2026-11-01")
    check("the same reminder asked twice isn't added twice", r2.id == r.id and "already" in note, f"#{r2.id} {note!r}")
    for kwargs, word in [({"message": ""}, "message"), ({"message": "x"}, "time"),
                         ({"message": "x", "at": "9:00", "in_minutes": 5}, "not both"),
                         ({"message": "x", "in_minutes": 0}, "more than 0"),
                         ({"message": "x", "in_minutes": 30, "repeat": "daily"}, "one-time"),
                         ({"message": "x", "at": "9:00", "repeat": "custom"}, "days"),
                         ({"message": "x", "at": "9:00", "repeat": "daily", "days": "mon"}, "custom"),
                         ({"message": "x", "at": "9:00", "repeat": "hourly"}, "repeat must be one of"),
                         ({"message": "x", "at": "9:00", "day": "2028-01-01"}, "year")]:
        msg = refused(store.add, **kwargs)
        check(f"refused: {kwargs} (says {word!r})", word in msg, repr(msg))
    print()

    print("Reminders: telling the person (claim, then delivered)")
    store, clock, path = fresh()
    once, _ = store.add("pizza", in_minutes=20)
    daily, _ = store.add("pills", at="10:30", repeat="daily")
    check("nothing is due before its time", store.claim() == [])
    clock.advance(minutes=31)  # 10:31: both are due (10:20 and 10:30)
    due = store.claim()
    check("at its time, it is due (both, in time order)", [d["id"] for d in due] == [once.id, daily.id], str(due))
    check("... and Gemini is told what to say and that it isn't the person speaking",
          "[REMINDER" in due[0]["announce"] and '"pizza"' in due[0]["announce"] and "not from the person" in due[0]["announce"],
          due[0]["announce"])
    check("a claimed reminder isn't handed out twice (two tabs, or the fallback)", store.claim() == [])
    clock.advance(seconds=reminder_store.CLAIM_S + 1)
    again = store.claim()
    check("not confirmed within a minute, it is due again (never lost)", len(again) == 2, str(again))
    check("... and its announcement says it is late", "late" in again[0]["announce"], again[0]["announce"])
    store.delivered(once.id, how="gemini")
    check("a one-time reminder is done once delivered", once.id not in store.reminders)
    store.delivered(daily.id, how="gemini")
    check("a daily one moves to tomorrow 10:30", store.reminders[daily.id].next_at == "2026-10-23T10:30:00",
          store.reminders[daily.id].next_at)
    clock.advance(days=3)  # the robot was off for three days
    due = store.claim()
    check("missed repeats are told once, not one per missed day", len(due) == 1, str(due))
    store.delivered(daily.id)
    check("... then it moves to the next one after now", store.reminders[daily.id].next_at == "2026-10-26T10:30:00",
          store.reminders[daily.id].next_at)
    check("delivered() for a reminder cancelled meanwhile is harmless", store.delivered(999) is None)
    mins = store.add("late one", in_minutes=1)[0]
    clock.advance(minutes=1, seconds=30)
    check("claim(min_late_s=60) skips a reminder only 30 s late (the session may come back)",
          [d["id"] for d in store.claim(min_late_s=60)] == [])
    clock.advance(seconds=40)
    check("... and takes it once it is over a minute late", [d["id"] for d in store.claim(min_late_s=60)] == [mins.id])
    print()

    print("Reminders: dismissing from the screen")
    dstore, dclock, _ = fresh()
    daily, _ = dstore.add("pills", at="10:01", repeat="daily")
    once, _ = dstore.add("pizza", at="10:02")
    dclock.advance(minutes=3)
    for d in dstore.claim():
        dstore.delivered(d["id"], how="gemini")
    check("told reminders are on the screen", sorted(e["id"] for e in dstore.on_screen()) == [daily.id, once.id])
    gone = dstore.dismiss(daily.id)
    check("dismiss(id) takes that one off the screen", [e["id"] for e in gone] == [daily.id]
          and [e["id"] for e in dstore.on_screen()] == [once.id])
    check("... and the web app is told to drop it", dstore.dismissed_ids() == [daily.id], str(dstore.dismissed_ids()))
    check("... but a repeating reminder is NOT cancelled", daily.id in dstore.reminders
          and dstore.reminders[daily.id].next_at == "2026-10-23T10:01:00")
    check("dismissing it again finds nothing showing", dstore.dismiss(daily.id) == [])
    dclock.advance(days=1)
    for d in dstore.claim():
        dstore.delivered(d["id"])
    check("the next day it is told and shows again (yesterday's dismissal doesn't hide it)",
          daily.id in [e["id"] for e in dstore.on_screen()] and daily.id not in dstore.dismissed_ids())
    check("dismiss() with no id clears everything showing", dstore.dismiss() and dstore.on_screen() == [])
    dstore.add("tea", in_minutes=1)
    dclock.advance(minutes=2)
    tea = dstore.claim()[0]
    dstore.delivered(tea["id"])
    dclock.advance(seconds=reminder_store.SCREEN_S + 1)
    check("after SCREEN_S a told reminder is off the screen by itself", dstore.on_screen() == [])
    print()

    print("Reminders: saved across restarts")
    reloaded = ReminderStore(path, now=clock)
    check("a new store (a restarted server) has the same reminders",
          sorted(reloaded.reminders) == sorted(store.reminders), f"{sorted(reloaded.reminders)}")
    nxt, _ = reloaded.add("after restart", in_minutes=5)
    check("... and keeps numbering after them", nxt.id > max(store.reminders), f"#{nxt.id}")
    path.write_text("{not json")
    broken = ReminderStore(path, now=clock)
    check("a broken file doesn't stop the server: it starts empty and keeps the file aside",
          not broken.reminders and path.with_suffix(".bad.json").exists())
    print()

    print("Reminders: the tools Gemini calls")
    store, clock, _ = fresh()
    reminder_store.use_store(store)
    rig = Rig()
    try:
        decls = {d["name"]: d for d in tools.declarations(rig.ts)}
        for name in ("set_reminder", "list_reminders", "dismiss_reminder", "cancel_reminder"):
            d = decls.get(name, {})
            params = d.get("parameters", {}).get("properties", {})
            check(f"{name} is declared, with a description and every parameter described",
                  len(d.get("description", "")) >= 20 and all(len(p.get("description", "")) >= 3 for p in params.values()),
                  json.dumps(d)[:200])
        res = await rig.call("set_reminder", message="take your pills", at="08:00", repeat="weekdays")
        check("set_reminder: weekdays at 8 is accepted", res.ok, res.message)
        check("... and the reply says how often, when next, and the robot's time",
              "every weekday" in res.message and "tomorrow at 8:00 AM" in res.message and "It is now" in res.message,
              res.message)
        res = await rig.call("set_reminder", message="call mom", at="5pm", day="tomorrow")
        check("set_reminder: '5pm' 'tomorrow' is understood", res.ok and "tomorrow at 5:00 PM" in res.message, res.message)
        res = await rig.call("set_reminder", message="x", at="half past", day="tomorrow")
        check("set_reminder: an unreadable time fails, saying what works", not res.ok and "17:30" in res.message,
              res.message)
        res = await rig.call("set_reminder", message="x", at="9:00", repeat="sometimes")
        check("set_reminder: an unknown repeat is refused by its choices", not res.ok, res.message)
        res = await rig.call("list_reminders")
        check("list_reminders lists both, with numbers and the time now",
              res.ok and "#1" in res.message and "#2" in res.message and "It is now" in res.message, res.message)
        res = await rig.call("cancel_reminder", id=7)
        check("cancel_reminder of an unknown number fails and lists the real ones",
              not res.ok and "#1" in res.message, res.message)
        res = await rig.call("cancel_reminder", id=1.0)
        check("cancel_reminder(1.0) cancels #1 (a model may send whole numbers as 1.0)",
              res.ok and 1 not in store.reminders, res.message)
        res = await rig.call("dismiss_reminder")
        check("dismiss_reminder with nothing on screen says so (and succeeds)",
              res.ok and "nothing to dismiss" in res.message, res.message)
        res = await rig.call("dismiss_reminder", id=2)  # call mom: set, not gone off yet
        check("dismiss_reminder of one that hasn't gone off yet fails, pointing to cancel_reminder",
              not res.ok and "cancel_reminder" in res.message, res.message)
        weekly = store.add("water the plants", at="10:01", repeat="daily")[0]
        clock.advance(minutes=2)
        for d in store.claim():
            store.delivered(d["id"], how="gemini")
        res = await rig.call("dismiss_reminder", id=float(weekly.id))
        check("'got it' -> dismiss_reminder clears it and says it still repeats",
              res.ok and "still repeats" in res.message and weekly.id in store.reminders, res.message)
        res = await rig.call("dismiss_reminder", id=weekly.id)
        check("... a second time: already off the screen", res.ok and "already off" in res.message, res.message)
        rig.ts.following = True  # follow mode on: a reminder is talking, not motion
        await rig.call("set_reminder", message="drink water", in_minutes=30)
        check("setting a reminder keeps follow mode on (it doesn't move the robot)", rig.ts.following)
        check("... and queues nothing for the legs", rig.ts.commands.empty() and rig.ts.current is None)
    finally:
        rig.ts.following = False
        rig.close()
        reminder_store.use_store(ReminderStore(Path(tempfile.mkdtemp()) / "r.json"))
