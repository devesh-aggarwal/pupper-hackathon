"""Reminders: what Pupper should tell someone, and when (no ROS: the server, the lab tools
and the tests all use it).

A reminder is a message and a rule for when it is due:

  once       one date and time ("tomorrow at 17:00", or "in 20 minutes")
  daily      every day at a time
  weekdays   Monday to Friday at a time
  weekends   Saturday and Sunday at a time
  weekly     one weekday at a time (the weekday of its first time)
  custom     chosen weekdays at a time ("mon,thu")

Times are the robot's local wall-clock time: its timezone (timedatectl on the Pi)
must be right, and every reply says what time the robot thinks it is.

Gemini passes words ("tomorrow", "friday", "october 30", "5:30 pm"); the dates are
worked out here, from the robot's clock, so Gemini never has to know today's date.

Delivery, at least once. Nothing runs on a timer: a reminder is due once its
next time has passed. Whoever can tell the person (the web app, through Gemini;
or the server's spoken fallback) claims it, which hides it from other claimers
for CLAIM_S, tells the person, then confirms with delivered(). A one-time
reminder is then done; a repeating one moves to its next time after now (repeats
missed while the robot was off are told once, not one by one). With no
delivered() within CLAIM_S (the tab closed, the session dropped), it is due
again: a reminder is never lost, at worst told late, and the announcement says
how late. Every change is saved to a JSON file, so reminders survive restarts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable

log = logging.getLogger("reminders")

DEFAULT_FILE = Path(os.environ.get("PUPPER_REMINDERS_FILE",
                                   Path.home() / ".config" / "gemini-pupper" / "reminders.json"))
CLAIM_S = 60.0  # a claimed reminder is hidden this long; not confirmed by then, it is due again
SCREEN_S = 120.0  # a told reminder stays on the web app's screen this long, unless dismissed (web/App.tsx)
RECENT = 20  # told reminders remembered for dismissing (in memory: the screen doesn't survive a restart)
MAX_REMINDERS = 50
MAX_MESSAGE = 200  # characters
MAX_AHEAD_DAYS = 366  # the furthest a first time may be
MAX_IN_MINUTES = 7 * 24 * 60

REPEATS = ["once", "daily", "weekdays", "weekends", "weekly", "custom"]
DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_FIXED_DAYS = {"daily": list(range(7)), "weekdays": [0, 1, 2, 3, 4], "weekends": [5, 6]}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
TIME_FORMS = "24-hour like '17:30', or '5:30 pm', 'noon', 'midnight'"
DAY_FORMS = "'today', 'tomorrow', a weekday like 'friday', 'october 30', '10-30' or '2026-10-30'"


class ReminderError(ValueError):
    """A request that can't become a reminder; the message says why and what works."""


@dataclass
class Reminder:
    id: int
    message: str
    repeat: str  # one of REPEATS
    at: str  # "HH:MM" or "HH:MM:SS": the time of day it is due
    days: list[int]  # weekdays it repeats on (0 = Monday); empty for once
    next_at: str  # ISO local datetime: when it is next due
    created_at: str
    claimed_until: str | None = None  # ISO local datetime, while someone is telling the person
    delivered_count: int = 0
    last_delivered_at: str | None = None
    last_delivered_how: str = ""

    @property
    def next_dt(self) -> datetime:
        return datetime.fromisoformat(self.next_at)


# ============================================================ words -> times and dates
def parse_time(text: str) -> tuple[int, int]:
    """'17:30', '5:30 pm', '5pm', '0730', 'noon', 'midnight' -> (hour, minute)."""
    t = str(text).strip().lower().replace(".", "").replace(" ", "")
    if t in ("noon", "midday"):
        return 12, 0
    if t == "midnight":
        return 0, 0
    m = re.fullmatch(r"(\d{1,2})(?::?(\d{2}))?(am|pm)?", t)
    if not m:
        raise ReminderError(f"can't read the time {text!r}: give it {TIME_FORMS}")
    hour, minute, half = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if minute > 59:
        raise ReminderError(f"can't read the time {text!r}: minutes go up to 59; give it {TIME_FORMS}")
    if half:
        if not 1 <= hour <= 12:
            raise ReminderError(f"can't read the time {text!r}: with am/pm the hour is 1 to 12; give it {TIME_FORMS}")
        hour = hour % 12 + (12 if half == "pm" else 0)
    elif hour > 23:
        raise ReminderError(f"can't read the time {text!r}: give it {TIME_FORMS}")
    return hour, minute


def parse_weekday(word: str) -> int | None:
    """'mon', 'Monday', 'mondays', 'tues', 'thurs', 'weds' -> 0..6, else None."""
    w = word.strip().lower()
    if w.endswith("s") and w[:-1] in DAY_NAMES:  # "mondays"
        w = w[:-1]
    if w == "weds":
        return 2
    if len(w) < 2:  # "s" or "t" could be two days
        return None
    return next((i for i, name in enumerate(DAY_NAMES) if name.startswith(w)), None)


def parse_days(text: str) -> list[int]:
    """'mon,thu', 'Monday and Thursday', 'tue wed' -> sorted weekday numbers."""
    words = [w for w in re.split(r"[^a-z]+", str(text).lower()) if w and w not in ("and", "every", "on")]
    days = set()
    for w in words:
        d = parse_weekday(w)
        if d is None:
            raise ReminderError(f"can't read the day {w!r} in days={text!r}: use weekday names like 'mon,wed,fri'")
        days.add(d)
    if not days:
        raise ReminderError("days is empty: list weekdays like 'mon,wed,fri'")
    return sorted(days)


def _month(word: str) -> int | None:
    w = word.lower()
    for i, name in enumerate(MONTHS):
        if len(w) >= 3 and name.startswith(w):
            return i + 1
    return None


def _first_on_or_after(today: date, month: int, day: int) -> date:
    """That month and day this year, or next year if it already passed."""
    for year in (today.year, today.year + 1):
        try:
            d = date(year, month, day)
        except ValueError:
            continue
        if d >= today:
            return d
    raise ReminderError(f"there is no day {day} in month {month}")


def parse_day(text: str, hour: int, minute: int, now: datetime) -> date:
    """A day in words -> a date, from the robot's clock. A bare weekday that is today
    means today if the time is still ahead, else next week."""
    t = re.sub(r"\s+", " ", str(text).strip().lower().replace(",", " "))
    today = now.date()
    if t in ("today", "tonight", "this evening", "this afternoon", "this morning"):
        return today
    if t == "tomorrow":
        return today + timedelta(days=1)
    if t in ("day after tomorrow", "the day after tomorrow"):
        return today + timedelta(days=2)
    words = t.split(" ")
    lead = words[0] if len(words) == 2 else ""
    if len(words) == 1 or lead in ("next", "this", "on", "coming"):  # "friday", "next friday"
        wd = parse_weekday(words[-1])
        if wd is not None:
            ahead = (wd - today.weekday()) % 7
            if ahead == 0 and (words[0] == "next" or datetime.combine(today, time(hour, minute)) <= now):
                ahead = 7
            return today + timedelta(days=ahead)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            raise ReminderError(f"{text!r} is not a real date") from None
    m = re.fullmatch(r"(\d{1,2})[-/](\d{1,2})", t)
    if m:
        return _first_on_or_after(today, int(m.group(1)), int(m.group(2)))
    m = (re.fullmatch(r"(?:the )?([a-z]+) (\d{1,2})(?:st|nd|rd|th)?", t)
         or re.fullmatch(r"(?:the )?(\d{1,2})(?:st|nd|rd|th)? (?:of )?([a-z]+)", t))
    if m:
        a, b = m.groups()
        month, day = (_month(a), b) if a.isalpha() else (_month(b), a)
        if month:
            return _first_on_or_after(today, month, int(day))
    raise ReminderError(f"can't read the day {text!r}: give {DAY_FORMS}")


# ============================================================ words for Gemini
def clock(dt: datetime | time) -> str:
    """17:05 -> '5:05 PM'."""
    return f"{dt.hour % 12 or 12}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def when(dt: datetime, now: datetime) -> str:
    """'today at 5:00 PM', 'tomorrow at 8:00 AM', 'Monday Oct 27 at 8:00 AM'."""
    days = (dt.date() - now.date()).days
    day = "today" if days == 0 else "tomorrow" if days == 1 else f"{dt:%A %b} {dt.day}"
    if days > 300:
        day += f" {dt.year}"
    return f"{day} at {clock(dt)}"


def now_words(now: datetime) -> str:
    return f"It is now {now:%A %b} {now.day}, {clock(now)}."


def rule_words(r: Reminder) -> str:
    """How often: 'once', 'every weekday (Mon-Fri) at 8:00 AM', 'every Mon and Thu at 7:00 PM'."""
    at = clock(time.fromisoformat(r.at))
    if r.repeat == "once":
        return "once"
    if r.repeat == "daily":
        return f"every day at {at}"
    if r.repeat == "weekdays":
        return f"every weekday (Mon-Fri) at {at}"
    if r.repeat == "weekends":
        return f"every Saturday and Sunday at {at}"
    names = [DAY_NAMES[d].capitalize() if len(r.days) == 1 else DAY_NAMES[d][:3].capitalize() for d in r.days]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f"every {listed} at {at}"


def describe(r: Reminder, now: datetime) -> str:
    """'#3 "take your pills", every weekday (Mon-Fri) at 8:00 AM, next Monday Oct 27 at 8:00 AM'."""
    rule = rule_words(r)
    nxt = when(r.next_dt, now)
    if r.next_dt <= now:
        nxt = f"due now (was due {nxt})"
    return f'#{r.id} "{r.message}", ' + (f"{nxt}" if rule == "once" else f"{rule}, next {nxt}")


def _ago(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} h {minutes} min" if minutes else f"{hours} h"
    return f"{round(hours / 24)} days"


# ============================================================ the store
class ReminderStore:
    """Every reminder, saved to a JSON file after every change. Safe to use from the
    server's event loop and its worker threads at once."""

    def __init__(self, path: Path | str = DEFAULT_FILE, now: Callable[[], datetime] = datetime.now):
        self.path = Path(path)
        self.now = lambda: now().replace(microsecond=0)
        self.lock = threading.RLock()
        self.reminders: dict[int, Reminder] = {}
        self.next_id = 1
        # Reminders just told (newest last, one entry per id): what is on the screen, and
        # what was dismissed from it. {"id", "message", "told_at", "repeats", "dismissed"}
        self.recent: list[dict] = []
        self._load()

    # ---------------------------------------------------------------- saving
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.next_id = int(data.get("next_id", 1))
            for item in data.get("reminders", []):
                r = Reminder(**item)
                self.reminders[r.id] = r
            log.info("loaded %d reminders from %s", len(self.reminders), self.path)
        except Exception as e:  # a broken file must not stop the server; keep it for a person to look at
            bad = self.path.with_suffix(".bad.json")
            self.path.replace(bad)
            self.reminders, self.next_id = {}, 1
            log.warning("could not read %s (%s); moved it to %s and started empty", self.path, e, bad)

    def _save(self) -> None:
        """Write every reminder so that pulling the power at any moment leaves either the old
        file or the new one, complete, on the SD card: never half a file, never nothing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        data = {"next_id": self.next_id, "reminders": [asdict(r) for r in self.reminders.values()]}
        with open(tmp, "w") as f:
            f.write(json.dumps(data, indent=2))
            f.flush()
            os.fsync(f.fileno())  # the new contents are on the card, not just in memory
        tmp.replace(self.path)  # atomic: the name now points at the complete new file
        try:  # ... and the rename itself is on the card too
            fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass  # some filesystems can't sync a directory; the file itself is already safe

    # ---------------------------------------------------------------- times
    @staticmethod
    def _repeat_days(repeat: str, days: list[int]) -> list[int]:
        return _FIXED_DAYS.get(repeat, days)

    def _next_after(self, r: Reminder, after: datetime) -> datetime | None:
        """A repeating reminder's first time strictly after ``after`` (None for once)."""
        if r.repeat == "once":
            return None
        at = time.fromisoformat(r.at)
        days = self._repeat_days(r.repeat, r.days)
        for i in range(8):
            d = after.date() + timedelta(days=i)
            if d.weekday() in days:
                t = datetime.combine(d, at)
                if t > after:
                    return t
        raise AssertionError("a repeating reminder always has a time within a week")

    # ---------------------------------------------------------------- what the tools do
    def add(self, message: str, at: str | None = None, day: str | None = None, repeat: str = "once",
            days: str | None = None, in_minutes: float | None = None) -> tuple[Reminder, str]:
        """Make a reminder. Returns it and a note for Gemini ("" or e.g. that it moved to
        tomorrow). Raises ReminderError with the reason and what works."""
        with self.lock:
            now = self.now()
            message = " ".join(str(message or "").split())
            if not message:
                raise ReminderError("set_reminder needs a message: what should Pupper remind them about?")
            if len(message) > MAX_MESSAGE:
                raise ReminderError(f"the message is too long ({len(message)} characters; at most {MAX_MESSAGE})")
            repeat = (repeat or "once").strip().lower()
            if repeat not in REPEATS:
                raise ReminderError(f"repeat must be one of {REPEATS}, not {repeat!r}")
            if len(self.reminders) >= MAX_REMINDERS:
                raise ReminderError(f"there are already {MAX_REMINDERS} reminders, the most Pupper keeps;"
                                    " cancel some first (list_reminders shows them)")
            day = (day or "").strip() or None
            days_list: list[int] = []
            if days and str(days).strip():
                if repeat not in ("custom", "weekly"):
                    raise ReminderError(f"days is only for repeat='custom' (or 'weekly'), not repeat='{repeat}'")
                days_list = parse_days(days)
                if repeat == "weekly" and len(days_list) > 1:
                    repeat = "custom"
            elif repeat == "custom":
                raise ReminderError("repeat='custom' needs days, e.g. days='mon,thu'")
            note = ""

            if in_minutes is not None:
                if at or day:
                    raise ReminderError("give either in_minutes, or a time (and day), not both")
                if repeat != "once":
                    raise ReminderError("in_minutes is for one-time reminders; for a repeating one give a time and repeat")
                if not 0 < in_minutes <= MAX_IN_MINUTES:
                    raise ReminderError(f"in_minutes must be more than 0 and at most {MAX_IN_MINUTES} (a week);"
                                        " for later, give a day and a time")
                first = now + timedelta(seconds=round(in_minutes * 60))
                at_text = first.time().isoformat()
            else:
                if not at:
                    raise ReminderError(f"set_reminder needs a time ({TIME_FORMS}) or in_minutes."
                                        " If the person didn't say when, ask them")
                hour, minute = parse_time(at)
                at_text = f"{hour:02d}:{minute:02d}"
                if repeat == "once":
                    if day:
                        first = datetime.combine(parse_day(day, hour, minute, now), time(hour, minute))
                        if first <= now:
                            raise ReminderError(f"{when(first, now)} has already passed. {now_words(now)}"
                                                " Pick a later time or day")
                    else:
                        first = datetime.combine(now.date(), time(hour, minute))
                        if first <= now:
                            first += timedelta(days=1)
                            note = f"{clock(first)} has already passed today, so it is set for tomorrow."
                else:
                    start = parse_day(day, hour, minute, now) if day else now.date()
                    if repeat == "weekly" and not days_list:
                        # the weekday of its first time: the start day, or (no day given) today/tomorrow
                        first_day = start
                        if not day and datetime.combine(start, time(hour, minute)) <= now:
                            first_day = start + timedelta(days=1)
                        days_list = [first_day.weekday()]
                    probe = Reminder(0, message, repeat, at_text, days_list, "", "")
                    first = self._next_after(probe, max(now, datetime.combine(start, time.min) - timedelta(seconds=1)))
            if first > now + timedelta(days=MAX_AHEAD_DAYS):
                raise ReminderError(f"{when(first, now)} is more than a year away; Pupper keeps reminders up to a year ahead")

            stored_days = [] if repeat == "once" else days_list
            for r in self.reminders.values():  # the same reminder asked for twice
                if (r.message.lower() == message.lower() and r.repeat == repeat and r.at == at_text
                        and r.days == stored_days and r.next_dt == first):
                    return r, "That reminder was already set, so nothing new was added."
            r = Reminder(id=self.next_id, message=message, repeat=repeat, at=at_text, days=stored_days,
                         next_at=first.isoformat(), created_at=now.isoformat())
            self.reminders[r.id] = r
            self.next_id += 1
            self._save()
            log.info("set %s", describe(r, now))
            return r, note

    def list(self) -> list[Reminder]:
        with self.lock:
            return sorted(self.reminders.values(), key=lambda r: (r.next_at, r.id))

    def cancel(self, rid: int) -> Reminder:
        with self.lock:
            r = self.reminders.pop(rid, None)
            if r is None:
                listed = "; ".join(describe(x, self.now()) for x in self.list()) or "there are none"
                raise ReminderError(f"there is no reminder #{rid}. The reminders: {listed}")
            self._save()
            log.info("cancelled %s", describe(r, self.now()))
            return r

    # ---------------------------------------------------------------- delivery
    def claim(self, min_late_s: float = 0.0, lease_s: float = CLAIM_S) -> list[dict]:
        """Due reminders nobody else is telling, at least ``min_late_s`` overdue. Each is
        hidden from other claimers for ``lease_s``; confirm it with delivered()."""
        with self.lock:
            now = self.now()
            out = []
            for r in self.list():
                late = (now - r.next_dt).total_seconds()
                if late < min_late_s or (r.claimed_until and datetime.fromisoformat(r.claimed_until) > now):
                    continue
                r.claimed_until = (now + timedelta(seconds=lease_s)).isoformat()
                out.append(self.announcement(r, now, late))
            if out:
                self._save()
            return out

    def announcement(self, r: Reminder, now: datetime, late: float) -> dict:
        """What the web app tells Gemini (announce), and says itself if it must (speech)."""
        repeats = "" if r.repeat == "once" else f" It repeats {rule_words(r)}."
        lateness = ""
        if late >= 120:
            lateness = (f" It was due {when(r.next_dt, now)} ({_ago(late)} ago), while nobody could tell"
                        " the person; say that it is late.")
        announce = (f"[REMINDER #{r.id}] This message is from your reminder system, not from the person."
                    f" {now_words(now)} Tell the person this reminder out loud right now, warmly, in one or two"
                    f' short sentences: "{r.message}".{repeats}{lateness}')
        return {"id": r.id, "message": r.message, "due_at": r.next_at, "late_s": round(late),
                "repeat": rule_words(r), "announce": announce, "speech": f"Reminder: {r.message}"}

    def delivered(self, rid: int, how: str = "") -> Reminder | None:
        """The person was told: a one-time reminder is done, a repeating one moves to its
        next time after now. None if it was cancelled meanwhile."""
        with self.lock:
            r = self.reminders.get(rid)
            if r is None:
                return None
            now = self.now()
            r.delivered_count += 1
            r.last_delivered_at, r.last_delivered_how = now.isoformat(), how
            r.claimed_until = None
            nxt = self._next_after(r, now)
            if nxt is None:
                del self.reminders[rid]
                log.info('delivered #%d "%s" (%s); done', rid, r.message, how or "?")
            else:
                r.next_at = nxt.isoformat()
                log.info('delivered #%d "%s" (%s); next %s', rid, r.message, how or "?", when(nxt, now))
            self._save()
            # On the screen now. A repeating reminder told again replaces its old entry, so an
            # earlier dismissal never hides today's.
            self.recent = [e for e in self.recent if e["id"] != rid][-(RECENT - 1):]
            self.recent.append({"id": rid, "message": r.message, "told_at": now,
                                "repeats": "" if nxt is None else rule_words(r), "dismissed": False})
            return r

    # ---------------------------------------------------------------- the screen
    def on_screen(self) -> list[dict]:
        """Told reminders still showing: not dismissed, and told less than SCREEN_S ago."""
        with self.lock:
            now = self.now()
            return [e for e in self.recent
                    if not e["dismissed"] and (now - e["told_at"]).total_seconds() < SCREEN_S]

    def recently_told(self, rid: int) -> dict | None:
        with self.lock:
            return next((e for e in self.recent if e["id"] == rid), None)

    def dismiss(self, rid: int | None = None) -> list[dict]:
        """Take told reminders off the screen (one, or all showing). Returns those that were
        showing. Their future times are untouched: that is cancel's job."""
        with self.lock:
            showing = [e for e in self.on_screen() if rid is None or e["id"] == rid]
            for e in showing:
                e["dismissed"] = True
                log.info('dismissed #%d "%s" from the screen', e["id"], e["message"])
            return showing

    def dismissed_ids(self) -> list[int]:
        """Ids the web app should take off its screen (dismissed by voice or by another screen)."""
        with self.lock:
            return [e["id"] for e in self.recent if e["dismissed"]]


_store: ReminderStore | None = None


def get_store() -> ReminderStore:
    """The robot's reminders (made on first use, from DEFAULT_FILE)."""
    global _store
    if _store is None:
        _store = ReminderStore()
    return _store


def use_store(store: ReminderStore) -> None:
    """Use another store (the tests: a temporary file and a fake clock)."""
    global _store
    _store = store
