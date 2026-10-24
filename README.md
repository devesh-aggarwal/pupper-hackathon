<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Gemini Pupper (CS 123 Lab 6, finished)

Talk to Pupper, and it listens, looks and does what you ask. Gemini Live is the brain: a
foundation model that hears you, sees through the camera and talks back. Trained policies are
the muscles. In between are the tools: the only way Gemini can move the robot, and the only way
it learns what really happened.

This repo is a working copy of CS 123 Lab 6 (*Bringing Your Pupper to Life*) with every part
done: all 11 TODOs written, and the sit-stay-shake and spider tricks trained and installed. It is
also the starting point for our Gemma hackathon build (see [Hackathon plan](#hackathon-plan)).

**Gemini Pupper was created by [Peng Xu](https://scholar.google.com/citations?user=460NWeQAAAAJ&hl=en)
(Google DeepMind)**, original repo [sippeyxp/gemini-pupper](https://github.com/sippeyxp/gemini-pupper).
The CS 123 version is [cs123-stanford/gemini_pupper](https://github.com/cs123-stanford/gemini_pupper);
the lab handout is
[here](https://cs123-stanford.readthedocs.io/en/latest/schedule/labs/fall-26/lab-6.html).

## What it can do

By voice (or text, with `lab/chat.py`):

- **Walk, turn, run, sidestep and curve** (`move`, `turn`, `arc`), including shapes like squares
  and circles. Unsafe requests are refused with the limit that applies ("too fast: |vx|<=1.5 m/s").
- **Follow a person** (`follow`), using the Lab 5 follower in `~/follow_me_pupper` on the Pupper.
  Any motion ends follow mode, and Pupper says so first.
- **Play tricks** (`play_trick`): push-ups, swim, sneeze, twerk, downward dog and more.
- **Sit, stay and shake** (`sit_stay_shake`, trained here): sits and stays as long as you like,
  shakes either paw on request (`interactive_action`), and stands up when asked.
- **Be a spider** (`spider_walk`, trained here): lies down low with legs splayed, creeps around
  with `move`/`turn`/`arc` using its own walking policy, and wiggles back up to standing.
- **Remind you** (`set_reminder`, `list_reminders`, `dismiss_reminder`, `cancel_reminder`): "remind me to call mom
  tomorrow at 5", "in 20 minutes, take the pizza out", "every weekday at 8 am, take your pills",
  "trash goes out Monday and Thursday at 7 pm". At that time Pupper says it out loud and shows it
  on screen. See [Reminders](#reminders).
- **Report honestly**: every tool reply says what was queued or why it was refused, and
  `robot_status` reports what really happened.

And, for the hackathon (details in [Hackathon features](#hackathon-features)):

- **Remember people** (`remember`, `recall`, `forget`, `switch_profile`): who it is talking with,
  their facts, where their things were last seen. Survives a restart; one-click reset.
- **Hearing for people who can't**: a loud sound (knock, doorbell, crash) flashes the screen
  LOUD SOUND, Pupper says and shows what it was (`show_text`), and everything Pupper says is
  captioned on its face.
- **Not bumping into things**: an on-device object detector sees a person or chair close ahead,
  and the move is refused or stopped early, on the robot. A depth sense (`vision/eyes.py`) does the
  same for anything low it can't name: chair legs, cables, shoes.
- **Walking to things by sight** (`go_to`): "come here", "go to the red mug" names a goal; the eyes
  find it in the camera and the robot steers to it, turn by turn, and stops at arm's length.
- **Understanding the picture** (`look`, `what_is_around`): one question to a local Gemma vision
  model about the frame, and a scene description it keeps up while Pupper is idle.
- **Finding things** (`find`): "where is my cup?", "look around for the whiteboard with code on it":
  Pupper looks (its detector, Gemma, its memory), turns a full circle looking, then roams the room
  in legs toward open floor it hasn't been on, and walks up to the thing when it finds it.
- **Learning on the fly** (`learn_routine`, `do_routine`): "learn a happy dance: a push-up, then
  spin around" records real tool calls as a named routine, no retraining.
- **A local brain** (`robot/gemma_brain.py`): Gemma on a Mac (`./run_pupper.sh` there) or a
  Jetson, in place of Gemini Live: speech to text, Gemma, the same tools, text to speech, no
  cloud. See [The local brain on a Mac](#the-local-brain-on-a-mac) and
  [on a Jetson](#the-local-brain-gemma-on-a-jetson-by-ethernet).

## What's where

```
.
├── lab/                 the Lab 6 code (all TODOs done)
│   ├── prompt.md        Pupper's personality and rules
│   ├── basics.py        stand_up, lie_down, stop, robot_status
│   ├── part2_walk.py    check_move, walk, turn, move, arc
│   ├── part3_follow.py  prepare_to_move, follow
│   ├── part4_tricks.py  play_trick, play_policy_trick
│   ├── part5_interactive_action.py  interactive_action
│   ├── hackathon_memory.py     remember, recall, forget, switch_profile (+ the memory file and CLI)
│   ├── hackathon_alerts.py     the event bus, show_text, sound alerts
│   ├── reminders.py            set_reminder, list_reminders, dismiss_reminder, cancel_reminder (robot/reminder_store.py)
│   ├── hackathon_vision.py     the obstacle rules (detector + depth), the follow guard, found_object
│   ├── hackathon_learn.py      learn_routine, finish_learning, do_routine
│   ├── hackathon_eyes.py       go_to, look, what_is_around (steering on the eyes service's target)
│   ├── hackathon_explore.py    find: look, sweep in place, roam in legs (a smart Roomba, no map), walk up to it
│   ├── hackathon_log.py        the session log: tool calls, events, speech, frames (~/pupper-logs)
│   ├── test.py          python3 lab/test.py      (tests on a fake robot: the lab's and the hackathon's)
│   └── chat.py          python3 lab/chat.py      (talk to Gemini in text)
├── jetson/              the local brain's setup: setup.sh (the Jetson), setup_pi.sh (the Pi), link.sh (the cable)
├── vision/              the eyes service: eyes.py (depth + Gemma, beside the robot server), test_eyes.py
├── tricks/              one folder per trick (policy.json + trick.yaml)
│   ├── sit_stay_shake/  trained here; covers sit and stand_sit_shake_sit_stand, so it hides them
│   └── spider_walk/     trained here: clip policy + walk_policy.json; covers spider, so it hides it
├── robot/               the Robot API, robot server, launch files, setup/start/stop scripts
│                        (test_tools.py: the lab's tests; test_hackathon.py: the hackathon's)
│                        gemma_brain.py: the local brain (Gemma on the Jetson); test_gemma_brain.py
├── web/                 the web app (face + debug view, Gemini Live client; services/soundMonitor.ts,
│                        objectDetector.ts and robotEvents.ts are the hackathon's senses;
│                        localBrain.ts is the face for the local brain, ?brain=local)
├── notebooks/           Meet the Loop, plus training/ (the code used to train the tricks)
├── docs/                architecture notes and Peng Xu's write-up
├── run_pupper.sh        one command: on the Pupper, the kiosk (Gemini); on the Mac, the local brain
└── tools/               hackathon dev tools: ./tools/pup, brain_on_mac.sh (the local brain on a Mac),
                         brain_bench.py (which Gemma is the best brain), display examples
```

Each file in `lab/` has two halves: first **what the robot does** (check the request, queue a
job, report what happened), then **what Gemini reads** (the `@tool` name, description and
parameters). The Robot API (`robot.queue`, `robot.switch_to`, ...) is documented at the top of
`robot/robot_api.py`.

## Setup on the Pupper (once)

Clone this whole repo onto the Pupper and run the setup script from its root:

```bash
git clone https://github.com/codergautam/pupper-hackathon-setup.git ~/pupper-hackathon-setup
cd ~/pupper-hackathon-setup
robot/setup.sh
```

`setup.sh` installs Node in your home folder, builds the web app, checks the Python packages and
asks for your Gemini API key (from [Google AI Studio](https://aistudio.google.com/apikey)). The
key is saved in `~/.config/gemini-pupper/api_key`, never in this repo. It is safe to run again.

Keep `~/follow_me_pupper` (Lab 5) on the Pupper: follow mode runs it. The ROS 2 workspace that
`start.sh` sources is `~/pupperv3-monorepo`.

To update the Pupper later: `git pull` in `~/pupper-hackathon-setup` (or `./tools/pup sync` from a
laptop), then restart.

## Running

**With the local brain (no cloud):** on the Mac, `./run_pupper.sh`. That's all; see
[The local brain on a Mac](#the-local-brain-on-a-mac).

**With Gemini Live:** on the Pupper, on the ground, with clear space around it, one command:

```bash
./run_pupper.sh           # kiosk on Pupper's screen, volume 30
./run_pupper.sh 50        # speaker volume 0-100
./run_pupper.sh --laptop  # no kiosk: open the app from your laptop through the SSH tunnel
```

It does every setup step for you: puts Node on the PATH, rebuilds the web app if a
source file is newer than the build (or runs `robot/setup.sh` if there is no Node yet), installs
missing Python packages, checks the mic (a warning, not a stop), sets the volume, prints who
Pupper remembers, starts everything, prints things to try, and then shows every tool call, event
(reminders, sound alerts, search results) and outcome live. Ctrl+C stops everything (`robot/stop.sh`).

Or run the pieces yourself:

```bash
robot/start.sh --kiosk    # robot + the app on Pupper's own screen (its mic and speaker)
robot/start.sh            # robot only; open the app from your laptop (below)
robot/stop.sh             # relax the motors and stop everything (also if Pupper overheats)
```

Wait for `[3/3] ready`. From your laptop: `ssh -N -L 8000:localhost:8000 pi@pupper5.local`, open
http://localhost:8000, open **Settings**, paste your key and press **WAKE PUPPER**. The debug view
shows every tool call and its result.

The e-stop is the gamepad's right stick (press it): it relaxes the motors, whatever Gemini is
doing.

Things to try: "sit, and give me your paw", "other paw", "stand up", "be a spider, walk forward a
bit, turn around, now stand up", "run as a spider" (refused: the spider can't run), "walk in a
square", "follow me".

Debugging: `tail -f /tmp/gemini-pupper/server.log` (your `lab/` code) and
`/tmp/gemini-pupper/robot.log` (ROS 2); `curl localhost:8000/system/mode` shows the real state.

## Testing without the robot

```bash
python3 -m pip install --user google-genai pyyaml
python3 lab/test.py            # all 559 checks, on a fake robot: the lab's 258 (with the reminders), the hackathon's 232, then the local brain's 69
python3 lab/test.py todo9      # one TODO
python3 lab/test.py reminders  # the reminders (robot/test_reminders.py)
python3 lab/test.py hackathon  # memory, screen, sound, vision, eyes, find (explore), guard, learning (or: hackathon explore)
python3 lab/test.py brain      # the local brain with a fake Ollama: tools, fallbacks, the loop, the ears (or: brain voice)
python3 vision/test_eyes.py    # the depth sense on synthetic floors, chair legs, cables and walls (needs numpy)
python3 lab/chat.py            # your prompt and tools, on a fake robot: nothing moves
python3 lab/chat.py --show     # print every tool description as Gemini reads it, then chat
python3 lab/chat.py --robot    # your tools on the real Pupper (robot server running): it moves!
```

`chat.py` needs a Gemini key in `GEMINI_API_KEY` or `~/.config/gemini-pupper/api_key`. In text
mode it uses your real memory file (set `PUPPER_MEMORY_FILE` for a scratch one), fires due
reminders when you press Enter, and shows the events Gemini gets.

### The whole app on your laptop (voice, face, reminders), with a fake robot

```bash
./run_local.sh                 # then open http://localhost:8000
PORT=8080 ./run_local.sh       # if 8000 is taken
./run_local.sh --build         # force a rebuild of the web app
```

This runs the real robot server, your `lab/` tools and prompt, the reminders and the web app on
your laptop; only ROS is replaced by the fake robot from the tests, so **nothing moves**: every
motion is printed instead (`[fake robot] legs: gemini_walk took over`, `[fake robot] walking
vx=0.4 ...`). Open the page, go to **Settings**, paste your Gemini key, press **WAKE PUPPER**, and
talk through your laptop's mic (allow mic and camera when Chrome asks). It builds the web app
first if needed (needs Node.js), and needs `fastapi uvicorn psutil pydantic pyyaml`; on NixOS,
without python3/npm, it reruns itself in a `nix-shell` that has them. Reminders made here go to
`~/.config/gemini-pupper/reminders-laptop.json`, separate from the Pupper's.

Testing reminders on the laptop:

- **With Gemini:** say "remind me in one minute to drink water". About a minute later Pupper
  says it, the banner shows, and the server prints `reminders INFO ... delivered (gemini)`.
- **Without a key** (the app open, not connected): set one by hand, then after it is a minute
  overdue the browser says it with its own voice and shows the banner:
  `curl -X POST localhost:8000/tools/set_reminder -H 'content-type: application/json' -d '{"message":"drink water","in_minutes":0.25}'`
- **With no app open:** the server says it with `espeak-ng` or `spd-say` after 90 s, if one is installed.
- `curl localhost:8000/reminders` lists them.

| You changed | Then |
| --- | --- |
| `lab/prompt.md` | try it in text, then reconnect in the app (no restart) |
| a `.py` file in `lab/` | `python3 lab/test.py`, then `robot/stop.sh && robot/start.sh` |
| a trick (`get_trick.py` or a `trick.yaml`) | `robot/stop.sh && robot/start.sh` |

## Reminders

Ask Pupper to remind you of something, once or repeating, and it tells you out loud at that
time. Nothing about it moves the robot, so it never ends follow mode.

- **Setting one.** Gemini calls `set_reminder` with the message, a time (`"17:30"`, `"5:30 pm"`)
  and a day in words (`"tomorrow"`, `"friday"`, `"october 30"`), or `in_minutes`, and how often:
  `once`, `daily`, `weekdays`, `weekends`, `weekly`, or `custom` with `days="mon,thu"`. The robot
  works out the date from its own clock, so Gemini never needs today's date, and the reply says
  exactly when it goes off ("next Monday Oct 26 at 8:00 AM. It is now Friday Oct 23, 10:26 PM.")
  so Pupper can confirm it. A time that already passed today moves to tomorrow (and says so); an
  unreadable or impossible request is refused with what works. If you don't say when, or AM/PM is
  unclear, the prompt has Gemini ask first. `list_reminders` and `cancel_reminder` (by number)
  manage them; to change one, cancel it and set a new one.
- **Saved** in `~/.config/gemini-pupper/reminders.json` on the Pupper (not in this repo; set
  `PUPPER_REMINDERS_FILE` to move it), so reminders survive restarts. At most 50, up to a year ahead.
- **Being told.** The web app asks the robot every 5 s for due reminders
  (`POST /reminders/claim`). If Gemini is connected, it sends Gemini a `[REMINDER #n] ...` message
  and Pupper says it in its own voice; it also shows the reminder on screen (tap anywhere on the
  screen, or say "got it": Gemini calls `dismiss_reminder`, which clears it without cancelling
  future repeats; it also clears by itself after 2 minutes), then
  confirms (`POST /reminders/<id>/delivered`). A one-time reminder is then done; a repeating one
  moves to its next time. Repeats missed while the robot was off are told once, late, and Pupper
  says it's late.
- **So it is never lost.** A claimed reminder that isn't confirmed within a minute (tab closed,
  session dropped) is due again. If the app isn't connected to Gemini, it waits a minute for the
  session to come back, then says it with the browser's own speech. If no app is open at all, the
  robot server says it with `espeak-ng` after 90 s, if installed (`sudo apt install espeak-ng`);
  otherwise it waits for the app. Watch it in `/tmp/gemini-pupper/server.log` (lines from
  `reminders`, also shown live by `run_pupper.sh`), or `curl localhost:8000/reminders`.
- **Before using it on the Pupper**, check its clock and timezone: `timedatectl` (fix with
  `sudo timedatectl set-timezone America/Los_Angeles`). With no internet the Pi can't sync its
  clock (NTP), so check the time after a power cycle at the hackathon. Every reminder reply says
  the time the robot thinks it is, so a wrong clock shows up right away.

| You changed | Then |
| --- | --- |
| `lab/reminders.py` or `robot/reminder_store.py` | `python3 lab/test.py reminders`, then restart |
| the web app (`web/`) | `robot/setup.sh` (rebuilds `web/dist/`), then restart |

## Tricks and how they were trained

Every trick is a policy, so the robot is always in one simple state: locomotion, one trick, motors
idle, or fallen. Each folder under `tricks/` with a `trick.yaml` is a trick Gemini can play
(`robot/tricks.py` lists the fields). A playback trick plays one motion once; an interactive
trick is a state Pupper stays in, with `actions` Gemini can request and an `exit_clip` that
stands back up. A `replaced_by` field hides a trick while another installed trick covers it.

`sit_stay_shake` and `spider_walk` were trained in Colab on a G4 GPU with
[pupper-mjlab](https://github.com/cs123-stanford/pupper-mjlab) (the code is in
`notebooks/training/`) and fetched from W&B with `robot/get_trick.py`:

| Trick | Policy | Physics check | W&B run |
| --- | --- | --- | --- |
| `sit_stay_shake` | tracking policy on 5 clips: `sit_down`, `sit_idle` (loops), `shake_left`, `shake_right` (the left shake mirrored), `stand_up` | plain replay falls; policy stays up (worst 3.7 cm / 3.2 deg) | `nru36bv1` |
| `spider_walk` clips | tracking policy on `spider_down`, `spider_idle` (loops), `spider_up` (`spider_down` reversed) | policy stays up (worst 4.0 cm / 6.2 deg) | `lkio9x6h` |
| `spider_walk` walking | `Mjlab-SpiderWalk-Flat-Pupper-v3`, body at 4.2 cm; capped at 0.12 m/s forward, 0.08 sideways, 25 deg/s | holds height, knees off the ground | `a0tpt3sl` |

New trick policies should get a TA's OK (score + simulation viewer) before their first run on
the robot. To fetch a different run:

```bash
python3 robot/get_trick.py <entity>/mjlab/<run-id> --name sit_stay_shake
python3 robot/get_trick.py <clip-run> --walk <spider-walk-run> --name spider_walk
```

The Pupper must be logged into W&B (`wandb login`) with the account the runs are on.

## Connecting to the Pupper

- Host: `pupper5.local` (hostname `pupper5`, Raspberry Pi 5, Debian Bookworm aarch64)
- User: `pi` (the password is the one set when the Pupper was imaged; ask your TA)

```sh
ssh pi@pupper5.local
```

The laptop and the Pupper must be on the same network for `.local` (mDNS) to resolve. To skip the
password from then on, set up key-based login once: `ssh-copy-id pi@pupper5.local` (or
`./tools/pup setup`).

## `./tools/pup`: quick testing on the Pupper

`./tools/pup` runs commands on the Pupper, copies files, and launches scripts, including GUI apps
on its 720×720 screen. Run `./tools/pup setup` once per laptop: it creates `~/.ssh/pupper_ed25519`
and installs it on the Pi. The user `pi` has passwordless `sudo`.

```sh
./tools/pup setup                          # one-time key setup (asks for the Pupper's password)
./tools/pup                                # interactive shell
./tools/pup sh 'free -h; uptime'           # run any command
./tools/pup run my_test.py arg1            # copy a local script to /tmp/pup on the Pi and run it
./tools/pup gui tools/examples/hello_pygame.py   # run it on Pupper's display
./tools/pup sync                           # rsync this repo to ~/pupper-hackathon-setup on the Pi
./tools/pup put ./file ~/dest  /  ./tools/pup get ~/file .
./tools/pup kill hello_pygame              # stop a runaway process
```

`tools/examples/` has display demos: `hello_pygame.py` (a 10-second smoke test) and
`crying_eyes.py`. Set `PUPPER_HOST` to target a different Pupper.

## Hackathon features

Everything below works with the Lab 6 architecture unchanged: new tools in `lab/`, a few
endpoints on the robot server, and the web app's senses. None of it depends on which brain
calls the tools, so it carries over when Gemini Live is swapped for a local Gemma.

**Events.** A reminder comes due, the mic hears a bang, a search finds the cup,
a walk stops early: the robot puts an *event* on a bus (`lab/hackathon_alerts.py`, `EVENTS`). The
web app polls `GET /events?after=<id>` once a second, paints the event's `display` on Pupper's
screen (even while Pupper sleeps), and hands its `for_gemini` text to Gemini as a turn, so Pupper
reacts without anyone speaking first. Those texts start with `[reminder]`, `[sound alert]`,
`[search result]` or `[robot]`, and the prompt tells Gemini they come from its own body, not the
person. `lab/chat.py` shows them in text mode. So a job's outcome reaches Gemini, not only the debug
log: a search reports back.

**Memory** (`lab/hackathon_memory.py`). A small JSON file on the robot,
`~/.config/gemini-pupper/memory.json` (or `$PUPPER_MEMORY_FILE`), never in the repo: one
profile per person (facts, where their things were last seen), the active profile,
learned routines, settings. The robot server appends "What you remember" to the prompt at
connect, so a new session starts knowing the person; during a session Gemini uses `remember`,
`recall`, `forget` (`what='everything'` clears the person; the prompt makes Gemini ask first) and
`switch_profile` ("I'm Devesh"). Reset, three ways: by voice, the **Memory** card in the app's
debug view (switch person, reset this person, reset all), or on the robot:

```bash
python3 lab/hackathon_memory.py show      # or: summary
python3 lab/hackathon_memory.py reset     # the active person;  reset all  wipes everything
python3 lab/hackathon_memory.py switch Devesh
```

**Reminders** (`lab/reminders.py`, `robot/reminder_store.py`).
`set_reminder` takes a day in words and a time ("tomorrow at 7:30", "in 20 minutes", "every weekday
at 9"), once or repeating; `list_reminders` numbers them; `cancel_reminder` removes one;
`dismiss_reminder` clears one from the screen when the person says "got it". The robot keeps them
across restarts. The web app asks the robot for due ones every few seconds: connected, Gemini is
told and says it in Pupper's voice; not connected, the browser's own speech says it after a minute.
Either way it stays on the screen until someone taps anywhere, says "got it", or two minutes pass,
and the robot is told it was delivered. `python3 lab/test.py reminders` tests them.

**The screen, captions and hearing** (`lab/hackathon_alerts.py`, `web/services/soundMonitor.ts`).
`show_text(text, detail, color, seconds)` shows a card at the top of the face (red, orange, yellow,
green, blue, purple, white, black); reminders are a white card too, and a tap clears either. Only
sound alerts take over the whole screen. The face blinks and changes emotion by itself (sleepy when
not connected, listening, happy while talking, excited when a search finds something, surprised at
a sound alert, sad on a fault). **Demo without a robot:** `cd web && npm run dev`, then open
`https://localhost:5173/?demo` for a scripted loop of all of these (`web/services/demo.ts`; nothing
moves, no Gemini). With **Captions** on (Settings, default on) everything Pupper says is also
written on its face (Gemini Live's output transcription). With **Sound alerts** on
(default on), the browser measures the mic (dB above a slowly adapting background; Pupper's own
voice is ignored) and posts loud moments to `POST /alerts/sound`; the robot decides: alerts off
(`sound_alerts(enabled=false)`, "stop listening"), under the sensitivity (`low`/`medium`/`high`
= +25/+18/+12 dB, kept in memory), or Pupper was moving (its motors), and nothing happens;
otherwise the screen flashes LOUD SOUND (orange; VERY LOUD SOUND red), and Gemini is asked to say
what it most likely was and to `show_text` a short label (DOORBELL). One alert at a time.

**Seeing things** (`lab/hackathon_vision.py`, `web/services/objectDetector.ts`). With **On-device
object detection** on (default on), the browser runs MediaPipe's EfficientDet-Lite0 (80 everyday
classes: person, chair, cup, bottle, cell phone, laptop, sports ball, backpack ...) on the camera
frames it already fetches for Gemini and posts `{label, score, box}` to `POST /vision/detections`.
The model (about 4 MB) downloads from Google the first time and is cached; without internet, or
with the box unticked, the rule below stays out of the way and Gemini's own eyes are what there
is. Two behaviors use it:

- *The obstacle rule.* `walk` (so `move`, `turn` and `arc`) asks `path_blocked(vx, vy)`: something
  close (its box is tall, or sits at the bottom of the frame) in the direction of the move refuses
  it ("blocked: a chair is right in front of me, close; I won't walk into it ..."). Something that
  steps in mid-walk on a forward walk or arc (no sideways part) is gone round: `detour` in
  `lab/part2_walk.py` turns toward the more open side (`clearer_side`) until the way ahead is
  clear, at most 90 degrees, walks 0.5 m past it, turns back by the same angle and carries on, up
  to 3 times a move (`MAX_DETOURS`, `DETOUR_MAX_DEG`, `PASS_M`); the time spent going round doesn't
  count against the move. Every step of it runs the same checks, and when there is no way round
  (both sides blocked, no fresh depth sense, or still blocked after the turn) the walk stops early
  and tells Gemini, as does anything in the way of any other move. The camera can't see beside the
  robot, so 0.5 m is a guess at "past it". Turning in place and backing away are always fine. The prompt adds the soft version: look at the camera before walking forward.
- *Finding things* is one tool, `find` (`lab/hackathon_explore.py`, below); `found_object(name,
  where)` remembers where a thing is when Gemini sees it in the camera or a person says.

**The eyes service: depth, go_to, look** (`vision/eyes.py`, `lab/hackathon_eyes.py`). The detector
above knows 80 classes and nothing about distance, so on its own it cannot see chair legs, cables or
shoes. The eyes service is a second pair of eyes that runs beside the robot server, on a laptop on
the same network (fast) or on the Pi (slower). A few times a second it fetches the camera frame the
server already serves (`GET /camera/image`), runs a monocular depth model on it (Depth Anything V2
Small, ONNX, CPU) and posts a *free-space summary* to `POST /vision/freespace`: for five strips of
the view cut by bearing (within 15 degrees of straight on, 15 to 45, and 45 to 80 on each side),
where the nearest thing standing on the floor meets the floor. The depth model's output is relative
(an unknown scale and offset), so the distance comes from geometry, not from the model's numbers: on
an empty floor the closeness falls steadily going up the image, and anything standing on the floor
is a run of rows with about the same closeness (a vertical surface) sitting on that curve. In each
strip the eyes service finds the lowest such run, the nearest surface (a near leg in front of a far
wall is the leg), and posts where its base is.

The rule lives in `lab/hackathon_vision.py`, next to the detector's: `path_blocked` refuses or stops
a move when the depth sense shows something in its way, naming it if the detector can ("a chair"),
otherwise "something low". It is class-agnostic by design: no list of things to avoid. "In the way"
is measured the way Lab 5 measures distance: the base of the nearest surface in a strip, through the
calibrated lens, as a row of the undistorted image (`nearness`; `lab/fisheye_geom.py`), and a base
at row 0.68 or lower blocks (`BLOCK_NEARNESS`). On real frames, standing: a cable hanging in front
0.73, a backpack at the feet 0.70 to 0.73, a chair at arm's length 0.70, a bottle a metre off 0.58,
chairs a metre off 0.54 to 0.56. Lying down the camera is lower and everything reads farther (the
same backpack 0.63), which is one more reason walks only happen standing. Things lower than a few
centimetres (a cable lying flat) are below what it can tell from floor. Each strip also carries a
rough `closest_m` (camera height / tan(elevation), with the height the robot server sends for its
state: lying 0.12 m, standing 0.22 m, `CAM_HEIGHT_*` in `lab/hackathon_vision.py`, not measured);
the fake robot's tests post that form, checked against `STOP_M`.

Three tools use the eyes service's other half: the same 80-class COCO detector the web app runs
(EfficientDet-Lite0 through MediaPipe, 15 ms a frame) and a local **Gemma** vision model in Ollama:

- `go_to(what)`: the brain names a goal in plain words ("the chair", "the red mug", "Devesh"). The
  robot server tells the eyes service what is wanted (`GET /vision/wanted`). A thing with a COCO class
  (person, chair, backpack, cup, bottle, laptop, ball ...; look-alikes count, a backpack on the floor
  reads as "suitcase") gets the detector's box every frame, nearest first (`POST /vision/target`);
  the detector comes first because Gemma's pointing is unreliable when the thing is big and close
  (it lands on a neighbouring chair). Anything else, or a classed thing the detector cannot find for
  3 s (it misses a dark backpack at some angles), goes to Gemma: first "is there a X in the picture,
  yes or no", then its centre point, posted once two fixes in a row agree (every 2-3 s). A job on the
  robot, shaped like `walk`, reads the latest fix every tick (20 Hz), aims at it and dead-reckons the
  aim from the turns it sends between fixes, walks when roughly facing it, slows down near it, and
  stops when the thing's base is within arm's length on several ticks and two different fixes, when
  the thing blocks the way itself, when it loses sight of it for 6 s, when something else is in the
  way (both obstacle rules run inside the loop: that is a stop, not an arrival), or at the move
  limits (18 s, 7 m). The range is Lab 5's, with Lab 5's calibration (`lab/fisheye_geom.py`): the
  lens is the double sphere model from `follow_me_pupper/robot/camera_params.yaml`, every pixel maps
  to a latitude and longitude as in Lab 5's undistorted 700 x 572 equirectangular image, and the row
  of a thing's bottom edge, where it meets the floor, is its distance: lower in the image is nearer.
  Bearings are longitudes. Arrival is Lab 5's `STOP_BOTTOM` 454 of 572 (row 0.794, the follower's
  polite distance at a person's feet, measured on the robot); "about there", for a target that
  blocks the way itself, is the row that blocks a walk (0.68); slowing starts `SLOWDOWN_PX` before
  the stop. Only the thing's own bottom edge can say "arrived"; the depth strips measure the nearest
  surface in a direction, a chair leg as often as the thing. A model that only pointed (an
  implausibly small box) gives a bearing and no range, and such a thing is reached only when it is
  dead ahead and the way is blocked. Between fixes the row is pushed down the image by the walking
  since the fix (dead reckoning), as the aim is by the turning. No metres anywhere in the rules: the
  camera's pitch and the lens are inside the calibration, and a level-camera formula (height /
  tan(elevation)) does not work here, since a 180-degree fisheye squeezes everything beyond a metre
  into a few percent of the frame below the horizon. The result comes back as a `[robot]` event, so
  the brain hears it, and the place is remembered. The brain never sets a speed: a language model
  decides about once a second, the legs need a speed twenty times a second, and `cmd_vel_mux` drops
  a source that goes quiet for half a second.
- `look(question)`: one question about the current frame ("what is on the floor ahead?", "is anyone
  waving?"), answered by Gemma through the eyes service's own port (`POST /look`, default 8100).
- `what_is_around`: the scene description the eyes service posts every 15 s while nothing is wanted
  (`POST /vision/scene`), plus the detector and the depth sense; free to call before planning a walk.

The eyes service also posts its detections on the browser's endpoint (`POST /vision/detections`),
so `find`'s looks and `what_is_around` work without the web app. Its Gemma prompts name no example
things (a prompt that lists "chair legs, cables, shoes" makes Gemma report exactly those in every
picture), ask for what is really there, and ask it to say when the floor is not visible. Asked for
a box, Gemma draws one for almost anything, while asked yes or no it is reliable; so the eyes
service asks whether the thing is there before asking where, and posts a box only when two fixes in
a row agree.

Running it, from a laptop (one tunnel both ways: the frames come down, `/look` goes up):

```bash
python3 -m pip install -r vision/requirements.txt          # numpy, pillow, onnxruntime
ollama pull gemma3:4b                                        # once; a vision model, on-device
ssh -N -L 8000:localhost:8000 -R 8100:localhost:8100 pi@pupper5.local
python3 vision/eyes.py --fetch-depth-model                   # first time: downloads the ~99 MB model
python3 vision/eyes.py                                       # then: depth at 5 Hz, Gemma on demand
python3 vision/eyes.py --no-gemma                            # depth only (no Ollama)
curl localhost:8100/health                                   # what it sees, and how fast
```

On the Pi itself the same commands work (`--advertise http://127.0.0.1:8100` is the default), with
Ollama installed there or `--no-gemma`. Without the eyes service nothing changes: the depth rule
stays out of the way, `go_to` falls back to the COCO detector, and `look` says it has no eyes.

Speeds: depth takes about 60 ms a frame at 252 px on a laptop (Apple Silicon CPU), so 5 Hz is easy,
and 320-350 ms on the Pi 5. `gemma3:4b` through Ollama takes 1.6 s for a yes/no question and 2.2 s
for a box or a description.

To check the threshold on a new robot or room, grab a frame with
`curl localhost:8000/camera/image -o frame.jpg` and run `python3 vision/eyes.py --image frame.jpg`:
it prints each strip's row and whether it blocks. An empty floor should stay clear in every strip
and a chair leg 30-40 cm ahead should block (`BLOCK_NEARNESS` in `lab/hackathon_vision.py`;
`DEPTH_INPUT`, `NEAR_RATIO` in `vision/eyes.py`). The camera is a 180-degree fisheye that draws a
bright disc in the middle of a black frame: the eyes service finds the disc (`find_circle`), crops
to it, ignores the black border, cuts its strips by bearing (at -45, -15, +15 and +45 degrees, so
"ahead" is what a forward step can hit) and posts the disc's centre and radius with the free space,
and `go_to` measures bearings from that disc (`HFOV_DEG`), so a box at the disc's edge is 90 degrees
off. The lens's UV layer also leaves the camera's auto white balance out of range (pink frames):
`robot/camera_setup.sh`, run by `start.sh` once the stack is up, sets Lab 5's fixed gains on the
camera node (red 1.4, blue 3.3; `auto` to undo), so Gemini, the browser's detector and the eyes
service all get the same colors; the eyes service also applies a gray-world correction of its own,
only when a frame is still pink. Nothing here adds a new policy, so there is nothing to replay in
pupper-mjlab: `go_to` only sends speeds the locomotion policy already walks at, within
`check_move`'s limits, and the desk tests (`python3 lab/test.py hackathon eyes`) drive it against a
kinematic stand-in for the camera (a thing at a bearing and a distance that every speed command
moves) to check it turns the right way, converges, slows, stops at arm's length, gives up when it
loses the thing, and obeys the obstacle rules mid-walk. Test it on the ground with clear space and a
hand near the e-stop; the first `go_to` should be to something soft, a metre away.

**The eyes service must be running, and the Pi alone is slow at it.** The depth sense, `go_to`,
`find` and the follow guard below are only as alive as the eyes service: without it the obstacle
rule has nothing to see and `look` fails. So `robot/start.sh` starts `vision/eyes.py` on the Pi
itself (`run_pupper.sh` installs its packages the first time; log `/tmp/gemini-pupper/eyes.log`;
`curl localhost:8100/health` shows it; `stop.sh` stops it; `PUPPER_EYES=off` skips it). With the
whole stack up and busy the Pi 5 can reach 84 °C and throttle to 1.5 GHz, and a depth frame then
takes about a second, so the Pi's eyes run at 1 Hz and 224 px (the smallest size that still finds a
backpack on the floor; 196 does not) and are a fallback, not a sense to walk fast on. **For the
demo, run the eyes on a laptop** (5 Hz, the commands above) or on the Jetson; its posts take
precedence. Every post carries its frame's age, so a slow poster can never make old data look
fresh: the robot trusts a summary for 2.5 s from its frame, and stale means blind, never wrong.
A laptop's eyes with Gemma also serve `look` and the non-detector targets; the Pi's posts no
`/look` address, so the laptop's stays known to the robot.

Two things keep the Pi cool without changing what the robot does. *Attention gating:* the eyes
service only runs at full rate while the robot needs eyes (a motion running or queued, follow
mode on, or any tool called in the last 10 s: the robot server says so in every reply and in
`GET /vision`); idle, it takes one frame every 5 s; and while another eyes service (a laptop's,
the Jetson's) is posting, it does nothing at all. It polls that cheap status every half second, so
it is awake within half a second of a tool call, before the legs move; the first check of a move
asked for while idle may find the summary stale (blind, never wrong), and the walk re-checks every
tick once the eyes are up. For the same reason `find`'s pre-check warns that the depth sense is not
posting only when no eyes service has posted for 15 s. `--always` turns the gating off. *The
camera's frame rate:* `robot/camera_setup.sh` (which also sets the white balance) asks the camera
node for 10 frames a second instead of the sensor's full rate, since nothing reads more than a few;
encoding 1400 x 1050 JPEGs at full rate costs the Pi half a core. With both, robot lying still: the
camera node uses about 20 % of a core, the eyes are idle (one frame every 5 s) and out of the top
CPU users, and the Pi runs at its full 2.4 GHz, unthrottled. A tool call puts the Pi's eyes in the
full state within 1.5 s and back to idle 10 s later; a laptop's eyes coming up put them in standby
within 6 s; stopping the laptop's eyes brings them back to idle within 5 s. `curl
localhost:8100/health` shows the eyes' state and `top` the rest.

Until the Jetson is here, the Pi's eyes service can use a model through Google's API instead
(`--gemma-api`, with the Gemini key in `~/.config/gemini-pupper/api_key`; `start.sh` does this when
the key file exists, `PUPPER_EYES_GEMMA=off` for depth and detector only). This is cloud, so it is
a stand-in for Gemma on the Jetson, not the final build, and the rest of the robot cannot tell the
difference. From the Pi on a real frame, the API's Gemma 4 (`gemma-4-26b-a4b-it`) thinks before
every answer, cannot be told not to, and takes 15-20 s, too slow for the eyes;
`gemini-3.5-flash-lite` answers "is there a backpack" in 1 s and places it in 1 s, so it is the
default (`--gemma-api-model` picks another).

**The follow guard** (`lab/hackathon_vision.py`, `follow_guard_loop`). Follow mode is the Lab 5
person follower, a ROS node that drives the legs itself and never asks the robot server, so the
rules in `walk` don't reach it. The robot server runs a guard ten times a second: while following,
something in the way of a forward walk (the depth sense or the detector) switches the follower off
and tells Gemini it paused; once the way has been clear for a second it switches it back on and says
so; a stop, another motion, `follow(enabled=false)` or 20 s of blockage end it for good. Tests:
`python3 lab/test.py hackathon guard`.

**The session log** (`lab/hackathon_log.py`). Answers "did the tool even run?": every tool call and
its reply, every event (an obstacle stop, a search result, a reminder, a sound alert), and both
sides of the conversation (the web app posts Gemini's transcriptions of the person and of Pupper to
`POST /transcript`) go to `~/pupper-logs/<day>/transcript.jsonl`, one JSON line each, with the
camera frame of that moment saved next to it in `frames/` for tool calls and for what the person
said (the newest 600 are kept). Read it on the robot with `python3 lab/hackathon_log.py 80`, through
the server with `GET /transcript?n=80`, or fetch the folder with `./tools/pup get ~/pupper-logs .`.

**Finding things** (`lab/hackathon_explore.py`). `find(what, max_seconds)` is the one search tool.
Cheapest step first: look (the detector's word for a thing with a class, else Gemma's yes or no
through the eyes service's `POST /exists`, and memory for where it was last seen), then a full circle
in place in 60-degree steps looking after each, then the roam. Pupper has no map and no odometry
worth the name, so this is a smart Roomba, not a Matic: it moves in legs and thinks at every stop.
At a stop it stands still for the senses to settle and asks the eyes whether the thing is in view.
Not found: it reads the depth sense's five strips and scores each direction by how open it is,
whether it leads somewhere new, a small preference for straight on, and a little noise, turns that
way and walks one leg of at most 1.5 m with both obstacle rules running every tick, then stops and
looks again. "Somewhere new" is a 1 m grid of visited cells kept by dead reckoning from the speeds
sent, which drifts, so it is a preference, never a claim about where the robot is. Nothing open
ahead means a turn about; four dead ends or legs that could not start in a row means "boxed in",
and `max_seconds` (default 120, at most 300) ends it with how much floor it covered. A `stop`
cancels it like any job, and it only ever sends the speeds `walk` and `turn` use. Found, at any
stage: a `[search result]` event says where, the place is remembered, and a `go_to` is queued to
walk up to it.

It is tested in the fake robot's kinematic room (`python3 lab/test.py hackathon explore`: four
walls, things standing in the room, the depth strips by ray casting and the detector when a thing is
in view): it finds a chair 3.5 m behind it and walks up to it, finds a Gemma-only "whiteboard with
code" across a 6 x 5 m room by asking at each stop, covers new cells when there is nothing to find
and says so when time is up, gives up as "boxed in" in a 0.7 m box without walking, and never comes
within 0.2 m of a wall. It has not run on the robot yet. On the robot: clear floor, a person
watching with a hand near the e-stop, and start with `max_seconds=30` in a small space; the one
hazard the depth sense cannot see is a drop (stairs, a stage edge).

**Learning on the fly** (`lab/hackathon_learn.py`). `learn_routine(name)` starts recording;
every motion tool Gemini calls after that is done for real and written down (accepted calls only;
a refused move is not part of the trick); `finish_learning()` saves it (`save=false` discards);
`do_routine(name)` replays it, each step through the same tool and the same checks, and stops at
the first refused step and says so. Routines are in the memory file and in the prompt's memory
section. The recorder is a `CALL_HOOKS` entry in `robot/tool_registry.py`, the one change to the
provided registry besides the list of modules.

**Calibrated hearing** (`hearing/`). When the robot server has it (numpy, scipy and
sounddevice, all on the Pupper), sound alerts come from `hearing/listener.py` instead of the
browser's measurement. It listens through PipeWire, next to the kiosk:

- **Bangs and shouts:** the calibrated meter (`hearing/meter.py`, `hearing/detect.py`), with
  thresholds in dB above the room's noise floor.
- **Sirens, alarms, doorbells, crashes, knocking and a baby crying:** Google's YAMNet
  (`hearing/classify.py`, AudioSet's 521 classes through LiteRT, on-device, about 1.5 ms per
  0.5 s).

What it hears goes to `lab/hackathon_alerts.py`:

- **Urgent sounds** (a hardcoded list, `URGENT`) turn the screen red with what they were (SIREN,
  DOORBELL, KNOCKING, BABY CRYING...) for 10 s.
- **Everyday sounds** show as a small label in the top-left corner (`GET /hearing/now`); silence
  shows nothing.
- **Gemini is never told about sounds:** Pupper shows them, it doesn't talk about them.

Sound alerts are **opt in**: off at every start, and only `sound_alerts(enabled=true)` (a person
asking) turns them on and opens the mic for them.

Setup and tuning:

- **Install YAMNet (once, on the Pupper):** `python3 hearing/classify.py --install` puts LiteRT in
  its own folder and the model in `~/.config/gemini-pupper/hearing/yamnet/`, so the robot's
  Python packages don't change.
- **Thresholds:** they live in `~/.config/gemini-pupper/hearing/calibration.json`, made with the
  meter page, `python3 hearing/mic_lab.py` (kiosk stopped; open http://localhost:8001 through an
  SSH tunnel). It shows the loudness, the detections and what YAMNet hears. Sensitivity low/high
  moves the thresholds +5/-5 dB.
- **Checks:** `GET /hearing` shows the detector's state, and `python3 hearing/test_signal.py`
  checks the signal chain.
- **Mic gain:** the kiosk browser starts with `--disable-features=WebRtcAllowInputVolumeAdjustment`,
  so its auto gain can't move the mic volume.

**Before using it on the robot.** Nothing here moves the robot on its own: reminders and sound
alerts only speak and paint the screen, and anything physical is Gemini's call through the normal
tools. A scan and a `go_to` are motions: clear space around Pupper, hand near the e-stop. A forward
walk can turn up to 90 degrees on its own to go round something, so keep clear space to its
sides too. The detector's sense of "close" is from box size, not depth: it errs toward refusing. In a loud room, set
`sound_alerts(sensitivity='low')` or untick Sound alerts.

## The AI stack: where Gemini is, where Gemma is

Two models, two jobs, and they are kept apart on purpose:

| | Model | Runs | Talks to the robot through |
| --- | --- | --- | --- |
| **The brain**: hears, talks, decides, calls tools | Gemini Live (cloud API), or for the hackathon Gemma 4 + whisper + Piper (local) | the browser (`web/`); or `robot/gemma_brain.py` on the Pi, with Gemma and whisper on the Mac or the Jetson | `GET /tools`, `POST /tools/<name>` |
| **The eyes**: depth, "where is X in the frame", "what is around" | Depth Anything V2 Small (ONNX) + Gemma 3 4B (Ollama) | `vision/eyes.py`, on a laptop today, the Jetson tomorrow | `POST /vision/*`, `GET /vision/wanted`, `/look` |
| **The legs**: walking, tricks | the trained policies | ROS 2 on the Pi | `robot_api.py` |

Why the split. The brain needs to be good at conversation, voice and tool use, and its job is one
decision at a time; Gemini Live is the best of that we can use today, but it is cloud, which the
hackathon forbids, and it cannot run the camera at a useful rate anyway. The eyes need to be fast,
local and honest, and they never decide anything: they post numbers (free space, a box) and sentences
(a description, an answer) to the robot server, and the rules in `lab/` decide what to do with them.
That is also why nothing in the eyes depends on which brain is calling the tools: `go_to`, `look` and
`what_is_around` are ordinary tools, and the obstacle rule is enforced on the robot whoever asks for
a move.

The Jetson that arrives at the hackathon changes where things run, not how they connect. Today the
eyes service runs on a laptop on the same network (Ollama on the laptop, frames through the SSH
tunnel), because the Pi 5's CPU cannot run Gemma at a useful speed. On the Jetson, mounted on the
Pupper, the same `vision/eyes.py` runs with Ollama there (`--robot http://<pi>:8000`), and the whole
eyes half is on-device. The brain moves too: `robot/gemma_brain.py` replaces Gemini Live with Gemma
on the Jetson (plus whisper for speech to text and Piper for text to speech), see
[The local brain](#the-local-brain-gemma-on-a-jetson-by-ethernet). One Gemma instance can serve both
the brain and the eyes, since the eyes are a few calls a second at most.

## The local brain on a Mac

The brain's heavy half on a Mac, the rest on the Pupper, over WiFi: one command, on the Mac.

```bash
./run_pupper.sh          # or ./run_pupper.sh 50 for the speaker's volume; Ctrl+C stops everything
```

```
  Mac (Apple silicon)                                Pupper (Pi 5)
  Ollama + Gemma 4 26B     :11435   <~~ WiFi ~~>      robot/server.py :8000 (localhost only)
  whisper.cpp (Metal)      :8178                     robot/gemma_brain.py, mic, speaker, Piper
```

What it does, in order (`tools/brain_on_mac.sh`; `run_pupper.sh` runs it on a Mac):

1. Checks it can reach the Pupper over ssh, and copies this checkout to it (`tools/pup sync`), so
   the Pupper runs exactly what you see. Nothing else starts if the Pupper can't be reached.
2. Starts an Ollama of its own on port 11435, reachable from the network, with the context the
   brain needs (16k tokens) and Gemma kept loaded. An Ollama app you already run is left alone:
   they share the downloaded models.
3. Starts whisper.cpp's server on the Mac's GPU (installing it with Homebrew, and its `base.en`
   model, the first time).
4. Runs `./run_pupper.sh` on the Pupper with the brain pointed at the Mac. There it checks it can
   reach Gemma and whisper (`gemma_brain.py --check --voice`) before anything starts, installs
   Piper the first time (`robot/setup_voice.sh`), starts the robot with the kiosk as the brain's
   face (`robot/start.sh --face`) and the eyes asking the Mac's Gemma, then runs the brain:
   Pupper's mic, whisper, Gemma, Piper, Pupper's speaker.

**The face works as with Gemini.** The kiosk opens the app with `?brain=local`
(`web/services/localBrain.ts`): no Gemini, no key, no WAKE PUPPER button. The brain posts what it
is doing (`POST /brain/state`: listening, speaking, the caption), and the face shows it: awake
while the brain runs, a moving mouth and captions while Pupper talks. The app's own jobs run as
before: the screen's cards (`show_text`, sound alerts, found it), the sound monitor on the mic,
the on-device object detector on the camera, and reminders. A due reminder's card shows on the
face, and the app hands it to the brain to say (`POST /brain/announce`, an event the brain reads
like any other). Without a face, the brain claims due reminders itself, a minute late.

Ctrl+C on the Mac stops the brain and the robot (`robot/stop.sh`), and the Mac's servers. So does
the ssh connection dropping. Run it with Pupper on the floor and a hand near the e-stop: it moves
when you ask it to.

Once per laptop: `./tools/pup setup` (the ssh key) and Ollama (`brew install ollama` or the app).
If `pupper5.local` doesn't resolve on the WiFi: `PUPPER_HOST=pi@<the Pupper's IP> ./run_pupper.sh`.

**The model.** `gemma4:26b`: a mixture of experts with 4B parameters active per token, so it is
fast for its size, and on a 48 GB M4 Pro it leaves room for whisper. Until it is pulled
(`ollama pull gemma4:26b`, about 17 GB, once, on a fast network), the script uses the best Gemma
the Mac already has and says so; `PUPPER_GEMMA=<model> ./run_pupper.sh` picks another.
Which build of it is best is measured, not guessed: `tools/brain_bench.py` gives every model the
same 17 commands through the real prompt and tools, on the fake robot, and scores what it does.

```bash
python3 tools/brain_bench.py gemma4:26b gemma4:26b-a4b-it-qat gemma4:26b-mlx \
    hf.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M
```

The candidates: Ollama's own build (`gemma4:26b`, with Ollama's Gemma 4 tool calling), Google's
quantization-aware 4-bit (`-it-qat`, usually the most accurate at 4 bits), the MLX build
(`-mlx`, Apple's engine) and Unsloth's dynamic 4-bit GGUF (tool calling through Ollama is
reportedly unreliable for Hugging Face GGUFs). Measured so far, on the M4 Pro: `gemma3:4b`, 14/17
right, 11/17 clean (nothing extra that moves the robot), 0.7 s to reply, 8.5 s for the first turn
(it reads the whole prompt once); it has no tool calling, so it answers in JSON. Run the bench on
the 26B builds once they are pulled, and set the winner with `PUPPER_GEMMA`.

Things to know:

- **The network.** The Mac and the Pupper must reach each other. Venue WiFi often blocks that
  (client isolation): the Pupper's check then fails with that hint. Use a phone hotspot.
- **macOS may ask** to let `ollama` and `whisper-server` accept incoming connections: allow it.
  While this runs, both are reachable by anyone on the same network.
- **The Mac's terminal** shows what Pupper hears, says and does, besides the face.
- **Is a laptop "on-device"?** The model runs on the table in front of you, with no cloud. Check with
  the organizers that this counts; the Jetson setup below is the stricter answer.

## The local brain: Gemma on a Jetson, by Ethernet

The hackathon forbids the cloud, and Gemini Live is cloud. `robot/gemma_brain.py` is the brain
without it: `lab/chat.py`'s loop with Gemma (in Ollama) instead of Gemini, plus ears and a voice. It
reads the same prompt and tools (`GET /tools`), sends every call to the robot server
(`POST /tools/<name>`) and gives Gemma the `{success, message}` word for word, so the `lab/` tools,
the queue, the limits and the obstacle rules are exactly the ones Gemini drives.

```
  laptop ~~WiFi~~ Pi 5 (pupper5.local)                      Jetson (Thor or Orin Nano)
                  robot/server.py :8000  (localhost only)    Ollama + Gemma 4   10.10.10.1:11434
                  robot/gemma_brain.py   ==== Ethernet ====  whisper.cpp server 10.10.10.1:8080
                  mic, speaker, Piper    10.10.10.2          (vision/eyes.py too, if you like)
```

The brain runs on the Pi, beside the mic and the speaker. Only the heavy parts are on the Jetson:
Gemma, and whisper (speech to text, on its GPU). Both listen only on the cable. The robot server
stays on the Pi's localhost, so nothing on any network can move the robot, and no audio has to cross
the cable on the way back (Piper speaks on the Pi).

**Which Jetson.** Use the Thor if you have the choice. The two run different Gemmas, and the
difference is in how well they pick tools:

| | Thor | Orin Nano (8 GB) |
| --- | --- | --- |
| Gemma | `gemma4:26b` (a mixture of experts: 4B active per token, so it is fast) | `gemma4:e2b-it-qat` (4.3 GB; `e4b` is 6.6 GB and doesn't fit beside whisper) |
| Tool calling | good | weak: give it fewer tools (`--only`) |
| whisper | `small.en` | `base.en` |
| Where | on the table, on wall power, the cable to Pupper | can ride on Pupper, with its own battery (not Pupper's) |

`gemma_brain.py` picks the biggest Gemma Ollama has, so the same code runs on either.

**Wiring.** One Ethernet cable straight from the Pi's port to the Jetson's. No switch, and any
cable works (both ends swap pairs by themselves). This is the port Lab 1 used for the laptop, so the
laptop now reaches the Pi over WiFi (`ssh pi@pupper5.local`). `jetson/link.sh pi down` puts the Pi
back to Lab 1's setup (DHCP on the cable, the laptop sharing its internet). The Jetson runs on its
own power supply.

**Setup, once** (each needs the internet for its downloads, over WiFi; after that, nothing goes out):

```bash
# on the Jetson, in a clone of this repo: the cable (10.10.10.1), Ollama + Gemma, whisper.cpp on the GPU
jetson/setup.sh                 # or: jetson/setup.sh gemma4:e4b   (another model)
# on the Pi: the cable (10.10.10.2), Piper and its voice, espeak-ng
jetson/setup_pi.sh
# on the Pi: is everything reachable, how fast is Gemma, can it call a tool?
python3 robot/gemma_brain.py --check
jetson/link.sh status           # on either end: the address, and a ping to the other end
```

**Running.** Try it without the robot first, then in text on the robot, then by voice:

```bash
python3 robot/gemma_brain.py                     # type; your tools on the fake robot: nothing moves
robot/start.sh                                   # on the Pupper: the robot, no kiosk (see below)
python3 robot/gemma_brain.py --robot             # type to the real robot: it moves!
python3 robot/gemma_brain.py --robot --voice     # talk to it: it moves!
python3 robot/gemma_brain.py --robot --voice --camera   # Gemma also sees the newest frame each turn
python3 robot/gemma_brain.py --only stand_up,lie_down,stop,move,turn,play_trick,set_reminder,show_text
```

With `--robot`, Pupper is on the ground with a hand near the e-stop, as always. Start the robot
with the face only (`robot/start.sh --face`), never `--kiosk`: that kiosk connects Gemini Live (a
second brain, and cloud). Saying just "stop" (or "Pupper,
stop") stops the robot at once, without waiting for Gemma. The gamepad's e-stop works whatever the
brain does.

What it does that the web app did: robot events (reminders, sound alerts, search results) become
turns, so Pupper speaks up by itself; due reminders are claimed and delivered (`/reminders/claim`);
what was said goes to the session log (`POST /transcript`). How Gemma calls tools: natively
(Ollama's tools); if it writes a call out as text instead, that still runs; and with a model or an
Ollama that can't do tools at all, `--tools auto` switches to JSON replies (`{say, calls}`, held to a
schema). Arguments like `"1.5"` for a number are fixed by the declaration; anything else wrong is
refused by the robot, with the reason, as with Gemini.

What it does not do yet, compared with Gemini Live:

- **No interrupting.** The mic is off while Pupper talks (or it would hear itself), so "stop" isn't
  heard mid-sentence; the e-stop is.
- **Slower turns**: the end of your sentence (0.8 s of quiet), whisper, then Gemma. `--check`
  prints Gemma's times; the first call after a start loads the model (10-30 s).
- **Not tried on a real Jetson yet.** The loop is tested against a fake Ollama
  (`python3 lab/test.py brain`), and the setup scripts follow Ollama's and whisper.cpp's docs.
  Run `--check` first, and expect to adjust `jetson/setup.sh` on the day (the Thor's JetPack 7
  in particular).

## Hackathon plan

Google Gemma Hardware Hackathon:

- **When/where:** Sat Oct 24, 2026, AGI House (1868 Floribunda Ave, Hillsborough). Doors 10:00,
  building from 12:15, **save the project draft by 7:00 PM** (finalists are picked from it),
  demos at 8:00.
- **Hard rule:** every project must use a Gemma model and run **entirely on-device, with no cloud
  in the loop**.
- **Our project:** turn this Pupper into a proactive AI helper dog that notices what's going on
  and helps without being asked, with Gemma running locally.
- **Track:** Physical AI, IoT & Robotics (1st place $7.5k). Also eligible for Best Use of LiteRT
  ($7.5k 1st) if we run Gemma through LiteRT-LM, Best Use of Raspberry Pi (5× Pi 5), and the
  Qualcomm hardware awards.
- **Biggest change from Lab 6:** Lab 6 is driven by **Gemini Live, a cloud API**, which the
  hackathon forbids. We keep the Lab 6 architecture (tools, the job queue, policies, tricks) and
  replace Gemini Live with a **local Gemma** brain, plus local speech-to-text and text-to-speech:
  `robot/gemma_brain.py` on the Pi, with Gemma and whisper on a Jetson by an Ethernet cable
  ([The local brain](#the-local-brain-gemma-on-a-jetson-by-ethernet)). The features above are
  already brain-agnostic: tools, events and endpoints on the robot; the object detector and the
  sound monitor already run on-device.

## Safety

- Test on the ground, never on a table, with a hand near the e-stop.
- Coding agents must not move the robot (`start.sh`, `chat.py --robot`, POST `curl`,
  `ros2 topic pub`), raise limits, or weaken tests without explicit approval at the robot.
- Never commit an API key. The Gemini key lives in `~/.config/gemini-pupper/api_key`.

## More

- [Peng Xu's write-up: Interactive Robot Dog with Gemini Live](docs/interactive-robot-dog.md)
- [System architecture](docs/architecture.md)
- [Notes for coding agents](AGENTS.md)
