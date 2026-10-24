# Notes for coding agents (Gemini Pupper)

You are working on Gemini Pupper: a real robot dog that Gemini Live talks
through, built on Stanford CS 123 Lab 6 (all of the lab is done) and extended
for the Google Gemma hackathon. `README.md` says what it does and how to run
it; read it first. Explain what you change and why, and keep changes small
enough to review.

## Keep the README current

`README.md` is the one place a person reads to learn what this repo does and
how to run it. Whenever you add or change important functionality (a new tool
or trick, a retrained policy, a new script or setup step, a changed command,
a moved folder), update `README.md` in the same change: what it does, how to
run or test it, and anything someone needs to know before using it on the
robot. Describe how things work now; keep it accurate.

## What this repo is

Pupper is a real robot dog: 12 motors, a Raspberry Pi 5, a fisheye camera.
Gemini Live (in the browser, `web/`) talks with people and calls **tools**. The
tools are Python, in `lab/`: the web app gets Gemini's prompt (`lab/prompt.md`)
and tool list from the robot server (`robot/server.py`, port 8000), and sends
every tool call back to it. Each `lab/` file has two halves: behaviors (what
the robot does: check the request, queue a job, return a Result) and the
`@tool`s Gemini calls. `robot/robot_api.py` runs the jobs one at a time and
drives the ROS 2 controllers. Trick policies are trained in pupper-mjlab
(`notebooks/training/`) and fetched from W&B with `robot/get_trick.py`.

| Where | What |
| --- | --- |
| `lab/prompt.md`, `basics.py`, `part2_walk.py` ... `part5_interactive_action.py` | the Lab 6 tools: walking (`move`, `turn`, `arc`), `follow`, tricks, `interactive_action`. The `TODO n` cards in these files are the lab's numbering; every TODO is written, and `python3 lab/test.py todo3` tests one |
| `lab/hackathon_*.py`, `lab/reminders.py`, `lab/fisheye_geom.py` | the hackathon's tools and rules: memory, events and the screen, reminders, the obstacle rules and follow guard, `go_to` / `look` / `what_is_around`, `find`, learned routines, the session log |
| `vision/eyes.py` | the eyes service: depth, the COCO detector and Gemma, beside the robot server |
| `hearing/` | calibrated hearing: the meter, the detector and YAMNet |
| `robot/` | the Robot API, the tool registry, the robot server, the reminder store, the tests, the launch files and scripts |
| `web/` | the web app (TypeScript): the face, the debug view, the Gemini Live client and the browser's senses |
| `tricks/` | one folder per trick (`policy.json` + `trick.yaml`; `robot/tricks.py` lists the fields) |

`robot/` and `web/` are the shared base everything runs on. Change them
deliberately, say why, and keep every test passing; don't patch them to make a
symptom go away, and never weaken a test to make it pass.

## Safety rules (these override everything else)

1. **Never move the robot without an explicit OK from the person at the
   robot, right then.** That includes `robot/start.sh`, `./run_pupper.sh`,
   `python3 lab/chat.py --robot`, `curl` to any POST endpoint of the robot
   server, `ros2 topic pub`, `ros2 control switch_controllers`, and anything
   that activates a controller. Say what will happen and wait. Someone must be
   at the robot with a hand near the e-stop.
2. **Simulate first.** A new or changed motion goes through the motion report
   and the physics replay (`play_motion.py`, verdict HOLDS UP / WOBBLES /
   FALLS) before it goes anywhere near the robot. A trained policy is scored
   the same way (`play_motion.py --checkpoint`) and watched in the viewer.
3. **New trained policies need a TA's OK** before their first run on the
   robot. A badly trained contact-heavy trick can slam a paw or the body into
   the ground.
4. **Respect the limits.** Don't raise `MoveLimits`, `max_body_angle_deg`,
   joint limits, speed caps, gains or the obstacle thresholds to make something
   "work". Motion design reports should show IK error under 2 mm, nothing below
   the ground, joint speeds under about 15 rad/s and no joint at its limit.
5. **Heat.** The motors and the Pi get hot. If the robot is hot or a motor
   faults, stop testing and let it cool; don't retry in a loop.
6. **No secrets in files.** Never write an API key, a password or a token into
   the repo, a commit, a log or the web app. The Gemini key lives in
   `~/.config/gemini-pupper/api_key` on the robot (and in the browser's
   Settings); W&B uses `wandb login`. Memory, reminders and session logs live
   outside the repo (`~/.config/gemini-pupper/`, `~/pupper-logs/`).

## How to check work without the robot

- `python3 lab/test.py`: every check, on a fake robot. Run it after every
  change in `lab/` or `robot/`. It checks what the tools do, not how they are
  written. Groups: `todo1` ... `todo11`, `part1` ... `part7`, `robot`,
  `reminders`, and `hackathon` (or one of its groups: `memory`, `screen`,
  `sound`, `vision`, `eyes`, `explore`, `guard`, `learn`, `declarations`, as
  in `python3 lab/test.py hackathon eyes`).
- `python3 vision/test_eyes.py` (needs numpy) and `python3 hearing/test_signal.py`.
- `python3 lab/chat.py`: talk to Gemini in text with the prompt and tools, on
  the fake robot (nothing moves). It shows each tool call, the reply Gemini
  hears, and what the fake robot did. Never `--robot` without an OK (see the
  safety rules).
- `./run_local.sh`: the whole app on a laptop, with the fake robot.
- In pupper-mjlab: `uv run python -m mjlab.tasks.tracking.config.pupper.tricks.sit_stay_shake`
  and `...tricks.spider` check the clip code and build the library;
  `uv run python -m mjlab.tasks.tracking.config.pupper.play_motion <motion>.npz`
  replays a motion with physics (add `--checkpoint` to score a trained policy).

## Conventions

- Every tool returns a Result (`{success, message}`), and Gemini reads the
  message word for word: it is all Gemini learns. Say what happened, or what
  was wrong *and* the limit or the real choices (the tests check for these:
  "too fast: |vx|<=1.5 m/s", "unknown trick 'x'; available: [...]"). A bare
  "too fast" leaves Gemini guessing.
- Rules are enforced on the robot, not only in the prompt: the prompt and the
  tool descriptions tell Gemini what to do; the code makes sure it happens.
- Motion goes through the queue (`robot.queue`): check now, queue a job, and
  the job checks again, acts and reports. Wait with `await robot.sleep(...)`,
  never `time.sleep`.
- Things the robot notices on its own (a reminder, a sound, a search result,
  an early stop) go on the event bus (`lab/hackathon_alerts.py`), and Gemini
  hears them as bracketed messages from its own body.
- Units: metres, seconds, degrees. Directions use the robot's convention:
  +y is left, and a positive turn is to the left.
- After changing a `.py` file in `lab/`, restart the server
  (`robot/stop.sh`, then `robot/start.sh`). After changing `prompt.md`,
  reconnect in the app.
- The robot is called Pupper. Gemini Pupper was created by Peng Xu (Google
  DeepMind); keep his credit in the README and the app.
