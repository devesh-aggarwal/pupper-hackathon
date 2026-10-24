"""Part 5: interactive actions. TODO 9 (interactive_action), then TODO 10 (what Gemini reads about it).

An interactive trick (part4_tricks.py explains them) is one Pupper stays in
and that takes requests: its *actions*, clips Gemini asks for while Pupper is in
it. interactive_action plays them, for every interactive trick: the shakes and
the stand-up of sit_stay_shake (Part 5), and spider_up, the spider's way back
to standing (Part 6). You train sit_stay_shake in the notebook and fetch it with
robot/get_trick.py; until then, the tests use a made-up interactive trick
called hold_pose.
"""

from __future__ import annotations

from robot_api import LOCOMOTION, LOCOMOTION_CONTROLLER, Result, Robot, not_written  # noqa: F401
from tool_registry import ACTION_IDS, Param, tool


# ============================================================ what the robot does
async def interactive_action(robot: Robot, action: str) -> Result:
    """Play one of an interactive trick's actions (e.g. shake a paw while sitting).

    TODO 9 (Part 5) ------------------------------------------------------------
    Now:                 robot.action_ids() maps each action to (trick, clip). An
                         unknown action fails, listing the real ones. An action
                         only works in its trick's state: refuse unless
                         robot.state_after_queue is trick, and tell Gemini which
                         trick to play first.
    Later (the job):     check again, about right now: robot.state_now is trick
                         (Pupper may have fallen, been e-stopped or stopped
                         since). Then play the clip with robot.play_clip. The
                         exit clip is different: it leaves the trick, so instead
                         hand the legs back with
                         robot.switch_to(LOCOMOTION_CONTROLLER), which plays the
                         exit clip on the way out.
    Gemini hears now:    robot.queue(f"{trick.name} {clip}", job,
                         trick.clip_time_estimate(clip), then=...):
                         LOCOMOTION after the exit clip, else trick.
    Why check twice:     "sit, then shake" said in one breath queues both before
                         the sit has happened (state_after_queue knows it will);
                         a fall or a stop can still happen in between (state_now
                         knows when the job runs).
    Use:                 robot.action_ids, robot.state_after_queue,
                         robot.state_now, robot.play_clip, robot.switch_to
    Tests check:         - an unknown action fails, and the message lists the real ones.
                           Passes: "unknown interactive action fetch_the_ball; available: ['wave_left', ...]"
                         - an action while Pupper isn't (going to be) in its trick fails
                           right away, names the trick, and queues nothing.
                           Passes: "wave_left only works while hold_pose is running;
                                    play it first (play_trick hold_pose)"
                           Fails:  "not sitting" (Gemini can't tell which trick to play)
                         - in the trick: the clip plays and Pupper stays in the trick
                         - the exit clip: locomotion takes the legs back, and later
                           actions are refused (then=LOCOMOTION)
                         - "sit, then shake" queued back to back plays both
                           (the now-check must use state_after_queue)
                         - if the trick was relaxed before the queue got there, the
                           job refuses (the later check: robot.state_now)
    Test:                python3 lab/test.py todo9
    """
    # ========== TODO 9: YOUR CODE HERE ==========
    found = robot.action_ids().get(action)
    if found is None:
        return Result(False, f"unknown interactive action {action}; available: {list(robot.action_ids())}")
    trick, clip = found
    if robot.state_after_queue is not trick:
        return Result(False, f"{action} only works while {trick.name} is running;"
                             f" play it first (play_trick {trick.name})")
    exits = clip == trick.exit_clip

    async def job() -> Result:
        if robot.state_now is not trick:
            return Result(False, f"{action} skipped: {trick.name} is no longer running;"
                                 f" play it first (play_trick {trick.name})")
        if exits:
            res = await robot.switch_to(LOCOMOTION_CONTROLLER)
            if not res.ok:
                return Result(False, f"{clip} did not finish: {res.message}")
            return Result(True, f"played {clip}; left {trick.name}, back in locomotion mode")
        return await robot.play_clip(trick, clip)

    return await robot.queue(f"{trick.name} {clip}", job, trick.clip_time_estimate(clip),
                             then=LOCOMOTION if exits else trick)
    # ====================================


# ============================================================ what Gemini reads
# TODO 10: describe interactive_action for Gemini, and write its body (it calls
# interactive_action above). {actions} in a description becomes the list of installed
# actions, each with its trick and its description from trick.yaml. Say what has
# to happen first, and what ends it. (Also TODO 10: the words in
# tricks/sit_stay_shake/trick.yaml, once you have fetched your policy.)
#   Tests check:
#   - the description is 20+ characters, and action is described
#   - it contains {actions} (each action's description appears in it)
#   - action keeps enum=ACTION_IDS, and the tool plays the clip
#   - trick.yaml (once fetched): no TODO left, description 20+ characters, and
#     every action described in 10+ characters.
#     Passes: shake_left: "Shakes the left paw, then keeps sitting."  Fails: shake_left: "left"
#   Test: python3 lab/test.py todo10
# ========== TODO 10: YOUR CODE HERE ==========
@tool(
    "interactive_action",
    "Ask an interactive trick for one of its actions, e.g. shake a paw while sitting. Each action"
    " only works while its trick is running: play that trick first with play_trick (you can call"
    " both in one go: play_trick, then interactive_action). Pupper stays in the trick after an"
    " action; its exit action, or any other motion tool, ends the trick. Actions:\n{actions}",
    action=Param("string", "The action's name, exactly as listed.", enum=ACTION_IDS),
)
async def interactive_action_tool(robot: Robot, action: str) -> Result:
    return await interactive_action(robot, action)
# =====================================
