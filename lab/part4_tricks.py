"""Part 4: Tricks. TODO 7 (play_trick), then TODO 8 (what Gemini reads about it).

Read it top to bottom: play_trick (yours) checks the request and queues the
job; play_policy_trick (finished: read it) is that job; the play_trick tool
(yours: its words) is what Gemini calls.

Two kinds of trick, both played by a trick policy (trick mode). A playback trick
(trick.interactive is False), like sit, plays one reference motion from start to
finish; its tracking policy keeps Pupper balanced while it plays. An interactive
trick (trick.interactive is True), like sit_stay_shake, is one Pupper stays in:
it plays its opening clips, then loops one (the "stay") until Gemini asks for
another (interactive_action, Part 5), or any other motion leaves it through its exit
clip (trick.exit_clip).
"""

from __future__ import annotations

from basics import ENDS_FOLLOW
from part3_follow import prepare_to_move
from robot_api import FELL_MESSAGE, LOCOMOTION, LOCOMOTION_CONTROLLER, WALK_READY_S, Result, Robot, not_written, with_note  # noqa: F401
from tool_registry import TRICK_NAMES, Param, tool
from tricks import Trick


# ============================================================ what the robot does
async def play_trick(robot: Robot, name: str) -> Result:
    """Play a trick from <repo>/tricks. Every trick in the lab is played by trick
    policies on their own controllers (the job is play_policy_trick below).

    TODO 7 (Part 4) ------------------------------------------------------------
    Now:                 an unknown name fails with f"unknown trick '{name}';
                         available: {list(robot.tricks)}"; leave follow mode.
    Later (the job):     play_policy_trick(robot, trick), below: read it.
    Gemini hears now:    robot.queue(f"trick {name}", job, trick.total_time,
                         then=...) with the note in front.
                         then is the state Pupper is in once the job is done
                         (play_policy_trick's step 4): LOCOMOTION after a
                         playback trick that ends standing (the job hands the
                         legs back), else the trick itself (an interactive trick
                         stays in it; any other trick holds its end pose).
    Tests check:         - an unknown name fails, and the message lists the real names.
                           Passes: "unknown trick 'backflip'; available: ['bow', ...]"
                           Fails:  "unknown trick"
                         - a real trick is accepted right away ("trick bow queued; ...")
                         - the queued job is play_policy_trick: the trick's
                           controller takes the legs for trick.total_time, and the
                           job's outcomes ("back in locomotion mode", "holds its
                           final pose", "already doing ...") come from it
                         - then=: LOCOMOTION after a trick that ends standing, else
                           the trick (asking for an interactive trick again while
                           Pupper is in it says "already doing ...")
                         - with follow mode on, the reply contains "follow" (the note)
    Test:                python3 lab/test.py todo7

    Optional, after the lab: the jump. Pupper's default locomotion policy can
    already jump (you have seen it on the gamepad). Give Gemini that jump:
    rename tricks/jump/trick.yaml.optional to trick.yaml, and make a trick of
    trick.kind == "reference" play with robot.play_reference_trick(trick)
    instead. Test: python3 lab/test.py optional_jump
    """
    # ========== TODO 7: YOUR CODE HERE ==========
    trick = robot.tricks.get(name)
    if trick is None:
        return Result(False, f"unknown trick '{name}'; available: {list(robot.tricks)}")
    note = await prepare_to_move(robot)
    job = lambda: play_policy_trick(robot, trick)  # noqa: E731
    then = LOCOMOTION if not trick.interactive and trick.end_pose == "standing" else trick
    return with_note(note, await robot.queue(f"trick {name}", job, trick.total_time, then=then))
    # ====================================


async def play_policy_trick(robot: Robot, trick: Trick) -> Result:
    """The job that plays a policy trick (finished: read it).

    Your play_trick (TODO 7) queues this; it runs when the queue gets to the
    trick. It gives the legs to the trick's controller (trick mode), waits while
    the trick plays, then leaves the robot in the right state, and says which:
    Gemini (and robot_status) can only plan the next move if it knows. The four
    steps:

    1. An interactive trick Pupper is already in: nothing to do.
    2. A playback trick still holding the end of its last run (Pupper is still
       in its state): relax first. Switching to a controller that is already on
       does nothing at all, so without this "swim again" would just keep lying
       there.
    3. Give the legs to the trick, and wait while it plays.
    4. Afterwards: an interactive trick stays (Pupper is in it until another
       motion), and one that walks in its own stance (the spider, Part 6) hands
       the legs to its walking policy; a playback trick that ends standing hands
       the legs back to the locomotion policy; any other trick holds its final
       pose.

    Examples (the handout has them as a table):
      sit (ends standing), in locomotion   -> "played sit; back in locomotion mode"
      swim (ends lying), in locomotion     -> "played swim; it holds its final pose (lying)"
      swim again, still lying from swim    -> relaxes first (step 2), then the same
      sit_stay_shake (interactive)         -> "played sit_stay_shake; it now stays sitting. Trick
                                              actions: shake_left, shake_right, stand_up. ..."
      sit_stay_shake, already in it        -> "already doing sit_stay_shake"
    """
    if trick.interactive and robot.state_now is trick:  # 1. already in it
        return Result(True, f"already doing {trick.name}")
    if not trick.interactive and robot.state_now is trick:
        # 2. Still holding this trick's end pose: relax, so switching to it restarts it.
        res = await robot.switch_to(None)
        if not res.ok:
            return res
    res = await robot.switch_to(trick.controller)  # 3. the trick takes the legs
    if not res.ok:
        return res
    await robot.sleep(trick.play_time + 0.3)  # ... and plays
    if robot.fallen:
        return Result(False, f"{trick.name} ended in a fall: {FELL_MESSAGE}")
    if trick.interactive:  # 4. it stays in its looping clip until asked for an action
        actions = ", ".join(trick.actions) or "none"
        leave = f" Any other motion first plays {trick.exit_clip}." if trick.exit_clip else ""
        if trick.walks:  # ... and a trick that walks in its stance hands the legs to its walking policy
            res = await robot.switch_to(trick.walk_controller)
            if not res.ok:
                return Result(False, f"played {trick.name}, but its walking policy did not take over: {res.message}")
            await robot.sleep(WALK_READY_S)  # it eases into the stance, then fades its policy in
            return Result(True, f"played {trick.name}; it now stays in its {trick.end_pose} stance."
                                f" {trick.walk.describe()} Interactive actions: {actions}.{leave}")
        return Result(True, f"played {trick.name}; it now stays {trick.end_pose}. Interactive actions: {actions}.{leave}")
    if trick.end_pose == "standing":  # 4. hand back to the locomotion policy
        res = await robot.switch_to(LOCOMOTION_CONTROLLER)
        if not res.ok:
            return Result(False, f"played {trick.name} but could not return to locomotion mode: {res.message}")
        return Result(True, f"played {trick.name}; back in locomotion mode")
    return Result(True, f"played {trick.name}; it holds its final pose ({trick.end_pose})")  # 4. it stays


# ============================================================ what Gemini reads
# TODO 8: describe play_trick for Gemini. {tricks} in a description becomes the
# list of installed tricks (each with its description from its trick.yaml).
# Gemini needs to know that each trick says whether it starts standing or lying
# down, and that the robot keeps a trick's final pose until the next motion.
#   Tests check:
#   - the description is 20+ characters, and name is described
#   - it contains {tricks} (every trick's own description appears in it).
#     Fails: listing the tricks by hand (it goes stale when a trick is added)
#   - name keeps enum=TRICK_NAMES
#   - it contains the word "pose": "A trick keeps its final pose until the next motion."
#   Test: python3 lab/test.py todo8
# ========== TODO 8: YOUR CODE HERE ==========
@tool(
    "play_trick",
    "Play one of Pupper's installed tricks by name. Each trick below says whether it starts"
    " standing or lying down and how it ends. A trick keeps its final pose until the next motion"
    " (stand_up gets Pupper walking again). A trick that starts standing may look wrong if Pupper is"
    " lying down: call stand_up first. Interactive tricks stay in their pose and take"
    " interactive_action requests. Only these tricks exist; for anything else, say Pupper doesn't"
    " know it yet." + ENDS_FOLLOW + "\nTricks:\n{tricks}",
    name=Param("string", "The trick's name, exactly as listed.", enum=TRICK_NAMES),
)
# ====================================
async def play_trick_tool(robot: Robot, name: str) -> Result:
    return await play_trick(robot, name)
