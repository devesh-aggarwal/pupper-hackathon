"""Part 3: Follow. TODO 5 (the rule that ends follow mode), then TODO 6 (the follow tool).

The person follower itself is provided (robot.set_following). The rule is yours:

    Questions and conversation keep following. Any motion (walk, turn, arc, trick,
    stand up, lie down) ends follow mode, and Pupper says so first:
    "Let me stop following you and ..."
"""

from __future__ import annotations

from robot_api import Result, Robot, not_written  # noqa: F401
from tool_registry import Param, tool


# ============================================================ what the robot does
async def prepare_to_move(robot: Robot) -> str:
    """Get the robot ready for a new motion. Every motion calls this first,
    before it queues anything: look for it in stand_up (basics.py) and walk
    (part2_walk.py).

    What it is for: stopping anything that runs on its own, outside the queue,
    and would fight with the new motion. In Gemini Pupper that is the person
    follower, so its one job is to end follow mode; it is also where anything
    like it you add later goes (a patrol loop, dancing to music). Getting the
    legs ready is a different job, done later in the job by switch_to.

    TODO 5 (Part 3) ------------------------------------------------------------
    The rule:            talking and questions keep following. Any motion (walk,
                         turn, trick, stand up, lie down) ends follow mode right
                         away, not when the queue gets to the motion.
    Returns:             a note that goes in front of the tool's reply, e.g.
                         "Stopped following the person first.", or "" if the
                         robot wasn't following.
    Why right away:      the person asked for something else *now*; the robot
                         shouldn't keep chasing them until the queue gets there.
    Why a note:          Gemini can only say "Let me stop following you and ..."
                         if the reply tells it.
    Use:                 robot.following, await robot.stop_following(),
                         robot.send_speed(0, 0, 0) (stand still)
    Tests check:         - stand_up, lie_down and walk stop the follower right
                           away, before the queue gets to them
                         - their reply contains the word "follow".
                           Passes: "Stopped following the person first. activate walking queued; ..."
                           Fails:  "Done. activate walking queued; ..."
                         - when not following, the reply does NOT contain "follow"
                         - robot_status (a question) keeps following
    Test:                python3 lab/test.py todo5
    """
    # ========== TODO 5: YOUR CODE HERE ==========
    if not robot.following:
        return ""
    await robot.stop_following()
    robot.send_speed(0, 0, 0)
    return "Stopped following the person first."
    # ====================================


# ============================================================ what Gemini reads
# TODO 6: describe follow for Gemini, and write its body: robot.set_following(enabled)
# (provided) starts or stops the person follower. Gemini should know what keeps
# follow mode on and what ends it (your rule in TODO 5).
#   Tests check:
#   - follow's description is 20+ characters, enabled is described, and the
#     description says what ends it: it contains "end" or "stop".
#     Passes: "... any motion tool (walk, turn, ...) ends it."  Fails: "Follow the person."
#   - prompt.md's Follow mode section contains "end" or "stop"
#   - follow(enabled=True) starts the follower, follow(enabled=False) stops it
#   Test: python3 lab/test.py todo6
# ========== TODO 6: YOUR CODE HERE ==========
@tool(
    "follow",
    "Start or stop follow mode: Pupper walks after the person it sees with its camera, keeping a"
    " short distance. Talking and questions (including robot_status and what you see) keep it on."
    " follow(enabled=false) stops it, and any motion tool (move, turn, arc, play_trick, stand_up,"
    " lie_down, interactive_action) ends it; say \"Let me stop following you and ...\" before calling one."
    " Example: 'follow me' -> follow(enabled=true); 'stay here' -> follow(enabled=false).",
    enabled=Param("boolean", "true to start following the person, false to stop and stand still."),
)
async def follow_tool(robot: Robot, enabled: bool) -> Result:
    return await robot.set_following(enabled)
# ====================================
