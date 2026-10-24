"""Robot stack for Gemini Pupper.

Includes the course's neural_controller launch (walking policies, camera, joystick
e-stop) exactly as pupper_gait_deploy installs it, and adds:
  - cmd_vel_mux, so the joystick, Gemini and the person follower share /cmd_vel
  - the person follower: your Lab 5 follow-me (follow_me_pupper/robot/follow_service.py)
    when that repo is on the robot, else the monorepo's simple follower
  - gemini_walk, the walking controller Gemini drives (default policy, running
    allowed, reference tricks such as the jump enabled), and one controller per
    policy trick in <repo>/tricks (see robot/tricks.py). Every motion Gemini asks
    for is one of these policies; there is no open-loop trick player.
The joystick still drives the robot, now through the mux, where it outranks Gemini.

    ros2 launch robot/launch/gemini_pupper.launch.py
"""
import os
import sys
from pathlib import Path

import yaml

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

HERE = Path(__file__).resolve().parent
PARAMS = str(HERE / "gemini_pupper.yaml")
sys.path.insert(0, str(HERE.parent))
from spawn_controllers import PARAM_FILE as TRICK_PARAMS  # noqa: E402
from tricks import WALK_CONTROLLER, controller_params, load_tricks, walk_controller_params, walk_policy_params  # noqa: E402

COURSE_CONFIG = PathJoinSubstitution([FindPackageShare("neural_controller"), "launch", "config.yaml"])


def gemini_controllers():
    """Write one parameter file for the controllers Gemini adds, and spawn them inactive:
    gemini_walk (the course default policy, able to run and play reference tricks)
    and one NeuralController per policy trick. All start from the course default
    controller's settings; the course's own controllers are left as they are."""
    from ament_index_python.packages import get_package_share_directory

    course = Path(get_package_share_directory("neural_controller")) / "launch" / "config.yaml"
    base = yaml.safe_load(course.read_text())["neural_controller_default"]["ros__parameters"]
    kind = {"type": "neural_controller/NeuralController"}
    params = {WALK_CONTROLLER: {"ros__parameters": {**kind, **walk_controller_params(base)}}}
    tricks = [t for t in load_tricks().values() if t.kind == "policy"]
    for t in tricks:
        params[t.controller] = {"ros__parameters": {**kind, **controller_params(t, base)}}
        if t.walks:  # a trick that walks in its own stance: its walking policy too
            params[t.walk_controller] = {"ros__parameters": {**kind, **walk_policy_params(t, base)}}
    # The course config writes paths as $(find-pkg-share pkg)/...; the spawner's
    # parameter file isn't substituted, so expand them here.
    for entry in params.values():
        p = entry["ros__parameters"]
        path = p.get("model_path", "")
        while "$(find-pkg-share " in path:
            start = path.index("$(find-pkg-share ")
            end = path.index(")", start)
            pkg = path[start + len("$(find-pkg-share "):end].strip()
            path = path[:start] + get_package_share_directory(pkg) + path[end + 1:]
        p["model_path"] = path
    TRICK_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    # ROS parameter files reject YAML aliases, which safe_dump emits for the lists
    # the controllers share with the course config.
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    TRICK_PARAMS.write_text(yaml.dump(params, Dumper=NoAliasDumper, sort_keys=False))
    # After the course's own controllers, and checked one by one: see spawn_controllers.py.
    spawn = HERE.parent / "spawn_controllers.py"
    return [ExecuteProcess(cmd=["python3", "-u", str(spawn), str(TRICK_PARAMS), *params], output="both")]


def person_follower():
    """The follower behind activate/deactivate_person_following.

    Your Lab 5 follow-me when ~/follow_me_pupper (or $FOLLOW_ME_DIR) has
    robot/follow_service.py and the YOLO model next to it (an older clone has
    both at the top level): it opens the Hailo only while
    following, with obstacle avoidance off. FOLLOW_ALWAYS_DETECT=1 keeps it
    detecting (slowly) between follows, so the on-screen eyes track people all
    the time; the robot runs warmer. Without that repo, the monorepo's Hailo
    detector (always on) and its simpler follower, which chases whichever
    person comes first and has no stop distance.
    """
    repo = Path(os.environ.get("FOLLOW_ME_DIR", Path.home() / "follow_me_pupper")).expanduser()
    service = next((d for d in (repo / "robot", repo)
                    if (d / "follow_service.py").exists() and (d / "yolov8n_seg.hef").exists()), None)
    if service is not None:
        always = os.environ.get("FOLLOW_ALWAYS_DETECT", "0").lower() in ("1", "true", "yes")
        print(f"[gemini_pupper] person follower: {service}/follow_service.py"
              + (" (always detecting, for the eyes)" if always else ""))
        return [ExecuteProcess(
            cmd=["python3", "-u", "follow_service.py", "--ros-args", "-r", "__node:=person_follower_node",
                 "-p", f"always_detect:={'true' if always else 'false'}"],
            cwd=str(service), output="both", respawn=True, respawn_delay=3.0)]
    print(f"[gemini_pupper] {repo} has no follow_service.py / yolov8n_seg.hef: "
          "using the monorepo person follower")
    return [
        Node(package="hailo", executable="hailo_detection", output="both", respawn=True, respawn_delay=3.0),
        Node(package="person_follower", executable="person_follower_node", output="both"),
    ]


def generate_launch_description():
    course = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("neural_controller"), "launch", "launch.py"])),
        launch_arguments={"teleop": "False"}.items(),  # re-added below, routed through the mux
    )
    return LaunchDescription([
        course,
        Node(package="teleop_twist_joy", executable="teleop_node", name="teleop_twist_joy_node",
             parameters=[COURSE_CONFIG], remappings=[("cmd_vel", "teleop_cmd_vel")], output="both"),
        Node(package="cmd_vel_mux", executable="cmd_vel_mux_node", parameters=[PARAMS], output="both"),
        Node(package="topic_tools", executable="throttle", name="joint_state_throttler",
             parameters=[PARAMS], arguments=["messages"], output="both"),
        *person_follower(),
        *gemini_controllers(),
    ])
