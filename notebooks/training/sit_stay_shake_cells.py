# Sit, Stay, Shake notebook cells (CS123_Pupper_Sit_Stay_Shake.ipynb, TODOs 1-4).
# Pasted into the notebook, where md, np, t (tricks.sit_stay_shake), whole and left are defined.

def cut(motion: md.Motion, start_s: float, end_s: float) -> md.Motion:
  # Seconds -> frame numbers; +1 keeps the end frame, so neighbours share it.
  i0 = round(start_s * motion.fps)
  i1 = round(end_s * motion.fps)
  return md.Motion(
    fps=motion.fps,
    root_pos=motion.root_pos[i0 : i1 + 1].copy(),
    root_quat=motion.root_quat[i0 : i1 + 1].copy(),
    joint_pos=motion.joint_pos[i0 : i1 + 1].copy(),
  )


def hold(motion: md.Motion, at_s: float, seconds: float) -> md.Motion:
  # Repeat one frame: seconds * fps + 1 frames, so it starts and ends on that pose.
  i = round(at_s * motion.fps)
  n = round(seconds * motion.fps) + 1
  return md.Motion(
    fps=motion.fps,
    root_pos=np.repeat(motion.root_pos[i : i + 1], n, axis=0),
    root_quat=np.repeat(motion.root_quat[i : i + 1], n, axis=0),
    joint_pos=np.repeat(motion.joint_pos[i : i + 1], n, axis=0),
  )


def mirror(motion: md.Motion) -> md.Motion:
  # Left and right legs are mounted as mirror images: in the default stand the
  # right front leg is (0.26, 0, -0.52) and the left (-0.26, 0, 0.52). So every
  # joint (abduction, hip, knee) flips sign when it moves to the other side.
  root_pos = motion.root_pos.copy()
  root_pos[:, 1] *= -1
  root_quat = motion.root_quat * np.array([1.0, -1.0, 1.0, -1.0])
  names = list(md.JOINT_NAMES)
  joint_pos = np.empty_like(motion.joint_pos)
  for i, name in enumerate(names):
    other = name.replace("_l_", "_X_").replace("_r_", "_l_").replace("_X_", "_r_")
    joint_pos[:, i] = -motion.joint_pos[:, names.index(other)]
  return md.Motion(fps=motion.fps, root_pos=root_pos, root_quat=root_quat, joint_pos=joint_pos)


def clips() -> list[md.Clip]:
  sitting = whole
  shake_l = left
  return [
    md.Clip("sit_down", cut(sitting, 0.0, t.SITTING_FROM_S), next="sit_idle"),
    md.Clip("sit_idle", hold(sitting, t.SITTING_FROM_S, 2.0), loop=True, next="sit_idle"),
    md.Clip("shake_left", shake_l, next="sit_idle"),
    md.Clip("shake_right", mirror(shake_l), next="sit_idle"),
    md.Clip("stand_up", cut(sitting, t.SITTING_UNTIL_S, sitting.duration), next=None),
  ]


# Every change the robot can make: sit_down->idle, idle->idle (the loop),
# idle->each shake, each shake->idle, shake->shake via idle, idle->stand_up.
ORDER = [
  "sit_down", "sit_idle", "sit_idle", "shake_left", "sit_idle", "shake_right",
  "sit_idle", "shake_left", "sit_idle", "shake_right", "sit_idle", "stand_up",
]
