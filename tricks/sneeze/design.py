"""The sneeze, designed here (the monorepo's recording of it barely moves).

    cd pupper-mjlab && uv run python <this file> <out.npz>

ah... ah... (nose up, a slow inhale) CHOO! (the front legs fold fast, nose down)
then a tiny hop on the front feet, and back to a level stand. Trained with
Mjlab-Tracking-Flat-Pupper-v3 plus one extra reward: the front lower legs tracked
to 2 cm (the whole-body term allows 10), or the policy skips the 2.5 cm hop.
"""

import sys

from mjlab.tasks.tracking.config.pupper import motion_design as md

HOP = 0.025  # front feet lift in the hop (m)
DROP = 0.035  # root drop at the CHOO (m)
CHOO_PITCH = 22.0  # nose down at the CHOO (deg)
CHOO_S = 0.18  # how fast the front collapses (s); faster raises the joint speed past 14 rad/s
HOP_S = 0.20  # collapse -> top of the hop (s)


def keyframes() -> list[md.Keyframe]:
  stand = md.stand_feet()
  fr, fl, br, bl = (stand[leg] for leg in md.LEGS)
  h = md.STAND_HEIGHT
  planted = {"front_r": fr, "front_l": fl, "back_r": br, "back_l": bl}
  lifted = dict(planted, front_r=(fr[0], fr[1], fr[2] + HOP), front_l=(fl[0], fl[1], fl[2] + HOP))
  choo = 1.05 + CHOO_S
  top = choo + HOP_S
  return [
    md.stand(0.0),
    md.stand(0.3),
    md.Keyframe(0.75, (-0.01, 0.0, h + 0.004), (0, -8, 0), feet=planted),  # ah...
    md.Keyframe(1.05, (-0.015, 0.0, h + 0.006), (0, -10, 0), feet=planted),  # ...ah (inhale)
    md.Keyframe(choo, (0.01, 0.0, h - DROP), (0, CHOO_PITCH, 0), feet=planted),  # CHOO: front legs fold
    md.Keyframe(top, (0.0, 0.0, h + 0.012), (0, -6, 0), feet=lifted),  # a tiny hop on the front feet
    md.Keyframe(top + 0.14, (0.0, 0.0, h - 0.008), (0, 3, 0), feet=planted),  # land, absorb
    md.Keyframe(top + 0.44, (0.0, 0.0, h), feet=planted),
    md.stand(top + 0.94),
  ]


if __name__ == "__main__":
  m = md.build_motion(keyframes())
  m.report = md.check_motion(m)
  for k, v in m.report.items():
    print(f"{k:36} {v}")
  print("saved", m.save_npz(sys.argv[1] if len(sys.argv) > 1 else "sneeze.npz"))
