"""Keep-out zones: coordinates the arm may never occupy.

Each zone is an axis-aligned box in one of two frames:
  "world" — fixed to the grid (§1.9 frame 1): floor, rails, frame posts
  "bot"   — fixed to the arm's own bot body (§1.9 frame 2), so it moves and
            turns with the bot: the chassis/yaw housing above the arm base

The arm is checked as capsules: shoulder->elbow, elbow->wrist, wrist->tip,
each with its link radius, plus MARGIN. When the claw's box corners are
given (claw_pts), the last capsule stops at the roll joint and the claw is
checked as its real geometry at the live opening, with CLAW_MARGIN — tight
enough for the pads to straddle a cube on a table. Pure numpy — on hardware the points
come from FK on read-back instead of MuJoCo.

Edit ZONES to add places (e.g. a socket rack, the charging bay).
"""

from dataclasses import dataclass

import numpy as np

BIG = 5.0                      # "unbounded" extent for half-spaces (m)
MARGIN = 0.010                 # clearance on top of the link radius (m)
LINK_R = (0.032, 0.028, 0.025) # upper arm, forearm, wrist+gripper (model_fleet.xml)
STEP = 0.005                   # capsule sampling along each link (m)
WRIST_LEN = 0.090              # wrist pitch -> roll joint; the claw starts here
CLAW_MARGIN = 0.005            # clearance for the claw's own geometry (m)


@dataclass
class Zone:
    name: str
    lo: tuple
    hi: tuple
    frame: str = "world"       # "world" | "bot"
    note: str = ""


ZONES = [
    # ---- world (grid frame) ----
    Zone("floor", (-BIG, -BIG, -BIG), (BIG, BIG, -0.50),
         note="floor plane z=-0.50; raise hi-z to the board top once the board is modeled"),
    Zone("rail_level", (-BIG, -BIG, 0.30), (BIG, BIG, BIG),
         note="rails/turntables bottom flange at z=0.360, minus 60 mm clearance"),
    Zone("post_SW", (0.021, 0.031, -0.50), (0.041, 0.051, 0.50), note="frame corner post"),
    Zone("post_NW", (0.021, 0.731, -0.50), (0.041, 0.751, 0.50), note="frame corner post"),
    Zone("post_SE", (1.781, 0.031, -0.50), (1.801, 0.051, 0.50), note="frame corner post"),
    Zone("post_NE", (1.781, 0.731, -0.50), (1.801, 0.751, 0.50), note="frame corner post"),
    Zone("table_A", (1.800, 0.131, -0.50), (1.960, 0.331, -0.25),
         note="lane-A end table: top z=-0.25, legs to the floor (solid box). "
              "Its cubes are work objects, not keep-outs"),
    Zone("table_B", (1.800, 0.451, -0.50), (1.960, 0.651, -0.25),
         note="lane-B end table, same as table_A"),
    # ---- bot frame (origin = bot body at the axle plane, z up) ----
    Zone("own_housing", (-0.085, -0.085, -0.260), (0.085, 0.085, 0.0), "bot",
         note="chassis + yaw stage O170 (as a box); bottom 5 mm above the J1 plane"),
    Zone("own_camera", (0.085, -0.010, -0.225), (0.109, 0.010, -0.205), "bot",
         note="down-looking USB camera on the stage side"),
]


def _box_dist(p, lo, hi):
    """Distance from points p (N,3) to an axis-aligned box (0 inside)."""
    d = np.maximum(np.maximum(lo - p, 0.0), p - hi)
    return np.linalg.norm(d, axis=1)


def _sample(a, b):
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / STEP)) + 1)
    return a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]


def check(pts, bot_pos, bot_mat, claw_pts=None, zones=ZONES, margin=MARGIN):
    """pts = [shoulder, elbow, wrist, tip] in world coords.
    bot_pos (3,), bot_mat (3,3) = the bot body's world pose.
    claw_pts (N,3) = claw box corners in world coords, or None to treat the
    claw as part of the wrist capsule (conservative).
    Returns the sorted names of zones the arm enters ([] = clear)."""
    pts = np.asarray(pts, float).copy()
    R = np.asarray(bot_mat, float).reshape(3, 3)
    if claw_pts is not None:                   # wrist capsule ends at the roll joint
        pts[3] = pts[2] + (pts[3] - pts[2]) * WRIST_LEN / np.linalg.norm(pts[3] - pts[2])
    groups = [(_sample(pts[k], pts[k + 1]), r + margin) for k, r in enumerate(LINK_R)]
    if claw_pts is not None:
        groups.append((np.asarray(claw_pts, float), CLAW_MARGIN))
    hit = set()
    for seg_w, clear in groups:
        seg_b = (seg_w - bot_pos) @ R          # world -> bot frame
        for z in zones:
            seg = seg_b if z.frame == "bot" else seg_w
            if z.name not in hit and np.any(_box_dist(seg, np.array(z.lo), np.array(z.hi)) < clear):
                hit.add(z.name)
    return sorted(hit)
