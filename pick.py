"""Pick & place v2 — top grasp of a perceived cube from a parked bot, put
down anywhere on the table.

Picking (steps 2-8):
  look      camera frame with the arm stowed -> detections (perception.py)
  plan      choose the cube; jaw direction = cube yaw or yaw+90 deg, whichever
            keeps the open pads clear of neighbouring cubes; IK for the
            waypoints with the claw vertical; keep-out check on every pose
            AND along every joint-space leg between them (as move_to will);
            the other cubes in view are obstacles for the closed-claw legs
  move      up (claw pulled back and raised) -> rise (to travel height, still
            short of the table) -> hover (over the cube) -> open claw ->
            pre (6 cm above) -> descend -> close
  verify    claw angle: ~asin(gap) for a cube-width grip = holding;
            ~0 = missed; anything else = bad grasp
  lift      back to pre, re-check the claw -> HOLDING

Placing (place_at(x, y[, yaw]), or automatically when given a dest):
  plan      the target must be on a table (the cube's corners inset from the
            edge), clear of the other cubes, with room for the open jaws; the
            cube is placed at `yaw` (default: as held). The cube turns with the
            jaws, so rotation = wrist roll; a cube looks the same every 90 deg,
            so the jaws go along the held direction, or turned 90 deg if that
            is blocked (choose_jaw — the UI previews the same choice); the carry
            legs are checked with the held cube's volume, the retreat with the
            newly placed cube as an obstacle
  move      hover -> over the target -> pre -> set down -> open -> pre ->
            hover -> close -> rise -> up -> stow
  verify    look again with the arm stowed: where did the cube really land?
place_back() is the same, to the spot it was picked from.
rotate_held(yaw) turns the held cube in the air (wrist roll) — the manual dial.

The Picker is a non-blocking state machine: call step(now) every physics
step. It only talks to the bot through the Hooks, so the same code drives
the fleet UI and the offline tests (and later the real bot).
"""

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

import perception

STOW = np.array([0.0, 0.3, -0.6, 0.0, 0.0])
CLAW_OPEN, CLAW_CLOSED = 1.2, 0.0
CLAW_LINK = 0.035                 # parallelogram link length (model_fleet.xml)
CLAW_GAP0 = 0.002                 # pad gap when closed
PAD_W = 0.010                     # pad half-width across the jaw axis
CLAW_PIVOT = 0.020                # link pivot offset from the claw centre
CLAW_OUTER = 0.0025               # link half-thickness: outermost claw edge beyond the jaw
UP_AHEAD, UP_Z = 0.03, -0.19      # "up" waypoint: tip just ahead of the bot, raised.
                                  # The stowed claw hangs BELOW the tabletop and 61 mm
                                  # ahead: swinging it forward first clips the table
RISE_AHEAD = 0.09                 # "rise" waypoint: travel height, just short of the table
HOVER_Z = -0.15                   # travel height over the table (closed pads ~50 mm over cube tops)
PRE_DZ = 0.06                     # pre-grasp height above the cube centre
GRASP_DZ = 0.003                  # tip target above the cube centre
OBST_MARGIN = 0.005               # clearance kept between open pads and other cubes
TABLE_EDGE = 0.005                # placed cube's corners stay this far inside the table edge
PLACE_GAP = 0.005                 # min gap between a placed cube and its neighbours
HELD_DZ = -0.004                  # held cube centre relative to the tip (measured: sits ~4 mm low)
HELD_R = perception.CUBE / 2 * np.sqrt(2)   # held cube's footprint radius, any yaw in the jaws


def claw_angle_for(width):
    """Claw link angle at which the pads are `width` apart."""
    return float(np.arcsin(np.clip((width - CLAW_GAP0) / (2 * CLAW_LINK), -1, 1)))


Q_HOLD = claw_angle_for(perception.CUBE)      # ~0.57 rad on a 40 mm cube
PRESHAPE_CLEAR = 0.008                        # per side, covers arm + perception error
Q_PRESHAPE = claw_angle_for(perception.CUBE + 2 * PRESHAPE_CLEAR)   # ~0.88 rad, 56 mm:
# open only as wide as the cube needs — a smaller footprint fits between
# neighbours, and the parallelogram's pads rise less (35*(1-cos q) mm)
HOLD_TOL = 0.12                               # rad
MISS_BELOW = 0.20                             # rad


@dataclass
class Hooks:
    arm: object                          # ArmController (move_to / done / keepout / be)
    set_claw: Callable[[float], None]
    claw_q: Callable[[], float]
    bot_pose: Callable[[], tuple]        # (x, y, yaw) from the rail encoders
    look: Callable[[], list]             # -> [perception.Detection]


def _wrap(a, period):
    return (a + period / 2) % period - period / 2


def _in_square(pts, det, grow):
    """Points (N,2) inside detection det's top face grown by `grow`."""
    c, s = np.cos(det.yaw), np.sin(det.yaw)
    local = (pts - [det.x, det.y]) @ np.array([[c, -s], [s, c]])
    return np.all(np.abs(local) <= perception.CUBE / 2 + grow, axis=1)


def _square_pts(det, n=5):
    """Grid of points covering det's top face (edges included)."""
    h = perception.CUBE / 2
    g = np.linspace(-h, h, n)
    uu, vv = np.meshgrid(g, g)
    c, s = np.cos(det.yaw), np.sin(det.yaw)
    return np.column_stack([det.x + c * uu.ravel() - s * vv.ravel(),
                            det.y + s * uu.ravel() + c * vv.ravel()])


def overlaps(a, b, gap):
    """Do cubes a and b (top faces) come within `gap` of each other?"""
    return bool(np.any(_in_square(_square_pts(a), b, gap)) or np.any(_in_square(_square_pts(b), a, gap)))


def pads_clear(det, jaw_yaw, others, q_open=Q_PRESHAPE):
    """Footprint of the open jaws around det (pad faces out to the links'
    outer edge — the jaw plates and links sink below neighbouring cube tops
    as the claw closes), jaws along jaw_yaw, vs other cubes."""
    inner = CLAW_GAP0 / 2 + CLAW_LINK * np.sin(q_open)
    outer = CLAW_PIVOT + CLAW_LINK * np.sin(q_open) + CLAW_OUTER
    u = np.linspace(inner, outer, 6)
    v = np.linspace(-PAD_W, PAD_W, 5)
    uu, vv = np.meshgrid(np.concatenate([u, -u]), v)
    c, s = np.cos(jaw_yaw), np.sin(jaw_yaw)
    pts = np.column_stack([det.x + c * uu.ravel() - s * vv.ravel(),
                           det.y + s * uu.ravel() + c * vv.ravel()])
    return not any(np.any(_in_square(pts, o, OBST_MARGIN)) for o in others)


def jaw_extent(q_open=Q_PRESHAPE):
    """(inner, outer, half-width) of each open jaw's footprint, measured from
    the claw centre along / across the jaw axis: pad face to the links'
    outer edge (what pads_clear checks)."""
    return (CLAW_GAP0 / 2 + CLAW_LINK * np.sin(q_open),
            CLAW_PIVOT + CLAW_LINK * np.sin(q_open) + CLAW_OUTER, PAD_W)


def jaw_rects(det, jaw_yaw, q_open=Q_PRESHAPE):
    """The two open jaws' footprints around det as (4,2) world corners."""
    inner, outer, w = jaw_extent(q_open)
    c, s = np.cos(jaw_yaw), np.sin(jaw_yaw)
    out = []
    for sg in (1.0, -1.0):
        uv = [(sg * inner, -w), (sg * outer, -w), (sg * outer, w), (sg * inner, w)]
        out.append(np.array([[det.x + c * u - s * v, det.y + s * u + c * v] for u, v in uv]))
    return out


def choose_jaw(det, base_jaw, others):
    """Jaw direction used to set det down: base_jaw if its jaws are clear of
    the other cubes, else base_jaw + 90 deg (the cube looks the same).
    Returns (jaw, clear). plan_place tries the same order, so the preview
    shows what will happen."""
    for k in (0, 1):
        jaw = base_jaw + k * np.pi / 2
        if pads_clear(det, jaw, others):
            return jaw, True
    return base_jaw, False


class Picker:
    def __init__(self, model, data, hooks, name=None, dest=None, tables=()):
        self.m, self.d, self.h = model, data, hooks
        self.want = name                 # cube colour to pick, or None = nearest
        self.dest = dest                 # (x, y) to place at after the pick, or None = hold
        self.tables = tables             # [(lo_xy, hi_xy, top_z)] — where cubes may be placed
        self.state, self.reason = "look", ""
        self.t0 = 0.0
        self.target = None               # Detection being picked
        self.dets = []                   # the look this pick was planned from
        self.jaw = None                  # jaw direction of the grasp (world yaw)
        self.dest_det = None             # where the cube is being put (planned pose)
        self.placed = None               # where the camera saw it land
        self.hold_yaw = None             # held cube's world yaw after rotate_held()
        self.wp = {}                     # waypoint name -> joint angles
        self._queue = []
        self._cur = ""
        self._scratch = mujoco.MjData(model)
        # link lengths from the model: shoulder->elbow, elbow->wrist pitch,
        # wrist pitch->tip (through the roll joint)
        be = hooks.arm.be
        body = [model.jnt_bodyid[j] for j in be.jids]
        self._links = (np.linalg.norm(model.body_pos[body[2]]),
                       np.linalg.norm(model.body_pos[body[3]]),
                       np.linalg.norm(model.body_pos[body[4]]) + np.linalg.norm(model.site_pos[be.tip]))

    # ---- status ----
    @property
    def busy(self):
        return self.state not in ("holding", "done", "failed")

    def label(self):
        s = self.state.upper()
        if self.state in ("place", "retreat") and self._cur:
            s += f" ({self._cur})"
        if self.target is not None:
            s += f" {self.target.name}"
        return s + (f" — {self.reason}" if self.reason else "")

    # ---- planning ----
    def ik(self, target, jaw_yaw, seed=None):
        """Tip -> target (world) with the claw vertical, jaws opening along
        world angle jaw_yaw. Analytic, §1.10: wrist centre = tip + L3 up;
        base yaw points the arm plane at it; 2-link planar solve for
        shoulder/elbow in the ELBOW-BACK branch (the elbow rises behind the
        shoulder, away from the table; the forearm reaches forward and down);
        wrist pitch = -(q2+q3) keeps the claw vertical; roll sets the jaws.
        Checked by FK on the model. Returns (q, ok)."""
        be, s = self.h.arm.be, self._scratch
        L1, L2, L3 = self._links
        byaw = self.h.bot_pose()[2]
        sh = self._shoulder()
        w = np.asarray(target, float) + [0.0, 0.0, L3]
        hx, hy = w[0] - sh[0], w[1] - sh[1]
        a, b = np.hypot(hx, hy), sh[2] - w[2]           # forward, drop
        D = (a * a + b * b - L1 * L1 - L2 * L2) / (2 * L1 * L2)
        if abs(D) > 1.0:
            return STOW.copy(), False
        beta = np.arccos(D)                             # elbow-back branch
        alpha = np.arctan2(a, b) - np.arctan2(L2 * np.sin(beta), L1 + L2 * np.cos(beta))
        q1 = _wrap(np.arctan2(hy, hx) - byaw, 2 * np.pi)
        q2, q3 = -alpha, -beta
        q = np.array([q1, q2, q3, -(q2 + q3), _wrap(jaw_yaw - q1 - byaw, np.pi)])
        s.qpos[:] = self.d.qpos
        for adr, v in zip(be.qadr, q):
            s.qpos[adr] = v
        mujoco.mj_kinematics(self.m, s)
        lo, hi = self.h.arm.soft_lo, self.h.arm.soft_hi
        ok = (np.linalg.norm(s.site_xpos[be.tip] - target) < 1e-3
              and np.all(q >= lo) and np.all(q <= hi))
        return q, ok

    def _shoulder(self):
        """Shoulder (J2) world position at the bot's current pose — on the
        base-yaw axis, so independent of q1."""
        return self.h.arm.be.fk()[0][0]

    def leg_hits(self, qa, qb, obstacles=(), carrying=False):
        """Keep-out zones — and obstacle cubes, checked against the claw's
        box corners at its CURRENT opening (plus the held cube's volume when
        carrying) — that a joint-space leg qa -> qb passes through (same
        sampling as ArmController.move_to)."""
        n = max(2, int(np.ceil(np.max(np.abs(qb - qa)) / 0.02)) + 1)
        hits = set()
        g = np.linspace(-1.0, 1.0, 3)
        held = np.array([[x * HELD_R, y * HELD_R, z * perception.CUBE / 2 + HELD_DZ]
                         for x in g for y in g for z in (-1.0, 1.0)])
        for s in np.linspace(0.0, 1.0, n):
            q = qa + (qb - qa) * s
            hits |= set(self.h.arm.keepout(q))
            if obstacles:
                pts, _, _, claw = self.h.arm.be.fk(q, claw=True)
                if carrying:
                    claw = np.vstack([claw, pts[3] + held])
                for o in obstacles:
                    inz = (claw[:, 2] >= o.z - perception.CUBE - OBST_MARGIN) & (claw[:, 2] <= o.z + OBST_MARGIN)
                    if np.any(inz & _in_square(claw[:, :2], o, OBST_MARGIN)):
                        hits.add(f"cube {o.name}")
        return sorted(hits)

    def plan(self, dets):
        bx, by, byaw = self.h.bot_pose()
        pool = [d for d in dets if self.want in (None, d.name)]
        if not pool:
            return f"{self.want or 'no'} cube in view"
        pool.sort(key=lambda d: np.hypot(d.x - bx, d.y - by))
        why = ""
        for det in pool:
            others = [o for o in dets if o is not det]
            for jaw in (det.yaw, det.yaw + np.pi / 2):
                if not pads_clear(det, jaw, others):
                    why = f"{det.name}: neighbours block the pads"
                    continue
                zc = det.z - perception.CUBE / 2          # cube centre height
                fwd = np.array([np.cos(byaw), np.sin(byaw)])
                goals = [("up", [*(np.array([bx, by]) + UP_AHEAD * fwd), UP_Z]),
                         ("rise", [*(np.array([bx, by]) + RISE_AHEAD * fwd), HOVER_Z]),
                         ("hover", [det.x, det.y, HOVER_Z]),
                         ("pre", [det.x, det.y, zc + PRE_DZ]),
                         ("grasp", [det.x, det.y, zc + GRASP_DZ])]
                wp, seed, bad = {}, STOW, None
                for nm, g in goals:
                    q, ok = self.ik(np.array(g), jaw, seed)
                    if not ok:
                        bad = f"{det.name}: {nm} out of reach"
                        break
                    hits = self.h.arm.keepout(q)
                    if hits:
                        bad = f"{det.name}: {nm} in keep-out {', '.join(hits)}"
                        break
                    wp[nm], seed = q, q
                if not bad:
                    # closed-claw travel legs see the other cubes; the open-claw
                    # descent is vertical over the target (pads_clear covers it)
                    legs = [("stow", STOW, wp["up"], others), ("up", wp["up"], wp["rise"], others),
                            ("rise", wp["rise"], wp["hover"], others),
                            ("hover", wp["hover"], wp["pre"], ()), ("pre", wp["pre"], wp["grasp"], ())]
                    for nm, qa, qb, obs in legs:
                        hits = self.leg_hits(qa, qb, obs)
                        if hits:
                            bad = f"{det.name}: path from {nm} crosses {', '.join(hits)}"
                            break
                if bad:
                    why = bad
                    continue
                self.target, self.wp, self.jaw, self.dets = det, wp, jaw, dets
                return None
        return why

    def table_of(self, x, y):
        for lo, hi, top in self.tables:
            if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]:
                return lo, hi, top
        return None

    def held_yaw(self):
        """World yaw the held cube has now (as picked, or as rotated)."""
        return self.target.yaw if self.hold_yaw is None else self.hold_yaw

    def _jaw_for(self, cube_yaw):
        """Jaw direction that puts the cube at cube_yaw (the cube keeps its
        orientation relative to the jaws from the grasp)."""
        return cube_yaw + (self.jaw - self.target.yaw)

    def plan_place(self, x, y, yaw=None):
        """Plan putting the held cube down at (x, y) with world yaw `yaw`
        (None = as held). Returns None (waypoints d_hover / d_pre / d_place
        stored) or the reason it can't."""
        src = self.target
        yaw = self.held_yaw() if yaw is None else yaw
        tbl = self.table_of(x, y)
        if tbl is None:
            return "that spot isn't on a table"
        lo, hi, top = tbl
        dest = perception.Detection(src.name, x, y, top + perception.CUBE, yaw, 0, None)
        corners = dest.corners_xy()
        if np.any(corners < np.array(lo) + TABLE_EDGE) or np.any(corners > np.array(hi) - TABLE_EDGE):
            return "too close to the table edge"
        others = [o for o in self.dets if o.name != src.name]
        for o in others:
            if overlaps(dest, o, PLACE_GAP):
                return f"too close to {o.name}"
        zc = dest.z - perception.CUBE / 2
        why = ""
        for k in (0, 1):                       # same order as choose_jaw
            jaw = self._jaw_for(yaw) + k * np.pi / 2
            if not pads_clear(dest, jaw, others):
                why = "no room for the jaws there"
                continue
            wp, bad = {}, None
            for nm, g in (("d_hover", [x, y, HOVER_Z]), ("d_pre", [x, y, zc + PRE_DZ]),
                          ("d_place", [x, y, zc + GRASP_DZ])):
                q, ok = self.ik(np.array(g), jaw)
                if not ok:
                    bad = f"{nm[2:]} out of reach"
                    break
                hits = self.h.arm.keepout(q)
                if hits:
                    bad = f"{nm[2:]} in keep-out {', '.join(hits)}"
                    break
                wp[nm] = q
            if not bad:
                # carry with the cube (other cubes are obstacles), set down
                # vertically (pads_clear + the overlap test cover it), retreat
                # with the claw closed past the cube just placed
                legs = [("carry", self.wp["pre"], self.wp["hover"], others, True),
                        ("carry", self.wp["hover"], wp["d_hover"], others, True),
                        ("lower", wp["d_hover"], wp["d_pre"], others, True),
                        ("set down", wp["d_pre"], wp["d_place"], (), False),
                        ("retreat", wp["d_hover"], self.wp["rise"], others + [dest], False)]
                for nm, qa, qb, obs, carry in legs:
                    hits = self.leg_hits(qa, qb, obs, carrying=carry)
                    if hits:
                        bad = f"{nm} path crosses {', '.join(hits)}"
                        break
            if bad:
                why = bad
                continue
            self.wp.update(wp)
            self.place_jaw = jaw
            self.dest_det = dest
            return None
        return why

    def rotate_held(self, now, yaw):
        """Turn the held cube (in the air, at the pre-grasp height) to world
        yaw `yaw` with the wrist roll. Of the equivalent rolls (every 90 deg)
        the one nearest the current roll is used, so a continuously turned
        dial gives smooth motion. Returns None, or why it can't."""
        if self.state != "holding":
            return "not holding a cube"
        q = self.wp["pre"].copy()
        byaw = self.h.bot_pose()[2]
        base = _wrap(self._jaw_for(yaw) - q[0] - byaw, np.pi / 2)
        cands = [base + n * np.pi / 2 for n in range(-4, 5)]
        cands = [r for r in cands if self.h.arm.soft_lo[4] <= r <= self.h.arm.soft_hi[4]]
        q[4] = min(cands, key=lambda r: abs(r - self.h.arm.wanted[4]))
        if self.h.arm.move_to(q, now) is None:
            return f"keep-out {', '.join(self.h.arm.refused)}"
        self.wp["pre"] = q
        self.hold_yaw = yaw
        return None

    # ---- execution ----
    def _go(self, state, now):
        self.state, self.t0 = state, now

    def _move(self, q, now):
        T = self.h.arm.move_to(q, now)
        if T is None:
            self.reason = f"{self.state} refused — keep-out {', '.join(self.h.arm.refused)}"
            self._fail(now)
            return False
        self._T = T
        return True

    def _arrived(self, now):
        arm = self.h.arm
        err = np.max(np.abs(arm.be.read()[:4] - arm.target[:4]))
        return (arm.done(now) and err < 0.03) or now - self.t0 > self._T + 2.0

    def _fail(self, now):
        """Retreat to hover (if we got that far) and stow, claw closed."""
        # retrace the approach (hover -> up -> stow): a direct hover -> stow
        # leg swings the claw through the table
        back = {"rise": ["up"], "hover": ["hover", "rise", "up"], "open": ["rise", "up"],
                "pre": ["hover", "rise", "up"], "descend": ["hover", "rise", "up"],
                "close": ["pre", "hover", "rise", "up"],
                "lift": ["pre", "hover", "rise", "up"]}.get(self.state, [])
        self._queue = [("move", w) for w in back] + [("claw", CLAW_CLOSED), ("move", "stow")]
        self._after = "failed"
        self._go("retreat", now)
        self._next(now)

    def _next(self, now):
        """Run the queued (kind, arg) actions of a retreat/place sequence."""
        if not self._queue:
            self._go(self._after, now)
            return
        kind, arg = self._queue.pop(0)
        self._cur = arg if kind == "move" else ("open" if arg > 0 else "close")
        self.t0 = now
        if kind == "move":
            q = STOW if arg == "stow" else self.wp[arg]
            self._sub = "move"
            if self.h.arm.move_to(q, now) is None:
                self.reason += f"; {arg} refused ({', '.join(self.h.arm.refused)})"
                self._go("failed", now)
                return
            self._T = self.h.arm.move.T
        else:
            self._sub = "claw"
            self.h.set_claw(arg)

    def place_at(self, now, x, y, yaw=None):
        """Put the held cube down at (x, y) with world yaw `yaw` (None = as
        held), release, stow, then look to verify. Returns None when
        started, else the reason it can't."""
        if self.state != "holding":
            return "not holding a cube"
        err = self.plan_place(x, y, yaw)
        if err:
            return err
        self._queue = [("move", "hover"), ("move", "d_hover"), ("move", "d_pre"),
                       ("move", "d_place"), ("claw", Q_PRESHAPE), ("move", "d_pre"),
                       ("move", "d_hover"), ("claw", CLAW_CLOSED), ("move", "rise"),
                       ("move", "up"), ("move", "stow")]
        self._after = "verify"
        self.reason = ""
        self._go("place", now)
        self._next(now)
        return None

    def place_back(self, now, keep_reason=False):
        """Put the held cube down where it was picked, release, stow, verify."""
        if self.state != "holding":
            return False
        reason = self.reason
        self.dest_det = self.target
        self._queue = [("move", "grasp"), ("claw", Q_PRESHAPE), ("move", "pre"),
                       ("move", "hover"), ("claw", CLAW_CLOSED), ("move", "rise"),
                       ("move", "up"), ("move", "stow")]
        self._after = "verify"
        self.reason = reason if keep_reason else ""
        self._go("place", now)
        self._next(now)
        return True

    def step(self, now):
        st = self.state
        if st == "look":
            err = self.plan(self.h.look())
            if not err and self.dest is not None:
                # the place plan needs only the pick plan: check it BEFORE
                # moving, rather than finding out while holding the cube
                err = self.plan_place(*self.dest)
                if err:
                    err = f"can't place {self.target.name} at ({self.dest[0]:.3f}, {self.dest[1]:.3f}): {err}"
            if err:
                self.reason = err
                self._go("failed", now)
                return
            self._go("up", now)
            self._move(self.wp["up"], now)
        elif st in ("up", "rise", "hover", "pre", "descend"):
            if not self._arrived(now):
                return
            if st == "up":
                self._go("rise", now); self._move(self.wp["rise"], now)
            elif st == "rise":
                self._go("hover", now); self._move(self.wp["hover"], now)
            elif st == "hover":
                self._go("open", now); self.h.set_claw(Q_PRESHAPE)
            elif st == "pre":
                self._go("descend", now); self._move(self.wp["grasp"], now)
            else:
                self._go("close", now); self.h.set_claw(CLAW_CLOSED)
        elif st == "open":
            if now - self.t0 > 0.8:
                self._go("pre", now); self._move(self.wp["pre"], now)
        elif st == "close":
            if now - self.t0 < 1.0:
                return
            q = self.h.claw_q()
            if q < MISS_BELOW:
                self.reason = f"missed (claw closed to {np.degrees(q):.0f}°)"
                self._fail(now)
            elif abs(q - Q_HOLD) > HOLD_TOL:
                self.reason = f"bad grasp (claw {np.degrees(q):.0f}°, want {np.degrees(Q_HOLD):.0f}°)"
                self.h.set_claw(Q_PRESHAPE)
                self._fail(now)
            else:
                self._go("lift", now); self._move(self.wp["pre"], now)
        elif st == "lift":
            if not self._arrived(now):
                return
            q = self.h.claw_q()
            if abs(q - Q_HOLD) > HOLD_TOL:
                self.reason = f"dropped during lift (claw {np.degrees(q):.0f}°)"
                self._fail(now)
            else:
                self._go("holding", now)
                if self.dest is not None:
                    err = self.place_at(now, *self.dest)
                    if err:
                        self.reason = (f"can't place at ({self.dest[0]:.3f}, {self.dest[1]:.3f}): "
                                       f"{err} — put it back")
                        self.place_back(now, keep_reason=True)
        elif st == "verify":                  # arm stowed: where did it really land?
            seen = [d for d in self.h.look() if d.name == self.target.name]
            goal = self.dest_det
            if seen:
                self.placed = min(seen, key=lambda d: np.hypot(d.x - goal.x, d.y - goal.y))
                err = np.hypot(self.placed.x - goal.x, self.placed.y - goal.y) * 1000
                yerr = np.degrees(perception.yaw_err(self.placed.yaw, goal.yaw))
                note = (f"placed at ({self.placed.x:.3f}, {self.placed.y:.3f}) "
                        f"{np.degrees(self.placed.yaw):+.0f}°, {err:.1f} mm / {yerr:+.1f}° from target")
            else:
                note = "placed — but the camera can't see it now"
            self.reason = f"{self.reason}; {note}" if self.reason else note
            self._go("done", now)
        elif st in ("retreat", "place"):
            if self._sub == "claw":
                if now - self.t0 > 0.8:
                    self._next(now)
            elif self._arrived(now):
                self._next(now)
