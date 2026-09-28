"""LocomotionBot FLEET v2 — space-time reservation planning ("string diagram").

Run:  python run_fleet2.py          (add --log to record joint world coords from t=0;
                                     or press l in the window to start/stop)

THE THEORY (Sai's): know grid + positions + targets -> simulate each plan into
timestamped positions -> overlay -> find collisions in (space x time) ->
insert WAITs (or reroute / shunt) -> verify clean -> only then move.

Panels:  left = 3D | middle = map (click to dispatch, 1/2/0 select) |
         bottom = live STRING DIAGRAM (x-position vs time; amber = station
         locks; vertical line = now).

Architecture change vs run_fleet.py:
  - Conflicts are found AT PLANNING TIME by simulate_plan() + conflict();
    resolution = start-WAIT search x station choice, then plan-time shunt of
    an idle blocker, else the task is REFUSED with a reason.
  - The reactive rules (R1 proximity + escape, R2 zone, R5 rotated table)
    remain armed as a pure SEATBELT with no timers/resolvers. Each bot counts
    reflex activations: a verified schedule should keep that count at ZERO.
"""

import sys
import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from arm_control import ArmController, SimBackend, model_sag, move_time
from arm_log import JointLogger
import keepout
import perception
import pick
from matplotlib.patches import Polygon, Circle, Rectangle

# ---------------- world constants (drawing-verified) ----------------
LANE_A, LANE_B = 0.231, 0.551
ST_LEFT, ST_RIGHT = 0.211, 1.251
X_MIN, X_MAX = 0.09, 1.70
TURN_TIME = 3.0
DEG90 = 1.5708
V_MAX = 0.13
STATION_ZONE = 0.27         # swept segment radius + bot half-length
BODY_CLEAR = 0.29           # inflated body-body clearance
T_BUF = 1.5                 # time buffer (s) on every reservation
SETTLE = 0.6                # extra seconds per drive segment (decel/approach)
DT = 0.25                   # planning sample step (s)
HORIZON_PAD = 8.0
# arm envelope (as-drawn links): upper / forearm / wrist+gripper
L1, L2, L3 = 0.1357, 0.1861, 0.245
POSES = {"STOW": (0.0, 0.3, -0.6, 0.0, 0.0),
         "WORK": (0.0, -0.9, 0.6, -0.2, 0.0)}
CLAW_OPEN, CLAW_CLOSED = 1.2, 0.0   # claw link angle (rad): ~67 mm / 2 mm parallel pad gap
ARM_PAD = 0.8               # per ARM prim: settle/certify time on top of the computed trajectory T
WORK_R = 0.30               # work-disc radius (WORK pose reach + margin)
STOW_R_MAX = 0.15           # certified-stowed envelope ceiling
SEG_SWEEP = 0.146           # turntable segment swing radius
WORK_DWELL = 8.0            # demo work time at the target

ARM_HW = 0.06               # arm capsule half-width (link dia + sway margin)
Z_BASE = 0.145              # J2 shoulder height (world z)
CH_R, CH_Z = 0.17, (0.23, 0.49)     # chassis slab: vertical capsule
COL_R, COL_Z = 0.14, (-0.31, 0.23)  # hanging stowed-arm column

def reach_of(q2, q3, q4):
    """Horizontal arm reach from joint feedback (signed magnitude)."""
    return abs(L1 * np.sin(q2) + L2 * np.sin(q2 + q3) + L3 * np.sin(q2 + q3 + q4))

def fk3(bx, by, heading, q2, q3, q4):
    """3D arm points [shoulder, elbow, tip] in world frame."""
    ch, sh = np.cos(heading), np.sin(heading)
    pts = [(bx, by, Z_BASE)]
    ang, r, z = 0.0, 0.0, 0.0
    for L, q in ((L1, q2), (L2 + 0.0, q3), (L3, q4)):
        ang += q
        r -= L * np.sin(ang)      # MuJoCo hinge: +pitch tips the link toward -x
        z -= L * np.cos(ang)
        pts.append((bx + r * ch, by + r * sh, Z_BASE + z))
    return [pts[0], pts[1], pts[3]]     # shoulder, elbow, tip (forearm+wrist merged)

def seg3_dist(a0, a1, b0, b1):
    """Min distance between 3D segments (standard closed-form)."""
    a0, a1, b0, b1 = map(np.asarray, (a0, a1, b0, b1))
    d1, d2, r = a1 - a0, b1 - b0, a0 - b0
    A, E, F = d1 @ d1, d2 @ d2, d2 @ r
    if A < 1e-12 and E < 1e-12:
        return float(np.linalg.norm(r))
    if A < 1e-12:
        s, t = 0.0, np.clip(F / E, 0.0, 1.0)
    else:
        C = d1 @ r
        if E < 1e-12:
            t, s = 0.0, np.clip(-C / A, 0.0, 1.0)
        else:
            B = d1 @ d2
            den = A * E - B * B
            s = np.clip((B * F - C * E) / den, 0.0, 1.0) if den > 1e-12 else 0.0
            t = (B * s + F) / E
            if t < 0.0:
                t, s = 0.0, np.clip(-C / A, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((B - C) / A, 0.0, 1.0)
    return float(np.linalg.norm(a0 + d1 * s - (b0 + d2 * t)))

def seg_pt_dist(x0, y0, x1, y1, px, py):
    dx, dy = x1 - x0, y1 - y0
    L2s = dx * dx + dy * dy
    if L2s < 1e-12:
        return float(np.hypot(px - x0, py - y0))
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / L2s))
    return float(np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)))

def seg_seg_dist(a, b):
    """Min distance between segments a=(x0,y0,x1,y1), b likewise."""
    return min(seg_pt_dist(*a, b[0], b[1]), seg_pt_dist(*a, b[2], b[3]),
               seg_pt_dist(*b, a[0], a[1]), seg_pt_dist(*b, a[2], a[3]))

model = mujoco.MjModel.from_xml_path("model_fleet.xml")
data = mujoco.MjData(model)
TT_ACT = {ST_LEFT: model.actuator("tt_left").id, ST_RIGHT: model.actuator("tt_right").id}
TT_SEN = {ST_LEFT: model.sensor("tt_left_pos").id, ST_RIGHT: model.sensor("tt_right_pos").id}

# ---------------- route construction (pure) ----------------
def plan_route(cx, cy, tx, target_lane, via_station=None):
    plan = []
    on_lane = min((LANE_A, LANE_B), key=lambda L: abs(cy - L))
    if abs(cy - on_lane) > 0.02:
        st = ST_LEFT if abs(cx - ST_LEFT) < abs(cx - ST_RIGHT) else ST_RIGHT
        plan += [("DRIVE_Y", target_lane), ("TURN", st, 0.0)]
        on_lane = target_lane
    if abs(on_lane - target_lane) > 0.01:
        st = via_station if via_station is not None else \
             min((ST_LEFT, ST_RIGHT), key=lambda s: abs(cx - s) + abs(tx - s))
        plan += [("DRIVE_X", st), ("TURN", st, DEG90),
                 ("DRIVE_Y", target_lane), ("TURN", st, 0.0)]
    plan += [("DRIVE_X", tx)]
    return plan

def detour_route(x, y, tx, lane, S1, S2):
    oth = LANE_B if abs(lane - LANE_A) < 0.01 else LANE_A
    return [("DRIVE_X", S1), ("TURN", S1, DEG90), ("DRIVE_Y", oth), ("TURN", S1, 0.0),
            ("DRIVE_X", S2), ("TURN", S2, DEG90), ("DRIVE_Y", lane), ("TURN", S2, 0.0),
            ("DRIVE_X", tx)]

def simulate_plan(x, y, t0, plan, q_arm=POSES["STOW"]):
    """Pure arithmetic: plan -> samples, station locks, work discs
    [(cx,cy,r,ta,tb)]. q_arm = the arm's commanded pose at t0: ARM prims take
    the same computed trajectory T the executor will run, plus ARM_PAD."""
    samples, locks, discs = [(t0, x, y)], [], []
    disc_open = None
    t = t0
    for prim in plan:
        if prim[0] == "ARM":
            dur = move_time(q_arm, POSES[prim[1]]) + ARM_PAD
            q_arm = POSES[prim[1]]
            end = t + dur
            while t < end:
                t = min(t + DT, end)
                samples.append((t, x, y))
            if prim[1] == "WORK":
                disc_open = (x, y, t - dur - T_BUF)
            elif prim[1] == "STOW" and disc_open is not None:
                # planned envelope: the WORK pose's real 3D arm, heading +x
                p = fk3(disc_open[0], disc_open[1], 0.0,
                        POSES["WORK"][1], POSES["WORK"][2], POSES["WORK"][3])
                for a, b in ((p[0], p[1]), (p[1], p[2])):
                    discs.append((*a, *b, ARM_HW, disc_open[2], t + T_BUF))
                disc_open = None
        elif prim[0] == "WAIT":
            end = t + prim[1]
            while t < end:
                t = min(t + DT, end)
                samples.append((t, x, y))
            end = t
        elif prim[0] in ("DRIVE_X", "DRIVE_Y"):
            axis = 0 if prim[0] == "DRIVE_X" else 1
            start = x if axis == 0 else y
            dist = abs(prim[1] - start)
            sgn = 1 if prim[1] > start else -1
            seg_t0, end = t, t + dist / V_MAX
            while t < end - 1e-9:
                t = min(t + DT, end)
                p = start + sgn * V_MAX * (t - seg_t0)
                samples.append((t, p, y) if axis == 0 else (t, x, p))
            if axis == 0:
                x = prim[1]
            else:
                y = prim[1]
            t = end + SETTLE
            samples.append((t, x, y))
        elif prim[0] == "TURN":
            end = t + TURN_TIME + 0.5
            locks.append((prim[1], t - T_BUF, end + T_BUF))
            while t < end:
                t = min(t + DT, end)
                samples.append((t, x, y))
    if disc_open is not None:            # never-stowed safety: envelope to horizon
        p = fk3(disc_open[0], disc_open[1], 0.0,
                POSES["WORK"][1], POSES["WORK"][2], POSES["WORK"][3])
        for a, b in ((p[0], p[1]), (p[1], p[2])):
            discs.append((*a, *b, ARM_HW, disc_open[2], t + 1e5))
    return samples, locks, discs, t

def pos_at(samples, t):
    """Position at time t; held at endpoints (a parked bot stays parked)."""
    if t <= samples[0][0]:
        return samples[0][1], samples[0][2]
    if t >= samples[-1][0]:
        return samples[-1][1], samples[-1][2]
    ts = [s[0] for s in samples]
    i = int(np.searchsorted(ts, t))
    a, b = samples[i - 1], samples[i]
    f = (t - a[0]) / max(b[0] - a[0], 1e-9)
    return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])

def conflict(sA, lA, dA, sB, lB, dB):
    """First conflict between two reservations, or None. Bodies, station
    locks, and work discs — one overlap rule for all of them."""
    t0 = min(sA[0][0], sB[0][0])
    t1 = max(sA[-1][0], sB[-1][0]) + HORIZON_PAD
    t = t0
    while t <= t1:
        xb, yb = pos_at(sB, t)
        for db in (-T_BUF, 0.0, T_BUF):
            xa, ya = pos_at(sA, t + db)
            if abs(xa - xb) < BODY_CLEAR and abs(ya - yb) < BODY_CLEAR:
                return ("body", t, xa, ya)
        for st, ta, tb in lA:
            if ta <= t <= tb and abs(xb - st) < STATION_ZONE:
                return ("lockA", t, st, xb)
        xa, ya = pos_at(sA, t)
        for st, ta, tb in lB:
            if ta <= t <= tb and abs(xa - st) < STATION_ZONE:
                return ("lockB", t, st, xa)
        def arm_vs_bot(entries, own_entries, px, py):
            """3D: arm segments vs (chassis slab + stow column when the
            target bot's arm is tucked; its own capsules cover it otherwise)."""
            col = not any(e[7] <= t <= e[8] for e in own_entries)
            for x0, y0, z0, x1, y1, z1, hw, ta, tb in entries:
                if not (ta <= t <= tb):
                    continue
                if seg3_dist((x0, y0, z0), (x1, y1, z1),
                             (px, py, CH_Z[0]), (px, py, CH_Z[1])) < hw + CH_R:
                    return True
                if col and seg3_dist((x0, y0, z0), (x1, y1, z1),
                                     (px, py, COL_Z[0]), (px, py, COL_Z[1])) < hw + COL_R:
                    return True
            return False
        if arm_vs_bot(dA, dB, xb, yb):
            return ("armA", t, xb, yb)
        if arm_vs_bot(dB, dA, xa, ya):
            return ("armB", t, xa, ya)
        t += DT
    for stA, a0, a1 in lA:
        for stB, b0, b1 in lB:
            if stA == stB and a0 < b1 and b0 < a1:
                return ("locklock", max(a0, b0), stA, 0)
    for c1 in dA:                                # arm vs arm — full 3D
        for c2 in dB:
            if c1[7] < c2[8] and c2[7] < c1[8] and \
                    seg3_dist(c1[0:3], c1[3:6], c2[0:3], c2[3:6]) < c1[6] + c2[6]:
                return ("armarm", max(c1[7], c2[7]), c1[0], c1[1])
    for x0, y0, z0, x1, y1, z1, hw, ta, tb in dA + dB:   # capsule vs lock (x-extent)
        for st, l0, l1 in lA + lB:
            if ta < l1 and l0 < tb and \
                    min(x0, x1) - hw < st + SEG_SWEEP + 0.03 and \
                    max(x0, x1) + hw > st - SEG_SWEEP - 0.03:
                return ("armlock", max(ta, l0), st, x0)
    return None

def make_arm_ctl(be):
    """Controller with keep-out checking: keepout(q) -> zone names the arm
    would enter at angles q (q=None -> the live measured pose)."""
    return ArmController(be, sag=model_sag(be),
                         keepout=lambda q: keepout.check(*be.fk(q, claw=True)))

# ---------------- per-bot executor (reflexes = seatbelt only) ----------------
class Bot:
    def __init__(self, name, x0, y0, act, sen, arm, tip, body, color, pri):
        self.name, self.x0, self.y0, self.color, self.pri = name, x0, y0, color, pri
        self.A = {k: model.actuator(v).id for k, v in act.items()}
        self.S = {k: model.sensor(v).id for k, v in sen.items()}
        self.arm_ctl = make_arm_ctl(SimBackend(model, data, arm, tip=tip, body=body))
        self.arm_T = 0.0                 # computed T of the running ARM prim
        self.queue, self.prim = [], None
        self.prim_t0 = self.turn_from = 0.0
        self.turn_started = False
        self.target = None
        self.reflex_holds = 0            # seatbelt activations (want: 0)
        self.plan_samples = None         # committed reservation, for drawing
        self.plan_locks = []
        self.plan_discs = []
        self._stow_latch = True          # hysteresis: certify <0.13, revoke >0.19
        self._calm_since = None          # ARM-STOW settle window tracker

    def xy(self):
        return (self.x0 + data.sensordata[self.S["x"]],
                self.y0 + data.sensordata[self.S["y"]])

    def yaw(self):
        return data.sensordata[self.S["yaw"]]

    def claw_q(self):
        """Measured claw opening (rad); stalls short of CLAW_CLOSED on an object."""
        return data.sensordata[self.S["claw"]]

    def set_claw(self, v):
        """Claw servo goal — its own channel, not part of the arm spline (§1.8)."""
        data.ctrl[self.A["claw"]] = float(np.clip(v, CLAW_CLOSED, CLAW_OPEN))

    def arm_q(self):
        return (data.sensordata[self.S["q2"]],
                data.sensordata[self.S["q3"]],
                data.sensordata[self.S["q4"]])

    def arm_segments(self):
        """Live 3D envelope: [shoulder->elbow, elbow->tip] in world frame."""
        x, y = self.xy()
        heading = self.yaw() + data.sensordata[self.S["q1"]]
        p = fk3(x, y, heading, *self.arm_q())
        return [(p[0], p[1]), (p[1], p[2])]

    def reach(self):
        return reach_of(*self.arm_q())

    def stowed(self):
        """The pose contract with hysteresis — servo overshoot cannot
        flicker the certification (certify < 0.13 m, revoke > 0.19 m)."""
        r = self.reach()
        if r < 0.13:
            self._stow_latch = True
        elif r > 0.19:
            self._stow_latch = False
        # mid-pick / holding a cube the arm is out on purpose, even when its
        # reach is short: never certified (seatbelt holds, reservation carries a disc)
        return self._stow_latch and not picking(self)

    def idle(self):
        return self.prim is None and not self.queue

    def turning(self):
        return self.prim is not None and self.prim[0] == "TURN"

    def stop_drives(self):
        data.ctrl[self.A["dx"]] = data.ctrl[self.A["dy"]] = 0.0

    def reservation(self, now):
        """Committed samples, or a parked point if idle. An idle bot with an
        un-stowed arm carries a live disc — it blocks like a working bot."""
        if self.plan_samples and not self.idle():
            return self.plan_samples, self.plan_locks, self.plan_discs
        x, y = self.xy()
        discs = []
        if not self.stowed():
            for a, b in self.arm_segments():
                discs.append((*a, *b, ARM_HW, now - 1.0, now + 1e5))
        return [(now, x, y)], [], discs

    def start_next(self):
        self.prim = self.queue.pop(0) if self.queue else None
        self.prim_t0 = data.time
        self.turn_started = False
        self._calm_since = None
        if self.prim is not None and self.prim[0] == "ARM":
            T = self.arm_ctl.move_to(POSES[self.prim[1]], data.time)
            if T is None:
                say(f"{self.name}: ARM {self.prim[1]} refused — keep-out "
                    f"{', '.join(self.arm_ctl.refused)}", 5.0)
                if self.prim[1] != "STOW":
                    return self.start_next()     # skip the extend, arm stays put
                T = 0.0                          # a refused STOW holds as a fault
            self.arm_T = T
        if self.prim is None:
            self.target = None

    def step(self, other):
        if self.prim is None:
            self.stop_drives()
            return
        x, y = self.xy()
        kind = self.prim[0]
        if kind == "WAIT":
            self.stop_drives()
            if data.time - self.prim_t0 > self.prim[1]:
                self.start_next()
        elif kind == "ARM":
            self.stop_drives()
            tgt = POSES[self.prim[1]]
            traj_done = self.arm_ctl.done(data.time)    # trajectory ran to T
            q2, q3, q4 = self.arm_q()
            settled = traj_done and max(abs(q2 - tgt[1]), abs(q3 - tgt[2]),
                                        abs(q4 - tgt[3])) < 0.12
            if self.prim[1] == "STOW":
                # ringing must DIE before certification: reach continuously
                # under 0.13 for 0.6 s (post-overshoot), not just touched once
                if self.reach() < 0.13:
                    if self._calm_since is None:
                        self._calm_since = data.time
                else:
                    self._calm_since = None
                settled = (traj_done and self._calm_since is not None
                           and data.time - self._calm_since > 0.6)
            # the timeout NEVER bypasses a STOW: a jammed stow holds as a
            # visible fault rather than releasing an extended arm into motion
            timeout = (self.prim[1] != "STOW"
                       and data.time - self.prim_t0 > self.arm_T + 3.0)
            if (settled and data.time - self.prim_t0 > 0.4) or timeout:
                self.start_next()
        elif kind in ("DRIVE_X", "DRIVE_Y"):
            axis = 0 if kind == "DRIVE_X" else 1
            cur = x if axis == 0 else y
            err = self.prim[1] - cur
            ox, oy = other.xy()
            perp = abs(y - oy) if axis == 0 else abs(x - ox)
            gap = (ox - x) if axis == 0 else (oy - y)
            blocked = (perp < 0.24 and abs(gap) < 0.32
                       and np.sign(gap) == np.sign(err) and abs(err) > 0.004)
            if blocked:
                tgt = self.prim[1]
                oax = ox if axis == 0 else oy
                if abs(tgt - oax) >= 0.27 and (tgt - oax) * ((x if axis == 0 else y) - oax) > 0:
                    blocked = False
            if not blocked and axis == 0:
                for st in (ST_LEFT, ST_RIGHT):
                    ang = data.sensordata[TT_SEN[st]]
                    peer_turn = other.turning() and other.turn_started and other.prim[1] == st
                    if ang > 0.05 or peer_turn:
                        gap_st = st - x
                        if abs(gap_st) < STATION_ZONE + 0.06 and np.sign(gap_st) == np.sign(err):
                            blocked = True
                            break
            if not self.stowed():                # SEATBELT — never drive extended
                self.stop_drives()
                self.reflex_holds += 1
                return
            if blocked:                          # SEATBELT — schedule failed us
                self.stop_drives()
                self.reflex_holds += 1
                return
            v = float(np.clip(3.0 * err, -V_MAX, V_MAX))
            data.ctrl[self.A["dx"] if axis == 0 else self.A["dy"]] = v
            data.ctrl[self.A["dy"] if axis == 0 else self.A["dx"]] = 0.0
            if abs(err) < 0.004:
                self.start_next()
        elif kind == "TURN":
            _, st, tgt = self.prim
            self.stop_drives()
            if not self.stowed():                # SEATBELT — never turn extended
                self.reflex_holds += 1
                return
            ox, oy = other.xy()
            busy = (abs(ox - st) < STATION_ZONE) or \
                   (other.turning() and other.turn_started and other.prim[1] == st)
            if not self.turn_started:
                if busy:                         # SEATBELT
                    self.reflex_holds += 1
                    return
                self.turn_started = True
                self.prim_t0 = data.time
                self.turn_from = data.sensordata[TT_SEN[st]]
            s = min((data.time - self.prim_t0) / TURN_TIME, 1.0)
            cmd = self.turn_from + (tgt - self.turn_from) * s
            data.ctrl[TT_ACT[st]] = cmd
            data.ctrl[self.A["yawh"]] = cmd
            if (data.time - self.prim_t0) > TURN_TIME + 0.5:
                self.start_next()

bots = [
    Bot("bot1", 0.741, 0.231,
        dict(dx="drive_x", dy="drive_y", yawh="yaw_hold", claw="claw"),
        dict(x="bot_x_pos", y="bot_y_pos", yaw="bot_yaw_pos",
             q1="q1_pos", q2="q2_pos", q3="q3_pos", q4="q4_pos", claw="claw_pos"),
        ["a1", "a2", "a3", "a4", "a5"], "tip", "bot", "#1a8c99", 0),
    Bot("bot2", 1.0, 0.551,
        dict(dx="drive2_x", dy="drive2_y", yawh="yaw2_hold", claw="claw_b2"),
        dict(x="b2_x_pos", y="b2_y_pos", yaw="b2_yaw_pos",
             q1="q1_b2_pos", q2="q2_b2_pos", q3="q3_b2_pos", q4="q4_b2_pos",
             claw="claw_b2_pos"),
        ["a1_b2", "a2_b2", "a3_b2", "a4_b2", "a5_b2"], "tip_b2", "bot2", "#d85a30", 1),
]

# ---------------- THE PLANNER ----------------
def commit(bot, plan, now):
    x, y = bot.xy()
    s, l, d, _ = simulate_plan(x, y, now, plan, bot.arm_ctl.target)
    bot.plan_samples, bot.plan_locks, bot.plan_discs = s, l, d
    bot.queue = list(plan)
    bot.start_next()

def try_schedule(bot, other, tx, lane, now, work=False):
    """Search: (route candidates) x (start WAIT 0..24 s); best = earliest
    finish. work=True appends extend -> dwell -> stow at the target."""
    x, y = bot.xy()
    oS, oL, oD = other.reservation(now)
    on_lane = min((LANE_A, LANE_B), key=lambda L: abs(y - L))
    lanes_differ = abs(on_lane - lane) > 0.01
    cands = []
    if not lanes_differ and abs(y - on_lane) < 0.02:
        cands.append(plan_route(x, y, tx, lane))                    # direct
        cands.append(detour_route(x, y, tx, lane, ST_LEFT, ST_RIGHT))
        cands.append(detour_route(x, y, tx, lane, ST_RIGHT, ST_LEFT))
    else:
        st0 = min((ST_LEFT, ST_RIGHT), key=lambda s: abs(x - s) + abs(tx - s))
        st1 = ST_LEFT if st0 == ST_RIGHT else ST_RIGHT
        cands.append(plan_route(x, y, tx, lane, via_station=st0))
        cands.append(plan_route(x, y, tx, lane, via_station=st1))
    prefix = [] if bot.stowed() else [("ARM", "STOW")]
    suffix = [("ARM", "WORK"), ("WAIT", WORK_DWELL), ("ARM", "STOW")] if work else []
    best, best_end = None, None
    for base in cands:
        for w in np.arange(0.0, 24.1, 1.0):
            plan = prefix + ([("WAIT", float(w))] if w > 0 else []) + base + suffix
            s, l, d, tend = simulate_plan(x, y, now, plan, bot.arm_ctl.target)
            if conflict(s, l, d, oS, oL, oD) is None:
                if best_end is None or tend < best_end:
                    best, best_end = plan, tend
                break        # larger waits on this candidate only end later
    return best

def assign(bot, tx, lane, work=False):
    """Sai's loop: map -> overlay -> collisions -> pauses -> recheck -> move."""
    other = bots[0] if bot is bots[1] else bots[1]
    now = data.time
    plan = try_schedule(bot, other, tx, lane, now, work=work)
    if plan is None and other.idle():
        # plan-time shunt: search VALID parking spots for the idle blocker —
        # clear of stations, of my position, and of my intended target —
        # and schedule the shunt with the same verified planner.
        bx, by = bot.xy()
        ox, oy = other.xy()
        oth_lane = LANE_B if abs(oy - LANE_A) < 0.12 else LANE_A
        own_lane = LANE_A if oth_lane == LANE_B else LANE_B
        spots = []
        for sl in (oth_lane, own_lane):
            for sxc in (0.60, 0.90, 1.60, 0.35, 1.05):
                if any(abs(sxc - st) < 0.30 for st in (ST_LEFT, ST_RIGHT)):
                    continue
                if abs(sl - lane) < 0.01 and abs(sxc - tx) < 0.35:
                    continue                      # not on my target
                if abs(sl - by) < 0.12 and abs(sxc - bx) < 0.35:
                    continue                      # not on me
                spots.append((sxc, sl))
        if picking(other):
            spots = []                            # never shunt a bot mid-pick
        for sxc, sl in spots:
            shunt = try_schedule(other, bot, sxc, sl, now)
            if shunt is None:
                continue
            commit(other, shunt, now)
            other.target = (sxc, sl)
            plan = try_schedule(bot, other, tx, lane, now, work=work)
            if plan is not None:
                break
            # that spot scheduled but still blocks me: undo and try next
            other.queue = []
            other.prim = None
            other.target = None
            other.plan_samples = None
            other.plan_locks = []
            other.plan_discs = []
    if plan is None:
        return False
    commit(bot, plan, now)
    bot.target = (tx, lane)
    return True

# ---------------- window: 3D + map + string diagram ----------------
W3D, H3D = 560, 420
renderer = mujoco.Renderer(model, H3D, W3D)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.azimuth, cam.elevation = 1.8, 135, -28

from matplotlib.widgets import Slider

# matplotlib's own shortcuts would fire alongside ours (p = pan tool, o = zoom,
# s = save dialog, g = grid, l = log scale): take our keys out of its keymap
_OUR_KEYS = {" ", "0", "1", "2", "s", "l", "g", "p", "o", "escape", "[", "]"}
for _k in [k for k in plt.rcParams if k.startswith("keymap.")]:
    plt.rcParams[_k] = [v for v in plt.rcParams[_k] if v not in _OUR_KEYS]

fig = plt.figure(figsize=(16.5, 8.2))
fig.canvas.manager.set_window_title("LocoBot Fleet v2 — reservation planning")
gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1.0],
                      width_ratios=[1.0, 1.30, 0.62],
                      left=0.035, right=0.985, top=0.96, bottom=0.07,
                      hspace=0.28, wspace=0.14)
ax3d = fig.add_subplot(gs[0, 0])
ax = fig.add_subplot(gs[0, 1])
axarm = fig.add_subplot(gs[0, 2])
axt = fig.add_subplot(gs[1, :2])
axcam = fig.add_subplot(gs[1, 2])

im3d = ax3d.imshow(np.zeros((H3D, W3D, 3), dtype=np.uint8))
ax3d.set_xticks([]); ax3d.set_yticks([])
ax3d.set_title("3D · click=move · s=stow g=claw p=pick o=put back · colour + dial + table click = place", fontsize=8)

ax.set_aspect("equal"); ax.set_xlim(-0.05, 2.0); ax.set_ylim(-0.06, 0.85)
ax.set_facecolor("#f4f3ef")
ax.add_patch(Rectangle((0.031, 0.041), 1.76, 0.70, fill=False, lw=2, ec="#222"))
ax.add_patch(Rectangle((1.42, 0.041), 0.37, 0.70, color="#dff0df", zorder=0))
for z in keepout.ZONES:                      # bounded world keep-outs (x-y footprint)
    if z.name.startswith("table"):           # below the arm's reach: furniture, not a hazard
        ax.add_patch(Rectangle((z.lo[0], z.lo[1]), z.hi[0] - z.lo[0], z.hi[1] - z.lo[1],
                               fc="#d9c3a5", ec="#8a6a48", lw=1, alpha=0.7, zorder=0.5))
        ax.text((z.lo[0] + z.hi[0]) / 2, z.lo[1] + 0.012, z.name.replace("_", " "), ha="center",
                fontsize=7, color="#8a6a48", zorder=0.6)
    elif z.frame == "world" and z.hi[0] - z.lo[0] < 2 * keepout.BIG:
        ax.add_patch(Rectangle((z.lo[0], z.lo[1]), z.hi[0] - z.lo[0], z.hi[1] - z.lo[1],
                               fc="#c33", ec="#800", hatch="xx", alpha=0.6, zorder=7))
for L, nm in ((LANE_A, "lane A"), (LANE_B, "lane B")):
    for off in (-0.0768, 0.0768):
        ax.plot([0.065, 1.74], [L + off, L + off], color="#999", lw=2, zorder=1)
    ax.text(0.90, L + 0.095, nm, ha="center", fontsize=9, color="#666")
tt_lines = {}
for st in (ST_LEFT, ST_RIGHT):
    ax.add_patch(Rectangle((st - 0.28, 0.06), 0.56, 0.66, color="#c98383",
                           alpha=0.10, zorder=0))
    for L in (LANE_A, LANE_B):
        ax.add_patch(Circle((st, L), 0.062, fill=False, ec="#555", lw=1.5, zorder=2))
        tt_lines[(st, L)] = ax.plot([], [], color="#c33", lw=3, zorder=3)[0]
ax.text(ST_LEFT, 0.045, "no parking", ha="center", fontsize=7, color="#a66")
ax.text(ST_RIGHT, 0.045, "no parking", ha="center", fontsize=7, color="#a66")
# cubes: live markers from the free bodies' world positions
CUBES = [model.body(i) for i in range(model.nbody) if model.body(i).name.startswith("cube_")]
cube_marks = [ax.plot([], [], "s", ms=5, mec="#222", mew=0.6, zorder=4,
                      color=model.geom_rgba[model.geom(c.name).id][:3])[0] for c in CUBES]
BOT_HX, BOT_HY = 0.1235, 0.114
polys, tmarks, routelines = [], [], []
for b in bots:
    p = Polygon([[0, 0]], closed=True, fc=b.color, ec="#222", lw=1, zorder=5)
    ax.add_patch(p); polys.append(p)
    m, = ax.plot([], [], "x", color=b.color, ms=14, mew=3, zorder=6)
    tmarks.append(m)
    rl, = ax.plot([], [], "--", color=b.color, lw=1.2, alpha=0.55, zorder=3)
    routelines.append(rl)

# ---- pseudo-arm panel: live side-view schematic + joint sliders ----
axarm.set_facecolor("#faf9f6")
axarm.set_aspect("equal")
axarm.set_xlim(-0.55, 0.55)
axarm.set_ylim(-0.65, 0.08)
axarm.set_xticks([]); axarm.set_yticks([])
# stow contract markers: chassis extent + certify/revoke thresholds
axarm.axvspan(-0.1235, 0.1235, color="#B4B2A9", alpha=0.18, zorder=0)
# keep-outs in the side view (origin = shoulder, z up): bot-frame boxes, plus
# world z half-spaces (floor / rail level) as bands
_pts, _bpos, _ = bots[0].arm_ctl.be.fk()
SH_BOT_Z = _pts[0][2] - _bpos[2]              # shoulder height in the bot frame
for z in keepout.ZONES:
    if z.frame == "bot":
        lo, hi = z.lo[2] - SH_BOT_Z, z.hi[2] - SH_BOT_Z
        axarm.add_patch(Rectangle((z.lo[0], lo), z.hi[0] - z.lo[0], hi - lo,
                                  fc="none", ec="#c33", hatch="///", lw=1, zorder=1))
    elif z.hi[0] - z.lo[0] >= 2 * keepout.BIG:
        zlo, zhi = z.lo[2] - _pts[0][2], z.hi[2] - _pts[0][2]
        axarm.axhspan(max(zlo, -2), min(zhi, 2), fc="none", ec="#c33", hatch="///", lw=0, zorder=1)
for r0, c in ((0.13, "#2a8"), (0.19, "#c33")):
    axarm.axvline(+r0, color=c, lw=1, ls=":")
    axarm.axvline(-r0, color=c, lw=1, ls=":")
arm_line, = axarm.plot([], [], "-o", color="#1a8c99", lw=3, ms=5)
arm_tip, = axarm.plot([], [], "s", color="#EF9F27", ms=8)
arm_title = axarm.set_title("", fontsize=9)

def arm_side_view(bot):
    """Joint positions in the reach(r)-drop(z) plane from live feedback."""
    q2, q3, q4 = bot.arm_q()
    pts = [(0.0, 0.0)]
    ang, r, z = 0.0, 0.0, 0.0
    for L, q in ((L1, q2), (L2, q3), (L3, q4)):
        ang += q
        r -= L * np.sin(ang)      # match MuJoCo: screen-right = bot forward
        z -= L * np.cos(ang)
        pts.append((r, z))
    return pts

# sliders live under the schematic
_p = axarm.get_position()
axarm.set_position([_p.x0, _p.y0 + 0.52 * _p.height, _p.width, 0.48 * _p.height])
slider_defs = [("q1 yaw", -3.1, 3.1, 0.0), ("q2 shoulder", -1.9, 1.9, 0.3),
               ("q3 elbow", -2.4, 2.4, -0.6), ("q4 wrist", -1.9, 1.9, 0.0),
               ("q5 roll", -3.1, 3.1, 0.0), ("claw", CLAW_CLOSED, CLAW_OPEN, CLAW_CLOSED)]
sliders = []
for i, (nm, lo, hi, v0) in enumerate(slider_defs):
    sa = fig.add_axes([_p.x0 + 0.055, _p.y0 + (0.40 - i * 0.075) * _p.height,
                       _p.width - 0.075, 0.045 * _p.height])
    s = Slider(sa, nm, lo, hi, valinit=v0)
    s.label.set_fontsize(7); s.valtext.set_fontsize(7)
    sliders.append(s)

def arm_bot():
    return bots[selected - 1] if selected else bots[0]

slider_sync = {"on": False}         # set_val() from code must not issue moves

def on_slider(_):
    if slider_sync["on"]:
        return
    b = arm_bot()
    if not b.idle() or picking(b):
        say(f"{b.name} is busy (plan or pick) — arm control only while idle")
        return
    if b.arm_ctl.move_to([s.val for s in sliders[:5]], data.time) is None:
        say(f"{b.name}: move refused — keep-out {', '.join(b.arm_ctl.refused)}")
for s in sliders[:5]:
    s.on_changed(on_slider)

def on_claw(v):
    if not slider_sync["on"]:
        if picking(arm_bot()):
            say(f"{arm_bot().name} is picking — the claw is under its control"); return
        arm_bot().set_claw(v)
sliders[5].on_changed(on_claw)

def toggle_claw():
    b = arm_bot()
    if picking(b):
        say(f"{b.name} is picking — the claw is under its control"); return
    v = CLAW_CLOSED if data.ctrl[b.A["claw"]] > 0.5 * CLAW_OPEN else CLAW_OPEN
    b.set_claw(v)
    slider_sync["on"] = True
    sliders[5].set_val(v)
    slider_sync["on"] = False
    say(f"{b.name}: claw {'open' if v == CLAW_OPEN else 'close'}", 2.0)

# string diagram
axt.set_facecolor("#faf9f6")
axt.set_ylim(0.0, 1.8)
axt.set_ylabel("x position (m)", fontsize=9)
axt.set_xlabel("sim time (s)", fontsize=9)
for st in (ST_LEFT, ST_RIGHT):
    axt.axhspan(st - STATION_ZONE, st + STATION_ZONE, color="#B4B2A9", alpha=0.18, zorder=0)
plan_lines = [axt.plot([], [], color=b.color, lw=2)[0] for b in bots]
now_line = axt.axvline(0, color="#666", lw=1, ls="--")
lock_patches = []
disc_map_patches = []

# ---- perception: selected bot's down_cam -> cube poses (perception.py) ----
PERC_PERIOD = 0.25                  # wall seconds between camera frames
TABLES = [((z.lo[0], z.lo[1]), (z.hi[0], z.hi[1]), z.hi[2])
          for z in keepout.ZONES if z.name.startswith("table")]      # the prior
CAMS = ("down_cam", "down_cam_b2")
finders = {b.name: perception.CubeFinder(model, cam, TABLES) for b, cam in zip(bots, CAMS)}
cam_renderer = mujoco.Renderer(model, perception.H, perception.W)
imcam = axcam.imshow(np.zeros((perception.H, perception.W, 3), dtype=np.uint8))
axcam.set_xticks([]); axcam.set_yticks([])
cam_title = axcam.set_title("", fontsize=8)
perc = {"t": 0.0, "artists": [], "dets": []}

def perceive():
    """Grab a frame from the selected bot's camera, detect cubes, draw the
    overlay (camera panel + estimated outlines on the map). The sim truth is
    used only for the error readout."""
    b = arm_bot()
    cam = CAMS[bots.index(b)]
    cam_renderer.update_scene(data, camera=cam)
    rgb = cam_renderer.render()
    x, y = b.xy()
    dets = finders[b.name].detect(rgb, (x, y, b.yaw()))
    registry.update(dets, data.time)
    for a in perc["artists"]:
        a.remove()
    perc["artists"], perc["dets"] = [], dets
    imcam.set_data(rgb)
    worst = 0.0
    for dt in dets:
        tx, ty, tyaw = perception.truth(model, data, dt.name)
        e = np.hypot(dt.x - tx, dt.y - ty) * 1000
        worst = max(worst, e)
        uv = np.vstack([dt.corners_uv, dt.corners_uv[:1]])
        perc["artists"] += axcam.plot(uv[:, 0], uv[:, 1], color="#f0f", lw=1.5)
        perc["artists"].append(axcam.text(
            uv[:4, 0].mean(), uv[:4, 1].min() - 6,
            f"{dt.name} ({dt.x:.3f}, {dt.y:.3f}) {np.degrees(dt.yaw):+.0f}°",
            color="#f0f", fontsize=6, ha="center", va="bottom"))
        xy = np.vstack([dt.corners_xy(), dt.corners_xy()[:1]])
        perc["artists"] += ax.plot(xy[:, 0], xy[:, 1], color="#f0f", lw=1, zorder=8)
    cam_title.set_text(f"{b.name} {cam} · {len(dets)} cube{'s' * (len(dets) != 1)}"
                       + (f" · err {worst:.1f} mm · click to pick" if dets else
                          " · none (arm out? park at a lane end)"))

# ---- picking & placing (pick.py) ----
pickers = {}                        # bot name -> its latest Picker (kept for the status)
tasks = {}                          # bot name -> (colour, dest): driving to its table first
registry = perception.CubeRegistry(model)   # where each cube is believed to be
PARK_X = X_MAX                      # lane end: every table spot within reach from here
TABLE_LANE = [min((LANE_A, LANE_B), key=lambda L: abs((lo[1] + hi[1]) / 2 - L))
              for lo, hi, _ in TABLES]
pending = {"colour": None, "yaw": None}   # colour chosen (button) + placing yaw (dial; None = keep)

def picking(b):
    """Mid-pick or holding a cube: the bot must not be dispatched or shunted."""
    p = pickers.get(b.name)
    return p is not None and (p.busy or p.state == "holding")

def table_at(x, y):
    for i, (lo, hi, _) in enumerate(TABLES):
        if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]:
            return i
    return None

def parked_at_table(b, lane):
    x, y = b.xy()
    return b.idle() and abs(y - lane) < 0.01 and x > PARK_X - 0.02

def start_pick(name=None, dest=None, bot=None):
    b = bot or arm_bot()
    if not b.idle():
        say(f"{b.name} is executing a plan — pick when it's parked"); return
    if picking(b):
        say(f"{b.name} is already picking/holding — o puts the cube back"); return
    if not b.stowed():
        say(f"{b.name}: stow the arm first (s) — the camera needs a clear view"); return
    cam = CAMS[bots.index(b)]

    def look():
        cam_renderer.update_scene(data, camera=cam)
        x, y = b.xy()
        dets = finders[b.name].detect(cam_renderer.render(), (x, y, b.yaw()))
        registry.update(dets, data.time)
        return dets
    hooks = pick.Hooks(arm=b.arm_ctl, set_claw=b.set_claw, claw_q=b.claw_q,
                       bot_pose=lambda: (*b.xy(), b.yaw()), look=look)
    pickers[b.name] = pick.Picker(model, data, hooks, name=name, dest=dest, tables=TABLES)
    pickers[b.name].announced = False
    say(f"{b.name}: picking {name or 'nearest cube'}"
        + (f" -> ({dest[0]:.3f}, {dest[1]:.3f})" if dest else ""), 2.5)

def request(colour, dest=None, yaw=None):
    """Pick `colour` and put it at dest (world yaw `yaw`, None = keep) — or
    just pick and hold it when dest is None: find its table (registry), send
    that table's bot to the lane end if it isn't there, then pick (& place)."""
    s = registry.cubes.get(colour)
    ti = table_at(s.x, s.y) if s else None
    if ti is None:
        say(f"{colour}: not on a table as far as I know"); return
    lane = TABLE_LANE[ti]
    if dest is None:                                     # pick & hold
        return dispatch_task(colour, None, lane)
    if table_at(*dest) != ti:
        say(f"{colour} is on table {'AB'[ti]} — moving cubes between tables isn't supported yet"); return
    # cheap checks against what we believe is on the table (the Picker
    # re-checks against a fresh look, including jaw room, before moving)
    goal_yaw = s.yaw if yaw is None else yaw
    problem, _, jaw_ok = spot_check(colour, dest[0], dest[1], goal_yaw, goal_yaw)
    if problem or not jaw_ok:
        say(f"{colour}: {problem or 'no room for the jaws there (too close to a neighbour)'}"); return
    dispatch_task(colour, (*dest, yaw) if yaw is not None else dest, lane)

def spot_check(name, x, y, yaw, base_jaw):
    """Can cube `name` go at (x, y, yaw)? Uses what the registry believes is
    on that table. Returns (problem or None, (det, jaw), jaws_clear)."""
    ti = table_at(x, y)
    top = TABLES[ti][2] + perception.CUBE
    goal = perception.Detection(name, x, y, top, yaw, 0, None)
    others = [o for o in registry.as_detections(top) if o.name != name and table_at(o.x, o.y) == ti]
    jaw, jaw_ok = pick.choose_jaw(goal, base_jaw, others)
    lo, hi = np.array(TABLES[ti][0]), np.array(TABLES[ti][1])
    c = goal.corners_xy()
    if np.any(c < lo + pick.TABLE_EDGE) or np.any(c > hi - pick.TABLE_EDGE):
        return "too close to the table edge", (goal, jaw), jaw_ok
    for o in others:
        if pick.overlaps(goal, o, pick.PLACE_GAP):
            return f"that spot is too close to {o.name}", (goal, jaw), jaw_ok
    return None, (goal, jaw), jaw_ok

def dispatch_task(colour, target, lane):
    """Give the task to the lane's bot: pick now if it's parked at the table,
    else drive it there first (step_tasks starts the pick on arrival)."""
    free = [b for b in bots if not picking(b) and b.name not in tasks]
    if not free:
        say("both bots are busy"); return
    b = max([b for b in free if abs(b.xy()[1] - lane) < 0.02] or free, key=lambda b: b.xy()[0])
    if parked_at_table(b, lane):
        start_pick(colour, target, bot=b)
    elif assign(b, PARK_X, lane):
        tasks[b.name] = (colour, target, lane)
        say(f"{b.name}: driving to the table for {colour}", 3.0)
    else:
        say(f"{b.name} can't get to the table right now (no clean schedule)")

def step_tasks():
    for name in list(tasks):
        b = next(bb for bb in bots if bb.name == name)
        if not b.idle():
            continue
        colour, target, lane = tasks.pop(name)
        if parked_at_table(b, lane):
            start_pick(colour, target, bot=b)
        else:
            say(f"{name} didn't reach the table — {colour} task dropped")

def table_click(x, y):
    """A click on a table: place the chosen colour there, or put down the
    cube a parked bot is holding."""
    if pending["colour"]:
        colour, yaw = pending["colour"], pending["yaw"]
        select_colour(None)
        ghost.set_data([], [])
        request(colour, (x, y), yaw)
        return
    for b in bots:
        p = pickers.get(b.name)
        if p is not None and p.state == "holding":
            err = p.place_at(data.time, x, y)
            if err:
                say(f"{b.name}: can't put {p.target.name} there — {err}")
            else:
                p.announced = False
                say(f"{b.name}: putting {p.target.name} at ({x:.3f}, {y:.3f})", 2.5)
            return
    say("choose a colour first (buttons under the camera), then click the table")

def place_back():
    b = arm_bot()
    p = pickers.get(b.name)
    if p is None or not p.place_back(data.time):
        say(f"{b.name} isn't holding a cube"); return
    p.announced = False
    say(f"{b.name}: putting {p.target.name} back", 2.0)

def step_pickers():
    step_tasks()
    for name, p in pickers.items():
        if p.busy:
            p.step(data.time)
        elif not p.announced:
            p.announced = True
            if p.state == "failed":
                say(f"{name}: {p.reason}", 6.0)
            elif p.state == "holding":
                dial.set(p.held_yaw())
                dial.set_jaws(p._jaw_for(p.held_yaw()) - p.held_yaw(), "ok")
                dial.title.set_text(f"rotate held {p.target.name}")
                say(f"{name}: holding {p.target.name} — dial rotates it, click the table to put it down", 5.0)
            elif p.state == "done":
                say(f"{name}: {p.target.name} {p.reason}", 5.0)

# colour buttons under the camera panel: choose a cube, then click the table
from matplotlib.widgets import Button
_cp = axcam.get_position()
colour_buttons = []
_names = [c.name[5:] for c in CUBES]
for i, nm in enumerate(_names):
    bw = _cp.width / len(_names)
    bax = fig.add_axes([_cp.x0 + i * bw + 0.002, _cp.y0 - 0.05, bw - 0.004, 0.032])
    rgba = model.geom_rgba[model.geom(f"cube_{nm}").id]
    btn = Button(bax, nm, color=tuple(rgba[:3]), hovercolor=tuple(np.minimum(rgba[:3] + 0.15, 1)))
    btn.label.set_fontsize(7)
    btn.label.set_color("white" if rgba[:3].mean() < 0.55 else "black")

    def _choose(_evt, nm=nm):
        select_colour(nm)
    btn.on_clicked(_choose)
    colour_buttons.append(btn)
dest_artists = []


def select_colour(nm):
    """Colour button: this cube is selected. p = pick & hold it; a table
    click = pick & place it there (dial = angle); Esc = clear."""
    pending.update(colour=nm, yaw=None)
    for b in colour_buttons:                    # the selected button gets a thick frame
        on = b.label.get_text() == nm
        for sp in b.ax.spines.values():
            sp.set_linewidth(3.0 if on else 0.8)
            sp.set_edgecolor("black" if on else "#666")
    if nm is None:
        dial.set_jaws(0.0, "hidden")
        return
    dial.set(registry.cubes[nm].yaw)
    dial.set_jaws(0.0, "ok")
    dial.title.set_text(f"place angle: {nm}")
    say(f"{nm} selected: p = pick it · click its table = pick & place there · dial = angle · Esc = clear", 8.0)


class Dial:
    """Circular rotation control, like the sliders: drag the knob (5° detents)
    or press [ / ] for 15° steps. 0° = map +x, counter-clockwise positive —
    the same convention as the map. The small square shows the cube."""

    def __init__(self, axd, on_change):
        self.ax, self.cb = axd, on_change
        axd.set_aspect("equal")
        axd.set_xlim(-1.4, 1.4); axd.set_ylim(-1.9, 1.4)
        axd.axis("off")
        th = np.linspace(0, 2 * np.pi, 181)
        axd.plot(np.cos(th), np.sin(th), color="#888", lw=2)
        for deg in range(0, 360, 15):
            a, r0 = np.radians(deg), (0.78 if deg % 90 == 0 else 0.88)
            axd.plot([r0 * np.cos(a), np.cos(a)], [r0 * np.sin(a), np.sin(a)],
                     color="#888", lw=1.6 if deg % 90 == 0 else 0.8)
        for deg, t in ((0, "0°"), (90, "90°"), (180, "180°"), (270, "-90°")):
            a = np.radians(deg)
            axd.text(1.22 * np.cos(a), 1.22 * np.sin(a), t, ha="center", va="center",
                     fontsize=6, color="#666")
        self.square, = axd.plot([], [], color="#444", lw=1.5)
        # the open jaws, to scale with the mini cube (half-side 0.35 = 20 mm)
        self.jaw_bars = [axd.fill([0], [0], color="#777", alpha=0.85, zorder=3)[0] for _ in range(2)]
        self.jaw_rel, self.jaw_state = 0.0, "hidden"   # jaw angle relative to the cube; hidden|ok|blocked
        self.note = axd.text(0, -1.78, "", ha="center", fontsize=6, color="#666")
        self.needle, = axd.plot([], [], color="#c33", lw=2)
        self.knob, = axd.plot([], [], "o", color="#c33", ms=9)
        self.text = axd.text(0, -1.5, "", ha="center", fontsize=8)
        self.title = axd.set_title("rotate cube", fontsize=8)
        self.value, self.dragging = 0.0, False
        self.set(0.0)

    def set(self, v, notify=False):
        self.ax.set_xlim(-1.4, 1.4); self.ax.set_ylim(-1.9, 1.4)   # undo any stray pan/zoom
        self.value = float((v + np.pi) % (2 * np.pi) - np.pi)
        c, s = np.cos(self.value), np.sin(self.value)
        self.needle.set_data([0, 0.9 * c], [0, 0.9 * s])
        self.knob.set_data([c], [s])
        h = 0.35
        sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]]) @ np.array([[c, s], [-s, c]])
        self.square.set_data(sq[:, 0], sq[:, 1])
        self.text.set_text(f"{np.degrees(self.value):+.0f}°")
        self._draw_jaws()
        if notify:
            self.cb(self.value)

    def set_jaws(self, rel, state):
        """Jaw bars at `rel` (rad) relative to the cube; state: hidden | ok |
        blocked (red = they'd hit a neighbour at the previewed spot)."""
        self.jaw_rel, self.jaw_state = rel, state
        self._draw_jaws()

    def _draw_jaws(self):
        if self.jaw_state == "hidden":
            for bar in self.jaw_bars:
                bar.set_visible(False)
            self.note.set_text("")
            return
        k = 0.35 / (perception.CUBE / 2)             # dial units per metre
        inner, outer, w = (v * k for v in pick.jaw_extent())
        a = self.value + self.jaw_rel
        c, s = np.cos(a), np.sin(a)
        col = "#c33" if self.jaw_state == "blocked" else "#777"
        for bar, sg in zip(self.jaw_bars, (1.0, -1.0)):
            uv = np.array([(sg * inner, -w), (sg * outer, -w), (sg * outer, w), (sg * inner, w)])
            bar.set_xy(uv @ np.array([[c, s], [-s, c]]))
            bar.set_color(col)
            bar.set_visible(True)
        self.note.set_text("jaws BLOCKED here" if self.jaw_state == "blocked" else "bars = open jaws")
        self.note.set_color(col)

    def press(self, e):
        self.dragging = True
        self.move(e)

    def move(self, e):
        if not self.dragging or e.inaxes != self.ax or e.xdata is None:
            return
        if np.hypot(e.xdata, e.ydata) < 0.15:
            return                                    # too close to the centre to read an angle
        a = np.radians(round(np.degrees(np.arctan2(e.ydata, e.xdata)) / 5) * 5)
        if abs((a - self.value + np.pi) % (2 * np.pi) - np.pi) > 1e-6:
            self.set(a, notify=True)

    def release(self, _e):
        self.dragging = False


def holder():
    """The bot holding a cube the dial should turn: the selected one if it's
    holding, else any holding bot."""
    order = [arm_bot()] + [b for b in bots if b is not arm_bot()]
    for b in order:
        p = pickers.get(b.name)
        if p is not None and p.state == "holding":
            return b
    return None


def on_dial(v):
    b = holder()
    if b is not None:                                   # live: turn the held cube
        err = pickers[b.name].rotate_held(data.time, v)
        if err:
            say(f"{b.name}: can't turn it there — {err}")
    elif pending["colour"]:                             # the angle to place at
        pending["yaw"] = v
    else:
        say("the dial sets a cube's angle: choose a colour or pick a cube first")


_tp = axt.get_position()
axt.set_position([_tp.x0, _tp.y0, _tp.width - 0.125, _tp.height])
axdial = fig.add_axes([_tp.x0 + _tp.width - 0.11, _tp.y0, 0.105, _tp.height])
dial = Dial(axdial, on_dial)
# controls and image panels must ignore the toolbar's pan/zoom tools — with
# pan on, dragging the knob would drag the whole dial. The map and the
# string diagram stay zoomable.
for _a in (axdial, axcam, ax3d, axarm):
    _a.set_navigate(False)
ghost, = ax.plot([], [], ":", lw=1.6, color="#555", zorder=9)   # placement preview under the mouse
ghost_jaws = [ax.fill([0], [0], color="#777", alpha=0.6, zorder=9, visible=False)[0] for _ in range(2)]


def placing_now():
    """(cube name, yaw, jaw) of what a table click would place, or None.
    Selected colour: jaws along its faces (the pick decides which pair).
    Holding: the jaws as they grip it now."""
    b = holder()
    if pending["colour"]:
        nm = pending["colour"]
        yaw = pending["yaw"] if pending["yaw"] is not None else registry.cubes[nm].yaw
        return nm, yaw, yaw
    if b is not None:
        p = pickers[b.name]
        return p.target.name, p.held_yaw(), p._jaw_for(p.held_yaw())
    return None

def update_ghost(event):
    """Dotted outline where (and at what angle) a click would put the cube,
    with the open jaws around it: grey = clear, red = they'd hit a neighbour
    (a red cube outline = the spot itself is bad). The dial mirrors the jaws."""
    what = placing_now()
    if what and event.inaxes == ax and event.xdata is not None \
            and table_at(event.xdata, event.ydata) is not None:
        nm, yaw, base_jaw = what
        problem, (g, jaw), jaw_ok = spot_check(nm, event.xdata, event.ydata, yaw, base_jaw)
        c = g.corners_xy()
        if problem:          # bad spot: black outline crossed out (a red outline would be ambiguous for the red cube)
            xy = np.vstack([c, c[:1], c[2:3], [[np.nan, np.nan]], c[1:2], c[3:4]])
            ghost.set_color("#000")
            ghost.set_linestyle("-")
        else:
            xy = np.vstack([c, c[:1]])
            ghost.set_color(model.geom_rgba[model.geom(f"cube_{nm}").id][:3])
            ghost.set_linestyle(":")
        ghost.set_data(xy[:, 0], xy[:, 1])
        for poly, rect in zip(ghost_jaws, pick.jaw_rects(g, jaw)):
            poly.set_xy(rect)
            poly.set_color("#777" if jaw_ok else "#c33")
            poly.set_visible(True)
        dial.set_jaws(jaw - yaw, "ok" if jaw_ok else "blocked")
        return
    ghost.set_data([], [])
    for poly in ghost_jaws:
        poly.set_visible(False)
    dial.set_jaws(what[2] - what[1] if what else 0.0, "ok" if what else "hidden")


def redraw():
    for c, cm in zip(CUBES, cube_marks):
        cm.set_data([data.xpos[c.id][0]], [data.xpos[c.id][1]])
    for a in dest_artists:                       # where cubes are headed
        a.remove()
    dest_artists.clear()
    goals = [(p.dest_det, p.target.name) for p in pickers.values()
             if p.busy and p.dest_det is not None and p.state != "verify"]
    goals += [(perception.Detection(c, t[0], t[1], 0, t[2] if len(t) > 2 else registry.cubes[c].yaw, 0, None), c)
              for c, t, _ in tasks.values() if t is not None]
    for g, nm in goals:
        xy = np.vstack([g.corners_xy(), g.corners_xy()[:1]])
        dest_artists.extend(ax.plot(xy[:, 0], xy[:, 1], "--", lw=1.2, zorder=9,
                                    color=model.geom_rgba[model.geom(f"cube_{nm}").id][:3]))
    for b, p, m in zip(bots, polys, tmarks):
        x, y = b.xy(); yw = b.yaw()
        c, s = np.cos(yw), np.sin(yw)
        corners = np.array([[BOT_HX, BOT_HY], [BOT_HX, -BOT_HY],
                            [-BOT_HX, -BOT_HY], [-BOT_HX, BOT_HY]])
        p.set_xy(corners @ np.array([[c, -s], [s, c]]).T + [x, y])
        p.set_linewidth(3 if (selected and bots[selected - 1] is b) else 1)
        m.set_data(*([[b.target[0]], [b.target[1]]] if b.target else [[], []]))
        rl = routelines[bots.index(b)]
        if b.plan_samples and not b.idle():
            rl.set_data([s[1] for s in b.plan_samples[::4]],
                        [s[2] for s in b.plan_samples[::4]])
        else:
            rl.set_data([], [])
    for (st, L), line in tt_lines.items():
        a = data.sensordata[TT_SEN[st]]
        dx, dy = 0.062 * np.cos(a), 0.062 * np.sin(a)
        line.set_data([st - dx, st + dx], [L - dy, L + dy])
    parts = [f"{b.name}: {b.prim[0] if b.prim else 'idle'}"
             f"{'' if b.stowed() else ' [ARM OUT]'}  seatbelt={b.reflex_holds}"
             for b in bots]
    head = f"[dispatch: bot{selected}]  " if selected else "[dispatch: AUTO]  "
    if logger.active:
        head = f"● REC {logger.rows} rows  " + head
    if notice["text"] and time.time() < notice["until"]:
        ax.set_title("⛔ " + notice["text"], fontsize=9, color="#a33")
    else:
        ax.set_title(head + "   ".join(parts), fontsize=9, color="#222")
    # string diagram
    global lock_patches
    for b, line in zip(bots, plan_lines):
        if b.plan_samples:
            line.set_data([s[0] for s in b.plan_samples],
                          [s[1] for s in b.plan_samples])
    global disc_map_patches
    for pch in lock_patches:
        pch.remove()
    lock_patches = []
    for pch in disc_map_patches:
        pch.remove()
    disc_map_patches = []
    for b in bots:
        for st, ta, tb in (b.plan_locks or []):
            r = Rectangle((ta, st - STATION_ZONE), tb - ta, 2 * STATION_ZONE,
                          color="#EF9F27", alpha=0.30, zorder=1)
            axt.add_patch(r); lock_patches.append(r)
        for x0, y0, z0, x1, y1, z1, hw, ta, tb in (b.plan_discs or []):
            if tb < data.time:
                continue
            lo, hi = min(x0, x1) - hw, max(x0, x1) + hw
            rp = Rectangle((ta, lo), min(tb, ta + 300) - ta, hi - lo,
                           color="#7A5FB5", alpha=0.22, zorder=1)
            axt.add_patch(rp); lock_patches.append(rp)
            ln, = ax.plot([x0, x1], [y0, y1], color="#7A5FB5", lw=2 * hw * 500,
                          alpha=0.35 if ta <= data.time <= tb else 0.15,
                          solid_capstyle="round", zorder=4)
            disc_map_patches.append(ln)
        # live 3D envelope: deeper segments drawn fainter (z depth cue)
        if not b.stowed():
            for (a, bb) in b.arm_segments():
                depth = np.clip((0.15 - min(a[2], bb[2])) / 0.6, 0, 0.35)
                ln, = ax.plot([a[0], bb[0]], [a[1], bb[1]], color="#7A5FB5",
                              lw=2 * ARM_HW * 500, alpha=0.55 - depth,
                              solid_capstyle="round", zorder=6)
                disc_map_patches.append(ln)
    now_line.set_xdata([data.time])
    axt.set_xlim(max(0, data.time - 12), data.time + 45)
    b = arm_bot()
    pts = arm_side_view(b)
    arm_line.set_data([p[0] for p in pts], [p[1] for p in pts])
    arm_line.set_color(b.color)
    arm_tip.set_data([pts[-1][0]], [pts[-1][1]])
    arm_title.set_text(f"{b.name} arm — reach {b.reach()*1000:.0f} mm  "
                       + ("· STOWED ✓" if b.stowed() else "· ARM OUT")
                       + ("" if b.arm_ctl.status == "ok" else f" · {b.arm_ctl.status.upper()}")
                       + f" · claw {np.degrees(b.claw_q()):.0f}°"
                       + (f"\n{pickers[b.name].label()[:70]}" if b.name in pickers
                          and pickers[b.name].state != "done" else ""))

# ---------------- interaction ----------------
selected = 0
paused = False
drag = {"on": False, "x": 0, "y": 0}
notice = {"text": None, "until": 0.0}

def say(msg, secs=3.5):
    notice["text"] = msg
    notice["until"] = time.time() + secs

def render_3d():
    (x1, y1), (x2, y2) = bots[0].xy(), bots[1].xy()
    cam.lookat[:] = [(x1 + x2) / 2, (y1 + y2) / 2, 0.25]
    renderer.update_scene(data, camera=cam)
    return renderer.render()

def on_press(event):
    global selected
    if event.inaxes == axdial:
        dial.press(event); return
    if event.inaxes == ax3d:
        drag.update(on=True, x=event.x, y=event.y); return
    if event.inaxes == axcam and event.xdata is not None:   # click a detected cube = pick it
        for dt in perc["dets"]:
            uv = dt.corners_uv
            if uv[:, 0].min() <= event.xdata <= uv[:, 0].max() and uv[:, 1].min() <= event.ydata <= uv[:, 1].max():
                select_colour(None)
                start_pick(dt.name)
                return
        return
    if event.inaxes != ax or event.xdata is None:
        return
    if table_at(event.xdata, event.ydata) is not None:     # a table spot = where to place
        table_click(event.xdata, event.ydata)
        return
    tx = float(np.clip(event.xdata, X_MIN, X_MAX))
    lane = min((LANE_A, LANE_B), key=lambda L: abs(event.ydata - L))
    for st in (ST_LEFT, ST_RIGHT):                       # R4
        if abs(tx - st) < 0.28:
            say("REFUSED: inside a turntable's swept zone — park clear of the shaded bands")
            return
    work = (event.button == 3)                      # right-click = WORK task
    bot = bots[selected - 1] if selected else \
        min(bots, key=lambda b: abs(b.xy()[0] - tx) + (0.5 if not b.idle() else 0))
    if picking(bot):
        say(f"{bot.name} is picking/holding a cube — o puts it back first"); return
    other = bots[0] if bot is bots[1] else bots[1]
    ot = other.target
    if ot and abs(tx - ot[0]) < 0.26 and abs(lane - ot[1]) < 0.01:
        say(f"REFUSED: that spot is {other.name}'s destination"); return
    # a target on/near the peer's CURRENT position is the planner's problem:
    # it can shunt an idle peer aside, or wait for a moving one to pass
    if not assign(bot, tx, lane, work=work):
        say("NO CLEAN SCHEDULE — task refused (try again later or elsewhere)")

def on_release(event):
    drag["on"] = False
    dial.release(event)
def on_motion(event):
    dial.move(event)
    update_ghost(event)
    if drag["on"]:
        cam.azimuth -= (event.x - drag["x"]) * 0.4
        cam.elevation = float(np.clip(cam.elevation + (event.y - drag["y"]) * 0.3, -89, 5))
        drag.update(x=event.x, y=event.y)
def on_scroll(event):
    if event.inaxes == ax3d:
        cam.distance = float(np.clip(cam.distance * (0.9 if event.button == "up" else 1.1), 0.4, 4.5))
def on_key(event):
    global paused, selected
    if event.key == " ": paused = not paused
    elif event.key in ("0", "1", "2"): selected = int(event.key)
    elif event.key == "s":
        b = arm_bot()
        if picking(b):
            say(f"{b.name} is picking/holding — o puts the cube back and stows")
        elif b.idle():
            if b.arm_ctl.move_to(POSES["STOW"], data.time) is None:
                say(f"{b.name}: stow refused — keep-out {', '.join(b.arm_ctl.refused)}")
                return
            slider_sync["on"] = True
            for s, v in zip(sliders, POSES["STOW"]):
                s.set_val(v)
            slider_sync["on"] = False
            say(f"{b.name}: stow commanded", 2.0)
        else:
            say(f"{b.name} is busy — stow is already managed by its plan")
    elif event.key == "g":
        toggle_claw()
    elif event.key == "p":
        if pending["colour"]:                  # the selected colour, by its table's bot
            colour = pending["colour"]
            select_colour(None)
            request(colour)
        else:                                  # nothing selected: nearest cube, selected bot
            start_pick()
    elif event.key == "o":
        place_back()
    elif event.key == "escape" and pending["colour"]:
        select_colour(None)
        ghost.set_data([], [])
        say("selection cleared", 1.5)
    elif event.key in ("[", "]"):
        dial.set(dial.value + np.radians(15 if event.key == "[" else -15), notify=True)
    elif event.key == "l":
        if logger.active:
            logger.stop()
            say(f"log saved: {logger.path}", 4.0)
            print(f"log saved: {logger.path}")
        else:
            say(f"logging joints -> {logger.start(data.time)}", 3.0)

for ev, fn in [("button_press_event", on_press), ("button_release_event", on_release),
               ("motion_notify_event", on_motion), ("scroll_event", on_scroll),
               ("key_press_event", on_key)]:
    fig.canvas.mpl_connect(ev, fn)

# ---------------- main loop ----------------
for b in bots:                         # start stowed, holding
    b.arm_ctl.be.set_pose(POSES["STOW"])
    mujoco.mj_forward(model, data)
    b.arm_ctl = make_arm_ctl(b.arm_ctl.be)
JOINTS = ["j1_yaw", "j2_shoulder", "j3_elbow", "j4_wrist_pitch", "j5_wrist_roll"]
logger = JointLogger(model, data, [(bots[0], JOINTS, "tip"),
                                   (bots[1], [j + "_b2" for j in JOINTS], "tip_b2")])
if "--log" in sys.argv:
    print(f"logging joints -> {logger.start(data.time)}")
_last_wall = time.time()

plt.ion(); plt.show()
while plt.fignum_exists(fig.number):
    now_wall = time.time()
    elapsed = min(now_wall - _last_wall, 0.20)     # cap catch-up bursts
    _last_wall = now_wall
    if not paused:
        for _ in range(int(elapsed / model.opt.timestep)):
            bots[0].step(bots[1])
            bots[1].step(bots[0])
            for b in bots:
                b.arm_ctl.tick(data.time)          # 50 Hz inside; no-op between ticks
            step_pickers()
            mujoco.mj_step(model, data)
            logger.sample(data.time)
    im3d.set_data(render_3d())
    if now_wall - perc["t"] >= PERC_PERIOD:
        perc["t"] = now_wall
        perceive()
    redraw()
    fig.canvas.draw_idle()
    plt.pause(0.001)
    time.sleep(0.005)
if logger.active:
    logger.stop()
    print(f"log saved: {logger.path}")
