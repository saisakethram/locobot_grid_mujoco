"""Arm control — Section 1 of Arm_Control_First_Principles_Section1.md, in code.

Layers (top to bottom):
  Move          one shared-T cubic/quintic segment for all 5 joints (§1.7–1.8)
  ArmController 50 Hz loop on ABSOLUTE ticks, wall-clock trajectory time (§1.4):
                traj(t) -> clamp wanted -> + sag -> clamp sent -> backend.write
                plus supervision from read-back (§1.5): blocked -> pause clock
  SimBackend    MuJoCo stand-in for the SC15 bus: zero-order hold between
                ticks, targets rounded to SC15 counts (210 deg / 1024)

The controller never knows which backend it drives: an SC15Backend with the
same read()/write()/limits() replaces SimBackend on hardware.

Units are radians throughout (the doc's degrees are for humans).
"""

import mujoco
import numpy as np

SC15_COUNT = np.deg2rad(210.0 / 1024.0)     # one SC15 count ~ 0.205 deg

# placeholder limits until the Phase 2 bench system-ID measures them
V_MAX = np.array([1.5, 1.5, 1.5, 2.0, 2.0])  # rad/s per joint (yaw, sh, el, wp, wr)
A_MAX = np.array([4.0, 4.0, 4.0, 6.0, 6.0])  # rad/s^2
SOFT_MARGIN = 0.05                          # soft limit inset from the hard range (rad)

# peak multipliers for a rest-to-rest move of size D over T (§1.7 table):
#   peak vel = K_V * D / T,   peak accel = K_A * D / T^2
PROFILES = {
    "cubic":   dict(K_V=1.5,   K_A=6.0),
    "quintic": dict(K_V=1.875, K_A=10.0 / np.sqrt(3.0)),   # ~5.77
}


def shape(s, profile):
    """Normalised position 0..1 for normalised time s in [0, 1]."""
    s = np.clip(s, 0.0, 1.0)
    if profile == "cubic":
        return 3 * s**2 - 2 * s**3
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def move_time(q_from, q_to, v_max=V_MAX, a_max=A_MAX, profile="cubic"):
    """One shared T, set by the slowest joint at the profile's PEAK (§1.8)."""
    d = np.abs(np.asarray(q_to, float) - np.asarray(q_from, float))
    k = PROFILES[profile]
    t_v = k["K_V"] * d / v_max
    t_a = np.sqrt(k["K_A"] * d / a_max)
    return float(max(np.max(t_v), np.max(t_a), 0.0))


class Move:
    """Rest-to-rest move A -> B, every joint on the same s(t) (§1.8)."""

    def __init__(self, q_from, q_to, T, profile="cubic"):
        self.A = np.asarray(q_from, float).copy()
        self.B = np.asarray(q_to, float).copy()
        self.T = T
        self.profile = profile

    def __call__(self, t):
        if self.T <= 0.0:
            return self.B.copy()
        return self.A + (self.B - self.A) * shape(t / self.T, self.profile)


class SimBackend:
    """MuJoCo stand-in for the SC15 bus. write() is the SYNC WRITE: all
    targets land together, rounded to whole counts, and hold until the next
    write (MuJoCo keeps ctrl between steps = the servo's zero-order hold)."""

    def __init__(self, model, data, actuator_names, tip=None, body=None):
        self.model, self.data = model, data
        self.act = [model.actuator(a).id for a in actuator_names]
        jids = [model.actuator(a).trnid[0] for a in actuator_names]
        self.jids = jids
        self.qadr = [model.jnt_qposadr[j] for j in jids]
        self.tip = model.site(tip).id if tip else None
        self.body = model.body(body).id if body else None
        self._scratch = mujoco.MjData(model)     # FK without touching the sim
        # claw geometry for keep-out: collidable boxes in the subtree below
        # the tip's body (palm, links, jaws, pads); corners follow the live
        # claw opening and wrist roll
        self.claw_geoms = []
        if tip:
            root = model.site_bodyid[self.tip]
            def below(b):
                while b > 0:
                    if b == root:
                        return True
                    b = model.body_parentid[b]
                return False
            self.claw_geoms = [g for g in range(model.ngeom)
                               if below(model.geom_bodyid[g])
                               and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                               and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self._corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        self.range = np.array([model.jnt_range[j] for j in jids])
        self.ctrlrange = np.array([model.actuator_ctrlrange[a] for a in self.act])

    def limits(self):
        """(soft_lo, soft_hi, sent_lo, sent_hi). Soft = hard range inset by
        SOFT_MARGIN; sent = the servo's own clamp (slightly wider, §1.6)."""
        return (self.range[:, 0] + SOFT_MARGIN, self.range[:, 1] - SOFT_MARGIN,
                self.ctrlrange[:, 0], self.ctrlrange[:, 1])

    def read(self):
        return np.array([self.data.qpos[a] for a in self.qadr])

    def write(self, q):
        q = np.round(np.asarray(q) / SC15_COUNT) * SC15_COUNT
        for a, v in zip(self.act, q):
            self.data.ctrl[a] = v

    def fk(self, q=None, claw=False):
        """World points [shoulder, elbow, wrist, tip] and the bot body pose
        (pos, mat) for arm angles q at the bot's CURRENT position. q=None =
        the live measured pose. (Hardware: chain FK from the rail encoder.)
        claw=True also returns the claw's box corners (N,3) at the live
        claw opening."""
        s = self._scratch
        s.qpos[:] = self.data.qpos
        if q is not None:
            for a, v in zip(self.qadr, q):
                s.qpos[a] = v
        mujoco.mj_kinematics(self.model, s)
        pts = [s.xanchor[j].copy() for j in self.jids[1:4]] + [s.site_xpos[self.tip].copy()]
        out = (pts, s.xpos[self.body].copy(), s.xmat[self.body].copy())
        if not claw:
            return out
        cp = [s.geom_xpos[g] + (self._corners * self.model.geom_size[g]) @ s.geom_xmat[g].reshape(3, 3).T
              for g in self.claw_geoms]
        return out + (np.vstack(cp) if cp else None,)

    def set_pose(self, q):
        """Sim-only: teleport the joints (initial conditions)."""
        for a, v in zip(self.qadr, q):
            self.data.qpos[a] = v


def model_sag(backend):
    """sag(q_wanted) for §1.6 feed-forward, from the MuJoCo model — the sim
    stand-in for the measured sag table. A P-servo (gain kp) holding against
    gravity torque g(q) settles at q = sent - g/kp, so sent = wanted + g/kp.
    g comes from the model at rest (arm masses + claw; payload not included).
    On hardware this is replaced by the bench lookup (angle x load)."""
    m, s = backend.model, mujoco.MjData(backend.model)
    kp = np.array([m.actuator_gainprm[a][0] for a in backend.act])
    dofs = [m.jnt_dofadr[j] for j in backend.jids]
    bias = np.zeros(m.nv)

    def sag(q):
        s.qpos[:] = backend.data.qpos
        for a, v in zip(backend.qadr, q):
            s.qpos[a] = v
        s.qvel[:] = 0.0
        mujoco.mj_kinematics(m, s)
        mujoco.mj_comPos(m, s)
        mujoco.mj_rne(m, s, 0, bias)          # qvel = 0 -> gravity only
        return bias[dofs] / kp
    return sag


class ArmController:
    """Outer loop: the servos control, this supervises (§1.5).

    tick(now) is safe to call every physics step — it only acts on 50 Hz
    tick boundaries, and evaluates the trajectory at the time it actually
    woke (Option A, §1.4), so a late wake self-corrects."""

    # supervision thresholds (rad, rad/s, s) — placeholders until bench data.
    # A P-servo trails a moving target: ~9 deg on the shoulder at 1.3 rad/s in
    # the sim (kp=20), so LAG_ERR must sit above that normal trailing error.
    BLOCK_ERR, BLOCK_SPEED, BLOCK_TIME = 0.15, 0.05, 0.3
    LAG_ERR, LAG_TIME = 0.20, 0.3

    def __init__(self, backend, rate_hz=50.0, profile="cubic",
                 v_max=V_MAX, a_max=A_MAX, sag=None, keepout=None):
        self.be = backend
        self.dt = 1.0 / rate_hz
        self.profile = profile
        self.v_max, self.a_max = v_max, a_max
        self.sag = sag                     # sag(q_wanted) -> offset; None = no FF
        self.keepout = keepout             # keepout(q) -> [zone names]; None = off
        self.refused = []                  # zones that refused the last move_to
        self._escape = set()               # zones the current move STARTED inside
        self.soft_lo, self.soft_hi, self.sent_lo, self.sent_hi = backend.limits()
        q0 = backend.read()
        self.move = Move(q0, q0, 0.0, profile)
        self.t_start = 0.0                 # move start (sim/wall clock)
        self.paused_for = 0.0              # clock time spent paused by supervision
        self.epoch = 0.0                   # tick grid origin
        self.next_tick = 0.0
        self.wanted = q0.copy()
        self.status = "ok"                 # ok | lag | blocked | keepout
        self._q_prev = q0.copy()
        self._bad_since = None
        self._lag_since = None

    # ---- commands ----
    def hold(self, q, now):
        """Stay at q (no motion)."""
        return self.move_to(q, now, T=0.0)

    def move_to(self, q_to, now, T=None):
        """Start a new move from the last commanded pose. Returns its T, or
        None if the path enters a keep-out zone (the arm keeps holding;
        the zone names are in self.refused)."""
        q_to = np.clip(np.asarray(q_to, float), self.soft_lo, self.soft_hi)
        if T is None:
            T = move_time(self.wanted, q_to, self.v_max, self.a_max, self.profile)
        move = Move(self.wanted, q_to, T, self.profile)
        escape = set()
        if self.keepout:
            # a move that starts inside a zone may only be an ESCAPE from it:
            # the end pose must be clear of every zone
            escape = set(self.keepout(self.wanted))
            hits = set(self.keepout(q_to))
            if not hits:
                n = max(2, int(np.ceil(np.max(np.abs(q_to - self.wanted)) / 0.02)) + 1)
                for s in np.linspace(0.0, 1.0, n)[1:-1]:
                    hits |= set(self.keepout(move(s * T))) - escape
            self.refused = sorted(hits)
            if hits:
                return None
        self._escape = escape
        self.move = move
        self.t_start, self.paused_for = now, 0.0
        self.epoch = self.next_tick = now  # tick immediately on the new grid
        self._bad_since = self._lag_since = None
        self.status = "ok"
        return T

    # ---- queries ----
    @property
    def target(self):
        return self.move.B

    def traj_time(self, now):
        return now - self.t_start - self.paused_for

    def done(self, now):
        """The trajectory has finished (the servos may still be settling)."""
        return self.traj_time(now) >= self.move.T and self.status not in ("blocked", "keepout")

    # ---- the loop ----
    def tick(self, now):
        if now + 1e-9 < self.next_tick:
            return False
        # absolute grid: epoch + k*dt, never now + dt (§1.4)
        k = np.floor((now - self.epoch) / self.dt + 1e-9) + 1
        self.next_tick = self.epoch + k * self.dt

        self._supervise(now)
        if self.status == "blocked":       # pause: freeze the trajectory clock
            self.paused_for += self.dt
        wanted = np.clip(self.move(self.traj_time(now)), self.soft_lo, self.soft_hi)
        if self.keepout and set(self.keepout(wanted)) - self._escape:
            # the path was clear when planned; something moved (the bot, or a
            # zone was edited) -> hold the last safe target and pause the clock
            self.status = "keepout"
            self.paused_for += self.dt
            return True
        sent = wanted + (self.sag(wanted) if self.sag else 0.0)
        sent = np.clip(sent, self.sent_lo, self.sent_hi)
        self.wanted = wanted
        self.be.write(sent)
        return True

    def _supervise(self, now):
        """Read-back is for things the servo can't know (§1.5), not a
        second position loop. Error is measured against the WANTED angle:
        with sag feed-forward on, a healthy joint reads back ~wanted."""
        q = self.be.read()
        e = np.abs(q - self.wanted)
        speed = np.abs(q - self._q_prev) / self.dt
        self._q_prev = q
        err = np.max(e)
        # per joint: THIS joint is off target AND not moving (others may be)
        if np.any((e > self.BLOCK_ERR) & (speed < self.BLOCK_SPEED)):
            self._bad_since = self._bad_since if self._bad_since is not None else now
            if now - self._bad_since >= self.BLOCK_TIME:
                self.status = "blocked"
            return
        self._bad_since = None
        if err > self.LAG_ERR:
            self._lag_since = self._lag_since if self._lag_since is not None else now
            self.status = "lag" if now - self._lag_since >= self.LAG_TIME else "ok"
        else:
            self._lag_since = None
            self.status = "ok"
