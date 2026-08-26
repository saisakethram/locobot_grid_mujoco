"""LocomotionBot FLEET v2 — space-time reservation planning ("string diagram").

Run:  python run_fleet2.py

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

import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
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

def simulate_plan(x, y, t0, plan):
    """Pure arithmetic: plan -> (samples [(t,x,y)], station locks [(st,ta,tb)])."""
    samples, locks = [(t0, x, y)], []
    t = t0
    for prim in plan:
        if prim[0] == "WAIT":
            end = t + prim[1]
            while t < end:
                t = min(t + DT, end)
                samples.append((t, x, y))
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
    return samples, locks, t

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

def conflict(sA, lA, sB, lB):
    """First conflict between two reservations, or None."""
    t0 = min(sA[0][0], sB[0][0])
    t1 = max(sA[-1][0], sB[-1][0]) + HORIZON_PAD
    t = t0
    while t <= t1:
        xb, yb = pos_at(sB, t)
        for db in (-T_BUF, 0.0, T_BUF):
            xa, ya = pos_at(sA, t + db)
            if abs(xa - xb) < BODY_CLEAR and abs(ya - yb) < BODY_CLEAR:
                return ("body", t, xa, ya)
        for st, ta, tb in lA:                    # A's lock vs B's body
            if ta <= t <= tb and abs(xb - st) < STATION_ZONE:
                return ("lockA", t, st, xb)
        xa, ya = pos_at(sA, t)
        for st, ta, tb in lB:                    # B's lock vs A's body
            if ta <= t <= tb and abs(xa - st) < STATION_ZONE:
                return ("lockB", t, st, xa)
        t += DT
    for stA, a0, a1 in lA:                       # lock vs lock, same shaft
        for stB, b0, b1 in lB:
            if stA == stB and a0 < b1 and b0 < a1:
                return ("locklock", max(a0, b0), stA, 0)
    return None

# ---------------- per-bot executor (reflexes = seatbelt only) ----------------
class Bot:
    def __init__(self, name, x0, y0, act, sen, arm, color, pri):
        self.name, self.x0, self.y0, self.color, self.pri = name, x0, y0, color, pri
        self.A = {k: model.actuator(v).id for k, v in act.items()}
        self.S = {k: model.sensor(v).id for k, v in sen.items()}
        self.arm = [model.actuator(a).id for a in arm]
        self.queue, self.prim = [], None
        self.prim_t0 = self.turn_from = 0.0
        self.turn_started = False
        self.target = None
        self.reflex_holds = 0            # seatbelt activations (want: 0)
        self.plan_samples = None         # committed reservation, for drawing
        self.plan_locks = []

    def xy(self):
        return (self.x0 + data.sensordata[self.S["x"]],
                self.y0 + data.sensordata[self.S["y"]])

    def yaw(self):
        return data.sensordata[self.S["yaw"]]

    def idle(self):
        return self.prim is None and not self.queue

    def turning(self):
        return self.prim is not None and self.prim[0] == "TURN"

    def stop_drives(self):
        data.ctrl[self.A["dx"]] = data.ctrl[self.A["dy"]] = 0.0

    def reservation(self, now):
        """Committed samples, or a parked point if idle."""
        if self.plan_samples and not self.idle():
            return self.plan_samples, self.plan_locks
        x, y = self.xy()
        return [(now, x, y)], []

    def start_next(self):
        self.prim = self.queue.pop(0) if self.queue else None
        self.prim_t0 = data.time
        self.turn_started = False
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
        dict(dx="drive_x", dy="drive_y", yawh="yaw_hold"),
        dict(x="bot_x_pos", y="bot_y_pos", yaw="bot_yaw_pos"),
        ["a1", "a2", "a3", "a4", "a5"], "#1a8c99", 0),
    Bot("bot2", 1.0, 0.551,
        dict(dx="drive2_x", dy="drive2_y", yawh="yaw2_hold"),
        dict(x="b2_x_pos", y="b2_y_pos", yaw="b2_yaw_pos"),
        ["a1_b2", "a2_b2", "a3_b2", "a4_b2", "a5_b2"], "#d85a30", 1),
]

# ---------------- THE PLANNER ----------------
def commit(bot, plan, now):
    x, y = bot.xy()
    s, l, _ = simulate_plan(x, y, now, plan)
    bot.plan_samples, bot.plan_locks = s, l
    bot.queue = list(plan)
    bot.start_next()

def try_schedule(bot, other, tx, lane, now):
    """Search: (station choice) x (start WAIT 0..24 s). Returns plan or None."""
    x, y = bot.xy()
    oS, oL = other.reservation(now)
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
    best, best_end = None, None
    for base in cands:
        for w in np.arange(0.0, 24.1, 1.0):
            plan = ([("WAIT", float(w))] if w > 0 else []) + base
            s, l, tend = simulate_plan(x, y, now, plan)
            if conflict(s, l, oS, oL) is None:
                if best_end is None or tend < best_end:
                    best, best_end = plan, tend
                break        # larger waits on this candidate only end later
    return best

def assign(bot, tx, lane):
    """Sai's loop: map -> overlay -> collisions -> pauses -> recheck -> move."""
    other = bots[0] if bot is bots[1] else bots[1]
    now = data.time
    plan = try_schedule(bot, other, tx, lane, now)
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
        for sxc, sl in spots:
            shunt = try_schedule(other, bot, sxc, sl, now)
            if shunt is None:
                continue
            commit(other, shunt, now)
            other.target = (sxc, sl)
            plan = try_schedule(bot, other, tx, lane, now)
            if plan is not None:
                break
            # that spot scheduled but still blocks me: undo and try next
            other.queue = []
            other.prim = None
            other.target = None
            other.plan_samples = None
            other.plan_locks = []
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

fig = plt.figure(figsize=(15, 8.2))
fig.canvas.manager.set_window_title("LocoBot Fleet v2 — reservation planning")
gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0], width_ratios=[1.0, 1.35],
                      left=0.04, right=0.985, top=0.96, bottom=0.07,
                      hspace=0.28, wspace=0.12)
ax3d = fig.add_subplot(gs[0, 0])
ax = fig.add_subplot(gs[0, 1])
axt = fig.add_subplot(gs[1, :])

im3d = ax3d.imshow(np.zeros((H3D, W3D, 3), dtype=np.uint8))
ax3d.set_xticks([]); ax3d.set_yticks([])
ax3d.set_title("3D (drag=orbit, scroll=zoom)", fontsize=9)

ax.set_aspect("equal"); ax.set_xlim(-0.05, 1.87); ax.set_ylim(-0.06, 0.85)
ax.set_facecolor("#f4f3ef")
ax.add_patch(Rectangle((0.031, 0.041), 1.76, 0.70, fill=False, lw=2, ec="#222"))
ax.add_patch(Rectangle((1.42, 0.041), 0.37, 0.70, color="#dff0df", zorder=0))
for L, nm in ((LANE_A, "lane A"), (LANE_B, "lane B")):
    for off in (-0.0768, 0.0768):
        ax.plot([0.065, 1.74], [L + off, L + off], color="#999", lw=2, zorder=1)
    ax.text(0.90, L + 0.095, nm, ha="center", fontsize=9, color="#666")
tt_lines = {}
for st in (ST_LEFT, ST_RIGHT):
    for L in (LANE_A, LANE_B):
        ax.add_patch(Circle((st, L), 0.062, fill=False, ec="#555", lw=1.5, zorder=2))
        tt_lines[(st, L)] = ax.plot([], [], color="#c33", lw=3, zorder=3)[0]
BOT_HX, BOT_HY = 0.1235, 0.114
polys, tmarks = [], []
for b in bots:
    p = Polygon([[0, 0]], closed=True, fc=b.color, ec="#222", lw=1, zorder=5)
    ax.add_patch(p); polys.append(p)
    m, = ax.plot([], [], "x", color=b.color, ms=14, mew=3, zorder=6)
    tmarks.append(m)

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

def redraw():
    for b, p, m in zip(bots, polys, tmarks):
        x, y = b.xy(); yw = b.yaw()
        c, s = np.cos(yw), np.sin(yw)
        corners = np.array([[BOT_HX, BOT_HY], [BOT_HX, -BOT_HY],
                            [-BOT_HX, -BOT_HY], [-BOT_HX, BOT_HY]])
        p.set_xy(corners @ np.array([[c, -s], [s, c]]).T + [x, y])
        p.set_linewidth(3 if (selected and bots[selected - 1] is b) else 1)
        m.set_data(*([[b.target[0]], [b.target[1]]] if b.target else [[], []]))
    for (st, L), line in tt_lines.items():
        a = data.sensordata[TT_SEN[st]]
        dx, dy = 0.062 * np.cos(a), 0.062 * np.sin(a)
        line.set_data([st - dx, st + dx], [L - dy, L + dy])
    parts = [f"{b.name}: {b.prim[0] if b.prim else 'idle'}  seatbelt={b.reflex_holds}"
             for b in bots]
    head = f"[dispatch: bot{selected}]  " if selected else "[dispatch: AUTO]  "
    ax.set_title(head + "   ".join(parts), fontsize=9)
    # string diagram
    global lock_patches
    for b, line in zip(bots, plan_lines):
        if b.plan_samples:
            line.set_data([s[0] for s in b.plan_samples],
                          [s[1] for s in b.plan_samples])
    for pch in lock_patches:
        pch.remove()
    lock_patches = []
    for b in bots:
        for st, ta, tb in (b.plan_locks or []):
            r = Rectangle((ta, st - STATION_ZONE), tb - ta, 2 * STATION_ZONE,
                          color="#EF9F27", alpha=0.30, zorder=1)
            axt.add_patch(r); lock_patches.append(r)
    now_line.set_xdata([data.time])
    axt.set_xlim(max(0, data.time - 12), data.time + 45)

# ---------------- interaction ----------------
selected = 0
paused = False
drag = {"on": False, "x": 0, "y": 0}

def render_3d():
    (x1, y1), (x2, y2) = bots[0].xy(), bots[1].xy()
    cam.lookat[:] = [(x1 + x2) / 2, (y1 + y2) / 2, 0.25]
    renderer.update_scene(data, camera=cam)
    return renderer.render()

def on_press(event):
    global selected
    if event.inaxes == ax3d:
        drag.update(on=True, x=event.x, y=event.y); return
    if event.inaxes != ax or event.xdata is None:
        return
    tx = float(np.clip(event.xdata, X_MIN, X_MAX))
    lane = min((LANE_A, LANE_B), key=lambda L: abs(event.ydata - L))
    for st in (ST_LEFT, ST_RIGHT):                       # R4
        if abs(tx - st) < 0.28:
            ax.set_title("REFUSED: can't park on a turntable", fontsize=9); return
    bot = bots[selected - 1] if selected else \
        min(bots, key=lambda b: abs(b.xy()[0] - tx) + (0.5 if not b.idle() else 0))
    other = bots[0] if bot is bots[1] else bots[1]
    ot = other.target
    ox, oy = other.xy()
    if (abs(tx - ox) < 0.28 and abs(lane - oy) < 0.12) or \
       (ot and abs(tx - ot[0]) < 0.28 and abs(lane - ot[1]) < 0.01):
        ax.set_title(f"REFUSED: conflicts with {other.name}", fontsize=9); return
    if not assign(bot, tx, lane):
        ax.set_title("NO CLEAN SCHEDULE — task refused (try later or elsewhere)",
                     fontsize=9)

def on_release(event): drag["on"] = False
def on_motion(event):
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

for ev, fn in [("button_press_event", on_press), ("button_release_event", on_release),
               ("motion_notify_event", on_motion), ("scroll_event", on_scroll),
               ("key_press_event", on_key)]:
    fig.canvas.mpl_connect(ev, fn)

# ---------------- main loop ----------------
for b in bots:
    data.ctrl[b.arm[1]] = 0.3
    data.ctrl[b.arm[2]] = -0.6
STEPS_PER_FRAME = 24

plt.ion(); plt.show()
while plt.fignum_exists(fig.number):
    if not paused:
        for _ in range(STEPS_PER_FRAME):
            bots[0].step(bots[1])
            bots[1].step(bots[0])
            mujoco.mj_step(model, data)
    im3d.set_data(render_3d())
    redraw()
    fig.canvas.draw_idle()
    plt.pause(0.001)
    time.sleep(0.005)