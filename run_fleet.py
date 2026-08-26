"""LocomotionBot FLEET console — two bots, one arbiter, click-to-dispatch.

Run:  python run_fleet.py

Right pane (map):  click a lane to dispatch. Keys 1 / 2 select which bot gets
                   the next click; 0 = AUTO (arbiter picks the cheaper bot).
Left pane (3D):    drag = orbit, scroll = zoom. Space = pause.

THE ARBITER (this is Grid Compute growing up) — three rules:
  R1 proximity hold : a bot driving toward the other bot on the same lane
                      stops at SAFE_DIST and waits until the path clears.
  R2 station mutex  : turntables at a station rotate BOTH lanes' segments
                      (shared shaft) — so a TURN is only granted when the
                      other bot is outside the whole station zone. A bot
                      denied the turn simply waits at the station.
  R3 dispatch check : a click whose target sits on top of the other bot's
                      position or target is refused outright.
  R4 no table parking: targets within the segment's swept zone of a station
                      are refused — the rotating rail would strike a parked bot.
R1 can produce an honest DEADLOCK (two bots commanded head-on): both stop
and the console says so. That's a feature — it shows exactly why the real
Grid Compute needs smarter reservations. Reroute one bot to resolve.
"""

import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle

# ---------------- world constants ----------------
LANE_A, LANE_B = 0.231, 0.551
ST_LEFT, ST_RIGHT = 0.211, 1.251
X_MIN, X_MAX = 0.09, 1.70
TURN_TIME = 3.0
DEG90 = 1.5708
SAFE_DIST = 0.32          # R1: stop this far from the other bot (center-center)
STATION_ZONE = 0.27       # R2: swept radius of the 292mm segment (0.146) + bot half-length (0.1235)
TARGET_MARGIN = 0.28      # R3: refuse targets this close to the other bot

model = mujoco.MjModel.from_xml_path("model_fleet.xml")
data = mujoco.MjData(model)

TT_ACT = {ST_LEFT: model.actuator("tt_left").id, ST_RIGHT: model.actuator("tt_right").id}
TT_SEN = {ST_LEFT: model.sensor("tt_left_pos").id, ST_RIGHT: model.sensor("tt_right_pos").id}

# ---------------- per-bot agent ----------------
class Bot:
    def __init__(self, name, x0, y0, act, sen, arm, color, pri):
        self.name, self.x0, self.y0, self.color, self.pri = name, x0, y0, color, pri
        self.A = {k: model.actuator(v).id for k, v in act.items()}
        self.S = {k: model.sensor(v).id for k, v in sen.items()}
        self.arm = [model.actuator(a).id for a in arm]
        self.queue, self.prim = [], None
        self.prim_t0 = self.turn_from = 0.0
        self.turn_started = False
        self.hold_since = None          # for deadlock detection
        self.resolved_once = False      # one auto-resolution attempt per block
        self.yielded = False            # one station-yield per task
        self.target = None

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

    def start_next(self):
        self.prim = self.queue.pop(0) if self.queue else None
        self.prim_t0 = data.time
        self.turn_started = False
        if self.prim is None:
            self.target = None

    def resolve_block(self, other):
        """Blocked > 4 s: try a detour via the free lane; else shunt an idle
        blocker onto the other lane. The Grid-Compute reflexes."""
        if self.target is None:
            return
        tx, tlane = self.target
        x, y = self.xy()
        ox, oy = other.xy()
        mylane = min((LANE_A, LANE_B), key=lambda L: abs(y - L))
        othlane = LANE_B if mylane == LANE_A else LANE_A
        # detour: same-lane block, both stations clear of the blocker
        if abs(tlane - mylane) < 0.01 and abs(oy - mylane) < 0.12:
            dirn = 1 if tx > x else -1
            S1 = ST_LEFT if dirn > 0 else ST_RIGHT
            S2 = ST_RIGHT if dirn > 0 else ST_LEFT
            if (abs(ox - S1) > STATION_ZONE and abs(ox - S2) > STATION_ZONE
                    and (S1 - ox) * (x - ox) > 0):   # S1 on MY side of the blocker
                self.queue = [("DRIVE_X", S1), ("TURN", S1, DEG90),
                              ("DRIVE_Y", othlane), ("TURN", S1, 0.0),
                              ("DRIVE_X", S2), ("TURN", S2, DEG90),
                              ("DRIVE_Y", mylane), ("TURN", S2, 0.0),
                              ("DRIVE_X", tx)]
                self.start_next()
                return
        self.shunt_peer(other)

    def shunt_peer(self, other):
        """Move an idle blocker to a clear spot on its opposite lane."""
        if not other.idle():
            return
        ox, oy = other.xy()
        sx = ox
        for st in (ST_LEFT, ST_RIGHT):
            if abs(sx - st) < 0.30:
                sx = st + (0.32 if sx >= st else -0.32)
        sx = float(np.clip(sx, X_MIN, X_MAX))
        slane = LANE_B if abs(oy - LANE_A) < 0.12 else LANE_A
        other.queue = list(plan_route(ox, oy, sx, slane, self))
        other.target = (sx, slane)
        other.start_next()

    def step(self, other):
        """One control tick. `other` is the other Bot — the arbiter's input."""
        if self.prim is None:
            self.stop_drives()
            self.hold_since = None
            return
        x, y = self.xy()
        kind = self.prim[0]

        if kind in ("DRIVE_X", "DRIVE_Y"):
            axis = 0 if kind == "DRIVE_X" else 1
            cur = x if axis == 0 else y
            err = self.prim[1] - cur
            # ---- R1: proximity hold — a true 2D check: the other bot only
            # blocks me if it is close on the PERPENDICULAR axis (bodies can
            # actually clash below ~0.24 m) AND ahead within SAFE_DIST along
            # my travel axis AND I am moving toward it.
            ox, oy = other.xy()
            perp = abs(y - oy) if axis == 0 else abs(x - ox)
            gap = (ox - x) if axis == 0 else (oy - y)
            blocked = (perp < 0.24 and abs(gap) < SAFE_DIST
                       and np.sign(gap) == np.sign(err) and abs(err) > 0.004)
            if blocked:
                # escape exception: the destination itself stays >= 0.27 m
                # clear of the blocker AND on MY side of it (e.g. a short
                # nudge to a station is not a collision course)
                tgt = self.prim[1]
                oax = ox if axis == 0 else oy
                cur = x if axis == 0 else y
                if abs(tgt - oax) >= 0.27 and (tgt - oax) * (cur - oax) > 0:
                    blocked = False
            # ---- R5: never drive into a station zone while its tables are
            # rotated or rotating — the lane segment there is swung away.
            if not blocked and axis == 0:
                for st in (ST_LEFT, ST_RIGHT):
                    ang = data.sensordata[TT_SEN[st]]
                    peer_turning = other.turning() and other.turn_started and other.prim[1] == st
                    if (ang > 0.05 or peer_turning):
                        gap_st = st - x
                        if abs(gap_st) < STATION_ZONE + 0.06 and np.sign(gap_st) == np.sign(err):
                            blocked = True
                            break
            if blocked:
                self.stop_drives()
                if self.hold_since is None:
                    self.hold_since = data.time
                elif data.time - self.hold_since > 4.0 and not self.resolved_once:
                    self.resolved_once = True
                    self.resolve_block(other)
                return
            self.hold_since = None
            self.resolved_once = False
            v = float(np.clip(3.0 * err, -0.13, 0.13))
            data.ctrl[self.A["dx"] if axis == 0 else self.A["dy"]] = v
            data.ctrl[self.A["dy"] if axis == 0 else self.A["dx"]] = 0.0
            if abs(err) < 0.004:
                self.start_next()

        elif kind == "TURN":
            _, st, tgt = self.prim
            self.stop_drives()
            # ---- R2: station mutex — the shared shaft moves BOTH lanes' tables.
            # Contested station: lower-priority bot YIELDS — backs out of the
            # zone, requeues its TURN, and retries once the station frees.
            ox, oy = other.xy()
            station_busy = (abs(ox - st) < STATION_ZONE) or \
                           (other.turning() and other.turn_started and other.prim[1] == st)
            if not self.turn_started:
                if station_busy:
                    contested = abs(oy - y) > 0.10        # peer on the other lane
                    if (self.pri > other.pri and contested and not self.yielded
                            and abs(x - st) < STATION_ZONE):
                        self.yielded = True
                        retreat = st + 0.40                # outward both stations
                        self.queue.insert(0, self.prim)
                        self.queue.insert(0, ("DRIVE_X", st))   # come BACK first
                        self.queue.insert(0, ("DRIVE_X", retreat))
                        self.start_next()
                        return
                    if self.hold_since is None:
                        self.hold_since = data.time
                    elif data.time - self.hold_since > 4.0 and not self.resolved_once:
                        self.resolved_once = True
                        self.shunt_peer(other)             # idle peer parked in zone
                    return
                self.hold_since = None
                self.resolved_once = False
                self.turn_started = True
                self.prim_t0 = data.time
                self.turn_from = data.sensordata[TT_SEN[st]]
            s = min((data.time - self.prim_t0) / TURN_TIME, 1.0)
            cmd = self.turn_from + (tgt - self.turn_from) * s
            data.ctrl[TT_ACT[st]] = cmd
            data.ctrl[self.A["yawh"]] = cmd
            if (data.time - self.prim_t0) > TURN_TIME + 0.5:
                self.start_next()

def station_time_cost(cx, tx, st, other):
    """Estimated SECONDS to transfer via station st, congestion-aware."""
    t = (abs(cx - st) + abs(tx - st)) / 0.13     # drive time
    t += 2 * TURN_TIME + 1.0                     # two turntable swings + settle
    if other is not None:
        ox, oy = other.xy()
        if abs(ox - st) < STATION_ZONE:
            t += 8.0 if other.idle() else 6.0    # zone occupied -> likely wait/shunt
        if other.turning() and other.prim[1] == st:
            t += 6.0                             # mid-rotation right now
        # peer's own route heading for this station
        if other.prim and other.prim[0] == "DRIVE_X" and any(
                p[0] == "TURN" and p[1] == st for p in [other.prim] + other.queue if len(p) > 2):
            t += 4.0
    return t

def plan_route(cx, cy, tx, target_lane, other=None, forbid_station=None):
    plan = []
    on_lane = min((LANE_A, LANE_B), key=lambda L: abs(cy - L))
    if abs(cy - on_lane) > 0.02:
        st = ST_LEFT if abs(cx - ST_LEFT) < abs(cx - ST_RIGHT) else ST_RIGHT
        plan += [("DRIVE_Y", target_lane), ("TURN", st, 0.0)]
        on_lane = target_lane
    if abs(on_lane - target_lane) > 0.01:
        options = [s for s in (ST_LEFT, ST_RIGHT) if s != forbid_station] \
                  or [ST_LEFT, ST_RIGHT]
        st = min(options, key=lambda s: station_time_cost(cx, tx, s, other))
        plan += [("DRIVE_X", st), ("TURN", st, DEG90),
                 ("DRIVE_Y", target_lane), ("TURN", st, 0.0)]
    plan += [("DRIVE_X", tx)]
    return plan

def break_deadlock():
    """Both bots holding > 5 s: force-replan the lower-priority lane-resident
    bot via the OTHER station. The circular-wait breaker."""
    if not all(b.hold_since is not None and data.time - b.hold_since > 5.0
               for b in bots):
        return
    for b in sorted(bots, key=lambda bb: -bb.pri):
        x, y = b.xy()
        on_lane = min(abs(y - LANE_A), abs(y - LANE_B)) < 0.02
        if on_lane and b.target:
            peer = bots[0] if b is bots[1] else bots[1]
            contested = min((ST_LEFT, ST_RIGHT), key=lambda s: abs(x - s))
            newq = list(plan_route(x, y, b.target[0], b.target[1],
                                   peer, forbid_station=contested))
            if not any(p[0] == "TURN" for p in newq):
                continue        # replan changes nothing (same-lane head-on)
            b.queue = newq
            b.hold_since = None
            b.yielded = False
            b.resolved_once = False
            b.start_next()
            return

def route_cost(bot, tx, lane):
    cx, cy = bot.xy()
    on = min((LANE_A, LANE_B), key=lambda L: abs(cy - L))
    cost = abs(tx - cx)
    if abs(on - lane) > 0.01:
        st = min((ST_LEFT, ST_RIGHT), key=lambda s: abs(cx - s) + abs(tx - s))
        cost = abs(cx - st) + abs(tx - st) + 0.5      # transfer penalty
    if not bot.idle():
        cost += 0.4                                   # busy penalty
    return cost

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
selected = 0            # 0 = AUTO, 1/2 = explicit bot
paused = False

# ---------------- offscreen 3D ----------------
W3D, H3D = 640, 480
renderer = mujoco.Renderer(model, H3D, W3D)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.azimuth, cam.elevation = 1.8, 135, -28

def render_3d():
    (x1, y1), (x2, y2) = bots[0].xy(), bots[1].xy()
    cam.lookat[:] = [(x1 + x2) / 2, (y1 + y2) / 2, 0.25]
    renderer.update_scene(data, camera=cam)
    return renderer.render()

# ---------------- window ----------------
fig, (ax3d, ax) = plt.subplots(1, 2, figsize=(15, 5.6), width_ratios=[1.05, 1.4])
fig.canvas.manager.set_window_title("LocoBot Fleet — 1/2 select bot, 0 auto, click to dispatch")
fig.subplots_adjust(left=0.02, right=0.99, wspace=0.06)
im3d = ax3d.imshow(np.zeros((H3D, W3D, 3), dtype=np.uint8))
ax3d.set_xticks([]); ax3d.set_yticks([])
ax3d.set_title("3D  (drag=orbit, scroll=zoom)", fontsize=9)

ax.set_aspect("equal"); ax.set_xlim(-0.05, 1.87); ax.set_ylim(-0.06, 0.85)
ax.set_facecolor("#f4f3ef")
ax.add_patch(Rectangle((0.031, 0.041), 1.76, 0.70, fill=False, lw=2, ec="#222"))
ax.add_patch(Rectangle((1.42, 0.041), 0.37, 0.70, color="#dff0df", zorder=0))
ax.text(1.60, 0.70, "charging bay", ha="center", fontsize=8, color="#2a6")
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

def redraw():
    for b, p, m in zip(bots, polys, tmarks):
        x, y = b.xy(); yaw = b.yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        corners = np.array([[BOT_HX, BOT_HY], [BOT_HX, -BOT_HY],
                            [-BOT_HX, -BOT_HY], [-BOT_HX, BOT_HY]])
        p.set_xy(corners @ np.array([[c, -s], [s, c]]).T + [x, y])
        p.set_linewidth(3 if (selected and bots[selected-1] is b) else 1)
        if b.target: m.set_data([b.target[0]], [b.target[1]])
        else: m.set_data([], [])
    for (st, L), line in tt_lines.items():
        a = data.sensordata[TT_SEN[st]]
        dx, dy = 0.062 * np.cos(a), 0.062 * np.sin(a)
        line.set_data([st - dx, st + dx], [L - dy, L + dy])
    parts = []
    for b in bots:
        if b.hold_since and data.time - b.hold_since > 3.0:
            parts.append(f"{b.name}: WAITING {data.time-b.hold_since:.0f}s")
        elif b.prim: parts.append(f"{b.name}: {b.prim[0]}")
        else: parts.append(f"{b.name}: idle")
    both_stuck = all(b.hold_since and data.time - b.hold_since > 3.0 for b in bots)
    head = "DEADLOCK — reroute one bot!  " if both_stuck else \
           (f"[dispatching: bot{selected}]  " if selected else "[dispatching: AUTO]  ")
    ax.set_title(head + "   ".join(parts), fontsize=9)

# ---------------- interaction ----------------
drag = {"on": False, "x": 0, "y": 0}

def on_press(event):
    global selected
    if event.inaxes == ax3d:
        drag.update(on=True, x=event.x, y=event.y); return
    if event.inaxes != ax or event.xdata is None:
        return
    tx = float(np.clip(event.xdata, X_MIN, X_MAX))
    lane = min((LANE_A, LANE_B), key=lambda L: abs(event.ydata - L))
    # ---- R4: turntables are infrastructure, not parking
    for st in (ST_LEFT, ST_RIGHT):
        if abs(tx - st) < 0.28:
            ax.set_title("REFUSED: can't park on a turntable — pick a spot clear of the station", fontsize=9)
            return
    if selected: bot = bots[selected - 1]
    else: bot = min(bots, key=lambda b: route_cost(b, tx, lane))
    other = bots[1] if bot is bots[0] else bots[0]
    if bot.turning():
        ax.set_title(f"{bot.name} BUSY turning — click ignored (interlock)", fontsize=9); return
    # ---- R3: dispatch conflict check
    ox, oy = other.xy()
    ot = other.target
    if (abs(tx - ox) < TARGET_MARGIN and abs(lane - oy) < 0.12) or \
       (ot and abs(tx - ot[0]) < TARGET_MARGIN and abs(lane - ot[1]) < 0.01):
        ax.set_title(f"REFUSED: target conflicts with {other.name}", fontsize=9); return
    cx, cy = bot.xy()
    bot.queue.clear(); bot.queue.extend(plan_route(cx, cy, tx, lane, other))
    bot.target = (tx, lane)
    bot.resolved_once = False
    bot.yielded = False
    bot.start_next()

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
for b in bots:                      # stow both arms
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
        break_deadlock()
    im3d.set_data(render_3d())
    redraw()
    fig.canvas.draw_idle()
    plt.pause(0.001)
    time.sleep(0.005)