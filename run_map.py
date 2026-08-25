"""LocomotionBot operator console — click-to-dispatch grid map.

Run:  python run_map.py

A top-down live map of the grid opens (matplotlib), alongside the 3D MuJoCo
viewer (if a display is available). Click anywhere on a lane: the click snaps
to the nearest lane centerline, the planner routes the bot there — including
driving to a station, rotating the turntables, crossing lanes, and rotating
back — and the bot executes the route on its own.

This IS the embryo of Grid Compute: click = task assignment, the planner =
traffic logic, the primitive executor = the per-bot agent.

Map:    green X = current target      teal rectangle = bot (live)
        circles = turntables (line shows current segment orientation)
        shaded right bay = charging area
Rules:  clicks are IGNORED while a turntable is rotating (the interlock!).
3D:     Space = pause physics, C = toggle chase/down camera.
"""

import time
import numpy as np
import mujoco
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle

# ---------------- world constants (drawing-verified) ----------------
LANE_A, LANE_B = 0.231, 0.551
ST_LEFT, ST_RIGHT = 0.211, 1.251
X_MIN, X_MAX = 0.09, 1.70          # usable travel span (segments + rails + CB)
BOT_X0, BOT_Y0 = 0.741, 0.231      # spawn pose (slide qpos is relative to this)
TURN_TIME = 3.0
DEG90 = 1.5708

model = mujoco.MjModel.from_xml_path("model_grid.xml")
data = mujoco.MjData(model)

A = {n: model.actuator(n).id for n in
     ["drive_x", "drive_y", "yaw_hold", "tt_left", "tt_right", "a2", "a3"]}
S = {n: model.sensor(n).id for n in
     ["bot_x_pos", "bot_y_pos", "bot_yaw_pos", "tt_left_pos", "tt_right_pos"]}

def bot_xy():
    return (BOT_X0 + data.sensordata[S["bot_x_pos"]],
            BOT_Y0 + data.sensordata[S["bot_y_pos"]])

# ---------------- the planner (Grid Compute logic) ----------------
def plan_route(cx, cy, target_x, target_lane):
    """Route from current (cx,cy) to (target_x, target_lane).
    Returns a list of primitives: ("DRIVE_X", x) ("DRIVE_Y", y) ("TURN", station, angle)."""
    plan = []
    on_lane = min((LANE_A, LANE_B), key=lambda L: abs(cy - L))
    mid_cross = abs(cy - on_lane) > 0.02          # caught between lanes

    if mid_cross:
        # finish the crossing toward whichever lane serves the goal, then unlock
        st = ST_LEFT if abs(cx - ST_LEFT) < abs(cx - ST_RIGHT) else ST_RIGHT
        plan += [("DRIVE_Y", target_lane), ("TURN", st, 0.0)]
        on_lane = target_lane

    if abs(on_lane - target_lane) > 0.01:
        # lane change needed: pick the station minimizing total X travel
        st = min((ST_LEFT, ST_RIGHT),
                 key=lambda s: abs(cx - s) + abs(target_x - s))
        plan += [("DRIVE_X", st),
                 ("TURN", st, DEG90),
                 ("DRIVE_Y", target_lane),
                 ("TURN", st, 0.0)]
    plan += [("DRIVE_X", target_x)]
    return plan

# ---------------- primitive executor (the bot agent) ----------------
queue = []            # pending primitives
prim = None           # active primitive
prim_t0 = 0.0
turn_from = 0.0
paused = False

def turning():
    return prim is not None and prim[0] == "TURN"

def start_next():
    global prim, prim_t0, turn_from
    prim = queue.pop(0) if queue else None
    prim_t0 = data.time
    if prim and prim[0] == "TURN":
        sid = S["tt_left_pos"] if prim[1] == ST_LEFT else S["tt_right_pos"]
        turn_from = data.sensordata[sid]

def agent_step():
    """One control tick: execute the active primitive. Sense -> think -> act."""
    global prim
    if prim is None:
        data.ctrl[A["drive_x"]] = data.ctrl[A["drive_y"]] = 0.0
        return
    x, y = bot_xy()
    kind = prim[0]
    if kind == "DRIVE_X":
        err = prim[1] - x
        data.ctrl[A["drive_x"]] = float(np.clip(3.0 * err, -0.13, 0.13))
        data.ctrl[A["drive_y"]] = 0.0
        if abs(err) < 0.004:
            start_next()
    elif kind == "DRIVE_Y":
        err = prim[1] - y
        data.ctrl[A["drive_y"]] = float(np.clip(3.0 * err, -0.13, 0.13))
        data.ctrl[A["drive_x"]] = 0.0
        if abs(err) < 0.004:
            start_next()
    elif kind == "TURN":
        _, st, target = prim
        act = "tt_left" if st == ST_LEFT else "tt_right"
        data.ctrl[A["drive_x"]] = data.ctrl[A["drive_y"]] = 0.0
        s = min((data.time - prim_t0) / TURN_TIME, 1.0)
        cmd = turn_from + (target - turn_from) * s
        data.ctrl[A[act]] = cmd
        data.ctrl[A["yaw_hold"]] = cmd            # soft lock: yaw follows table
        if (data.time - prim_t0) > TURN_TIME + 0.5:
            start_next()

# ---------------- the map ----------------
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.canvas.manager.set_window_title("LocoBot Grid — click to dispatch")
ax.set_aspect("equal")
ax.set_xlim(-0.05, 1.87); ax.set_ylim(-0.06, 0.85)
ax.set_facecolor("#f4f3ef")

# frame + lanes + stations + CB
ax.add_patch(Rectangle((0.031, 0.041), 1.76, 0.70, fill=False, lw=2, ec="#222"))
ax.add_patch(Rectangle((1.42, 0.041), 0.37, 0.70, color="#dff0df", zorder=0))
ax.text(1.60, 0.70, "charging bay", ha="center", fontsize=8, color="#2a6")
for L, name in ((LANE_A, "lane A"), (LANE_B, "lane B")):
    for off in (-0.0768, 0.0768):
        ax.plot([0.065, 1.74], [L + off, L + off], color="#999", lw=2, zorder=1)
    ax.text(0.90, L + 0.095, name, ha="center", fontsize=9, color="#666")
tt_lines = {}
for st in (ST_LEFT, ST_RIGHT):
    for L in (LANE_A, LANE_B):
        ax.add_patch(Circle((st, L), 0.062, fill=False, ec="#555", lw=1.5, zorder=2))
        tt_lines[(st, L)] = ax.plot([], [], color="#c33", lw=3, zorder=3)[0]
ax.plot([], [])

bot_poly = Polygon([[0, 0]], closed=True, fc="#1a8c99", ec="#0c555e", zorder=5)
ax.add_patch(bot_poly)
target_mark, = ax.plot([], [], "gx", ms=14, mew=3, zorder=6)
status = ax.set_title("click a lane to dispatch", fontsize=10)

BOT_HX, BOT_HY = 0.1235, 0.114   # half length / half width (as-drawn envelope)
def redraw():
    x, y = bot_xy()
    yaw = data.sensordata[S["bot_yaw_pos"]]
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[ BOT_HX,  BOT_HY], [ BOT_HX, -BOT_HY],
                        [-BOT_HX, -BOT_HY], [-BOT_HX,  BOT_HY]])
    R = np.array([[c, -s], [s, c]])
    bot_poly.set_xy(corners @ R.T + [x, y])
    for (st, L), line in tt_lines.items():
        a = data.sensordata[S["tt_left_pos"] if st == ST_LEFT else S["tt_right_pos"]]
        dx, dy = 0.062 * np.cos(a), 0.062 * np.sin(a)
        line.set_data([st - dx, st + dx], [L - dy, L + dy])
    if prim:
        status.set_text(f"{prim[0]} {['%.2f'%p if isinstance(p,float) else p for p in prim[1:]]}"
                        + (f"   (+{len(queue)} queued)" if queue else ""))
    else:
        status.set_text("idle — click a lane to dispatch")

def on_click(event):
    if event.inaxes != ax or event.xdata is None:
        return
    if turning():
        status.set_text("BUSY: turntable rotating — click ignored (interlock)")
        return
    tx = float(np.clip(event.xdata, X_MIN, X_MAX))
    lane = min((LANE_A, LANE_B), key=lambda L: abs(event.ydata - L))
    cx, cy = bot_xy()
    queue.clear()
    queue.extend(plan_route(cx, cy, tx, lane))
    target_mark.set_data([tx], [lane])
    start_next()

fig.canvas.mpl_connect("button_press_event", on_click)

# ---------------- optional 3D viewer ----------------
viewer = None
try:
    import mujoco.viewer
    def set_track(v):
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = model.body("bot").id
        v.cam.distance = 1.4; v.cam.elevation = -25
    def on_key(keycode):
        global paused
        if keycode == 32:
            paused = not paused
        elif keycode == ord('C') and viewer is not None:
            if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
                set_track(viewer)
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = model.camera("down_cam").id
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)
    set_track(viewer)
except Exception as e:
    print("3D viewer unavailable, running map-only:", e)

# ---------------- main loop ----------------
data.ctrl[A["a2"]] = 0.3          # stowed arm
data.ctrl[A["a3"]] = -0.6
STEPS_PER_FRAME = 16              # 16 x 2ms = 32ms sim per UI frame (~real time)

plt.ion(); plt.show()
try:
    while plt.fignum_exists(fig.number) and (viewer is None or viewer.is_running()):
        if not paused:
            for _ in range(STEPS_PER_FRAME):
                agent_step()
                mujoco.mj_step(model, data)
        redraw()
        if viewer is not None:
            viewer.sync()
        fig.canvas.draw_idle()
        plt.pause(0.001)
        time.sleep(0.01)
finally:
    if viewer is not None:
        viewer.close()