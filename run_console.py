"""LocomotionBot unified console — 3D view + dispatch map in ONE window.

Run:  python run_console.py

Left pane : live 3D render of the sim (drag = orbit, scroll = zoom,
            'c' key = toggle chase view / bot's down-camera)
Right pane: the grid map — click a lane to dispatch the bot; the planner
            routes through stations and turntables automatically.
Space     : pause physics.

How it works: MuJoCo renders OFFSCREEN (mujoco.Renderer) into a numpy image
every frame; matplotlib displays it next to the map. Matplotlib is the one
window that owns everything — the MuJoCo viewer window is not used at all.
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
BOT_X0, BOT_Y0 = 0.741, 0.231
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

# ---------------- planner + agent (same as run_map.py) ----------------
def plan_route(cx, cy, tx, target_lane):
    plan = []
    on_lane = min((LANE_A, LANE_B), key=lambda L: abs(cy - L))
    if abs(cy - on_lane) > 0.02:                       # stranded mid-cross
        st = ST_LEFT if abs(cx - ST_LEFT) < abs(cx - ST_RIGHT) else ST_RIGHT
        plan += [("DRIVE_Y", target_lane), ("TURN", st, 0.0)]
        on_lane = target_lane
    if abs(on_lane - target_lane) > 0.01:
        st = min((ST_LEFT, ST_RIGHT), key=lambda s: abs(cx - s) + abs(tx - s))
        plan += [("DRIVE_X", st), ("TURN", st, DEG90),
                 ("DRIVE_Y", target_lane), ("TURN", st, 0.0)]
    plan += [("DRIVE_X", tx)]
    return plan

queue, prim = [], None
prim_t0 = turn_from = 0.0
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
        if abs(err) < 0.004: start_next()
    elif kind == "DRIVE_Y":
        err = prim[1] - y
        data.ctrl[A["drive_y"]] = float(np.clip(3.0 * err, -0.13, 0.13))
        data.ctrl[A["drive_x"]] = 0.0
        if abs(err) < 0.004: start_next()
    else:
        _, st, target = prim
        act = "tt_left" if st == ST_LEFT else "tt_right"
        data.ctrl[A["drive_x"]] = data.ctrl[A["drive_y"]] = 0.0
        s = min((data.time - prim_t0) / TURN_TIME, 1.0)
        cmd = turn_from + (target - turn_from) * s
        data.ctrl[A[act]] = cmd
        data.ctrl[A["yaw_hold"]] = cmd
        if (data.time - prim_t0) > TURN_TIME + 0.5: start_next()

# ---------------- offscreen 3D renderer ----------------
W3D, H3D = 640, 480
renderer = mujoco.Renderer(model, H3D, W3D)
cam = mujoco.MjvCamera()                 # our own orbitable chase camera
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.azimuth, cam.elevation = 1.5, 135, -22
use_down_cam = False

def render_3d():
    if use_down_cam:
        renderer.update_scene(data, camera="down_cam")
    else:
        x, y = bot_xy()
        cam.lookat[:] = [x, y, 0.25]     # chase: always look at the bot
        renderer.update_scene(data, camera=cam)
    return renderer.render()

# ---------------- the window: 3D pane + map pane ----------------
fig, (ax3d, ax) = plt.subplots(
    1, 2, figsize=(15, 5.6), width_ratios=[1.05, 1.4])
fig.canvas.manager.set_window_title("LocoBot Console — one screen, both views")
fig.subplots_adjust(left=0.02, right=0.99, wspace=0.06)

im3d = ax3d.imshow(np.zeros((H3D, W3D, 3), dtype=np.uint8))
ax3d.set_xticks([]); ax3d.set_yticks([])
ax3d.set_title("3D  (drag=orbit, scroll=zoom, 'c'=bot cam)", fontsize=9)

ax.set_aspect("equal")
ax.set_xlim(-0.05, 1.87); ax.set_ylim(-0.06, 0.85)
ax.set_facecolor("#f4f3ef")
ax.set_title("click a lane to dispatch", fontsize=10)
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
bot_poly = Polygon([[0, 0]], closed=True, fc="#1a8c99", ec="#0c555e", zorder=5)
ax.add_patch(bot_poly)
target_mark, = ax.plot([], [], "gx", ms=14, mew=3, zorder=6)

BOT_HX, BOT_HY = 0.1235, 0.114
def redraw_map():
    x, y = bot_xy()
    yaw = data.sensordata[S["bot_yaw_pos"]]
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[ BOT_HX,  BOT_HY], [ BOT_HX, -BOT_HY],
                        [-BOT_HX, -BOT_HY], [-BOT_HX,  BOT_HY]])
    bot_poly.set_xy(corners @ np.array([[c, -s], [s, c]]).T + [x, y])
    for (st, L), line in tt_lines.items():
        a = data.sensordata[S["tt_left_pos"] if st == ST_LEFT else S["tt_right_pos"]]
        dx, dy = 0.062 * np.cos(a), 0.062 * np.sin(a)
        line.set_data([st - dx, st + dx], [L - dy, L + dy])
    if prim:
        ax.set_title(f"{prim[0]} {['%.2f'%p if isinstance(p,float) else p for p in prim[1:]]}"
                     + (f"  (+{len(queue)} queued)" if queue else ""), fontsize=10)
    else:
        ax.set_title("idle — click a lane to dispatch", fontsize=10)

# ---------------- interaction ----------------
drag = {"on": False, "x": 0, "y": 0}

def on_press(event):
    if event.inaxes == ax3d:
        drag.update(on=True, x=event.x, y=event.y)
    elif event.inaxes == ax and event.xdata is not None:
        if turning():
            ax.set_title("BUSY: turntable rotating — click ignored (interlock)", fontsize=10)
            return
        tx = float(np.clip(event.xdata, X_MIN, X_MAX))
        lane = min((LANE_A, LANE_B), key=lambda L: abs(event.ydata - L))
        cx, cy = bot_xy()
        queue.clear(); queue.extend(plan_route(cx, cy, tx, lane))
        target_mark.set_data([tx], [lane])
        start_next()

def on_release(event):
    drag["on"] = False

def on_motion(event):
    if drag["on"] and not use_down_cam:
        cam.azimuth   -= (event.x - drag["x"]) * 0.4
        cam.elevation = float(np.clip(cam.elevation + (event.y - drag["y"]) * 0.3, -89, 5))
        drag.update(x=event.x, y=event.y)

def on_scroll(event):
    if event.inaxes == ax3d and not use_down_cam:
        cam.distance = float(np.clip(cam.distance * (0.9 if event.button == "up" else 1.1),
                                     0.4, 4.0))

def on_key(event):
    global paused, use_down_cam
    if event.key == " ":
        paused = not paused
    elif event.key == "c":
        use_down_cam = not use_down_cam

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("scroll_event", on_scroll)
fig.canvas.mpl_connect("key_press_event", on_key)

# ---------------- main loop ----------------
data.ctrl[A["a2"]] = 0.3
data.ctrl[A["a3"]] = -0.6
STEPS_PER_FRAME = 24            # 48 ms sim per UI frame; UI ~15-20 fps => ~real time

plt.ion(); plt.show()
while plt.fignum_exists(fig.number):
    if not paused:
        for _ in range(STEPS_PER_FRAME):
            agent_step()
            mujoco.mj_step(model, data)
    im3d.set_data(render_3d())
    redraw_map()
    fig.canvas.draw_idle()
    plt.pause(0.001)
    time.sleep(0.005)