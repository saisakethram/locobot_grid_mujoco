"""LocomotionBot grid demo — full circulation loop with turntable transfers.

Run:  python run_grid.py

The bot circulates: lane A leftward -> left-station transfer to lane B ->
lane B rightward -> right-station transfer back to lane A -> repeat.
Turntables take TURN_TIME seconds per 90 deg (worm-geared: slow, stiff).
The bot's yaw is commanded to follow the turntable angle -- this is the
"soft lock": nothing physically holds the bot, exactly like the real design.

Space = pause. Camera tracks the bot.
Dimensions are real, extracted from Full_Grid.glb.
"""

import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("model_grid.xml")
data = mujoco.MjData(model)

# --- geometry constants (from the GLB) -----------------------------------
LANE_A, LANE_B = 0.231, 0.551
BOT_X0, BOT_Y0 = 0.741, 0.231   # bot start pose; slide qpos is RELATIVE to this
ST_LEFT, ST_RIGHT = 0.211, 1.251
DOCK_X = 1.60                      # charging bay parking spot (lane A only)
TURN_TIME = 3.0                    # seconds per 90 deg turntable swing
DEG90 = 1.5708

# --- actuator / sensor indices --------------------------------------------
A = {n: model.actuator(n).id for n in
     ["drive_x", "drive_y", "yaw_hold", "tt_left", "tt_right",
      "a1", "a2", "a3", "a4", "a5"]}
S = {n: model.sensor(n).id for n in
     ["bot_x_pos", "bot_y_pos", "bot_yaw_pos", "tt_left_pos", "tt_right_pos"]}

paused = False
_viewer_ref = {}          # filled after launch so the key callback can switch cameras

def set_track_cam(v):
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.trackbodyid = model.body("bot").id
    v.cam.distance = 1.4
    v.cam.elevation = -25

def set_down_cam(v):
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    v.cam.fixedcamid = model.camera("down_cam").id

def on_key(keycode):
    global paused
    if keycode == 32:                      # Space: pause
        paused = not paused
    elif keycode == ord('C'):              # C: toggle bot-tracking <-> down camera
        v = _viewer_ref.get("v")
        if v is None:
            return
        if v.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
            set_track_cam(v)
        else:
            set_down_cam(v)

def drive_to(axis, target, pos):
    """Simple proportional approach with the N20 speed cap. Returns (ctrl, arrived)."""
    err = target - pos
    if abs(err) < 0.004:
        return 0.0, True
    v = max(-0.13, min(0.13, 3.0 * err))
    return v, False

def ramp(t, T, a, b):
    """Linear ramp from a to b over T seconds (the worm drive's slow swing)."""
    s = max(0.0, min(t / T, 1.0))
    return a + (b - a) * s

# --- circulation plan: (state, params) ------------------------------------
# Each entry: what the bot-agent does. This IS the future A7Z state machine.
PLAN = [
    ("DRIVE_X", ST_LEFT),      # lane A -> left station
    ("TURN", ("tt_left", DEG90)),
    ("DRIVE_Y", LANE_B),       # cross to lane B on the rotated segments
    ("TURN", ("tt_left", 0.0)),
    ("DRIVE_X", ST_RIGHT),     # lane B -> right station
    ("TURN", ("tt_right", DEG90)),
    ("DRIVE_Y", LANE_A),       # cross back to lane A
    ("TURN", ("tt_right", 0.0)),
    ("DRIVE_X", DOCK_X),       # visit the charging bay
    ("DWELL", 2.0),            # "charge"
    ("DRIVE_X", 0.741),        # back to mid-grid, loop restarts
]

step_i, t0 = 0, 0.0
turn_from = 0.0

with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    _viewer_ref["v"] = viewer
    set_track_cam(viewer)

    # arm in stowed travel pose
    data.ctrl[A["a2"]] = 0.3
    data.ctrl[A["a3"]] = -0.6

    while viewer.is_running():
        if paused:
            viewer.sync(); time.sleep(0.05); continue

        t = data.time
        x   = BOT_X0 + data.sensordata[S["bot_x_pos"]]   # world coords
        y   = BOT_Y0 + data.sensordata[S["bot_y_pos"]]
        state, arg = PLAN[step_i]
        done = False

        if state == "DRIVE_X":
            data.ctrl[A["drive_x"]], done = drive_to("x", arg, x)
            data.ctrl[A["drive_y"]] = 0.0
        elif state == "DRIVE_Y":
            data.ctrl[A["drive_y"]], done = drive_to("y", arg, y)
            data.ctrl[A["drive_x"]] = 0.0
        elif state == "TURN":
            tt_name, target = arg
            data.ctrl[A["drive_x"]] = data.ctrl[A["drive_y"]] = 0.0
            cmd = ramp(t - t0, TURN_TIME, turn_from, target)
            data.ctrl[A[tt_name]] = cmd
            data.ctrl[A["yaw_hold"]] = cmd        # soft lock: bot yaw follows table
            done = (t - t0) > TURN_TIME + 0.5     # settle margin
        elif state == "DWELL":
            data.ctrl[A["drive_x"]] = data.ctrl[A["drive_y"]] = 0.0
            done = (t - t0) > arg

        if done:
            step_i = (step_i + 1) % len(PLAN)
            t0 = t
            nxt = PLAN[step_i]
            if nxt[0] == "TURN":
                turn_from = data.sensordata[S["tt_left_pos"] if nxt[1][0] == "tt_left"
                                            else S["tt_right_pos"]]
            print(f"t={t:6.1f}s  -> {nxt[0]} {nxt[1]}")

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)