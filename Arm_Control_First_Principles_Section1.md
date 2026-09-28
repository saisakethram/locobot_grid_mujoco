# Arm Control from First Principles — Section 1
*Brainstorm notes, 2026-08-25. Prototype arm: 6× Waveshare SC15 bus servos (210° / 1024 steps, 1 Mbps, up to 17 kg·cm at 8.4 V; 5 arm joints — base yaw, shoulder, elbow, wrist pitch, wrist roll — plus gripper; matches `j1`–`j5` in `model_fleet.xml`), A7Z host over UART4.*

## 1.1 What "control" means for one servo
- The SC15 closes its own position loop (PID on its internal position sensor). The A7Z never drives the motor — it drives a *controller*. It sends goals; the servo closes the loop.
- The servo has a raw frame (0–1023 counts over 210° in servo mode) and soft limits in that frame. The arm has a *joint* frame (e.g. shoulder-down = 0°). They're related per joint by an offset and a sign, **measured after the horn is mounted**, never assumed:
  `joint_angle = sign × (raw − raw_offset) × (210/1024)`   # ≈ 0.205° per count
- Soft limits are stored in joint degrees.
- The SC15 loop is closed on **position only**. Speed and torque are *limits*, not goals. It cannot be told "move at 30°/s".

## 1.2 Getting velocity out of a position servo
- Send a new target every `step_time`, each a small increment away. The servo is always "almost there" and moves at the velocity the increments imply: `v = Δangle / step_time`.
- `step_time` sets the loop rate (10–20 ms → 50–100 Hz). Bus bandwidth is fine at 1 Mbps; the sender now has a real-time job.
- The increments must come from a trajectory: `angle_j(t)` per joint.
- **Resolution floor:** one count ≈ 0.205°. At 50 Hz, any joint moving slower than ~10°/s advances less than one count per tick → the servo steps in visible stairs, not a smooth ramp. Slow approach/seat moves and the slow tail of every cubic/quintic hit this. Mitigations: round (don't truncate) targets to counts; accept the stair-step on slow tails; don't expect sub-count positioning.

## 1.3 Who owns the loop
- **A7Z**, not Nano. Reason is not timing quality (the Nano is the better clock) — it's that trajectories need arm state, vision, and task plan, all of which live on the A7Z, and UART4 is wired there. Nano has no spare UART.
- Price: Linux jitter — the loop will sometimes wake late by ms.

## 1.4 Late wake-ups: wall-clock vs step-count time
- **Option A (chosen):** send the angle the schedule says for *now* — `angle = f(t_now − t_start)`. Late wakes self-correct. Needed because Bot A / Bot B relay and gripper timing must stay tied to real time.
- **Option B (rejected):** send the next angle in the list. Time drifts with every late wake; breaks anything that must sync with the arm.
- Principle: **smoothness is a property of the trajectory function; timing is a property of the loop.** Keep them separate.

Controller skeleton:
```
t = now() − t_start
for each joint j:
    wanted_j = trajectory_j(t)
    wanted_j = clamp(wanted_j, soft_limits_j)        # clamp 1: wanted angle (§1.6)
    sent_j   = wanted_j + sag_j(θ, payload)           # feed-forward (§1.6)
    sent_j   = clamp(sent_j, sent_limits_j)           # clamp 2: slightly wider, near physical stops
SYNC WRITE sent_1..sent_5 (+ gripper)               # one bus packet, UART4
sleep until next absolute tick                      # t_start + k·step_time, not now() + step_time
```

## 1.5 What read-back is for
- **Not** a second position loop — two loops on one error oscillate.
- Read-back is for **supervision**: things the servo's loop can't know.
  - Blocked: target 40°, reports 35°, load high and position not changing → stop / back off / retry.
  - Lagging: consistently trails target → slow the trajectory.
  - State for FK, handoff-pose checks, sim comparison, collision swept-space.
- Rule: **inner loop (servo) controls; outer loop (A7Z) supervises.** A7Z touches the trajectory (pause / slow / abort), never the gains. Gain tuning is offline system-ID.

## 1.6 Gravity sag and feed-forward
- A horizontal joint under load settles *below* target — P-term needs nonzero error to hold torque; hobby servos run weak/no I-term. No fault is raised. Read-back shows the true (sagged) angle.
- Fix: **feed-forward**, not gain changes:
  `target_sent = target_wanted + sag(θ, payload)`
- `sag()` is a measured lookup per joint/angle/load: command 60°, read 58.5°, record +1.5°. That *is* the sag system-ID test. At 0.205°/count that's ~7 counts; sag below ~1 count is not measurable or correctable.
- Principle: **anything predictable → feed-forward (gravity, backlash, deadband); anything unpredictable → supervision (blocked, lag).** Characterize, don't fight.
- Two clamps needed: soft limit on the *wanted* angle before compensation; a slightly wider clamp on the *sent* angle near physical stops.
- Leader-follower note: leader carries no payload so its position reading is true; the follower must add sag or it droops under the leader.

## 1.7 Trajectory shape
- Straight line `θ = A + (B−A)·t/T`: continuous position, but velocity jumps at t=0 and t=T → torque spike → the "thump" that slips a piece.
- Chosen smoothness level: **continuous velocity, bounded acceleration**. No fast loads, no flexible links, no contact regulation.
- **Cubic** (4 constraints: start/end pos, zero start/end vel), s = t/T:
  `θ(s) = A + (B−A)·(3s² − 2s³)`
  Acceleration is bounded but steps at the ends.
- **Quintic** adds zero end-acceleration (6 constraints), costs nothing at runtime:
  `θ(s) = A + (B−A)·(10s³ − 15s⁴ + 6s⁵)`
- Peak values (Δ = B−A) — these, not the average Δ/T, are what must fit under the servo limits:

  | Profile | Peak velocity | Peak acceleration |
  |---|---|---|
  | Cubic | 1.5·Δ/T | 6·Δ/T² (at the ends) |
  | Quintic | 1.875·Δ/T | ≈5.77·Δ/T² |

  So a quintic needs ~25% more T than a cubic for the same velocity limit.
- Rule: **start cubic, upgrade to quintic on evidence.**

## 1.8 Five joints, one move
- **One shared T** for all joints, set by the slowest joint using the profile's *peak* (§1.7), not its average:
  `T = max over j of max( k_v·|Δθ_j| / v_max_j ,  sqrt(k_a·|Δθ_j| / a_max_j) )`
  with k_v = 1.5, k_a = 6 (cubic) or k_v = 1.875, k_a = 5.77 (quintic). Tip draws one clean arc. Per-joint T's curl the tip path.
- "Joint halts midway" is a *waypoint*, not a special joint: two segments, that joint's second segment has A = B.
- A move = list of waypoints (5 arm angles each), trajectory = chain of segments. The gripper is a separate channel (open/close events on the same clock), not part of the arm spline.
- Chained cubics stop at every interior waypoint. Two fixes:
  - **Blend velocities** at interior waypoints (cubic accepts nonzero end velocities).
  - **B-spline** over control points: continuous vel/accel by construction, no seams; loses exact waypoint hits unless forced. This is the learned-tier representation (arXiv:2607.09648).
- Split by move type:
  - Free-space transit (rack → relay): B-spline.
  - Approach / seat (last ~3 cm): exact waypoints, straight line in *tip space*, slow.

## 1.9 Coordinate frames
1. **World / Grid** — fixed to frame; sockets, rack, relay point.
2. **Bot** — chassis origin; World↔Bot = encoder position + per-lane constant.
3. **Arm base (shoulder)** — fixed offset from Bot origin, from CAD.
4. **Tip / gripper** — from arm base via five joint angles.

Only 3→4 is hard: **kinematics**.
- **FK** (angles → tip): walk the chain; unique; easy.
- **IK** (tip → angles): multiple solutions (elbow up/down), sometimes none (reach), ill-conditioned near full extension.
- Controller needs both: IK to turn socket pose into joint targets; FK to answer "where is the tip" from read-back.

## 1.10 Pose DOF decision
- Full pose = 6 numbers (x, y, z + 3 orientation). The arm has **5** joints (the 6th servo is the gripper) → it can reach at most a 5-DOF subset of poses. Base yaw is always set by the target's position (`atan2(y, x)`), so the approach axis can never tilt sideways out of the arm's vertical plane.
- **Prototype demo (flat board, top-down):** approach axis vertical → two orientation numbers pinned. Remaining: **x, y, z, yaw** — 4 numbers for 5 joints, fully solvable. Geometric IK:
  1. `base_yaw = atan2(y, x)`.
  2. **Wrist centre:** move the tip target up by the wrist-to-tip length L3 (154.84 mm) along the approach axis: `z_w = z + L3`, `r_w = sqrt(x² + y²)`. Subtract the base→shoulder offset (39.13 mm) from z_w.
  3. **2-link planar** on (r_w, z_w) with L1 = 135.73 mm, L2 = 186.1 mm (law of cosines) → shoulder, elbow. Choose elbow-up explicitly; no solution if `sqrt(r_w² + z_w²) > L1 + L2`.
  4. **Wrist pitch:** vertical pins the sum shoulder + elbow + wrist-pitch = −90° (sign per convention) → wrist pitch is what's left over.
  5. **Wrist roll:** tool yaw in world = base yaw + wrist roll (sign per axis convention), so `wrist_roll = yaw_desired − base_yaw`. Using wrist roll as the yaw directly is wrong everywhere except dead ahead.
- **Wall-facing (CNC → material holder, roadmap):** a second *fixed* approach direction (horizontal). Same steps with the wrist-centre offset horizontal (`r_w = r − L3`, `z_w = z`) and the pitch sum = 0°. **Constraint:** a 5-DOF arm can only approach horizontally along the radial direction from its base, so the bot must park so the holder face is perpendicular to the base→holder line. That's a bot-positioning requirement, not an arm one. Two hard-coded modes, not a general solver.
- **Arbitrary approach axis (tilted board — the stretch goal):** only reachable when the tilt lies in the arm's vertical plane (pitch sum = the tilt angle; same solver). A tilt with a sideways component is **not reachable by a 5-DOF arm** — general IK won't fix it; it needs either the bot to reposition/rotate so the tilt lines up with the arm plane, or a 6th arm joint.
- Real cost of the wall is reach/torque (200 g @ ≤127 mm ceiling) and a 2-D sag table (angle × elevation). Control stack unchanged; tables grow.
- Principle: **constrain the pose, solve the small problem, generalize only when the approach direction becomes a variable.**

## Open threads for Section 2
- Write the geometric IK (5 joints, steps 1–5 in §1.10).
- Converting the tip-space straight line (approach/seat) into joint targets — where IK meets the trajectory.
- Sag table measurement procedure (fits Phase 2 bench system-ID).
- Bot (rail) controls — not yet started.

---

## Addendum (2026-09-24) — Implementability checks

### A.1 B-spline trajectories — confirmed implementable
- Recipe: **clamped cubic B-spline** per joint, control points Q₀…Qₙ, θ(t) = Σ Bᵢ(t)·Qᵢ.
- Why it fits:
  - **C² by construction** — velocity and acceleration continuous everywhere; solves the chained-cubic seam problem without hand-blending velocities.
  - **Convex hull property** — the curve never leaves the hull of its control points → clamp *control points* to soft limits and the whole path is provably limit-safe before it runs.
  - **Closed-form derivatives** — the derivative is another B-spline of control-point differences → velocity/accel bounds checkable *before* execution.
- Trap: a plain uniform B-spline doesn't pass through its end control points. **Clamped** knots (repeated end knots) fix it: starts exactly at A, ends exactly at B.
- Second trap: clamping does **not** give zero end velocity. The end velocity is `p·(Q₁−Q₀)/(u_{p+1}−u₁)` (and likewise at the far end), zero only when **Q₁ = Q₀** and **Qₙ₋₁ = Qₙ**. Repeat the end control points on purpose. Also set **Q₂ = Q₀** and **Qₙ₋₂ = Qₙ** for zero end acceleration. Without this the move starts with a velocity step — the §1.7 thump.
- Runtime: `scipy.interpolate.BSpline`; the 50 Hz loop just evaluates it. `trajectory_j(t)` skeleton unchanged — the spline is one more implementation of it.
- Scope unchanged: B-spline for free-space transit; approach/seat stays exact-waypoint, tip-space straight line.
- Status: **deferred until robot-control work starts.**

### A.2 Flex-π (arXiv:2608.10860, UW / Dieter Fox group) — role assessed
- What it is: 6B-parameter world-action model; jointly denoises RGB + 3D pointmap + DINO semantics with actions in one latent space; one checkpoint runs any subset of streams (fast action-only → full joint generation).
- Headline result: 4 mm bit into 4.5 mm socket (±0.25 mm) at 55% vs 5% best baseline, with retry behaviour on misses.
- Three reads for this project:
  1. **Not deployable** now or near-term: 60 ms action-only on an RTX 5090; A7Z is out by orders of magnitude, and it's data-hungry (large absolute demo count).
  2. **Validates two bets:** tight insertion (socket-seating class) is where learned policies now beat classical baselines; and the training data is teleop demos — exactly what the leader-follower rig produces. The rig is the on-ramp to this tier.
  3. **Interface lesson:** these models output *action chunks* (short joint-target sequences). The trajectory layer (waypoints/splines → 50 Hz loop) is the natural adapter — a learned policy is just another *producer* of control points. Nothing below it changes; the classical stack is the substrate.
- Status: **watch-list, not roadmap.**
