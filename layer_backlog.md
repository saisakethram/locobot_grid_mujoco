# Layer backlog — pick & place

What exists today (see `perception.py`, `pick.py`, `run_fleet2.py`):
camera → cube pose (≤0.5 mm, ≤1.5°) → top grasp with the jaws aligned to the cube →
place anywhere on the **same** table at any yaw (≈1 mm, ≈0.1°) → camera re-check.
Choose a colour + click a spot; the table's bot drives to its lane end first.

Each layer below lists what it adds, what we already know constrains it, and a rough
size (S = a session, M = a few, L = needs design/hardware decisions).

---

## Manipulation

### 1. Placement rotation — S — ✅ DONE
Choose the final yaw, not just position.
- The cube turns with the jaws, so rotation = a wrist-roll change. Of the
  equivalent rolls (every 90° for a cube) the nearest to the current roll is used.
- The destination jaw-room check uses the *new* jaw direction.
- UI: circular **rotate dial** (drag, 5° detents; `[` / `]` = ±15°). Colour chosen →
  the dial sets the placing angle and a ghost outline follows the mouse over the
  table. Holding a cube → the dial turns it live in the air, then click to place.
- Measured: yaw error ≤ 0.2° (mean 0.1°), position ≤ 1.3 mm; the camera re-check
  reports both.
- Jaw preview: the dial draws the open jaws to scale around its mini cube; the map
  preview draws the jaw footprints around the ghost cube among the real neighbours
  (grey = clear, red = they'd hit a neighbour, crossed-out black outline = bad
  spot). If the jaws are blocked, the preview shows the 90°-turned pair the planner
  will use (`choose_jaw`, same order as `plan_place`).
- Still open: non-square objects need mod-180°/360° handling.

### 2. Tables at different heights — M
- Perception already takes a per-table top height (the prior); pick depths come from it.
- **Reach shrinks with depth:** with the claw vertical, reach is ~0.26 m from the
  shoulder at today's table (top z = −0.25). Lower tables lose reach fast; the tip
  can't go below z ≈ −0.371 at all. Need a reach map per height (the tooling exists).
- **Travel heights are hard-coded** (`UP_Z = −0.19`, `HOVER_Z = −0.15`); they must
  become "table top + clearance". A higher table crowds the up/rise path and the
  bot's own housing keep-out.
- The stow rule still holds at any height: nothing may sit under the stowed claw
  (61 mm ahead of the bot, hanging to z ≈ −0.35), so tables stay beyond x = 1.80.
- Camera: the expected cube pixel size and projection plane scale with depth. An
  unknown height could be *estimated* from apparent cube size and checked against the prior.

### 3. Moving cubes between tables — M/L
Right now this is refused. Options:
- **Transfer shelf** between the lanes (y ≈ 0.33–0.45, x 1.80–1.96) that both bots
  can reach: A places, B picks. Simplest, and it reuses everything.
- **Carry through a turntable:** a bot stows *with* the cube and changes lanes. Needs
  a stow-with-payload pose (the cube hangs ~20 mm below the claw) and certification rules.
- **Hand-to-hand handoff in the air:** the shoulders are ~0.32 m apart at the lane
  ends, so it's geometrically possible, but needs two-arm coordination.

### 4. Carry while driving — M
Pick at one table, drive, place elsewhere (depends on 3 for another lane).
- Stow-with-cube pose: the seatbelt and stow certification must accept a loaded arm.
- Sag feed-forward with payload (+30 g at the tip).
- Reservations: the held cube lengthens the stowed-arm column.

### 5. Stacking — M
Place on top of another cube.
- Target height = lower cube top + CUBE; placement accuracy (~1 mm) is plenty.
- Perception must handle cubes at several heights: the projection plane is per
  hypothesis (choose by apparent size), and a stacked cube hides the one below.
- Jaw-room rules change: the neighbours to worry about are at the new height.

### 6. Side grasp — L
Fallback when the top is blocked or an object is too tall.
- 5-DOF constraint (§1.10): the approach can only be radial from the arm base,
  so the bot has to reposition. That means planning the bot and arm together.

### 7. Nudge to make room — M
When neighbours block the jaws, push one aside with the closed claw first
(non-prehensile), then re-look and grasp.

### 8. Regrasp / flip — M
Rotate beyond one move's range or tip a cube onto another face: pick → place
rotated → pick again.

### 9. Other objects — M each
- Different cube sizes: preshape width and expected grip angle (`Q_HOLD`) come from the
  measured size instead of `CUBE`.
- Rectangular blocks: yaw mod 180°, and grip across the narrow side.
- Cylinders: any yaw works; roll grasp.
- Heavier items: claw torque limit (1.67 N·m ⇒ ~48 N), slip, sag with payload.

## Perception

### 10. Touching / clustered cubes — M
Touching cubes of different colours are fine. Same-colour cubes that touch merge into one blob:
split by size/shape (two squares) or fit multiple rectangles.

### 11. Identity beyond colour — M
Two cubes of the same colour need IDs and data association in the registry
(nearest-neighbour tracking across looks).

### 12. Colour-agnostic detection — M
Segment "anything that isn't table" (table colour/height model) instead of a
fixed palette; classify afterwards.

### 13. Camera realism + calibration — M/L
Noise, lens distortion, exposure/lighting changes, latency, the real
resolution/FOV (640×480 @ 55° is a guess). Add a calibration step with a fiducial
(e.g. an AprilTag on each table). It gives camera extrinsics and the table pose/height,
so the table prior can be removed.

### 14. Registry confidence — S
Sightings age, so decay the confidence and re-look when stale. Scan a table when a
requested colour isn't where it's believed to be.

### 15. Camera-verified grasp — S
After lifting, confirm the cube left its spot (today only the claw angle is used).
Slip detection during carry.

### 16. Final-approach visual servoing — L
The down camera is blocked once the arm is out. The last-mm correction needs a
wrist camera (hardware) or a look-at-pre-grasp pose that keeps the cube visible.

## Fleet & planning

### 17. Task queue & compound commands — M
Several tasks per bot, both bots working in parallel, priorities.
Compound commands built from primitives: "sort by colour", "line them up",
"put red on blue", "swap red and green".

### 18. PICK/PLACE inside the planner — M
Today the pick runs as a guarded state machine next to the planner. Making it
proper reservation primitives in `simulate_plan` would show it on the string
diagram and let the other bot's plans account for it in time, not just space.

### 19. Park where the target needs — S/M
Always parking at the lane end wastes reach flexibility, and parking 4–6 cm short
already causes "out of reach". Pick the park x from the reach map per target,
with a closed-loop final approach.

### 20. Tables elsewhere — M
Beside the lanes (outside the frame), along the lane ends, near stations.
Same stow-clearance rule; needs sideways reach (J1 yaw) and per-location reach maps.

## Robustness

### 21. Recovery — M
- Missed grasp: re-look and retry instead of fail and stow.
- Dropped cube: search the table, then the floor.
- Knocked-off cube: registry update plus a report.

### 22. Servo realism — S/M
- The claw currently stalls at full 1.67 N·m while holding. A real SC15 would overheat,
  so hold with a goal just past contact and less torque.
- A blocked joint mid-pick should take the abort path (supervision already pauses it).

### 23. Hardware bring-up — L
- Swap `SimBackend` for the SC15 bus backend, and add the claw servo channel.
- Replace `model_sag` with the bench-measured sag table (§1.6).
- Feed real camera frames to `CubeFinder`.

---

## Suggested order
1. **Rotation** (S): small, visible, reuses everything.
2. **Different table heights** (M): turns hard-coded heights into table-relative ones,
   which later layers need anyway.
3. **Task queue + PICK/PLACE in the planner** (M): the base for compound commands.
4. **Transfer shelf between lanes** (M): the cheapest way to move cubes between tables.
5. **Stacking** (M).
6. Then perception robustness (10, 11, 13) before any hardware work (23).
