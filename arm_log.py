"""Joint world-coordinate logger (sim).

One CSV row per bot per sample:
  t, bot, prim, status,
  <joint>_x/_y/_z   world position of each joint axis anchor (m)
  tip_x/_y/_z       gripper tip site (m)
  <joint>_q         measured angle (rad)
  <joint>_qw        controller's wanted angle (rad)
  keepout           keep-out zones the MEASURED arm is inside ("|"-joined, "" = clear)

World frame = MuJoCo world = the grid frame (§1.9 frame 1). On hardware the
same columns come from FK on read-back instead of MuJoCo's kinematics.
"""

import csv
import os
import time

import mujoco


class JointLogger:
    def __init__(self, model, data, arms, rate_hz=50.0, folder="logs"):
        """arms: [(bot, [joint names], tip site name), ...] where bot has
        .name, .prim and .arm_ctl (status, wanted, be.read())."""
        self.model, self.data = model, data
        self.dt = 1.0 / rate_hz
        self.arms = [(bot, [model.joint(j).id for j in joints], model.site(tip).id)
                     for bot, joints, tip in arms]
        self.names = [j[:-3] if j.endswith("_b2") else j for j in arms[0][1]]
        self.folder = folder
        self.path = None
        self._f = self._w = None
        self._next = 0.0
        self._last_flush = 0.0
        self.rows = 0

    @property
    def active(self):
        return self._f is not None

    def start(self, now):
        os.makedirs(self.folder, exist_ok=True)
        self.path = os.path.join(self.folder, time.strftime("arm_joints_%Y%m%d_%H%M%S.csv"))
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        head = ["t", "bot", "prim", "status"]
        for n in self.names:
            head += [f"{n}_x", f"{n}_y", f"{n}_z"]
        head += ["tip_x", "tip_y", "tip_z"]
        head += [f"{n}_q" for n in self.names] + [f"{n}_qw" for n in self.names]
        head += ["keepout"]
        self._w.writerow(head)
        self._next = self._last_flush = now
        self.rows = 0
        return self.path

    def stop(self):
        if self._f:
            self._f.close()
        self._f = self._w = None

    def sample(self, now):
        """Call every physics step; writes only on rate_hz boundaries."""
        if not self._f or now + 1e-9 < self._next:
            return
        self._next += self.dt * (1 + int((now - self._next) / self.dt))
        # after mj_step, xanchor lags qpos by one step: recompute kinematics so
        # positions and angles in a row describe the same instant
        mujoco.mj_kinematics(self.model, self.data)
        for bot, jids, tip in self.arms:
            ctl = bot.arm_ctl
            row = [f"{now:.4f}", bot.name,
                   ":".join(str(p) for p in bot.prim) if bot.prim else "idle", ctl.status]
            for j in jids:
                row += [f"{v:.5f}" for v in self.data.xanchor[j]]
            row += [f"{v:.5f}" for v in self.data.site_xpos[tip]]
            row += [f"{v:.5f}" for v in ctl.be.read()] + [f"{v:.5f}" for v in ctl.wanted]
            row += ["|".join(ctl.keepout(None)) if ctl.keepout else ""]
            self._w.writerow(row)
            self.rows += 1
        if now - self._last_flush > 1.0:       # survive a crash with <1 s lost
            self._f.flush()
            self._last_flush = now
