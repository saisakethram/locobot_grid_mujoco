"""Perception v1 — cubes from a bot's down-looking camera.

Pipeline (pure image + bot pose; never reads cube poses from the sim):
  1. label each saturated pixel with its NEAREST known cube hue (orange and
     yellow are only ~11 hue units apart: independent windows overlap),
     inside the table ROI (the prior)
  2. connected blobs, size-filtered against the expected cube size; blobs
     cut by the frame edge are dropped (a partial cube biases the estimate)
  3. keep the blob's TOP face: the brightest pixels (sides are shaded)
  4. back-project those pixels onto the plane z = table top + CUBE (the prior
     says where tables are and how tall cubes are)
  5. min-area rectangle of the projected points -> centre (x, y) and yaw
     (mod 90 deg: a cube looks the same every quarter turn); rejected unless
     it is a ~CUBE-sized, well-filled square (not a pad, hub or sliver)

Camera pose = bot pose (rail encoders: x, y, yaw) composed with the fixed
camera mount from the model — the same data the real bot has. In the sim the
image comes from MuJoCo's renderer; on hardware it's the USB camera frame.

The sim ground truth is used only by truth() for scoring.
"""

from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

CUBE = 0.040                     # cube edge (m), prefeed
W, H = 640, 480                  # camera frame (px) — GUESS until the real camera is known
HUE_TOL = 9                      # OpenCV hue units (0..180) around each reference hue
SAT_MIN, VAL_MIN = 110, 60       # reject table / floor / grey metal
TOP_FRAC = 0.80                  # top face = pixels >= TOP_FRAC x the blob's bright level
AREA_TOL = (0.35, 2.2)           # accepted blob area vs expected top-face area
SIDE_TOL = (0.75, 1.25)          # accepted rectangle sides vs CUBE
FILL_MIN = 0.80                  # projected pixel area / rectangle area


@dataclass
class Detection:
    name: str        # colour name, e.g. "red"
    x: float         # world (m), cube centre
    y: float
    z: float         # plane the top face was projected onto
    yaw: float       # rad, in [-pi/4, pi/4)
    px: int          # top-face pixel count
    corners_uv: np.ndarray   # (4,2) estimated top-face corners in the image, for overlay

    def corners_xy(self):
        """(4,2) estimated top-face corners in world x, y."""
        c, s, h = np.cos(self.yaw), np.sin(self.yaw), CUBE / 2
        return np.array([[self.x + c * dx - s * dy, self.y + s * dx + c * dy]
                         for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))])


def _rgb_to_hue(rgb):
    return int(cv2.cvtColor(np.uint8([[np.clip(rgb, 0, 1) * 255]]), cv2.COLOR_RGB2HSV)[0, 0, 0])


class CubeFinder:
    """One per bot camera. tables: list of (lo_xy, hi_xy, top_z) — the prior."""

    def __init__(self, model, cam, tables, w=W, h=H):
        self.model = model
        self.cid = model.camera(cam).id
        self.body = model.cam_bodyid[self.cid]
        self.w, self.h = w, h
        self.f = (h / 2) / np.tan(np.radians(model.cam_fovy[self.cid]) / 2)
        self.mount_p = model.cam_pos[self.cid].copy()
        self.mount_R = np.zeros(9)
        mujoco.mju_quat2Mat(self.mount_R, model.cam_quat[self.cid])
        self.mount_R = self.mount_R.reshape(3, 3)
        self.body_z = model.body_pos[self.body][2]        # rail height: bots don't move in z
        self.tables = tables
        # reference hues from the cube geoms' colours (the "colour calibration")
        self.palette = {}
        for g in range(model.ngeom):
            n = model.geom(g).name
            if n.startswith("cube_"):
                self.palette[n[5:]] = _rgb_to_hue(model.geom_rgba[g][:3])

    # ---- geometry ----
    def camera_pose(self, bx, by, byaw):
        c, s = np.cos(byaw), np.sin(byaw)
        Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return np.array([bx, by, self.body_z]) + Rb @ self.mount_p, Rb @ self.mount_R

    def project(self, uv, pose, z):
        """Pixels (N,2) -> points on the plane z (N,3). MuJoCo camera looks
        along its -Z, image right = +X, image up = +Y."""
        p, R = pose
        d = np.column_stack([(uv[:, 0] + 0.5 - self.w / 2) / self.f,
                             -(uv[:, 1] + 0.5 - self.h / 2) / self.f,
                             -np.ones(len(uv))]) @ R.T
        t = (z - p[2]) / d[:, 2]
        return p + d * t[:, None]

    def to_image(self, pts, pose):
        p, R = pose
        c = (np.asarray(pts) - p) @ R
        return np.column_stack([self.w / 2 + self.f * c[:, 0] / -c[:, 2] - 0.5,
                                self.h / 2 - self.f * c[:, 1] / -c[:, 2] - 0.5])

    # ---- detection ----
    def detect(self, rgb, bot_pose):
        pose = self.camera_pose(*bot_pose)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        H_, S_, V_ = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
        base = (S_ >= SAT_MIN) & (V_ >= VAL_MIN)
        names = list(self.palette)
        hues = np.array([self.palette[n] for n in names])
        dh = np.abs(H_[..., None] - hues)                  # (h, w, colours)
        dh = np.minimum(dh, 180 - dh)
        nearest, dmin = dh.argmin(-1), dh.min(-1)
        out = []
        for lo, hi, top in self.tables:
            z = top + CUBE
            roi = np.zeros((self.h, self.w), np.uint8)       # table top in the image
            quad = self.to_image([[lo[0], lo[1], z], [hi[0], lo[1], z],
                                  [hi[0], hi[1], z], [lo[0], hi[1], z]], pose)
            cv2.fillConvexPoly(roi, np.round(quad).astype(np.int32), 1)
            if not roi.any():
                continue
            # expected top-face area at this depth (px)
            depth = pose[0][2] - z
            a_exp = (CUBE * self.f / depth) ** 2
            for i, name in enumerate(names):
                mask = (base & (nearest == i) & (dmin <= HUE_TOL) & (roi > 0)).astype(np.uint8)
                n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for k in range(1, n):
                    if stats[k, cv2.CC_STAT_AREA] < AREA_TOL[0] * a_exp:
                        continue
                    bx, by, bw, bh = stats[k, :4]
                    if bx == 0 or by == 0 or bx + bw >= self.w or by + bh >= self.h:
                        continue                              # cut by the frame edge: partial
                    vs, us = np.nonzero(lab == k)
                    val = V_[vs, us]
                    keep = val >= TOP_FRAC * np.percentile(val, 95)
                    us, vs = us[keep], vs[keep]
                    if not (AREA_TOL[0] * a_exp <= len(us) <= AREA_TOL[1] * a_exp):
                        continue                              # occluded or not a cube
                    pts = self.project(np.column_stack([us, vs]).astype(float), pose, z)
                    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts[:, :2].astype(np.float32) * 1e4)
                    rw, rh = rw * 1e-4, rh * 1e-4
                    if not (SIDE_TOL[0] * CUBE <= min(rw, rh) and max(rw, rh) <= SIDE_TOL[1] * CUBE):
                        continue
                    px_area = len(us) * (depth / self.f) ** 2   # ground area of the kept pixels
                    if px_area < FILL_MIN * rw * rh:
                        continue
                    yaw = (np.radians(ang) + np.pi / 4) % (np.pi / 2) - np.pi / 4
                    det = Detection(name, cx * 1e-4, cy * 1e-4, z, yaw, len(us), None)
                    det.corners_uv = self.to_image(np.column_stack([det.corners_xy(), np.full(4, z)]), pose)
                    out.append(det)
        return out


@dataclass
class Sighting:
    x: float
    y: float
    yaw: float
    t: float          # sim time of the sighting
    source: str       # "prefeed" (layout file) | "camera"


class CubeRegistry:
    """Where each cube is believed to be: seeded from the prefeed layout
    (the cube positions written in the model file — what the operator
    loaded), then overwritten by every camera sighting. Used to decide which
    table/bot a colour belongs to before any camera has seen it."""

    def __init__(self, model):
        self.cubes = {}
        for b in range(model.nbody):
            n = model.body(b).name
            if n.startswith("cube_"):
                x, y, _ = model.body_pos[b]
                w, _, _, z = model.body_quat[b]
                yaw = (2 * np.arctan2(z, w) + np.pi / 4) % (np.pi / 2) - np.pi / 4
                self.cubes[n[5:]] = Sighting(float(x), float(y), float(yaw), 0.0, "prefeed")

    def update(self, dets, t):
        for d in dets:
            self.cubes[d.name] = Sighting(d.x, d.y, d.yaw, t, "camera")

    def as_detections(self, z):
        """Registry entries as Detections (top face at height z), for planning."""
        return [Detection(n, s.x, s.y, z, s.yaw, 0, None) for n, s in self.cubes.items()]


def truth(model, data, name):
    """Sim ground truth for scoring: (x, y, yaw mod 90 deg) of cube_<name>."""
    b = model.body(f"cube_{name}").id
    R = data.xmat[b].reshape(3, 3)
    yaw = (np.arctan2(R[1, 0], R[0, 0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return data.xpos[b][0], data.xpos[b][1], yaw


def yaw_err(a, b):
    """Smallest difference between two yaws that are equal mod 90 deg."""
    return (a - b + np.pi / 4) % (np.pi / 2) - np.pi / 4
