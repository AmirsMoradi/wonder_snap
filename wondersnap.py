"""WonderSnap — gesture-controlled GPU particle monuments.

All monuments are procedural point clouds.  No 3D model files are used.
The renderer keeps the heavy per-particle interpolation on the GPU.
"""

from __future__ import annotations

import argparse
import getpass
import math
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from urllib.parse import quote
from urllib.request import urlopen

# Keep third-party inference libraries quiet; actionable errors are still shown.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")


def _relaunch_in_project_venv() -> None:
    """Recover automatically when an IDE selects the global Python.

    This happens before NumPy/MediaPipe are imported, so binary wheels from the
    global environment cannot poison the process. Debuggers should still select
    the venv explicitly, but ordinary Run works without IDE configuration.
    """
    if os.environ.get("WONDERSNAP_VENV_RELAUNCHED") == "1":
        return
    project_dir = Path(__file__).resolve().parent
    local_python = project_dir / ".venv" / "Scripts" / "python.exe"
    if not local_python.is_file():
        return
    try:
        already_local = Path(sys.executable).resolve().samefile(local_python.resolve())
    except (OSError, FileNotFoundError):
        already_local = Path(sys.executable).resolve() == local_python.resolve()
    if already_local:
        return
    print(f"WonderSnap: switching to project environment: {local_python}", flush=True)
    child_env = os.environ.copy()
    child_env["WONDERSNAP_VENV_RELAUNCHED"] = "1"
    result = subprocess.run(
        [str(local_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        env=child_env,
    )
    raise SystemExit(result.returncode)


_relaunch_in_project_venv()

import numpy as np


SHAPES = ("TEMPLE GATE", "STATUE OF LIBERTY", "BIG BEN", "COLOSSEUM")
RTSP_MAIN_PATH = "/cam/realmonitor?channel=1&subtype=0"
RTSP_SUB_PATH = "/cam/realmonitor?channel=1&subtype=1"
RTSP_PASSWORD_ENV = "WONDERSNAP_RTSP_PASSWORD"
# Kept in the source so the demo runs with no setup, at the camera owner's
# request.  WONDERSNAP_RTSP_PASSWORD and --rtsp-password still take priority,
# so the credential can be moved back out without touching any other code.
DEFAULT_RTSP_PASSWORD = "your camera pass"
# FFmpeg queues decoded RTSP frames; these options keep that queue short and
# stop a dead camera from blocking a read forever.
RTSP_FFMPEG_OPTIONS = "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|flags;low_delay"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
# Additive blending sums every overlapping particle, so the per-particle
# contribution has to fall as the count rises.  Without this a dense monument
# saturates to a flat white silhouette and loses its colour entirely.
PARTICLE_GAIN_REFERENCE = 22_000.0
COLORS = (
    (1.00, 0.78, 0.05),
    (0.20, 1.00, 0.62),
    (0.10, 0.46, 1.00),
    (1.00, 0.08, 0.16),
)


# ---------------------------------------------------------------------------
# Procedural geometry


def _fit_count(parts: list[np.ndarray], count: int, rng: np.random.Generator) -> np.ndarray:
    points = np.concatenate([p for p in parts if len(p)], axis=0).astype("f4")
    if len(points) < count:
        points = np.concatenate(
            (points, points[rng.integers(0, len(points), count - len(points))]), axis=0
        )
    elif len(points) > count:
        points = points[rng.choice(len(points), count, replace=False)]
    rng.shuffle(points)
    return points


def _box_surface(
    rng: np.random.Generator,
    n: int,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
) -> np.ndarray:
    size_a = np.asarray(size, dtype="f4")
    center_a = np.asarray(center, dtype="f4")
    p = rng.uniform(-0.5, 0.5, (n, 3)).astype("f4") * size_a
    face = rng.integers(0, 3, n)
    p[np.arange(n), face] = rng.choice((-0.5, 0.5), n) * size_a[face]
    return p + center_a


def _elliptic_cylinder(
    rng: np.random.Generator,
    n: int,
    rx: float,
    rz: float,
    height: float,
    center: tuple[float, float, float],
    caps: bool = True,
) -> np.ndarray:
    theta = rng.uniform(0.0, math.tau, n)
    y = rng.uniform(-height / 2.0, height / 2.0, n)
    radial = np.ones(n)
    if caps:
        cap = rng.random(n) < 0.14
        y[cap] = rng.choice((-height / 2.0, height / 2.0), cap.sum())
        radial[cap] = np.sqrt(rng.random(cap.sum()))
    x = rx * radial * np.cos(theta)
    z = rz * radial * np.sin(theta)
    return np.column_stack((x, y, z)).astype("f4") + np.asarray(center, dtype="f4")


def _sphere(
    rng: np.random.Generator,
    n: int,
    radius: float,
    center: tuple[float, float, float],
) -> np.ndarray:
    v = rng.normal(size=(n, 3)).astype("f4")
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-7
    return v * radius + np.asarray(center, dtype="f4")


def _cone_surface(
    rng: np.random.Generator,
    n: int,
    r0: float,
    r1: float,
    height: float,
    center: tuple[float, float, float],
) -> np.ndarray:
    t = rng.random(n)
    theta = rng.uniform(0.0, math.tau, n)
    r = r0 + (r1 - r0) * t
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    y = (t - 0.5) * height
    return np.column_stack((x, y, z)).astype("f4") + np.asarray(center, dtype="f4")


def _tube_between(
    rng: np.random.Generator,
    n: int,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    a0, b0 = np.asarray(a, dtype="f4"), np.asarray(b, dtype="f4")
    axis = b0 - a0
    length = float(np.linalg.norm(axis))
    w = axis / max(length, 1e-7)
    helper = np.array((0.0, 1.0, 0.0), dtype="f4")
    if abs(float(np.dot(w, helper))) > 0.9:
        helper = np.array((1.0, 0.0, 0.0), dtype="f4")
    u = np.cross(w, helper)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    t = rng.random((n, 1))
    angle = rng.uniform(0.0, math.tau, (n, 1))
    shell = radius * (np.cos(angle) * u + np.sin(angle) * v)
    return (a0 + t * axis + shell).astype("f4")


def _ring(
    rng: np.random.Generator,
    n: int,
    radius: float,
    center: tuple[float, float, float],
    plane: str = "xy",
    thickness: float = 0.025,
) -> np.ndarray:
    theta = rng.uniform(0.0, math.tau, n)
    r = radius + rng.normal(0.0, thickness, n)
    p = np.zeros((n, 3), dtype="f4")
    if plane == "xy":
        p[:, 0], p[:, 1] = r * np.cos(theta), r * np.sin(theta)
    else:
        p[:, 0], p[:, 2] = r * np.cos(theta), r * np.sin(theta)
    return p + np.asarray(center, dtype="f4")


def _temple_gate(rng: np.random.Generator, n: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    # Stone podium, four pillars, three layered roofs and a glowing emblem.
    parts.append(_box_surface(rng, n * 10 // 100, (3.4, 0.28, 1.25), (0, -1.65, 0)))
    for x in (-1.35, -0.45, 0.45, 1.35):
        parts.append(_box_surface(rng, n * 8 // 100, (0.28, 2.45, 0.38), (x, -0.35, 0)))
    parts.append(_box_surface(rng, n * 11 // 100, (3.25, 0.38, 1.18), (0, 0.83, 0)))
    parts.append(_cone_surface(rng, n * 14 // 100, 2.1, 1.35, 0.42, (0, 1.12, 0)))
    parts.append(_box_surface(rng, n * 8 // 100, (2.0, 0.48, 0.90), (0, 1.48, 0)))
    parts.append(_cone_surface(rng, n * 12 // 100, 1.48, 0.75, 0.36, (0, 1.86, 0)))
    parts.append(_box_surface(rng, n * 5 // 100, (0.78, 0.36, 0.65), (0, 2.09, 0)))
    parts.append(_cone_surface(rng, n * 9 // 100, 0.82, 0.06, 0.42, (0, 2.43, 0)))
    parts.append(_ring(rng, n * 7 // 100, 0.28, (0, 0.92, -0.62), "xy"))
    return _fit_count(parts, n, rng)


def _statue(rng: np.random.Generator, n: int) -> np.ndarray:
    parts: list[np.ndarray] = [
        _box_surface(rng, n * 10 // 100, (1.45, 0.72, 1.18), (0, -1.86, 0)),
        _box_surface(rng, n * 8 // 100, (1.08, 0.44, 0.88), (0, -1.28, 0)),
        _cone_surface(rng, n * 30 // 100, 0.72, 0.34, 2.10, (0, -0.12, 0)),
        _elliptic_cylinder(rng, n * 12 // 100, 0.35, 0.24, 0.76, (0, 1.12, 0)),
        _sphere(rng, n * 9 // 100, 0.29, (0, 1.70, 0)),
        _tube_between(rng, n * 8 // 100, (0.22, 1.30, 0), (0.74, 2.17, 0), 0.12),
        _tube_between(rng, n * 7 // 100, (-0.20, 1.30, 0), (-0.68, 0.82, -0.06), 0.12),
        _cone_surface(rng, n * 5 // 100, 0.15, 0.08, 0.42, (0.76, 2.48, 0)),
        _sphere(rng, n * 4 // 100, 0.17, (0.76, 2.76, 0)),
    ]
    crown = (0, 1.74, 0)
    for i in range(7):
        angle = -0.9 + i * 0.3
        end = (0.52 * math.sin(angle), 2.22 + 0.12 * math.cos(angle), 0.02)
        parts.append(_tube_between(rng, n // 100, crown, end, 0.025))
    return _fit_count(parts, n, rng)


def _big_ben(rng: np.random.Generator, n: int) -> np.ndarray:
    parts: list[np.ndarray] = [
        _box_surface(rng, n * 10 // 100, (1.55, 0.34, 1.28), (0, -2.02, 0)),
        _box_surface(rng, n * 34 // 100, (1.08, 2.92, 0.92), (0, -0.40, 0)),
        _box_surface(rng, n * 15 // 100, (1.34, 1.02, 1.14), (0, 1.48, 0)),
        _cone_surface(rng, n * 13 // 100, 0.82, 0.14, 1.12, (0, 2.51, 0)),
        _tube_between(rng, n * 3 // 100, (0, 2.98, 0), (0, 3.55, 0), 0.055),
    ]
    for z in (-0.59, 0.59):
        parts.append(_ring(rng, n * 7 // 100, 0.38, (0, 1.53, z), "xy", 0.035))
    for x in (-0.43, 0.43):
        parts.append(_tube_between(rng, n * 2 // 100, (x, -1.75, -0.48), (x, 1.95, -0.48), 0.035))
    parts.append(_tube_between(rng, n * 2 // 100, (0, 1.53, -0.64), (0, 1.82, -0.65), 0.025))
    parts.append(_tube_between(rng, n * 2 // 100, (0, 1.53, -0.64), (0.25, 1.41, -0.65), 0.025))
    return _fit_count(parts, n, rng)


def _colosseum(rng: np.random.Generator, n: int) -> np.ndarray:
    parts: list[np.ndarray] = [
        _elliptic_cylinder(rng, n * 45 // 100, 2.15, 1.22, 1.75, (0, -0.34, 0), False),
        _elliptic_cylinder(rng, n * 18 // 100, 1.68, 0.86, 1.67, (0, -0.32, 0), False),
        _ring(rng, n * 5 // 100, 2.12, (0, -1.24, 0), "xz", 0.06),
        _ring(rng, n * 5 // 100, 2.12, (0, 0.57, 0), "xz", 0.06),
    ]
    # Three arcades represented by columns and arch rings on the front ellipse.
    for level in (-0.89, -0.28, 0.33):
        for x in np.linspace(-1.78, 1.78, 13):
            z = -1.22 * math.sqrt(max(0.0, 1.0 - (x / 2.15) ** 2))
            parts.append(_tube_between(rng, max(8, n // 850), (x, level - 0.27, z), (x, level + 0.22, z), 0.035))
        for x in np.linspace(-1.63, 1.63, 12):
            z = -1.23 * math.sqrt(max(0.0, 1.0 - (x / 2.15) ** 2))
            arc = _ring(rng, max(8, n // 550), 0.16, (x, level + 0.15, z - 0.01), "xy", 0.022)
            arc = arc[arc[:, 1] >= level + 0.14]
            parts.append(arc)
    # A deliberately uneven top edge gives the ruin its recognizable silhouette.
    rubble_n = n * 12 // 100
    rubble = _elliptic_cylinder(rng, rubble_n, 2.13, 1.20, 0.46, (0, 0.78, 0), False)
    angle = np.arctan2(rubble[:, 2] / 1.20, rubble[:, 0] / 2.13)
    keep = rubble[:, 1] < 0.88 + 0.26 * np.sin(angle * 2.0 + 0.7)
    parts.append(rubble[keep])
    return _fit_count(parts, n, rng)


def make_shape(index: int, count: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed + index * 997)
    generators = (_temple_gate, _statue, _big_ben, _colosseum)
    p = generators[index % len(generators)](rng, count)
    lo, hi = np.percentile(p, (0.4, 99.6), axis=0)
    p -= (lo + hi) / 2.0
    height = max(float((hi - lo)[1]), 1e-5)
    # The model spins around Y, so the horizontal room a monument needs is its
    # rotational radius, not its width.  Fitting on height alone scaled the
    # wide, low Colosseum until it overflowed the viewport on every side.
    radius = max(float(np.percentile(np.hypot(p[:, 0], p[:, 2]), 99.6)), 1e-5)
    p *= min(4.65 / height, 2.50 / radius)
    return np.ascontiguousarray(p, dtype="f4")


def cloud_points(count: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(count, 3)).astype("f4")
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    radius = np.cbrt(rng.random((count, 1))).astype("f4") * 2.25
    return np.ascontiguousarray(v * radius, dtype="f4")


# ---------------------------------------------------------------------------
# Gesture recognition


@dataclass
class HandState:
    gesture: str = "NO HAND"
    center: tuple[float, float] = (0.5, 0.52)
    size: float = 0.16
    snapped: bool = False
    landmarks: object | None = None


class GestureDetector:
    def __init__(self, mp_module, model_path: Path) -> None:
        self.mp = mp_module
        options = mp_module.tasks.vision.HandLandmarkerOptions(
            base_options=mp_module.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_module.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.58,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.hands = mp_module.tasks.vision.HandLandmarker.create_from_options(options)
        self.was_pinched = False
        self.pinch_started = 0.0
        self.cooldown_until = 0.0
        self.last_gesture = "NO HAND"
        self.stable_frames = 0
        self._last_timestamp_ms = -1

    @staticmethod
    def _distance(a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def process(self, rgb_frame: np.ndarray, now: float) -> HandState:
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        # VIDEO mode rejects a timestamp that does not advance, and two frames
        # can easily land inside the same millisecond.
        timestamp_ms = max(int(now * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self.hands.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            self.was_pinched = False
            return HandState()

        lm = result.hand_landmarks[0]
        palm_ids = (0, 5, 9, 13, 17)
        cx = sum(lm[i].x for i in palm_ids) / len(palm_ids)
        cy = sum(lm[i].y for i in palm_ids) / len(palm_ids)
        palm_size = max(self._distance(lm[5], lm[17]), 0.035)

        # Rotation-independent openness: a fingertip is extended when it is
        # substantially farther from the wrist than its PIP joint.
        extended = 0
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            if self._distance(lm[tip], lm[0]) > self._distance(lm[pip], lm[0]) + palm_size * 0.18:
                extended += 1
        thumb_open = self._distance(lm[4], lm[9]) > self._distance(lm[3], lm[9]) + palm_size * 0.08
        extended += int(thumb_open)

        raw = "OPEN" if extended >= 4 else "FIST" if extended <= 1 else "TRACKING"
        if raw == self.last_gesture:
            self.stable_frames += 1
        else:
            self.last_gesture, self.stable_frames = raw, 0
        gesture = raw if self.stable_frames >= 2 else "TRACKING"

        # Optical snap: thumb and middle finger touch, then separate rapidly.
        pinch = self._distance(lm[4], lm[12]) / palm_size
        snapped = False
        if pinch < 0.38 and not self.was_pinched:
            self.was_pinched = True
            self.pinch_started = now
        elif self.was_pinched and pinch > 0.72:
            if now - self.pinch_started < 0.48 and now >= self.cooldown_until:
                snapped = True
                self.cooldown_until = now + 0.75
            self.was_pinched = False
        elif self.was_pinched and now - self.pinch_started > 0.8:
            self.was_pinched = False

        return HandState(gesture, (cx, cy), palm_size, snapped, lm)

    def close(self) -> None:
        self.hands.close()


# ---------------------------------------------------------------------------
# ModernGL renderer


QUAD_VERTEX_SHADER = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 uv;
void main() { uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

QUAD_FRAGMENT_SHADER = """
#version 330
uniform sampler2D camera_tex;
in vec2 uv;
out vec4 frag;
void main() {
    vec3 c = texture(camera_tex, uv).rgb;
    c = pow(c, vec3(0.92)) * vec3(0.72, 0.78, 0.88);
    float vignette = smoothstep(1.15, 0.22, length(uv - 0.5));
    frag = vec4(c * (0.56 + 0.44 * vignette), 1.0);
}
"""

PARTICLE_VERTEX_SHADER = """
#version 330
in vec3 pos_from;
in vec3 pos_to;
uniform mat4 mvp;
uniform float morph;
uniform float spread;
uniform float time;
uniform float point_size;
uniform float trail_offset;
out float energy;

float hash(float n) { return fract(sin(n * 17.17) * 43758.5453); }

void main() {
    float id = float(gl_VertexID);
    float k = smoothstep(0.0, 1.0, clamp(morph - trail_offset * (0.5 + hash(id)), 0.0, 1.0));
    vec3 p = mix(pos_from, pos_to, k);
    vec3 dir = normalize(p + vec3(0.001));
    float wave = sin(time * 1.7 + id * 0.019) * 0.06;
    p += dir * spread * (0.35 + 1.35 * hash(id * 1.31));
    p += dir * wave * (0.2 + spread);
    vec4 clip = mvp * vec4(p, 1.0);
    gl_Position = clip;
    // clip.w is the eye-space depth; clip.z is already the mapped depth, and
    // using it pinned every point to the upper clamp regardless of distance.
    gl_PointSize = point_size * clamp(7.0 / max(0.001, clip.w), 0.65, 1.45);
    energy = 0.65 + 0.35 * hash(id * 0.71);
}
"""

PARTICLE_FRAGMENT_SHADER = """
#version 330
uniform vec3 color;
uniform float alpha;
in float energy;
out vec4 frag;
void main() {
    vec2 q = gl_PointCoord - vec2(0.5);
    float d = length(q) * 2.0;
    if (d > 1.0) discard;
    float core = pow(1.0 - d, 1.7);
    frag = vec4(color * (0.55 + 1.55 * core) * energy, alpha * (1.0 - d));
}
"""

LINE_VERTEX_SHADER = """
#version 330
in vec3 in_pos;
uniform mat4 mvp;
void main() { gl_Position = mvp * vec4(in_pos, 1.0); }
"""

LINE_FRAGMENT_SHADER = """
#version 330
uniform vec3 color;
uniform float alpha;
out vec4 frag;
void main() { frag = vec4(color, alpha); }
"""


def _perspective(fovy: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype="f4")
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _model_matrix(x: float, y: float, z: float, angle: float, scale: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    r = np.array(
        ((c, 0, s, 0), (0, 1, 0, 0), (-s, 0, c, 0), (0, 0, 0, 1)), dtype="f4"
    )
    r[:3, :3] *= scale
    r[:3, 3] = (x, y, z)
    return r


def _wire_sphere(radius: float = 2.48, segments: int = 96) -> np.ndarray:
    lines: list[tuple[float, float, float]] = []

    def add_loop(points: list[tuple[float, float, float]]) -> None:
        for i in range(len(points)):
            lines.extend((points[i], points[(i + 1) % len(points)]))

    for lat in (-60, -30, 0, 30, 60):
        phi = math.radians(lat)
        rr, y = radius * math.cos(phi), radius * math.sin(phi)
        add_loop([(rr * math.cos(t), y, rr * math.sin(t)) for t in np.linspace(0, math.tau, segments, endpoint=False)])
    for lon in range(0, 180, 30):
        theta = math.radians(lon)
        add_loop(
            [
                (radius * math.cos(p) * math.cos(theta), radius * math.sin(p), radius * math.cos(p) * math.sin(theta))
                for p in np.linspace(0, math.tau, segments, endpoint=False)
            ]
        )
    return np.asarray(lines, dtype="f4")


class Renderer:
    def __init__(self, ctx, width: int, height: int, count: int) -> None:
        import moderngl

        self.ctx, self.width, self.height, self.count = ctx, width, height, count
        self.gl = moderngl
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        self.quad_program = ctx.program(vertex_shader=QUAD_VERTEX_SHADER, fragment_shader=QUAD_FRAGMENT_SHADER)
        quad = np.array(
            [(-1, -1, 0, 1), (1, -1, 1, 1), (-1, 1, 0, 0), (1, 1, 1, 0)], dtype="f4"
        )
        self.quad_vbo = ctx.buffer(quad.tobytes())
        self.quad_vao = ctx.vertex_array(self.quad_program, [(self.quad_vbo, "2f 2f", "in_pos", "in_uv")])
        self.camera_tex = ctx.texture((width, height), 3, dtype="f1")
        self.camera_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.quad_program["camera_tex"].value = 0

        self.particle_program = ctx.program(
            vertex_shader=PARTICLE_VERTEX_SHADER, fragment_shader=PARTICLE_FRAGMENT_SHADER
        )
        initial = cloud_points(count)
        target = make_shape(0, count)
        self.from_vbo = ctx.buffer(initial.tobytes(), dynamic=True)
        self.to_vbo = ctx.buffer(target.tobytes(), dynamic=True)
        self.particle_vao = ctx.vertex_array(
            self.particle_program,
            [(self.from_vbo, "3f", "pos_from"), (self.to_vbo, "3f", "pos_to")],
        )
        self.from_points, self.to_points = initial, target
        self.shape_index = 0
        # perf_counter counts from boot, and a uniform is float32: after a day
        # of uptime its resolution is coarser than a frame and the shimmer
        # animation stutters.  Everything on the GPU uses time since startup.
        self.epoch = time.perf_counter()
        self.morph_started = self.epoch
        self.morph_duration = 1.55

        self.line_program = ctx.program(vertex_shader=LINE_VERTEX_SHADER, fragment_shader=LINE_FRAGMENT_SHADER)
        sphere = _wire_sphere()
        self.sphere_vbo = ctx.buffer(sphere.tobytes())
        self.sphere_vao = ctx.vertex_array(self.line_program, [(self.sphere_vbo, "3f", "in_pos")])

    def resize(self, width: int, height: int) -> None:
        if (width, height) == (self.width, self.height):
            return
        self.width, self.height = width, height
        self.camera_tex.release()
        self.camera_tex = self.ctx.texture((width, height), 3, dtype="f1")
        self.camera_tex.filter = (self.gl.LINEAR, self.gl.LINEAR)

    def select_shape(self, index: int, now: float) -> None:
        index %= len(SHAPES)
        elapsed = (now - self.morph_started) / self.morph_duration
        k = float(np.clip(elapsed, 0.0, 1.0))
        k = k * k * (3.0 - 2.0 * k)
        current = self.from_points * (1.0 - k) + self.to_points * k
        self.from_points = np.ascontiguousarray(current, dtype="f4")
        self.to_points = make_shape(index, self.count)
        self.from_vbo.write(self.from_points.tobytes())
        self.to_vbo.write(self.to_points.tobytes())
        self.shape_index = index
        self.morph_started = now

    def render(
        self,
        frame_rgb: np.ndarray,
        now: float,
        anchor: tuple[float, float],
        hand_size: float,
        spread: float,
    ) -> None:
        import cv2

        if frame_rgb.shape[1::-1] != (self.width, self.height):
            frame_rgb = cv2.resize(frame_rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.camera_tex.write(np.ascontiguousarray(frame_rgb).tobytes())
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(self.gl.DEPTH_TEST)
        self.ctx.disable(self.gl.BLEND)
        self.camera_tex.use(0)
        self.quad_vao.render(self.gl.TRIANGLE_STRIP)

        x = (anchor[0] - 0.5) * 3.8
        y = (0.5 - anchor[1]) * 2.15 + 0.10
        # Palm size contributes gently so noisy depth estimates do not jump.
        object_scale = float(np.clip(0.78 + hand_size * 1.45, 0.78, 1.08))
        elapsed_total = now - self.epoch
        model = _model_matrix(x, y, -6.5, elapsed_total * 0.22, object_scale)
        proj = _perspective(math.radians(48.0), self.width / max(1, self.height), 0.1, 100.0)
        mvp = np.ascontiguousarray((proj @ model).T, dtype="f4")

        color = COLORS[self.shape_index]
        self.ctx.enable(self.gl.BLEND)
        self.ctx.blend_func = self.gl.SRC_ALPHA, self.gl.ONE

        self.line_program["mvp"].write(mvp.tobytes())
        self.line_program["color"].value = tuple(c * 0.64 for c in color)
        self.line_program["alpha"].value = 0.13
        self.sphere_vao.render(self.gl.LINES)

        elapsed = (now - self.morph_started) / self.morph_duration
        morph = float(np.clip(elapsed, 0.0, 1.0))
        p = self.particle_program
        p["mvp"].write(mvp.tobytes())
        p["morph"].value = morph
        p["spread"].value = spread
        p["time"].value = elapsed_total
        p["color"].value = color

        gain = float(np.clip(PARTICLE_GAIN_REFERENCE / self.count, 0.06, 1.0))

        # Trails are earlier snapshots of the GPU interpolation; strongest only
        # during a transition, almost free when the monument is settled.
        if morph < 1.0:
            for offset, alpha, size in ((0.19, 0.025, 6.0), (0.12, 0.04, 5.0), (0.06, 0.07, 4.0)):
                p["trail_offset"].value = offset
                p["point_size"].value = size
                p["alpha"].value = alpha * gain
                self.particle_vao.render(self.gl.POINTS)

        p["trail_offset"].value = 0.0
        p["point_size"].value = 5.2
        p["alpha"].value = 0.18 * gain
        self.particle_vao.render(self.gl.POINTS)
        p["point_size"].value = 1.7
        p["alpha"].value = 0.88 * gain
        self.particle_vao.render(self.gl.POINTS)


# ---------------------------------------------------------------------------
# Application


def _draw_hud(
    cv2,
    frame: np.ndarray,
    state: HandState,
    shape_index: int,
    fps: float,
    source: str = "",
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (24, 22), (356, 150), (7, 10, 18), -1)
    cv2.rectangle(frame, (24, 22), (30, 150), tuple(int(c * 255) for c in COLORS[shape_index][::-1]), -1)
    cv2.putText(frame, "WONDER // SNAP", (48, 53), cv2.FONT_HERSHEY_DUPLEX, 0.72, (240, 245, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, SHAPES[shape_index], (48, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (190, 205, 225), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{state.gesture}   {fps:4.0f} FPS", (48, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (125, 240, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, source[:34], (48, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 165, 190), 1, cv2.LINE_AA)
    cv2.putText(frame, "OPEN: scatter   FIST: assemble   SNAP: next", (28, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 225, 236), 1, cv2.LINE_AA)


def _draw_hand(cv2, frame: np.ndarray, landmarks) -> None:
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (70, 225, 255), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        radius = 5 if index in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, point, radius + 2, (15, 35, 45), -1, cv2.LINE_AA)
        cv2.circle(frame, point, radius, (120, 255, 230), -1, cv2.LINE_AA)


def _synthetic_frame(width: int, height: int, t: float) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    glow = np.exp(-(((x / width - 0.5) ** 2) + ((y / height - 0.45) ** 2)) * 5.0)
    frame = np.zeros((height, width, 3), dtype="u1")
    frame[..., 0] = np.clip(7 + glow * 13, 0, 255)
    frame[..., 1] = np.clip(8 + glow * 11, 0, 255)
    frame[..., 2] = np.clip(13 + glow * (19 + 3 * math.sin(t)), 0, 255)
    return frame


def _ensure_hand_model(requested: str | None) -> Path:
    path = Path(requested) if requested else Path(__file__).resolve().parent / "assets" / "hand_landmarker.task"
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the official MediaPipe hand-landmarker model (7.5 MB)...")
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(HAND_MODEL_URL, timeout=45, context=context) as source, path.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the MediaPipe hand model. Download hand_landmarker.task "
            f"from {HAND_MODEL_URL} and pass it with --hand-model."
        ) from None
    return path


def _rtsp_password(args: argparse.Namespace) -> str:
    """Resolve the camera password without ever returning it to a print site."""
    password = args.rtsp_password or os.environ.get(RTSP_PASSWORD_ENV) or DEFAULT_RTSP_PASSWORD
    if password:
        return password
    try:
        return getpass.getpass(f"RTSP password for {args.rtsp_user}@{args.rtsp_host}: ")
    except (EOFError, KeyboardInterrupt):
        # No console attached (PyCharm Run, a double-clicked shortcut): an empty
        # password is still worth trying, cameras without auth accept it.
        return ""


def _rtsp_url(args: argparse.Namespace) -> str | None:
    """Build the stream URL.  Callers must never print the result."""
    if args.webcam or not args.rtsp_host:
        return None
    userinfo = quote(args.rtsp_user, safe="")
    password = _rtsp_password(args)
    if password:
        # safe="" percent-encodes '$', ':' and '@', so passwords with URL
        # metacharacters survive the round trip into FFmpeg.
        userinfo += ":" + quote(password, safe="")
    path = args.rtsp_path if args.rtsp_path.startswith("/") else "/" + args.rtsp_path
    return f"rtsp://{userinfo}@{args.rtsp_host}:{args.rtsp_port}{path}"


class CameraStream:
    """Newest-frame camera reader.

    OpenCV's FFmpeg backend queues decoded RTSP frames, so a render loop slower
    than the camera falls further behind every second (CAP_PROP_BUFFERSIZE is
    ignored by that backend).  Reading on a dedicated thread and keeping only
    the most recent frame bounds the delay to a single frame.  Downscaling and
    mirroring also happen here, so the render loop, the HUD overlay and
    MediaPipe all work on window-sized images instead of the 4 MP main stream.
    """

    def __init__(self, cv2_module, args: argparse.Namespace, size: tuple[int, int]) -> None:
        self.cv2 = cv2_module
        self.args = args
        self._size = size
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._first_frame = threading.Event()
        self.label = "no camera"
        self._sources = self._plan_sources()
        self._thread = threading.Thread(target=self._run, name="wondersnap-camera", daemon=True)
        self._thread.start()

    def _plan_sources(self) -> list[tuple[str, object, tuple[str, int] | None]]:
        """RTSP first, then the local webcam, so one bad camera is not fatal."""
        sources: list[tuple[str, object, tuple[str, int] | None]] = []
        url = _rtsp_url(self.args)
        if url:
            host, port = self.args.rtsp_host, self.args.rtsp_port
            sources.append((f"RTSP {host}:{port}", url, (host, port)))
        if url is None or not self.args.no_webcam_fallback:
            sources.append((f"webcam {self.args.camera}", self.args.camera, None))
        return sources

    @staticmethod
    def _reachable(address: tuple[str, int], timeout: float = 1.5) -> bool:
        try:
            with socket.create_connection(address, timeout):
                return True
        except OSError:
            return False

    def _open(self, target, probe: tuple[str, int] | None) -> object | None:
        cv2 = self.cv2
        if isinstance(target, str):
            # OpenCV hardcodes a 30 s FFmpeg open timeout that no capture option
            # overrides, so an offline camera would stall every retry.  A plain
            # TCP probe answers in milliseconds and costs nothing when it is up.
            if probe is not None and not self._reachable(probe):
                return None
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", RTSP_FFMPEG_OPTIONS)
            cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
        else:
            # DirectShow negotiates resolution on Windows far more reliably than
            # the MSMF default, but it is absent on some driver stacks.
            cap = cv2.VideoCapture(target, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(target)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _publish(self, frame: np.ndarray) -> None:
        cv2 = self.cv2
        with self._lock:
            width, height = self._size
        owned = False
        if frame.shape[1::-1] != (width, height):
            interp = cv2.INTER_AREA if frame.shape[1] > width else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (width, height), interpolation=interp)
            owned = True
        if not self.args.no_mirror:
            frame = cv2.flip(frame, 1)
            owned = True
        if not owned:
            # VideoCapture.read() reuses one internal buffer, so a frame that
            # needed neither step would mutate under the consumer.
            frame = frame.copy()
        with self._lock:
            self._frame = np.ascontiguousarray(frame)
        self._first_frame.set()

    def _run(self) -> None:
        cap = None
        index, failures, retry_delay = 0, 0, 1.0
        try:
            while not self._stop.is_set():
                if cap is None:
                    if not self._sources:
                        self.label = "synthetic background"
                        return
                    name, target, probe = self._sources[index % len(self._sources)]
                    cap = self._open(target, probe)
                    if cap is None:
                        index += 1
                        if index >= len(self._sources):
                            # Every source failed once; keep retrying the list
                            # instead of giving up, cameras do come back.
                            self.label = "reconnecting..."
                            self._stop.wait(retry_delay)
                            retry_delay = min(retry_delay * 2.0, 5.0)
                        continue
                    self.label, failures, retry_delay = name, 0, 1.0
                    print(f"Connected to {name}.", flush=True)

                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    failures += 1
                    # A capture that fails instantly would spin the CPU, so pace
                    # the retries; 45 of them is well under a second either way.
                    self._stop.wait(0.01)
                    # A handful of dropped packets is normal on RTSP; a long run
                    # of them means the stream is gone and needs a fresh session.
                    if failures >= 45:
                        cap.release()
                        cap, failures = None, 0
                        self.label = "reconnecting..."
                        print("Camera stream lost; reconnecting...", flush=True)
                    continue
                failures = 0
                self._publish(frame)
        finally:
            if cap is not None:
                cap.release()

    def wait_ready(self, timeout: float) -> bool:
        return self._first_frame.wait(timeout)

    def resize_output(self, width: int, height: int) -> None:
        with self._lock:
            self._size = (width, height)

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            frame = self._frame
        return (frame is not None), frame

    def close(self) -> None:
        self._stop.set()
        # A read blocked on a dead socket unblocks after stimeout; the thread is
        # a daemon, so a slow one can never hold up interpreter shutdown.
        self._thread.join(timeout=2.0)


def run(args: argparse.Namespace) -> int:
    try:
        import cv2
        import mediapipe as mp
        import moderngl
        import pygame
    except ImportError as exc:
        print(f"Missing dependency: {exc.name}")
        print("Run: python -m pip install -r requirements.txt")
        return 2

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    try:
        pygame.display.set_mode((args.width, args.height), flags, vsync=1)
    except pygame.error:
        # Some drivers reject a vsync-locked GL surface outright.
        pygame.display.set_mode((args.width, args.height), flags)
    pygame.display.set_caption("WonderSnap — Gesture Particle Monuments")

    ctx = moderngl.create_context(require=330)
    renderer = Renderer(ctx, args.width, args.height, args.particles)
    try:
        hand_model = _ensure_hand_model(args.hand_model)
    except RuntimeError as exc:
        print(exc)
        pygame.quit()
        return 3
    detector = GestureDetector(mp, hand_model)
    cap = CameraStream(cv2, args, (renderer.width, renderer.height))
    if not cap.wait_ready(8.0):
        print("No camera frame yet; starting in keyboard mode with a synthetic background.")
        print("For RTSP, verify the username/password and --rtsp-path.")

    clock = pygame.time.Clock()
    running, show_hand = True, True
    hand_state = HandState()
    target_spread, spread = 0.0, 0.0
    smooth_anchor = np.array((0.60, 0.50), dtype="f4")
    smooth_size = 0.16
    fps_smoothed = 60.0

    try:
        while running:
            now = time.perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    renderer.resize(max(320, event.w), max(180, event.h))
                    cap.resize_output(renderer.width, renderer.height)
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        renderer.select_shape(renderer.shape_index + 1, now)
                    elif pygame.K_1 <= event.key <= pygame.K_4:
                        renderer.select_shape(event.key - pygame.K_1, now)
                    elif event.key == pygame.K_o:
                        target_spread = 1.0
                    elif event.key == pygame.K_f:
                        target_spread = 0.0
                    elif event.key == pygame.K_h:
                        show_hand = not show_hand

            ok, frame = cap.read()
            if ok:
                # The reader hands over a private copy each frame, but the HUD
                # and the landmark overlay draw in place, so never share it.
                frame = frame.copy()
            else:
                frame = _synthetic_frame(renderer.width, renderer.height, now)

            if ok:
                # Landmark inference gains no useful accuracy above ~640 px, so
                # keep the normalized coordinates while cutting CPU load.
                track_width = min(args.tracking_width, frame.shape[1])
                track_height = max(1, round(frame.shape[0] * track_width / frame.shape[1]))
                tracking_frame = cv2.resize(frame, (track_width, track_height), interpolation=cv2.INTER_AREA)
                rgb_for_mp = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2RGB)
                hand_state = detector.process(rgb_for_mp, now)
            else:
                # Nothing to track on the synthetic background; skip inference.
                hand_state = HandState("KEYBOARD")

            if hand_state.landmarks is not None:
                smooth_anchor += (np.asarray(hand_state.center) - smooth_anchor) * 0.18
                smooth_size += (hand_state.size - smooth_size) * 0.14
                if hand_state.gesture == "OPEN":
                    target_spread = 1.0
                elif hand_state.gesture == "FIST":
                    target_spread = 0.0
                if hand_state.snapped:
                    renderer.select_shape(renderer.shape_index + 1, now)
                if show_hand:
                    _draw_hand(cv2, frame, hand_state.landmarks)

            spread += (target_spread - spread) * 0.075
            fps_now = clock.get_fps() or fps_smoothed
            fps_smoothed += (fps_now - fps_smoothed) * 0.06
            _draw_hud(cv2, frame, hand_state, renderer.shape_index, fps_smoothed, cap.label)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            renderer.render(frame_rgb, now, tuple(smooth_anchor), smooth_size, spread)
            pygame.display.flip()
            clock.tick(args.fps)
    finally:
        cap.close()
        detector.close()
        pygame.quit()
    return 0


def self_test(count: int) -> int:
    assert count >= 1000
    for i, name in enumerate(SHAPES):
        points = make_shape(i, count)
        assert points.shape == (count, 3), (name, points.shape)
        assert points.dtype == np.float32
        assert np.isfinite(points).all()
        height = float(np.ptp(points[:, 1]))
        radius = float(np.hypot(points[:, 0], points[:, 2]).max())
        # Every monument has to sit inside the reference globe, whatever its
        # aspect ratio, so check both extents rather than height alone.
        assert 2.5 < height < 5.4, (name, height)
        assert radius < 2.85, (name, radius)
        print(f"OK  {name:20s} {len(points):>8,} points  height={height:.2f}  radius={radius:.2f}")
    print("All procedural geometry checks passed.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--particles", type=int, default=250_000)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--webcam", action="store_true", help="use a local webcam instead of RTSP")
    parser.add_argument("--rtsp-host", default="1*")
    parser.add_argument("--rtsp-port", type=int, default=554)
    parser.add_argument("--rtsp-user", default="admin")
    parser.add_argument(
        "--rtsp-password",
        default=None,
        help=f"overrides the built-in password; {RTSP_PASSWORD_ENV} keeps it out of shell history",
    )
    parser.add_argument("--rtsp-path", default=RTSP_MAIN_PATH)
    parser.add_argument(
        "--substream",
        action="store_true",
        help="use the Dahua sub stream (much lower CPU than the 4 MP main stream)",
    )
    parser.add_argument(
        "--no-webcam-fallback",
        action="store_true",
        help="fail to the synthetic background instead of trying a local webcam",
    )
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the camera image")
    parser.add_argument("--hand-model", default=None, help="path to hand_landmarker.task")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--tracking-width", type=int, default=640)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.particles < 1_000:
        parser.error("--particles must be at least 1000")
    if args.substream and args.rtsp_path == RTSP_MAIN_PATH:
        args.rtsp_path = RTSP_SUB_PATH
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    raise SystemExit(self_test(cli_args.particles) if cli_args.self_test else run(cli_args))
