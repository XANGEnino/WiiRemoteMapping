"""Live 3D orientation and swing-trace window for the Wii remote.

Pitch and roll are derived from the accelerometer, which can only sense
the direction of gravity:

* Pitch (IR end up/down) and roll (twist around the long axis) are real
  measurements whenever the remote is still or moving gently.
* Yaw (turning around the vertical axis) leaves the gravity vector
  unchanged and is invisible to the accelerometer.  When the remote has
  a MotionPlus gyro (built into -TR remotes, a dongle on older ones)
  wiimote.py activates it and streams angular rates; the rate about the
  vertical axis is integrated here into a heading, so the model turns
  when the remote does.  Gyro integration drifts slowly — clicking the
  model re-centers the heading.  Without a MotionPlus the heading simply
  stays fixed, as before.
* During a swing the sensor measures hand acceleration on top of gravity,
  so orientation updates are throttled while the reading is far from 1 g
  and catch up once the remote settles.
* Left/right (and up/down, forward/back) swings are translations, and
  those ARE visible: the linear acceleration left over after subtracting
  gravity is rotated into the world frame and integrated twice, so the
  model slides the way the hand moves.  A spring-damper pulls it back to
  center because accelerometer integration drifts within seconds — the
  slide is a short-lived visualization, not position tracking.

The strip chart plots the same number the swing detector works with: the
magnitude of the deviation from the tracked gravity baseline, in g.  Wii
games detect swings the same way — an acceleration spike — so a golf
backswing whose trace crosses the trigger line is exactly what makes the
game fire the stroke early.

Axis convention (WiiBrew), remote pointing at the screen, buttons up:
+X = left, +Y = toward the user, +Z = up.
"""

import math
import time
from collections import deque

import tkinter as tk
from tkinter import ttk

from motion import SwingDetector

# ---- 3D model of the remote (body frame, roughly to scale) ----

HX, HY, HZ = 0.19, 0.74, 0.15  # half width / length / thickness

_CORNERS = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
]

# (corner indices wound so the normal points outward, base fill color)
_FACES = [
    ((4, 5, 6, 7), "#dde3ea"),  # top (buttons)
    ((0, 3, 2, 1), "#9aa3ad"),  # bottom
    ((0, 1, 5, 4), "#2b2f33"),  # front — the dark IR window end
    ((3, 7, 6, 2), "#c6cdd6"),  # back
    ((1, 2, 6, 5), "#c6cdd6"),  # +X side
    ((0, 4, 7, 3), "#c6cdd6"),  # -X side
]

# Button details on the top face, front (IR) to back: D-pad, A,
# minus/home/plus, 1, 2.  Coordinates are (x, y) on the z=+HZ plane.
_A_BUTTON = (0.0, -0.18, 0.10)         # (x, y, radius)
_DPAD = (0.0, -0.47, 0.10, 0.035)      # (x, y, arm length, arm half-width)
_SMALL_BUTTONS = [                     # (x, y, radius)
    (-0.12, 0.10, 0.035), (0.0, 0.10, 0.035), (0.12, 0.10, 0.035),
    (0.0, 0.38, 0.05), (0.0, 0.54, 0.05),
]

CAM_TILT = math.radians(22)  # look slightly down at the remote
_COS_T, _SIN_T = math.cos(CAM_TILT), math.sin(CAM_TILT)
CAM_DIST = 6.0
VIEW_SCALE = 150.0
# Light heading toward the scene from the viewer's upper left.
_LIGHT = (-0.30, 0.85, -0.44)

# Sway: the model slides with the hand by double-integrating linear
# acceleration.  SWAY_ACCEL converts g to model units/s² (the model is
# ~4.2 units per meter of real remote); the spring-damper recenters it
# since accelerometer integration drifts, and the deadband keeps sensor
# noise from making the model shimmer at rest.
SWAY_ACCEL = 41.0
SWAY_SPRING = 30.0
SWAY_DAMP = 9.0
SWAY_DEADBAND = 0.07              # g
SWAY_CLAMP = (0.95, 0.60, 1.40)   # max |offset| along world x, y, z

MODEL_W, MODEL_H = 400, 290
CHART_W, CHART_H = 640, 185

TRACE_SECONDS = 6.0
TRACE_MAX_G = 3.0
FRAME_MS = 33

ACCENT = "#1976d2"
TRIGGER_COLOR = "#c62828"
GESTURE_COLOR = "#2e7d32"
MUTED = "#888888"

# Flip to -1.0 if the model turns opposite to the remote — the raw
# MotionPlus rate sign conventions differ between remote revisions.
YAW_SIGN = 1.0
# A gyro sample this old means the MotionPlus stopped (or never started).
GYRO_TIMEOUT = 0.5

YAW_CAPTION_GYRO = ("+ = turned left (MotionPlus gyro) — the heading "
                    "drifts a little over time; click the model to "
                    "re-center it")
YAW_CAPTION_NONE = ("turning left/right needs a MotionPlus gyro and "
                    "none was detected — but swinging left/right still "
                    "slides the model sideways")

GOLF_TIP = (
    "Golf tip: the game fires the stroke when it sees an acceleration "
    "spike — i.e. when a trace like the blue line above crosses a "
    "threshold like the red line. A jerky start of the backswing, or "
    "stopping the backswing abruptly, makes such a spike, and the game "
    "mistakes it for the forward swing: that is the “random” "
    "early stroke. Draw back slowly enough that the line stays low, "
    "pause at the top, then accelerate smoothly through the ball. The "
    "height of the spike is your swing strength."
)


def _shade(color, brightness):
    """Scale a #rrggbb color by a 0..1 brightness factor."""
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return "#%02x%02x%02x" % tuple(
        max(0, min(255, int(c * brightness))) for c in (r, g, b)
    )


class MotionViewer(tk.Toplevel):
    """3D orientation display plus a rolling swing-acceleration chart.

    Feed it via feed_accel()/mark_gesture(); it redraws itself ~30x/s.
    """

    def __init__(self, master, on_close=None):
        super().__init__(master)
        self.title("Wiimote Motion Viewer")
        self.resizable(False, False)
        self._on_close_cb = on_close
        self._closed = False
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._trace = deque()    # (monotonic time, deviation in g)
        self._gestures = []      # (monotonic time, swing name)
        self._grav = None        # smoothed gravity vector, g units
        self._pitch = 0.0
        self._roll = 0.0
        self._yaw = 0.0          # gyro-integrated heading; 0 without gyro
        self._deviation = 0.0
        self._last_sample = 0.0
        self._last_gyro = 0.0    # monotonic time of the last gyro sample
        self._had_gyro = False
        self._pos = [0.0, 0.0, 0.0]  # sway offset of the model, world frame
        self._vel = [0.0, 0.0, 0.0]

        self._build_ui()
        self._tick()

    # ---- data in ----

    def feed_accel(self, x, y, z, deviation):
        now = time.monotonic()
        dt = min(now - self._last_sample, 0.05) if self._last_sample else 0.0
        self._last_sample = now
        self._deviation = deviation
        self._trace.append((now, deviation))
        cutoff = now - TRACE_SECONDS
        while self._trace and self._trace[0][0] < cutoff:
            self._trace.popleft()

        # Track gravity for the orientation display.  A reading is only
        # trustworthy as "gravity" when its magnitude is near 1 g; the
        # further off it is (hand acceleration), the less it is let in.
        mag = math.sqrt(x * x + y * y + z * z)
        if self._grav is None:
            self._grav = [float(x), float(y), float(z)]
        else:
            err = abs(mag - 1.0)
            alpha = 0.15 if err < 0.10 else 0.04 if err < 0.35 else 0.008
            g = self._grav
            g[0] += alpha * (x - g[0])
            g[1] += alpha * (y - g[1])
            g[2] += alpha * (z - g[2])
        gx, gy, gz = self._grav
        self._pitch = math.atan2(-gy, math.hypot(gx, gz))
        self._roll = math.atan2(gx, gz)

        # Sway: whatever the reading holds beyond gravity is the hand's
        # own acceleration (sensor frame).  Rotate it into the world
        # frame and integrate against the spring so the model slides in
        # the direction the remote moves — this is what makes a left
        # swing look different from a right swing.
        lx, ly, lz = x - gx, y - gy, z - gz
        if math.sqrt(lx * lx + ly * ly + lz * lz) < SWAY_DEADBAND:
            lx = ly = lz = 0.0
        bx, by, bz, _ = self._body_axes()
        accel = (
            lx * bx[0] + ly * by[0] + lz * bz[0],
            lx * bx[1] + ly * by[1] + lz * bz[1],
            lx * bx[2] + ly * by[2] + lz * bz[2],
        )
        for i in range(3):
            self._vel[i] += (accel[i] * SWAY_ACCEL
                             - SWAY_SPRING * self._pos[i]
                             - SWAY_DAMP * self._vel[i]) * dt
            p = self._pos[i] + self._vel[i] * dt
            limit = SWAY_CLAMP[i]
            self._pos[i] = max(-limit, min(limit, p))

    def feed_gyro(self, yaw_dps, pitch_dps, roll_dps, dt):
        """Integrate MotionPlus angular rates into the heading.

        The rates are about the remote's own axes; the display needs the
        rotation about the world-vertical axis, which is the body-rate
        vector projected onto "up".  Up in the sensor frame is exactly
        where the accelerometer's gravity estimate points, so tilting
        the remote automatically hands the heading over to whichever
        gyro axis is vertical at that moment.
        """
        self._last_gyro = time.monotonic()
        g = self._grav
        mag = math.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) if g else 0.0
        if mag < 0.3:  # free fall or no data: no usable "up" reference
            ux, uy, uz = 0.0, 0.0, 1.0
        else:
            ux, uy, uz = g[0] / mag, g[1] / mag, g[2] / mag
        # Body axes: pitch is about +X, roll about +Y, yaw about +Z.
        rate = pitch_dps * ux + roll_dps * uy + yaw_dps * uz
        self._yaw += math.radians(YAW_SIGN * rate) * dt
        self._yaw = (self._yaw + math.pi) % (2 * math.pi) - math.pi

    def mark_gesture(self, name):
        self._gestures.append((time.monotonic(), name))

    # ---- UI ----

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both")

        upper = ttk.Frame(main)
        upper.pack(fill="x")

        self.model = tk.Canvas(
            upper, width=MODEL_W, height=MODEL_H, bg="#f4f6f8",
            highlightthickness=1, highlightbackground="#cccccc",
        )
        self.model.pack(side="left")
        self.model.bind("<Button-1>", self._recenter_yaw)

        panel = ttk.Frame(upper, padding=(14, 0, 0, 0))
        panel.pack(side="left", anchor="n")
        value_font = ("Segoe UI", 16, "bold")
        caption_font = ("Segoe UI", 9)

        self._pitch_var = tk.StringVar(value="--")
        self._roll_var = tk.StringVar(value="--")
        self._yaw_var = tk.StringVar(value="--")
        self._motion_var = tk.StringVar(value="--")
        rows = [
            ("Pitch", self._pitch_var, "+ = pointing up"),
            ("Roll", self._roll_var, "+ = twisted clockwise"),
            ("Yaw", self._yaw_var, YAW_CAPTION_NONE),
            ("Motion", self._motion_var,
             "swing acceleration, in g"),
        ]
        for row, (title, var, caption) in enumerate(rows):
            ttk.Label(panel, text=title, font=caption_font).grid(
                row=row * 2, column=0, sticky="w", pady=(8 if row else 0, 0))
            label = ttk.Label(panel, textvariable=var, font=value_font)
            label.grid(row=row * 2 + 1, column=0, sticky="w")
            cap = ttk.Label(panel, text=caption, font=caption_font,
                            foreground=MUTED, wraplength=170,
                            justify="left")
            cap.grid(row=row * 2 + 1, column=1, sticky="w", padx=(8, 0))
            if title == "Motion":
                self._motion_label = label
            elif title == "Yaw":
                self._yaw_caption = cap

        self.chart = tk.Canvas(
            main, width=CHART_W, height=CHART_H, bg="white",
            highlightthickness=1, highlightbackground="#cccccc",
        )
        self.chart.pack(pady=(10, 0))

        ttk.Label(
            main, text=GOLF_TIP, wraplength=CHART_W, justify="left",
            foreground="#444444", font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(8, 0))

    def _recenter_yaw(self, _event=None):
        self._yaw = 0.0

    def _tick(self):
        if self._closed:
            return
        now = time.monotonic()
        stale = now - self._last_sample > 0.5
        have_gyro = not stale and now - self._last_gyro < GYRO_TIMEOUT
        if stale:
            self._pitch_var.set("--")
            self._roll_var.set("--")
            self._motion_var.set("--")
            self._pos = [0.0, 0.0, 0.0]
            self._vel = [0.0, 0.0, 0.0]
            self._yaw = 0.0
        else:
            self._pitch_var.set("%+d°" % round(math.degrees(self._pitch)))
            self._roll_var.set("%+d°" % round(math.degrees(self._roll)))
            self._motion_var.set("%.2f g" % self._deviation)
            self._motion_label.config(
                foreground=TRIGGER_COLOR
                if self._deviation >= SwingDetector.START else "")
        if have_gyro:
            self._yaw_var.set("%+d°" % round(math.degrees(self._yaw)))
        else:
            self._yaw_var.set("--" if stale else "—")
        if have_gyro != self._had_gyro:
            self._had_gyro = have_gyro
            self._yaw_caption.config(
                text=YAW_CAPTION_GYRO if have_gyro else YAW_CAPTION_NONE)
        self._draw_model(stale)
        self._draw_chart(stale)
        self.after(FRAME_MS, self._tick)

    # ---- 3D drawing ----

    def _body_axes(self):
        """Unit vectors of the body axes in world coordinates.

        World frame: x = viewer's right, y = up, z = into the screen.
        At rest the remote points into the screen, buttons up.  Pitch
        and roll come from gravity; yaw is the gyro-integrated heading
        (zero without a MotionPlus) and swings everything around the
        world-vertical axis, positive turning the IR end to the left.
        """
        sp, cp = math.sin(self._pitch), math.cos(self._pitch)
        sr, cr = math.sin(self._roll), math.cos(self._roll)
        sy, cy = math.sin(self._yaw), math.cos(self._yaw)

        def spin(v):  # rotate about the world-vertical (y) axis
            x, y, z = v
            return (x * cy - z * sy, y, x * sy + z * cy)

        forward = spin((0.0, sp, cp))          # where the IR end points
        # Positive roll is clockwise as seen from behind the remote (the
        # camera's view), so the buttons-up axis tilts toward +x.
        up = spin((sr, cr * cp, -cr * sp))
        right = spin((cr, -sr * cp, sr * sp))
        bx = (-right[0], -right[1], -right[2])  # body +X points left
        by = (-forward[0], -forward[1], -forward[2])  # +Y toward the user
        return bx, by, up, forward

    @staticmethod
    def _to_camera(p):
        """World point -> (screen x offset, screen y offset, depth)."""
        x, y, z = p
        # Tilt the camera down: far points drift up toward the horizon
        # and the top of the remote comes into view.
        yc = y * _COS_T + z * _SIN_T
        zc = -y * _SIN_T + z * _COS_T
        k = VIEW_SCALE * CAM_DIST / (CAM_DIST + zc)
        return x * k, -yc * k, zc

    def _draw_model(self, stale):
        c = self.model
        c.delete("all")
        cx, cy = MODEL_W / 2, MODEL_H / 2 + 14

        bx, by, bz, forward = self._body_axes()
        ox, oy, oz = self._pos

        def world(px, py, pz):
            return (
                px * bx[0] + py * by[0] + pz * bz[0] + ox,
                px * bx[1] + py * by[1] + pz * bz[1] + oy,
                px * bx[2] + py * by[2] + pz * bz[2] + oz,
            )

        def screen(p):
            sx, sy, _ = self._to_camera(p)
            return cx + sx, cy + sy

        # Horizon reference behind everything.
        c.create_line(10, cy, MODEL_W - 10, cy, fill="#dddddd", dash=(4, 4))
        c.create_text(MODEL_W - 12, cy - 8, text="level", anchor="e",
                      fill="#bbbbbb", font=("Segoe UI", 8))

        # While the model is swept away from home by a swing, mark the
        # home position and tether the model to it so the direction of
        # the movement is easy to read.
        if math.sqrt(ox * ox + oy * oy + oz * oz) > 0.06:
            hx, hy = screen((0.0, 0.0, 0.0))
            mx, my = screen((ox, oy, oz))
            c.create_line(hx, hy, mx, my, fill="#c9d6e4", width=2)
            c.create_line(hx - 7, hy, hx + 7, hy, fill="#b6bec8")
            c.create_line(hx, hy - 7, hx, hy + 7, fill="#b6bec8")

        # Pointing ray: from the IR end, along where the remote points.
        # Pitch is always real; the sideways component is the gyro
        # heading, so it only moves when a MotionPlus is present.
        tip = world(0, -HY, 0)
        far = (tip[0] + forward[0] * 2.6,
               tip[1] + forward[1] * 2.6,
               tip[2] + forward[2] * 2.6)
        end = screen(far)
        c.create_line(*screen(tip), *end, fill="#90b8e0", dash=(5, 3))
        c.create_oval(end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3,
                      outline="#90b8e0")

        corners = [world(sx * HX, sy * HY, sz * HZ)
                   for sx, sy, sz in _CORNERS]
        cam = [self._to_camera(p) for p in corners]

        visible = []
        for idx, (face, color) in enumerate(_FACES):
            v0, v1, v2 = cam[face[0]], cam[face[1]], cam[face[2]]
            e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
            e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
            n = (e1[1] * e2[2] - e1[2] * e2[1],
                 e1[2] * e2[0] - e1[0] * e2[2],
                 e1[0] * e2[1] - e1[1] * e2[0])
            # Screen y grows downward, so flip it for lighting/culling.
            n = (n[0], -n[1], n[2])
            if n[2] >= 0:  # facing away from the camera
                continue
            depth = sum(cam[i][2] for i in face) / 4
            norm = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) or 1.0
            lit = (n[0] * _LIGHT[0] + n[1] * _LIGHT[1]
                   + n[2] * _LIGHT[2]) / norm
            brightness = 0.62 + 0.38 * max(0.0, lit)
            visible.append((depth, idx, face, _shade(color, brightness)))

        top_visible = False
        for depth, idx, face, fill in sorted(visible, reverse=True):
            pts = []
            for i in face:
                pts.extend((cx + cam[i][0], cy + cam[i][1]))
            c.create_polygon(*pts, fill=fill, outline=_shade(fill, 0.75))
            if idx == 0:
                top_visible = True

        if top_visible:
            self._draw_top_details(c, world, screen, stale)

        if stale:
            c.create_text(
                cx, 24, fill=MUTED, font=("Segoe UI", 10),
                text="waiting for motion data — is the remote "
                     "connected?",
            )

    def _draw_top_details(self, c, world, screen, stale):
        def k_at(p):
            _, _, zc = self._to_camera(p)
            return VIEW_SCALE * CAM_DIST / (CAM_DIST + zc)

        ax, ay, ar = _A_BUTTON
        p = world(ax, ay, HZ)
        sx, sy = screen(p)
        r = ar * k_at(p)
        fill = "#7fa3cc" if not stale else "#b9c4d2"
        c.create_oval(sx - r, sy - r, sx + r, sy + r,
                      fill=fill, outline=_shade(fill, 0.7))

        dx, dy, arm, half = _DPAD
        for w, h in ((arm, half), (half, arm)):
            pts = []
            for qx, qy in ((-w, -h), (w, -h), (w, h), (-w, h)):
                pts.extend(screen(world(dx + qx, dy + qy, HZ)))
            c.create_polygon(*pts, fill="#5a626b", outline="")

        for sx0, sy0, sr in _SMALL_BUTTONS:
            p = world(sx0, sy0, HZ)
            px, py = screen(p)
            r = sr * k_at(p)
            c.create_oval(px - r, py - r, px + r, py + r,
                          fill="#aab2bc", outline="#8b939c")

    # ---- chart drawing ----

    def _draw_chart(self, stale):
        c = self.chart
        c.delete("all")
        left, right, top, bottom = 34, 10, 12, 18
        plot_w = CHART_W - left - right
        plot_h = CHART_H - top - bottom
        now = time.monotonic()

        def px(ts):
            return left + plot_w * (1 - (now - ts) / TRACE_SECONDS)

        def py(g):
            return top + plot_h * (1 - min(g, TRACE_MAX_G) / TRACE_MAX_G)

        for g in range(int(TRACE_MAX_G) + 1):
            y = py(g)
            if g:
                c.create_line(left, y, CHART_W - right, y, fill="#eeeeee")
            c.create_text(left - 5, y, text="%d g" % g, anchor="e",
                          fill=MUTED, font=("Segoe UI", 8))
        c.create_line(left, py(0), CHART_W - right, py(0), fill="#cccccc")

        y = py(SwingDetector.START)
        c.create_line(left, y, CHART_W - right, y,
                      fill=TRIGGER_COLOR, dash=(6, 3))
        c.create_text(CHART_W - right - 4, y - 8, anchor="e",
                      text="swing trigger (%.1f g)" % SwingDetector.START,
                      fill=TRIGGER_COLOR, font=("Segoe UI", 8))

        pts = []
        for ts, dev in self._trace:
            pts.extend((px(ts), py(dev)))
        if len(pts) >= 4:
            c.create_line(*pts, fill=ACCENT, width=2)

        self._gestures = [(ts, name) for ts, name in self._gestures
                          if now - ts <= TRACE_SECONDS]
        for ts, name in self._gestures:
            x = px(ts)
            c.create_line(x, top, x, py(0), fill=GESTURE_COLOR, dash=(3, 3))
            c.create_text(x + 3, top + 6, text=name.replace("Swing", ""),
                          anchor="w", fill=GESTURE_COLOR,
                          font=("Segoe UI", 8, "bold"))

        peak = max((dev for _, dev in self._trace), default=0.0)
        c.create_text(
            left + 4, top + 6, anchor="w", font=("Segoe UI", 8),
            fill=TRIGGER_COLOR if peak >= SwingDetector.START else MUTED,
            text="peak %.2f g (last %.0f s)" % (peak, TRACE_SECONDS),
        )

        if stale and not self._trace:
            c.create_text(left + plot_w / 2, top + plot_h / 2,
                          text="no motion data yet", fill=MUTED,
                          font=("Segoe UI", 10))

    # ---- lifecycle ----

    def _close(self):
        self._closed = True
        if self._on_close_cb is not None:
            self._on_close_cb()
        self.destroy()
