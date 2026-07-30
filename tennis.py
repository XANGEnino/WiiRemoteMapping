"""Wii Sports-style tennis swing tester on a full court.

The old practice wall is gone: the scene is a whole tennis court with a
net, an opponent on the far side and the player figure near the camera.
Press A (Wiimote, or Space/'a' on the keyboard) and the opponent feeds
you a ball; the D-pad or the keyboard arrow keys choose whether it comes
to your forehand or backhand side, and the HUD banner on top shows where
the next ball flies.  Swing as the ball reaches the hit zone to return
it over the net.

Strokes are recognized from the player's own recorded swings: feed_accel()
runs the raw stream through motion.StrokeCapture, and each captured
window goes to classifier.StrokeClassifier (trained by trainer.py on
recordings/swings.jsonl).  Each stroke then flies with its own character
and carries its own particle color, so you can see at a glance what the
classifier made of your swing:

    normal    no particles  ordinary arc, ordinary bounce
    topspin   orange        peaks early and dives, kicks forward a
                            touch off the bounce
    backspin  dark blue     floats and falls late, sits down a touch
                            off the bounce
    smash     red           hit overhead, dead straight to the ground,
                            bounces high
    lob       green         skyball arc, high bounce
    dropshot  cyan          slow, just clears the net, dies off the
                            bounce

Without a trained model the game falls back to the old heuristic
(motion.TennisSwingDetector: up = lob, down = smash, else drive).

The court is a simple painter's-algorithm perspective scene: z runs 0
(player) to 1 (opponent baseline), x is lateral in [-1, 1], y is height
in the same units.  Everything beyond the net draws before the net and
everything nearer after it, so the ball passes visibly behind the tape.
"""

import math
import random
import time

import tkinter as tk
from tkinter import ttk

from classifier import StrokeClassifier
from motion import StrokeCapture, TennisSwingDetector

W, H = 640, 430       # design size; drawing scales to the live canvas
HORIZON = 56          # y toward which the ground converges
NEAR_GROUND = H - 24  # screen y of the ground at the player (z = 0)
PERSP = 1.15          # perspective strength; scale k = 1 / (1 + PERSP*z)
X_NEAR = 335          # half court width in pixels at z = 0
Y_SCALE = 175         # pixels per height unit at z = 0

NET_Z = 0.5           # net depth, halfway down the court
NET_H = 0.17          # net height, court units
NET_X = 1.06          # net sticks out a little past the sidelines

RACKET_Y = 0.35       # height an ordinary ball is struck at
HIT_Z = 0.35          # swings connect once the incoming ball is this close
PERFECT_Z = (0.05, 0.20)  # the sweet part of the window
MISS_T = 1.12         # feed progress past which the ball is gone

FEED_SPEED = 0.62     # court lengths per second of the opponent's feed
FEED_Y = 0.55         # height the feed leaves the opponent's racket
FEED_BOUNCE_T = 0.65  # feed progress at which it bounces (z = 0.35)
FEED_RISE = 0.9       # how lively the feed comes up off its bounce
FEED_X = 0.45         # lateral feed target: +x forehand, -x backhand
FEED_JITTER = 0.10    # scatter on the feed target

BASE_SPEED = 0.9      # court lengths per second of an unhurried shot
SPEED_GAIN = 0.25     # extra ball speed per g*s of hand speed above minimum
                      # (real swings run ~1-2 g*s; only a genuinely hard
                      # one should reach the cap)
SPEED_MAX = 0.5
SIDE_GAIN = 1.4       # fallback: lateral aim per unit of sideways swing
LOB_UZ = 0.45         # fallback: unit vertical component beyond which a
                      # stroke is a lob (upward) or a smash (downward)

AIM_GAIN = 0.95       # court x per unit of the classifier's aim scalar:
                      # a committed cross/straight lands near the sideline
                      # and can sail past it
AIM_JITTER = 0.08     # lateral scatter with a trained model...
KIN_JITTER = 0.18     # ...and with the heuristic fallback
X_MAX = 1.20          # farthest a shot's lateral aim may reach
WIDE_X = 1.0          # a shot landing past this |x| is out wide
LAND_GAIN = 0.12      # extra landing depth per g*s of pace above minimum
LAND_CAP = 0.97       # depth is capped at the baseline: a shot can go out
                      # wide (controllable) but never long (not)

# stroke -> flight & bounce character (see _launch):
#   hit_y: contact height; land: landing depth; arc: extra arc height;
#   skew: arc time-warp (<1 peaks early then dives - topspin; >1 floats
#   and falls late - backspin); mult: speed multiplier; land_max: cap on
#   the pace-extended landing depth; bounce_h: bounce height; bounce_k:
#   fraction of pace kept after the bounce; color: particle color.
STROKES = {
    "normal":   dict(hit_y=0.35, land=0.80, arc=0.18, skew=1.00, mult=1.00,
                     land_max=1.10, bounce_h=0.30, bounce_k=1.00,
                     color=None),
    "topspin":  dict(hit_y=0.35, land=0.74, arc=0.22, skew=0.62, mult=1.05,
                     land_max=0.92, bounce_h=0.32, bounce_k=1.12,
                     color="#fb8c00"),
    "backspin": dict(hit_y=0.35, land=0.78, arc=0.15, skew=1.45, mult=0.80,
                     land_max=1.06, bounce_h=0.20, bounce_k=0.85,
                     color="#1a3fa0"),
    "smash":    dict(hit_y=0.85, land=0.70, arc=0.00, skew=1.00, mult=1.60,
                     land_max=0.98, bounce_h=0.62, bounce_k=1.00,
                     color="#e53935"),
    "lob":      dict(hit_y=0.30, land=0.85, arc=0.85, skew=1.00, mult=0.55,
                     land_max=0.96, bounce_h=0.55, bounce_k=0.95,
                     color="#43a047"),
    "dropshot": dict(hit_y=0.35, land=0.58, arc=0.30, skew=1.00, mult=0.55,
                     land_max=0.62, bounce_h=0.15, bounce_k=0.80,
                     color="#00bcd4"),
}
# Touch shots need pace to clear the net: below these hand speeds (g*s)
# their arc flattens toward - and under - the tape.
DROP_RISK = (0.35, 0.12)   # (full-pace speed, max arc penalty)
SLICE_RISK = (0.30, 0.05)
SIDE_NAMES = {"forehand": "FH", "backhand": "BH"}

BOUNCE_BASE_S = 0.55  # bounce-leg seconds for an ordinary 0.30 bounce
BOUNCE_DZ_MAX = 0.55  # farthest a bounce may carry the ball, court units
NET_DROP_S = 0.30     # seconds a netted ball takes to drop
FRAME_MS = 20
VERDICT_S = 1.6       # how long hit feedback stays on screen

TRAIL_PER_FRAME = 2   # particles shed by a colored ball each frame
BURST_N = 10          # particles thrown on bounce / net impact
PARTICLE_LIFE = (0.25, 0.45)

# ---- player figure & mirrored racket ----
# The racket is driven straight from the accel stream the way the Motion
# Viewer drives its 3D model: roll/pitch come from the tracked gravity
# vector, and the leftover linear acceleration is double-integrated
# against a spring-damper so the hand sways the way the remote moves.

PLAYER_Z = 0.02       # depth the player figure stands at
PLAYER_DX = -0.14     # body stands just left of the ball's line
HAND_X = 0.10         # resting hand offset from the body, toward the ball
HAND_Y = 0.34         # resting hand height, court units (= strike height)
RACKET_TILT = 0.45    # resting racket tilt away from vertical, rad
ROLL_CLAMP = 1.2      # how much remote roll the racket tilt may take, rad
PITCH_LIFT = 0.16     # hand raise for a fully pitched-up remote, court units
SWAY_ACCEL = 3.3      # court units/s^2 per g of linear hand acceleration
SWAY_SPRING = 30.0    # recenters the sway (accel integration drifts)
SWAY_DAMP = 9.0
SWAY_DEADBAND = 0.05  # g; keeps sensor noise from jittering the racket
SWAY_CLAMP = 0.30     # max |sway|, court units
SWING_ANIM_S = 0.30   # duration of the stroke animation
SWING_REACH = 0.30    # how far the animated stroke carries the hand
RING_START_Z = 0.85   # incoming-ball depth at which the timing ring appears

OPP_Z = 0.93          # depth the opponent figure stands at
OPP_ANIM_S = 0.30     # duration of the opponent's feed swing

SKY = "#dce8f2"
FENCE = "#2f5e3c"
COURT = "#4d8f5c"
COURT_OUT = "#3f7a4d"
COURT_ZONE = "#5fa36e"
COURT_LINE = "#e8f0e8"
NET_MESH = "#39404a"
NET_BAND = "#f4f6f8"
NET_POST = "#2a2e33"
HUD_BG = "#1f2933"
BALL = "#f2e34c"
BALL_EDGE = "#b3a52e"
SHADOW = "#3a6d47"
GOOD = "#2e7d32"
WARN = "#e07b00"
BAD = "#c62828"
MUTED = "#888888"
SKIN = "#e9c19b"
SKIN_EDGE = "#c9a27f"
SHIRT = "#3f6fb5"
OPP_SHIRT = "#b54040"
SHORTS = "#2c4f80"
RACKET_FRAME = "#8a4b2d"
RACKET_STRINGS = "#efe9db"


def _scale(z):
    return 1.0 / (1.0 + PERSP * z)


class TennisPractice(tk.Toplevel):
    """Full-court swing-testing window. Feed it the stream via feed_accel()
    and Wiimote buttons via on_button()."""

    def __init__(self, master, on_close=None):
        super().__init__(master)
        self.title("Wii Tennis Practice")
        self.minsize(420, 360)
        self._fullscreen = False
        self._on_close_cb = on_close
        self._closed = False
        self.protocol("WM_DELETE_WINDOW", self._close)

        # Ball state.  _t runs 0..1 along the current leg (past 1 while a
        # missed feed sails by); _x0/_x1 are the leg's lateral path.
        self._phase = "idle"     # idle | feed | shot | bounce | net
        self._t = 0.0
        self._leg_s = 1.0        # seconds per leg
        self._x0 = self._x1 = 0.0
        self._shot = None        # active stroke params (phase shot/bounce/net)
        self._bounce = None      # bounce-leg params
        self._net_drop = None    # netted-ball drop params
        self._feed_side = "forehand"
        self._stroke = "--"
        self._streak = 0         # consecutive feeds returned in
        self._best = 0
        self._last_speed = None  # hand speed of the last stroke, g*s
        self._verdict = ("", MUTED, 0.0)
        self._particles = []
        self._capture = StrokeCapture()
        self._classifier = StrokeClassifier()
        self._detector = TennisSwingDetector()  # fallback without a model
        self._last_frame = time.monotonic()

        # Racket mirroring (see the constants block above).
        self._grav = None            # smoothed gravity vector, g units
        self._roll = 0.0
        self._pitch = 0.0
        self._sway = [0.0, 0.0]      # hand offset (court x, y) from rest
        self._sway_v = [0.0, 0.0]
        self._player_x = 0.0         # where the figure stands, eased
        self._swing_anim = None      # (start time, court dx, court dy)
        self._opp_x = 0.0            # where the opponent stands, eased
        self._opp_anim = None        # start time of the opponent's feed swing

        self._set_viewport(W, H)
        self._build_ui()
        self.bind("<Left>", lambda e: self.on_button("Left"))
        self.bind("<Right>", lambda e: self.on_button("Right"))
        self.bind("<a>", lambda e: self.on_button("A"))
        self.bind("<space>", lambda e: self.on_button("A"))
        self.bind("<F11>", lambda e: self._set_fullscreen(not self._fullscreen))
        self.bind("<Escape>", lambda e: self._set_fullscreen(False))
        self.focus_set()
        self._tick()

    # ---- data in ----

    def on_button(self, name):
        """Wiimote A feeds a ball; the D-pad picks the feed side."""
        if name == "Left":
            self._feed_side = "backhand"
        elif name == "Right":
            self._feed_side = "forehand"
        elif name == "A":
            self._feed()

    def feed_accel(self, x, y, z, deviation):
        self._update_racket(x, y, z)
        if self._classifier.ok:
            out = self._capture.feed(x, y, z)
            if out is not None:
                res = self._classifier.classify(*out)
                if res is not None and \
                        res["speed"] >= TennisSwingDetector.MIN_SPEED:
                    self._on_swing(res)
            return
        swing = self._detector.feed(x, y, z)
        if swing is not None:
            (ux, uy, uz), speed = swing
            if uz >= LOB_UZ:
                stroke = "lob"
            elif uz <= -LOB_UZ:
                stroke = "smash"
            else:
                stroke = "normal"
            self._on_swing({"side": None, "stroke": stroke, "aim": None,
                            "direction": (ux, uy, uz), "speed": speed})

    def _update_racket(self, x, y, z):
        """Track gravity and hand motion so the racket mirrors the remote.

        A reading is only trustworthy as "gravity" when its magnitude is
        near 1 g, so the further off it is (hand acceleration), the less
        it is let into the gravity estimate.  Whatever the sample holds
        beyond gravity is the hand's own acceleration; its court-plane
        part is integrated against a spring-damper into a sway offset.
        """
        if self._grav is None:
            self._grav = [float(x), float(y), float(z)]
            return
        g = self._grav
        mag = (x * x + y * y + z * z) ** 0.5
        err = abs(mag - 1.0)
        alpha = 0.15 if err < 0.10 else 0.04 if err < 0.35 else 0.008
        g[0] += alpha * (x - g[0])
        g[1] += alpha * (y - g[1])
        g[2] += alpha * (z - g[2])
        self._roll = math.atan2(g[0], g[2])
        self._pitch = math.atan2(-g[1], math.hypot(g[0], g[2]))

        # Sensor +X is the player's left (court -x), +Z is up (court +y);
        # forward/back has no place on the 2D court and is ignored.
        lx, lz = x - g[0], z - g[2]
        if math.hypot(lx, lz) < SWAY_DEADBAND:
            lx = lz = 0.0
        dt = TennisSwingDetector.DT  # samples arrive at a steady ~100 Hz
        for i, a in enumerate((-lx, lz)):
            self._sway_v[i] += (a * SWAY_ACCEL
                                - SWAY_SPRING * self._sway[i]
                                - SWAY_DAMP * self._sway_v[i]) * dt
            p = self._sway[i] + self._sway_v[i] * dt
            self._sway[i] = max(-SWAY_CLAMP, min(SWAY_CLAMP, p))

    def _on_swing(self, res):
        stroke, speed = res["stroke"], res["speed"]
        ux, _, uz = res["direction"]
        # Play the stroke on the mirrored racket whatever the game state.
        self._swing_anim = (time.monotonic(),
                            max(-1.0, min(1.0, -ux)),
                            max(-1.0, min(1.0, uz)))
        self._last_speed = speed

        if res["aim"] is not None:
            # Cross goes to the player's off side: right-to-left for a
            # forehand, left-to-right for a backhand — court x is
            # negative on the left, so the sign flips per side.
            sign = 1.0 if res["side"] == "forehand" else -1.0
            side = res["aim"] * sign * AIM_GAIN
            self._stroke = "%s %s %+.1f" % (SIDE_NAMES[res["side"]],
                                            stroke, res["aim"])
        else:
            # Fallback: sensor +X is the player's left; court x is
            # negative on the left.
            side = max(-X_MAX, min(X_MAX, -ux * SIDE_GAIN))
            self._stroke = stroke

        if self._phase == "feed":
            z = 1.0 - self._t
            if z > HIT_Z:
                self._set_verdict("Too early!", WARN)
                return
            if PERFECT_Z[0] <= z <= PERFECT_Z[1]:
                self._set_verdict("Perfect!", GOOD)
            elif z > PERFECT_Z[1]:
                self._set_verdict("Early", WARN)
            else:
                self._set_verdict("Late", WARN)
            x, _, bz = self._ball_pos()
            self._launch(stroke, side, speed, from_x=x,
                         z0=max(0.02, min(bz, 0.30)), is_return=True)
            return

        if self._phase == "shot":
            return  # the last shot is still on its way

        # No ball in play: hit one up yourself.
        self._launch(stroke, side, speed, from_x=self._player_x,
                     z0=0.02, is_return=False)

    # ---- game state ----

    def _feed(self):
        """Have the opponent play a ball to the selected side."""
        if self._phase in ("feed", "shot"):
            return  # one ball at a time
        sign = 1.0 if self._feed_side == "forehand" else -1.0
        self._x0 = self._opp_x
        self._x1 = sign * FEED_X + random.uniform(-FEED_JITTER, FEED_JITTER)
        self._t = 0.0
        self._leg_s = 1.0 / FEED_SPEED
        self._phase = "feed"
        self._shot = None
        self._opp_anim = time.monotonic()

    def _launch(self, stroke, side, hand_speed, from_x, z0, is_return):
        """Send the ball over the net with this stroke's flight."""
        p = STROKES[stroke]
        arc = p["arc"]
        # Touch shots lose arc when hit without enough pace — a limp
        # dropshot (and, rarely, a limp slice) can clip the tape.
        if stroke == "dropshot":
            full, pen = DROP_RISK
            arc -= pen * max(0.0, 1.0 - hand_speed / full)
        elif stroke == "backspin":
            full, pen = SLICE_RISK
            arc -= pen * max(0.0, 1.0 - hand_speed / full)

        pace = max(hand_speed - TennisSwingDetector.MIN_SPEED, 0.0)
        land = min(p["land"] + pace * LAND_GAIN, p["land_max"], LAND_CAP)
        speed = (BASE_SPEED + min(pace * SPEED_GAIN, SPEED_MAX)) * p["mult"]

        self._phase = "shot"
        self._t = 0.0
        self._x0 = from_x
        jitter = AIM_JITTER if self._classifier.ok else KIN_JITTER
        self._x1 = max(-X_MAX, min(X_MAX,
                                   side + random.uniform(-jitter, jitter)))
        self._leg_s = max(0.15, (land - z0) / speed)
        self._shot = dict(p, stroke=stroke, arc=arc, land=land, z0=z0,
                          is_return=is_return)
        # Will it clear the tape?  Judge the crossing point up front.
        t_net = (NET_Z - z0) / (land - z0)
        self._shot["t_net"] = t_net
        self._shot["netted"] = self._shot_y(t_net) < NET_H

    def _shot_y(self, t):
        """Height of the active shot at leg progress t (0=hit, 1=landing).

        The base term slopes from the contact height to the ground; the
        arc term is a parabola whose time axis is warped by skew, which
        is what gives topspin its early peak and dive and backspin its
        float and late fall.  A smash has no arc: a straight line down.
        """
        s = self._shot
        ts = t ** s["skew"]
        return s["hit_y"] * (1.0 - t) + s["arc"] * 4.0 * ts * (1.0 - ts)

    def _ball_pos(self):
        """Current ball position as (x, y, z) in court units, or None."""
        t = self._t
        if self._phase == "feed":
            x = self._x0 + (self._x1 - self._x0) * min(t, 1.0)
            if t < FEED_BOUNCE_T:
                f = t / FEED_BOUNCE_T
                y = FEED_Y * (1.0 - f * f)
            else:
                s = (t - FEED_BOUNCE_T) / (1.0 - FEED_BOUNCE_T)
                y = max(0.0, RACKET_Y * s + FEED_RISE * s * (1.0 - s))
            return x, y, 1.0 - t
        if self._phase == "shot":
            sh = self._shot
            tc = min(t, 1.0)
            x = self._x0 + (self._x1 - self._x0) * tc
            z = sh["z0"] + (sh["land"] - sh["z0"]) * tc
            return x, max(0.0, self._shot_y(tc)), z
        if self._phase == "bounce":
            b = self._bounce
            s = min(t, 1.0)
            return (b["x0"] + b["dx"] * s,
                    b["h"] * 4.0 * s * (1.0 - s),
                    b["z0"] + b["dz"] * s)
        if self._phase == "net":
            n = self._net_drop
            s = min(t, 1.0)
            return n["x"], n["y0"] * (1.0 - s * s), NET_Z - 0.02
        return None  # idle: no ball in play

    def _set_verdict(self, text, color):
        self._verdict = (text, color, time.monotonic())

    def _land_shot(self):
        """The shot reached its landing point: judge it, start the bounce."""
        sh = self._shot
        if abs(self._x1) > WIDE_X:
            self._streak = 0
            self._set_verdict("Wide!", BAD)
        elif sh["is_return"]:
            self._streak += 1
            self._best = max(self._best, self._streak)
            self._set_verdict("In! — %d back" % self._streak, GOOD)
        else:
            self._set_verdict("In!", GOOD)

        # The bounce keeps the shot's pace times the stroke's bounce_k:
        # topspin kicks through, backspin and dropshots sit down.
        speed = (sh["land"] - sh["z0"]) / self._leg_s
        vx = (self._x1 - self._x0) / self._leg_s
        tb = BOUNCE_BASE_S * math.sqrt(sh["bounce_h"] / 0.30)
        self._bounce = dict(
            x0=self._x1, z0=sh["land"],
            dx=vx * sh["bounce_k"] * tb,
            dz=min(speed * sh["bounce_k"] * tb, BOUNCE_DZ_MAX),
            h=sh["bounce_h"],
        )
        self._burst(self._x1, 0.02, sh["land"], sh["color"])
        self._phase = "bounce"
        self._t = 0.0
        self._leg_s = tb

    def _net_shot(self):
        """The shot failed to clear the tape: drop it at the net."""
        sh = self._shot
        x = self._x0 + (self._x1 - self._x0) * sh["t_net"]
        self._net_drop = dict(x=x, y0=max(0.02, NET_H - 0.02))
        self._burst(x, NET_H, NET_Z - 0.02, sh["color"])
        self._streak = 0
        self._set_verdict("Into the net!", BAD)
        self._phase = "net"
        self._t = 0.0
        self._leg_s = NET_DROP_S

    def _tick(self):
        if self._closed:
            return
        now = time.monotonic()
        dt = min(now - self._last_frame, 0.05)
        self._last_frame = now

        if self._phase != "idle":
            self._t += dt / self._leg_s

        if self._phase == "feed" and self._t >= MISS_T:
            self._streak = 0
            self._set_verdict("Miss!", BAD)
            self._phase = "idle"
        elif self._phase == "shot":
            sh = self._shot
            if sh["netted"] and self._t >= sh["t_net"]:
                self._net_shot()
            elif self._t >= 1.0:
                self._land_shot()
        elif self._phase in ("bounce", "net") and self._t >= 1.0:
            self._phase = "idle"

        self._update_particles(dt)

        # The player runs to where the ball will arrive, Wii Sports
        # style; between points they drift back to the center.  The
        # opponent does the same on their side of the net.
        if self._phase == "feed":
            target, opp_target = self._x1, self._x0
        elif self._phase in ("shot", "bounce"):
            target, opp_target = self._x0, self._x1
        else:
            target = opp_target = 0.0
        self._player_x += (target - self._player_x) * min(1.0, dt * 5.0)
        self._opp_x += (opp_target - self._opp_x) * min(1.0, dt * 3.0)

        self._draw()
        self._update_stats()
        self.after(FRAME_MS, self._tick)

    # ---- particles ----

    def _make_particle(self, x, y, z, color, spread=0.25):
        return dict(
            x=x + random.uniform(-0.02, 0.02),
            y=max(0.0, y + random.uniform(-0.02, 0.02)),
            z=z + random.uniform(-0.02, 0.02),
            vx=random.uniform(-spread, spread),
            vy=random.uniform(-spread, spread),
            vz=random.uniform(-0.10, 0.10),
            age=0.0, life=random.uniform(*PARTICLE_LIFE), color=color,
        )

    def _burst(self, x, y, z, color):
        if color is None:
            return
        for _ in range(BURST_N):
            p = self._make_particle(x, y, z, color, spread=0.6)
            p["vy"] = abs(p["vy"])  # impact debris flies upward
            self._particles.append(p)

    def _update_particles(self, dt):
        alive = []
        for p in self._particles:
            p["age"] += dt
            if p["age"] >= p["life"]:
                continue
            p["x"] += p["vx"] * dt
            p["y"] = max(0.0, p["y"] + p["vy"] * dt)
            p["z"] += p["vz"] * dt
            alive.append(p)
        self._particles = alive

        # A colored ball sheds a trail while its stroke is in the air.
        if self._phase in ("shot", "bounce", "net") and \
                self._shot is not None and self._shot["color"] is not None:
            bx, by, bz = self._ball_pos()
            for _ in range(TRAIL_PER_FRAME):
                self._particles.append(
                    self._make_particle(bx, by, bz, self._shot["color"]))

    # ---- UI ----

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            main, width=W, height=H, bg=COURT_OUT,
            highlightthickness=1, highlightbackground="#cccccc",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        panel = ttk.Frame(main, padding=(0, 8, 0, 0))
        panel.pack(fill="x")
        value_font = ("Segoe UI", 16, "bold")
        caption_font = ("Segoe UI", 9)

        self._streak_var = tk.StringVar(value="0")
        self._best_var = tk.StringVar(value="0")
        self._stroke_var = tk.StringVar(value="--")
        self._speed_var = tk.StringVar(value="--")
        cols = [
            ("Returns in", self._streak_var),
            ("Best", self._best_var),
            ("Last stroke", self._stroke_var),
            ("Swing speed", self._speed_var),
        ]
        for col, (title, var) in enumerate(cols):
            ttk.Label(panel, text=title, font=caption_font,
                      foreground=MUTED).grid(
                row=0, column=col, sticky="w", padx=(0 if col == 0 else 24, 0))
            ttk.Label(panel, textvariable=var, font=value_font).grid(
                row=1, column=col, sticky="w",
                padx=(0 if col == 0 else 24, 0))

        self._help = ttk.Label(
            main, wraplength=W, justify="left",
            foreground="#444444", font=("Segoe UI", 9),
            text="Press A (or Space) and the opponent feeds you a ball; "
                 "the D-pad or arrow keys pick the forehand or backhand "
                 "side — the banner on top shows where the next ball "
                 "flies. Swing as the green ring meets the racket to "
                 "return it. Each stroke flies and bounces in character "
                 "and wears its color: topspin orange (dives, kicks off "
                 "the bounce), backspin dark blue (floats, sits down), "
                 "smash red (overhead, straight and hard, bounces high), "
                 "lob green (skyball), dropshot cyan (dies just past the "
                 "net), plain drives fly clean. With no ball in play, any "
                 "swing hits one over yourself. F11 toggles fullscreen.",
        )
        self._help.pack(anchor="w", pady=(8, 0))

    def _set_viewport(self, w, h):
        """Refresh the projection for a w-by-h canvas.

        Court geometry stretches with each axis independently; pixel
        sizes (radii, line widths, fonts) follow the smaller stretch so
        figures never look inflated on a wide window.
        """
        self._cw, self._ch = w, h
        self._kx = w / W
        self._ky = h / H
        self._s = min(self._kx, self._ky)
        self._ccx = w / 2
        self._horizon = HORIZON * self._ky
        self._near_ground = NEAR_GROUND * self._ky
        self._x_near = X_NEAR * self._kx
        self._y_scale = Y_SCALE * self._ky

    def _on_canvas_resize(self, event):
        self._set_viewport(max(event.width, 1), max(event.height, 1))
        self._help.configure(wraplength=max(event.width, 300))

    def _set_fullscreen(self, on):
        self._fullscreen = on
        self.attributes("-fullscreen", on)

    def _project(self, x, y, z):
        """Court point -> (screen x, screen y). Nearer points draw larger."""
        k = 1.0 / (1.0 + PERSP * z)
        ground_y = self._horizon + (self._near_ground - self._horizon) * k
        return (self._ccx + x * self._x_near * k,
                ground_y - y * self._y_scale * k)

    def _font(self, size, bold=False):
        f = ("Segoe UI", max(7, round(size * self._s)))
        return f + ("bold",) if bold else f

    def _update_stats(self):
        self._streak_var.set(str(self._streak))
        self._best_var.set(str(self._best))
        self._stroke_var.set(self._stroke)
        self._speed_var.set("--" if self._last_speed is None
                            else "%.1f m/s" % (self._last_speed * 9.81))

    # ---- drawing ----

    def _draw(self):
        c = self.canvas
        c.delete("all")
        now = time.monotonic()
        s = self._s

        # Sky, back fence, and the ground the court sits on.
        fence_top = 108 * self._ky
        _, ground_far = self._project(0, 0, 1.6)
        c.create_rectangle(0, 0, self._cw, fence_top, fill=SKY, outline="")
        c.create_rectangle(0, fence_top, self._cw, ground_far, fill=FENCE,
                           outline="")
        c.create_rectangle(0, ground_far, self._cw, self._ch,
                           fill=COURT_OUT, outline="")

        # Court surface with a little run-off beyond the lines.
        court = []
        for x, z in ((-1.12, -0.06), (1.12, -0.06),
                     (1.12, 1.10), (-1.12, 1.10)):
            court.extend(self._project(x, 0, z))
        c.create_polygon(*court, fill=COURT, outline="")

        # Hit zone: the strip of court where a swing connects.
        zone = []
        for x, z in ((-1.0, 0.03), (1.0, 0.03), (1.0, HIT_Z), (-1.0, HIT_Z)):
            zone.extend(self._project(x, 0, z))
        c.create_polygon(*zone, fill=COURT_ZONE, outline="")
        zx, zy = self._project(-1.0, 0, HIT_Z)
        c.create_text(zx + 6 * s, zy + 10 * s, text="hit zone", anchor="w",
                      fill=COURT_LINE, font=self._font(8))

        # Lines: sidelines, both baselines, service lines + center line.
        lw = max(1, round(2 * s))
        for x in (-1.0, 1.0):
            c.create_line(*self._project(x, 0, 0.03),
                          *self._project(x, 0, 0.97),
                          fill=COURT_LINE, width=lw)
        for z in (0.03, 0.97):
            c.create_line(*self._project(-1.0, 0, z),
                          *self._project(1.0, 0, z),
                          fill=COURT_LINE, width=lw)
        for z in (0.28, 0.72):
            c.create_line(*self._project(-1.0, 0, z),
                          *self._project(1.0, 0, z),
                          fill=COURT_LINE)
        c.create_line(*self._project(0, 0, 0.28), *self._project(0, 0, 0.72),
                      fill=COURT_LINE)

        # Painter's order across the net: far side, net, near side.
        ball = self._ball_pos()
        color = (self._shot["color"]
                 if self._shot is not None and
                 self._phase in ("shot", "bounce", "net") else None)

        self._draw_opponent(c, now)
        if ball is not None and ball[2] > NET_Z:
            self._draw_ball(c, *ball, color=color)
        self._draw_particles(c, near=False)

        self._draw_net(c)

        if ball is not None and ball[2] <= NET_Z:
            self._draw_ball(c, *ball, color=color)
        self._draw_particles(c, near=True)
        self._draw_player(c, now)

        self._draw_hud(c)

        text, vcolor, ts = self._verdict
        if text and now - ts <= VERDICT_S:
            c.create_text(self._ccx, 68 * self._ky, text=text, fill=vcolor,
                          font=self._font(16, bold=True))
        if self._phase == "idle":
            c.create_text(self._ccx, self._ch - 12 * s, fill=COURT_LINE,
                          font=self._font(11, bold=True),
                          text="Press A — the opponent feeds you a ball")

    def _draw_hud(self, c):
        """Banner showing whether the ball flies to forehand or backhand."""
        name = self._feed_side.upper()
        arrow = "▶" if self._feed_side == "forehand" else "◀"
        if self._phase == "feed":
            msg = "Ball to %s %s" % (name, arrow)
        else:
            msg = "Next feed: %s %s" % (name, arrow)
        s, cx = self._s, self._ccx
        c.create_rectangle(cx - 120 * s, 8 * s, cx + 120 * s, 32 * s,
                           fill=HUD_BG, outline="")
        c.create_text(cx, 20 * s, text=msg, fill="#ffffff",
                      font=self._font(11, bold=True))
        c.create_text(cx, 42 * s, text="◀ / ▶  switch side      A  feed ball",
                      fill="#f0f0f0", font=self._font(8))

    def _draw_net(self, c):
        s = self._s
        lbx, lby = self._project(-NET_X, 0, NET_Z)
        rbx, rby = self._project(NET_X, 0, NET_Z)
        ltx, lty = self._project(-NET_X, NET_H, NET_Z)
        rtx, rty = self._project(NET_X, NET_H, NET_Z)
        c.create_polygon(lbx, lby, rbx, rby, rtx, rty, ltx, lty,
                         fill=NET_MESH, stipple="gray50", outline="")
        c.create_line(lbx, lby, rbx, rby, fill=NET_POST)
        c.create_line(ltx, lty, rtx, rty, fill=NET_BAND,
                      width=max(2, round(3 * s)))
        c.create_line(*self._project(0, 0, NET_Z),
                      *self._project(0, NET_H, NET_Z),
                      fill=NET_BAND, width=max(1, round(2 * s)))
        for bx_, by_, tx_, ty_ in ((lbx, lby, ltx, lty),
                                   (rbx, rby, rtx, rty)):
            c.create_line(bx_, by_, tx_, ty_, fill=NET_POST,
                          width=max(2, round(3 * s)))

    def _draw_ball(self, c, bx, by, bz, color=None):
        s = self._s
        k = _scale(max(bz, -0.2)) * s
        sx, sy = self._project(bx, 0, bz)
        c.create_oval(sx - 10 * k, sy - 4 * k, sx + 10 * k, sy + 4 * k,
                      fill=SHADOW, outline="")
        px, py = self._project(bx, by, bz)
        r = 3 * s + 7 * k
        if color is not None:
            g = r + 3 * s
            c.create_oval(px - g, py - g, px + g, py + g,
                          outline=color, width=max(1, round(2 * s)))
        c.create_oval(px - r, py - r, px + r, py + r,
                      fill=BALL, outline=BALL_EDGE)

    def _draw_particles(self, c, near):
        for p in self._particles:
            if (p["z"] <= NET_Z) != near:
                continue
            k = _scale(max(p["z"], -0.1)) * self._s
            px, py = self._project(p["x"], p["y"], p["z"])
            f = 1.0 - p["age"] / p["life"]
            r = (1.0 + 3.0 * f) * k
            if f > 0.45:
                c.create_oval(px - r, py - r, px + r, py + r,
                              fill=p["color"], outline="")
            else:
                # Fading out: hollow ring since tk has no real alpha.
                c.create_oval(px - r, py - r, px + r, py + r,
                              outline=p["color"])

    def _draw_opponent(self, c, now):
        """The far-side opponent who feeds the balls."""
        z = OPP_Z
        k = _scale(z) * self._s
        x = self._opp_x

        hipx, hipy = self._project(x, 0.20, z)
        for foot in (-0.055, 0.055):
            c.create_line(hipx, hipy, *self._project(x + foot, 0.0, z),
                          fill=SHORTS, width=max(2, int(5 * k)),
                          capstyle="round")
        shx, shy = self._project(x, 0.46, z)
        c.create_line(shx, shy, hipx, hipy, fill=OPP_SHIRT,
                      width=max(3, int(11 * k)), capstyle="round")
        hdx, hdy = self._project(x, 0.56, z)
        hr = 8.5 * k
        c.create_oval(hdx - hr, hdy - hr, hdx + hr, hdy + hr,
                      fill=SKIN, outline=SKIN_EDGE)

        # Their hitting arm sweeps briefly when a feed leaves the racket.
        swing = 0.0
        if self._opp_anim is not None:
            u = (now - self._opp_anim) / OPP_ANIM_S
            if u >= 1.0:
                self._opp_anim = None
            else:
                swing = math.sin(math.pi * u)
        hx = x - 0.12 - 0.15 * swing
        hy = 0.35 + 0.10 * swing
        hpx, hpy = self._project(hx, hy, z)
        c.create_line(shx, shy, hpx, hpy, fill=SKIN,
                      width=max(2, int(4 * k)), capstyle="round")
        ang = -0.5 - 0.8 * swing
        dxs, dys = math.sin(ang), -math.cos(ang)
        handle = 18 * k
        head_r = 11 * k
        rcx = hpx + dxs * (handle + head_r)
        rcy = hpy + dys * (handle + head_r)
        c.create_line(hpx, hpy, hpx + dxs * handle, hpy + dys * handle,
                      fill=RACKET_FRAME, width=max(1, round(2 * self._s)))
        c.create_oval(rcx - head_r, rcy - head_r * 0.8,
                      rcx + head_r, rcy + head_r * 0.8,
                      fill=RACKET_STRINGS, outline=RACKET_FRAME)

    def _draw_player(self, c, now):
        """Player figure whose racket mirrors the remote, plus timing aids.

        The racket tilts with the remote's roll, rises with its pitch and
        sways with the hand; a detected stroke sweeps it through a full
        swing.  While a feed is incoming, a ring collapses onto the
        racket head: it turns orange once the ball is in reach and green
        in the sweet spot — swing when the green ring meets the racket.
        """
        z = PLAYER_Z
        k = _scale(z) * self._s
        px = self._player_x + PLAYER_DX

        sweep = 0.0
        anim_dx = anim_dy = 0.0
        if self._swing_anim is not None:
            t0, adx, ady = self._swing_anim
            u = (now - t0) / SWING_ANIM_S
            if u >= 1.0:
                self._swing_anim = None
            else:
                s = math.sin(math.pi * u)  # out through the ball and back
                anim_dx = adx * SWING_REACH * s
                anim_dy = ady * SWING_REACH * s
                sweep = adx * 1.5 * s

        hx = self._player_x + HAND_X + self._sway[0] + anim_dx
        hy = max(0.06, HAND_Y + self._pitch / (math.pi / 2) * PITCH_LIFT
                 + self._sway[1] + anim_dy)
        ang = (RACKET_TILT
               + max(-ROLL_CLAMP, min(ROLL_CLAMP, self._roll))
               + sweep)
        dxs, dys = math.sin(ang), -math.cos(ang)  # racket dir, screen px
        hpx, hpy = self._project(hx, hy, z)
        handle = 26 * k
        head_r = 17 * k
        rcx = hpx + dxs * (handle + head_r)
        rcy = hpy + dys * (handle + head_r)

        # Where is the feed, timing-wise?
        ball_z = None
        if self._phase == "feed" and self._t <= MISS_T:
            ball_z = 1.0 - self._t

        # Timing ring, collapsing onto the racket head as the ball comes.
        if ball_z is not None and ball_z <= RING_START_Z:
            mid = (PERFECT_Z[0] + PERFECT_Z[1]) / 2
            f = max(0.0, (ball_z - mid) / (RING_START_Z - mid))
            r = head_r + (6 + f * 78) * self._s
            if PERFECT_Z[0] <= ball_z <= PERFECT_Z[1]:
                color, width = GOOD, 3
            elif ball_z < PERFECT_Z[0]:
                color, width = BAD, 2       # late — ball nearly gone
            elif ball_z <= HIT_Z:
                color, width = WARN, 2      # in reach, sweet spot coming
            else:
                color, width = MUTED, 1
            c.create_oval(rcx - r, rcy - r, rcx + r, rcy + r,
                          outline=color,
                          width=max(1, round(width * self._s)))

        # Figure: legs, torso, head, then the arm out to the hand.
        hipx, hipy = self._project(px, 0.20, z)
        for foot in (-0.055, 0.055):
            c.create_line(hipx, hipy, *self._project(px + foot, 0.0, z),
                          fill=SHORTS, width=max(2, int(5 * k)),
                          capstyle="round")
        shx, shy = self._project(px, 0.46, z)
        c.create_line(shx, shy, hipx, hipy, fill=SHIRT,
                      width=max(3, int(11 * k)), capstyle="round")
        hdx, hdy = self._project(px, 0.56, z)
        hr = 8.5 * k
        c.create_oval(hdx - hr, hdy - hr, hdx + hr, hdy + hr,
                      fill=SKIN, outline=SKIN_EDGE)
        c.create_line(shx, shy, hpx, hpy, fill=SKIN,
                      width=max(2, int(4 * k)), capstyle="round")

        # Racket glow while the ball can be hit: orange = in reach,
        # green = perfect window.
        glow = None
        if ball_z is not None:
            if PERFECT_Z[0] <= ball_z <= PERFECT_Z[1]:
                glow = GOOD
            elif ball_z <= HIT_Z:
                glow = WARN
        if glow is not None:
            gr = head_r + 5 * self._s
            c.create_oval(rcx - gr, rcy - gr, rcx + gr, rcy + gr,
                          outline=glow, width=max(2, round(3 * self._s)))

        # Racket: handle, then the head as a rotated ellipse polygon.
        c.create_line(hpx, hpy, hpx + dxs * handle, hpy + dys * handle,
                      fill=RACKET_FRAME, width=max(2, int(3 * k)))
        pts = []
        a, b = head_r, head_r * 0.78
        for i in range(12):
            t = 2 * math.pi * i / 12
            ca, sb = math.cos(t) * a, math.sin(t) * b
            pts.extend((rcx + dxs * ca - dys * sb,
                        rcy + dys * ca + dxs * sb))
        c.create_polygon(*pts, fill=RACKET_STRINGS,
                         outline=RACKET_FRAME,
                         width=max(1, round(2 * self._s)))
        c.create_oval(hpx - 3 * k, hpy - 3 * k, hpx + 3 * k, hpy + 3 * k,
                      fill=SKIN, outline=SKIN_EDGE)

    # ---- lifecycle ----

    def _close(self):
        self._closed = True
        if self._on_close_cb is not None:
            self._on_close_cb()
        self.destroy()
