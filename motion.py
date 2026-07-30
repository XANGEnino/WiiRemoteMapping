"""Turn Wii remote accelerometer readings into swing gestures.

Readings arrive already normalized to g units: wiimote.py converts the raw
unsigned axes using the zero/+1g calibration stored in the remote's EEPROM,
the same normalization Dolphin applies to real remotes. At rest the vector
holds steady at gravity (magnitude ~1g); we track gravity with a slow
baseline and measure swings as deviations from it.

A swing produces two spikes: first in the direction of motion as the hand
accelerates, then an opposite (usually stronger) one as it stops. Direction
is therefore taken from the first spike only — the capture ends as soon as
the deviation reverses — and a cooldown swallows the deceleration spike so
it can't fire a second, opposite gesture. If the readings never settle back
to the old baseline (the remote came to rest in a new orientation, moving
gravity in the sensor frame), the current reading is adopted as the new
baseline after a timeout instead of staying stuck.

Axis convention (WiiBrew), remote held pointing at the screen, buttons up:
    +X = left, +Z = up. Y is forward/back and is ignored.
"""

import time
from collections import deque

import classifier


class SwingDetector:
    # In g units (deviation from the gravity baseline).
    START = 1.3            # net acceleration that begins a swing
    RESET = 0.6            # settled again once below this
    BASELINE_ALPHA = 0.05  # how fast the gravity estimate follows the remote

    CAPTURE_S = 0.10   # longest the initial-spike capture may run
    COOLDOWN_S = 0.30  # ignore the deceleration spike after firing
    STUCK_S = 0.80     # re-seed the baseline if the reading never settles

    def __init__(self):
        self.reset()

    def reset(self):
        self.deviation = 0.0     # |last sample - gravity baseline|, in g
        self._baseline = None    # slow [x, y, z] gravity estimate
        self._state = "idle"     # idle | capture | settle
        self._since = 0.0        # monotonic time of the last state change
        self._trigger = (0.0, 0.0, 0.0)  # deviation that started the swing
        self._peak_mag = 0.0
        self._peak = (0.0, 0.0, 0.0)

    def feed(self, x, y, z):
        """Feed one accelerometer sample; return a swing name or None.

        Names: "SwingUp", "SwingDown", "SwingLeft", "SwingRight".
        """
        now = time.monotonic()
        if self._baseline is None:
            self._baseline = [float(x), float(y), float(z)]
            self._since = now
            return None

        bx, by, bz = self._baseline
        dx, dy, dz = x - bx, y - by, z - bz
        mag = (dx * dx + dy * dy + dz * dz) ** 0.5
        self.deviation = mag

        if self._state == "idle":
            if mag >= self.START:
                self._state = "capture"
                self._since = now
                self._trigger = (dx, dy, dz)
                self._peak_mag = mag
                self._peak = (dx, dy, dz)
            elif mag < self.RESET:
                # Drift the gravity estimate only while at rest, so a swing
                # never contaminates the baseline it is measured against.
                a = self.BASELINE_ALPHA
                self._baseline = [bx + a * dx, by + a * dy, bz + a * dz]
                self._since = now
            elif now - self._since >= self.STUCK_S:
                # Neither at rest nor swinging for a while: the remote was
                # slowly re-oriented, so gravity moved in the sensor frame.
                self._baseline = [float(x), float(y), float(z)]
                self._since = now
            return None

        if self._state == "capture":
            tx, ty, tz = self._trigger
            along = dx * tx + dy * ty + dz * tz  # still the initial spike?
            if along > 0 and mag > self._peak_mag:
                self._peak_mag = mag
                self._peak = (dx, dy, dz)
            done = (
                along <= 0                        # deviation reversed: decel
                or mag < self.RESET               # spike over
                or now - self._since >= self.CAPTURE_S
            )
            if done:
                self._state = "settle"
                self._since = now
                return self._classify(*self._peak)
            return None

        # settle: wait out the deceleration spike, then require the reading
        # to calm down before another swing may begin.
        if now - self._since >= self.COOLDOWN_S and mag < self.RESET:
            self._state = "idle"
            self._since = now
        elif now - self._since >= self.STUCK_S:
            # Never settled: the remote is resting in a new orientation, so
            # adopt the current reading as the new gravity baseline.
            self._baseline = [float(x), float(y), float(z)]
            self._state = "idle"
            self._since = now
        return None

    @staticmethod
    def _classify(dx, dy, dz):
        ax, ay, az = abs(dx), abs(dy), abs(dz)
        if ax >= ay and ax >= az:
            return "SwingLeft" if dx > 0 else "SwingRight"
        if az >= ax and az >= ay:
            return "SwingUp" if dz > 0 else "SwingDown"
        return None  # forward/back jab — not one of the four swings


class TennisSwingDetector:
    """Recognize whole racket strokes rather than directional flicks.

    A tennis stroke is a sweeping arc lasting a few tenths of a second, not
    the sharp 0.1 s spike SwingDetector captures, and it isn't one of four
    directions — the ball should go wherever the player swung, as fast as
    they swung. So instead of classifying the peak of the initial spike,
    this detector integrates acceleration over the stroke's whole
    accelerating phase. The integral is the change in hand velocity: its
    direction is the true direction of the swing (even as the acceleration
    vector rotates through the arc) and its magnitude is the speed the hand
    reached.

    feed() returns None or ((ux, uy, uz), speed): the unit swing direction
    in the WiiBrew sensor frame (+X = left, +Y = forward, +Z = up) and the
    hand speed in g-seconds (1 g*s ~= 9.8 m/s).

    Timing is counted in samples (the remote streams at a steady ~100 Hz)
    rather than wall-clock time, because samples reach the consumer in
    bursts from the event queue.
    """

    DT = 0.01              # seconds per sample at the remote's ~100 Hz
    START = 1.1            # deviation (g) that begins a stroke
    RESET = 0.5            # settled again once below this
    BASELINE_ALPHA = 0.05  # how fast the gravity estimate follows the remote

    CAPTURE_S = 0.35   # an arc's accelerating phase; far longer than a flick
    COOLDOWN_S = 0.35  # ignore the deceleration spike after firing
    STUCK_S = 0.80     # re-seed the baseline if the reading never settles
    MIN_SPEED = 0.12   # discard twitches with less delta-v than this (g*s)

    def __init__(self):
        self.reset()

    def reset(self):
        self._baseline = None    # slow [x, y, z] gravity estimate
        self._state = "idle"     # idle | capture | settle
        self._elapsed = 0.0      # sample-counted seconds since state change
        self._v = [0.0, 0.0, 0.0]    # integrated delta-v of the stroke, g*s
        self._pre = [0.0, 0.0, 0.0]  # delta-v of the ramp before the trigger

    def feed(self, x, y, z):
        """Feed one g-normalized sample; return ((ux,uy,uz), speed) or None."""
        if self._baseline is None:
            self._baseline = [float(x), float(y), float(z)]
            return None

        bx, by, bz = self._baseline
        dx, dy, dz = x - bx, y - by, z - bz
        mag = (dx * dx + dy * dy + dz * dz) ** 0.5
        self._elapsed += self.DT

        if self._state == "idle":
            if mag >= self.START:
                self._state = "capture"
                self._elapsed = 0.0
                vx, vy, vz = dx * self.DT, dy * self.DT, dz * self.DT
                px, py, pz = self._pre
                if px * dx + py * dy + pz * dz > 0:
                    # The gentle ramp from rest up to the trigger is part
                    # of this same stroke; folding it in keeps slow swings
                    # from losing most of their measured speed.
                    vx, vy, vz = vx + px, vy + py, vz + pz
                self._v = [vx, vy, vz]
                self._pre = [0.0, 0.0, 0.0]
            elif mag < self.RESET:
                # Drift the gravity estimate only while at rest, so a stroke
                # never contaminates the baseline it is measured against.
                a = self.BASELINE_ALPHA
                self._baseline = [bx + a * dx, by + a * dy, bz + a * dz]
                self._elapsed = 0.0
                self._pre = [0.0, 0.0, 0.0]
            elif self._elapsed >= self.STUCK_S:
                # Neither at rest nor swinging: the remote was slowly
                # re-oriented, so gravity moved in the sensor frame.
                self._baseline = [float(x), float(y), float(z)]
                self._elapsed = 0.0
                self._pre = [0.0, 0.0, 0.0]
            else:
                px, py, pz = self._pre
                self._pre = [px + dx * self.DT,
                             py + dy * self.DT,
                             pz + dz * self.DT]
            return None

        if self._state == "capture":
            vx, vy, vz = self._v
            # Still driving the hand in the direction it has been going?
            # The decel spike points against the accumulated velocity.
            along = dx * vx + dy * vy + dz * vz
            if along > 0:
                self._v = [vx + dx * self.DT,
                           vy + dy * self.DT,
                           vz + dz * self.DT]
            done = (
                along <= 0
                or mag < self.RESET
                or self._elapsed >= self.CAPTURE_S
            )
            if not done:
                return None
            self._state = "settle"
            self._elapsed = 0.0
            vx, vy, vz = self._v
            speed = (vx * vx + vy * vy + vz * vz) ** 0.5
            if speed < self.MIN_SPEED:
                return None
            return (vx / speed, vy / speed, vz / speed), speed

        # settle: wait out the deceleration spike, then require the reading
        # to calm down before another stroke may begin.
        if self._elapsed >= self.COOLDOWN_S and mag < self.RESET:
            self._state = "idle"
            self._elapsed = 0.0
        elif self._elapsed >= self.STUCK_S:
            self._baseline = [float(x), float(y), float(z)]
            self._state = "idle"
            self._elapsed = 0.0
        return None


class StrokeCapture:
    """Cut the accelerometer stream into raw swing windows for the
    stroke classifier.

    Unlike the detectors above, this keeps no gravity baseline at all:
    the trigger is the acceleration magnitude's deviation from 1 g,
    which is zero at rest in any orientation — so a stale baseline can
    never mask or fake a swing.  The emitted window is the classifier's
    raw input: up to PRE samples of pre-trigger context from a ring
    buffer, the trigger sample, and exactly POST - 1 samples after it,
    segmented identically to how trainer.py segments the recorded
    corpus (train == serve).

    feed() returns None or (window, t0) where window is a list of
    [x, y, z] samples in g and t0 is the trigger index within it.
    Timing is counted in samples (~100 Hz), matching the other
    detectors.
    """

    SETTLE_N = 10        # consecutive quiet samples required to re-arm
    COOLDOWN_N = 50      # min samples between captures (0.5 s)
    FORCE_REARM_N = 150  # re-arm regardless after 1.5 s, so a restless
                         # rally can't wedge the detector shut

    def __init__(self):
        self._ring = deque(maxlen=classifier.PRE)
        self._state = "idle"    # idle | capture | settle
        self._window = []
        self._t0 = 0
        self._count = 0         # samples since entering settle
        self._quiet = 0         # consecutive samples below E_END

    def feed(self, x, y, z):
        """Feed one g-normalized sample; return (window, t0) or None."""
        s = [float(x), float(y), float(z)]
        e = classifier.magnitude_dev(s)

        if self._state == "idle":
            if e >= classifier.E_TRIG:
                self._window = [list(v) for v in self._ring] + [s]
                self._t0 = len(self._window) - 1
                self._state = "capture"
            self._ring.append(s)
            return None

        self._ring.append(s)
        if self._state == "capture":
            self._window.append(s)
            if len(self._window) - self._t0 >= classifier.POST:
                self._state = "settle"
                self._count = 0
                self._quiet = 0
                return self._window, self._t0
            return None

        # settle
        self._count += 1
        self._quiet = self._quiet + 1 if e < classifier.E_END else 0
        if (self._count >= self.COOLDOWN_N and self._quiet >= self.SETTLE_N) \
                or self._count >= self.FORCE_REARM_N:
            self._state = "idle"
        return None
