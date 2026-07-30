"""Stroke classification shared by the offline trainer and the live game.

This module is the single home of the preprocessing that turns a raw
accelerometer window into a feature vector, so training (trainer.py) and
serving (tennis.py via StrokeClassifier) can never drift apart.  It is
pure stdlib: numpy is only needed by trainer.py.

A swing window is a list of [x, y, z] samples in g (WiiBrew frame) around
a trigger index t0 — the first sample whose acceleration magnitude
deviates from 1 g by at least E_TRIG.  That trigger is orientation-free
(no gravity baseline involved), which matters because the per-record
baselines stored by the recorder turned out to be stale: the remote is
held in racket pose, not the pose the drifting baseline last settled in.

Preprocessing:
  1. take PRE samples before t0 and POST after (left-pad by repeating the
     first sample when fewer than PRE exist),
  2. subtract a fresh local gravity estimate taken from the quiet
     pre-trigger samples,
  3. divide by the RMS deviation magnitude so classification is
     speed-invariant (speed stays a free axis, mapped to shot power),
  4. resample to FEAT_T time steps and flatten to a FEAT_T*3 vector.

StrokeClassifier loads recordings/model.json (written by trainer.py) and
turns a window into (side, stroke, aim, speed).  The model is one
multinomial logistic regression over the 36 side*stroke*placement
classes; side and stroke come from the highest-probability side*stroke
group, and aim is the softmax expectation over that group's placements
with cross = -1, center = 0, straight = +1 — cross-validated on the
recorded corpus, hard cross<->straight confusions are ~1%, so the scalar
steers monotonically.
"""

import json
import math
import os

E_TRIG = 1.2       # |accel magnitude - 1 g| that marks the swing start
E_END = 0.4        # quiet again once below this (live re-arm)
E_REST = 0.5       # pre-trigger samples below this count as "at rest"
PRE = 10           # samples kept before the trigger
POST = 45          # samples kept after the trigger (0.45 s)
FEAT_T = 32        # time steps after resampling
DT = 0.01          # seconds per sample at the remote's ~100 Hz
SCALE_FLOOR = 0.5  # minimum RMS divisor so near-still windows don't blow up

WINDOW_N = PRE + POST

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "recordings", "model.json")


def magnitude_dev(sample):
    """Deviation of one sample's acceleration magnitude from 1 g."""
    x, y, z = sample[0], sample[1], sample[2]
    return abs(math.sqrt(x * x + y * y + z * z) - 1.0)


def find_trigger(samples, e_trig=E_TRIG):
    """Index of the first sample past the trigger threshold, or None."""
    for i, s in enumerate(samples):
        if magnitude_dev(s) >= e_trig:
            return i
    return None


def _padded_window(samples, t0, pre=PRE, post=POST):
    """The pre+post window around t0, left-padded to exactly pre+post."""
    lo = max(0, t0 - pre)
    win = [[float(s[0]), float(s[1]), float(s[2])]
           for s in samples[lo:t0 + post]]
    missing_pre = pre - (t0 - lo)
    if missing_pre > 0:
        win = [list(win[0])] * missing_pre + win
    if len(win) < pre + post:
        raise ValueError("window too short: %d samples after trigger,"
                         " need %d" % (len(win) - pre, post))
    return win


def _rest_reference(win, pre=PRE):
    """Fresh local gravity estimate from the quiet pre-trigger samples."""
    quiet = [s for s in win[:pre] if magnitude_dev(s) < E_REST]
    if not quiet:
        quiet = win[:3]
    n = float(len(quiet))
    return [sum(s[0] for s in quiet) / n,
            sum(s[1] for s in quiet) / n,
            sum(s[2] for s in quiet) / n]


def feature_vector(samples, t0, pre=PRE, post=POST, feat_t=FEAT_T,
                   subtract_gravity=True):
    """The (unstandardized) feat_t*3 feature vector for one swing window."""
    win = _padded_window(samples, t0, pre, post)
    if subtract_gravity:
        r = _rest_reference(win, pre)
    else:
        r = [0.0, 0.0, 0.0]
    d = [[s[0] - r[0], s[1] - r[1], s[2] - r[2]] for s in win]

    rms = math.sqrt(sum(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
                        for v in d) / len(d))
    scale = max(SCALE_FLOOR, rms)

    n = len(d)
    feat = []
    for k in range(feat_t):
        pos = k * (n - 1) / (feat_t - 1)
        i = min(int(pos), n - 2)
        f = pos - i
        a, b = d[i], d[i + 1]
        feat.append((a[0] + (b[0] - a[0]) * f) / scale)
        feat.append((a[1] + (b[1] - a[1]) * f) / scale)
        feat.append((a[2] + (b[2] - a[2]) * f) / scale)
    return feat


def swing_kinematics(samples, t0, pre=PRE, post=POST):
    """((ux, uy, uz), speed) of the swing — direction and hand speed.

    Same rule as motion.TennisSwingDetector: integrate acceleration (minus
    gravity) over the stroke's accelerating phase — while each new sample
    still pushes in the direction of the velocity built so far — folding
    in the pre-trigger ramp when it points the same way.  Speed is in
    g-seconds (1 g*s ~= 9.8 m/s).
    """
    win = _padded_window(samples, t0, pre, post)
    r = _rest_reference(win, pre)
    d = [[s[0] - r[0], s[1] - r[1], s[2] - r[2]] for s in win]

    px = sum(v[0] for v in d[:pre]) * DT
    py = sum(v[1] for v in d[:pre]) * DT
    pz = sum(v[2] for v in d[:pre]) * DT
    d0 = d[pre]
    if px * d0[0] + py * d0[1] + pz * d0[2] > 0:
        vx, vy, vz = px, py, pz
    else:
        vx, vy, vz = 0.0, 0.0, 0.0

    for v in d[pre:]:
        if (vx, vy, vz) != (0.0, 0.0, 0.0) and \
                vx * v[0] + vy * v[1] + vz * v[2] <= 0:
            break
        vx += v[0] * DT
        vy += v[1] * DT
        vz += v[2] * DT

    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    if speed < 1e-9:
        return (0.0, 0.0, 0.0), 0.0
    return (vx / speed, vy / speed, vz / speed), speed


class StrokeClassifier:
    """Classify swing windows using the model exported by trainer.py.

    Pure stdlib; the model loads lazily on first classify().  ok is False
    when the model file is missing or invalid, letting the game fall back
    to the old heuristic detector.
    """

    def __init__(self, path=MODEL_PATH):
        self._path = path
        self._model = None
        self._tried = False

    @property
    def ok(self):
        self._load()
        return self._model is not None

    @property
    def meta(self):
        self._load()
        return self._model["meta"] if self._model else {}

    AIM_VALUE = {"cross": -1.0, "center": 0.0, "straight": 1.0}

    def _load(self):
        if self._tried:
            return
        self._tried = True
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                model = json.load(f)
            if model.get("version") != 1:
                raise ValueError("unknown model version")
            if model["type"] != "logreg36":
                raise ValueError("unknown model type")
            classes = model["classes"]
            W, b = model["logreg"]["W"], model["logreg"]["b"]
            n_feat = len(model["feat_mean"])
            if not (len(classes) == len(W) == len(b) and
                    len(model["feat_std"]) == n_feat and
                    all(len(row) == n_feat for row in W)):
                raise ValueError("inconsistent model shapes")
            # group the 36 classes into 12 (side, stroke) groups of the
            # three placements, in file order
            groups = []
            for c0 in range(0, len(classes), 3):
                side, stroke, _ = classes[c0].split(":")
                groups.append((side, stroke,
                               [classes[c0 + j].split(":")[2]
                                for j in range(3)]))
            model["_groups"] = groups
            self._model = model
        except (OSError, ValueError, KeyError, IndexError):
            self._model = None

    def classify(self, samples, t0):
        """Classify one swing window; None when no model is loaded.

        Returns a dict: side ("forehand"/"backhand"), stroke, aim in
        [-1, 1] (cross -1 .. center 0 .. straight +1, as recorded for
        that side), direction unit vector, speed (g*s) and confidence
        in [0, 1].
        """
        self._load()
        m = self._model
        if m is None:
            return None
        cfg = m["config"]

        raw = feature_vector(samples, t0,
                             pre=cfg["pre"], post=cfg["post"],
                             feat_t=cfg["feat_t"],
                             subtract_gravity=cfg["subtract_gravity"])
        mean, std = m["feat_mean"], m["feat_std"]
        v = [(raw[i] - mean[i]) / std[i] for i in range(len(raw))]

        W, b = m["logreg"]["W"], m["logreg"]["b"]
        scores = [sum(Wc[i] * v[i] for i in range(len(v))) + bc
                  for Wc, bc in zip(W, b)]
        top = max(scores)
        p = [math.exp(s - top) for s in scores]
        total = sum(p)
        p = [x / total for x in p]

        best = max(range(len(m["_groups"])),
                   key=lambda g: p[3 * g] + p[3 * g + 1] + p[3 * g + 2])
        side, stroke, placements = m["_groups"][best]
        sub = p[3 * best:3 * best + 3]
        sub_total = sum(sub) or 1.0
        aim = sum(sub[j] / sub_total * self.AIM_VALUE[placements[j]]
                  for j in range(3))

        direction, speed = swing_kinematics(samples, t0,
                                            pre=cfg["pre"], post=cfg["post"])
        return {"side": side, "stroke": stroke, "aim": aim,
                "direction": direction, "speed": speed,
                "confidence": sub_total}
