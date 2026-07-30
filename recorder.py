"""Record labeled swing samples for stroke-recognition analysis.

Pick a stroke class (topspin, backspin, no-spin, smash, dropshot, lob)
and a placement (forehand/backhand x cross/straight/center), then hold
the remote's A button while you swing: recording starts when A goes
down and ends when it is released, so each hold is exactly one swing —
backswing included — segmented by you, not by a threshold.  A short
tail keeps streaming after the release so the deceleration spike at the
end of the stroke isn't cut off.  MotionPlus gyro samples, when the
remote has one, are captured over the same window.  Record as many
samples per stroke as you like; per-label counts accumulate across
sessions.

Every capture is appended as one JSON line to recordings/swings.jsonl:

    {"label": <placement>, "stroke": <class>, "t": ..., "rate_hz": 100,
     "trigger_i": <first sample whose deviation crossed START, else 0>,
     "release_i": <sample count when A was released>,
     "baseline": [bx, by, bz],          # gravity estimate at A-press
     "accel": [[x, y, z], ...],         # g units, sensor frame (WiiBrew)
     "gyro": [[yaw, pitch, roll, dt], ...] | null}   # deg/s

Gravity tracking copies the detectors in motion.py: the baseline drifts
only while the remote is at rest and is re-seeded if the readings sit
between rest and swing for too long (remote slowly re-oriented).
"""

import json
import os
import time

import tkinter as tk
from tkinter import ttk

RATE_HZ = 100          # the remote streams at a steady ~100 Hz
DT = 1.0 / RATE_HZ
START = 1.0            # deviation (g) marked as the swing trigger
RESET = 0.5            # calm enough to drift the gravity baseline
BASELINE_ALPHA = 0.05  # how fast the gravity estimate follows the remote
STUCK_S = 0.80         # re-seed the baseline if the reading never settles

TAIL_N = 30            # samples kept after A-release (the decel spike)
MIN_N = 15             # shorter holds than this are not saved
MAX_N = 800            # hard cap on one recording (8 s)

STROKES = [
    ("topspin", "Topspin"),
    ("backspin", "Backspin"),
    ("normal", "No-spin"),
    ("smash", "Smash"),
    ("dropshot", "Dropshot"),
    ("lob", "Lob"),
]

LABELS = [
    ("forehand_cross", "Forehand cross"),
    ("forehand_straight", "Forehand straight"),
    ("forehand_center", "Forehand center"),
    ("backhand_cross", "Backhand cross"),
    ("backhand_straight", "Backhand straight"),
    ("backhand_center", "Backhand center"),
]

REC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "recordings")
REC_FILE = os.path.join(REC_DIR, "swings.jsonl")

TRACE_W, TRACE_H = 400, 110
GOOD = "#2e7d32"
WARN = "#e07b00"
MUTED = "#888888"
TRACE = "#3f6fb5"
THRESH = "#c62828"

IDLE_TEXT = "Hold A on the remote to record a swing"


class SwingRecorder(tk.Toplevel):
    """Labeled-swing capture window.

    Feed it the accel/gyro stream via feed_accel()/feed_gyro() and the
    remote's A button via on_record_button() — recording runs while A
    is held.
    """

    def __init__(self, master, on_close=None):
        super().__init__(master)
        self.title("Swing Recorder")
        self.resizable(False, False)
        self._on_close_cb = on_close
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._baseline = None    # slow [x, y, z] gravity estimate
        self._elapsed = 0.0      # sample-counted seconds since a calm reading
        self._rec_accel = None   # samples of the hold in progress, or None
        self._rec_gyro = None
        self._rec_baseline = None
        self._release_i = None   # sample count when A was released
        self._tail_left = None   # post-release samples still to collect
        # (file size before the write, stroke, label) per save this
        # session, so "Discard last" can truncate bad swings off the file.
        self._saved = []

        self._counts = {(s, l): 0 for s, _ in STROKES for l, _ in LABELS}
        self._scan_existing()
        self._build_ui()

    # ---- data in ----

    def on_record_button(self, pressed):
        """A-button edge from the app: press starts, release ends."""
        if pressed:
            if self._rec_accel is not None:
                return
            if self._baseline is None:
                self._set_status("No motion data — is the remote "
                                 "connected?", WARN)
                return
            self._rec_accel = []
            self._rec_gyro = []
            self._rec_baseline = tuple(self._baseline)
            self._release_i = None
            self._tail_left = None
            self._set_status("Recording — swing, then release A", WARN)
        elif self._rec_accel is not None and self._tail_left is None:
            # Keep a short tail so the stroke's deceleration spike at
            # the moment of release still makes it into the sample.
            self._release_i = len(self._rec_accel)
            self._tail_left = TAIL_N

    def feed_accel(self, x, y, z, deviation):
        if self._baseline is None:
            self._baseline = [float(x), float(y), float(z)]
            return

        if self._rec_accel is not None:
            self._rec_accel.append((x, y, z))
            if self._tail_left is not None:
                self._tail_left -= 1
                if self._tail_left <= 0:
                    self._finish()
            elif len(self._rec_accel) >= MAX_N:
                self._release_i = len(self._rec_accel)
                self._finish()
            return

        bx, by, bz = self._baseline
        dx, dy, dz = x - bx, y - by, z - bz
        mag = (dx * dx + dy * dy + dz * dz) ** 0.5
        self._elapsed += DT
        if mag < RESET:
            # Drift the gravity estimate only while at rest, so a swing
            # never contaminates the baseline it is measured against.
            a = BASELINE_ALPHA
            self._baseline = [bx + a * dx, by + a * dy, bz + a * dz]
            self._elapsed = 0.0
        elif self._elapsed >= STUCK_S:
            # Neither at rest nor swinging: the remote was slowly
            # re-oriented, so gravity moved in the sensor frame.
            self._baseline = [float(x), float(y), float(z)]
            self._elapsed = 0.0

    def feed_gyro(self, yaw, pitch, roll, dt):
        if self._rec_gyro is not None:
            self._rec_gyro.append((yaw, pitch, roll, dt))

    # ---- capture -> file ----

    def _finish(self):
        accel = self._rec_accel
        gyro = self._rec_gyro
        baseline = self._rec_baseline
        release_i = self._release_i
        self._rec_accel = self._rec_gyro = self._rec_baseline = None
        self._release_i = self._tail_left = None
        self._elapsed = 0.0

        # Judge the hold itself, not the post-release tail — otherwise
        # the tail alone pushes an accidental tap past the minimum.
        if release_i < MIN_N:
            self._set_status("Too short — not saved. " + IDLE_TEXT, WARN)
            return

        bx, by, bz = baseline
        devs = [((x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2) ** 0.5
                for x, y, z in accel]
        trigger_i = next((i for i, d in enumerate(devs) if d >= START), 0)

        stroke = self._stroke_var.get()
        label = self._label_var.get()
        record = {
            "label": label,
            "stroke": stroke,
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rate_hz": RATE_HZ,
            "trigger_i": trigger_i,
            "release_i": release_i,
            "baseline": [round(v, 4) for v in baseline],
            "accel": [[round(v, 4) for v in s] for s in accel],
            "gyro": ([[round(v, 2) for v in s[:3]] + [round(s[3], 4)]
                      for s in gyro] if gyro else None),
        }
        os.makedirs(REC_DIR, exist_ok=True)
        with open(REC_FILE, "a", encoding="utf-8") as f:
            self._saved.append((f.tell(), stroke, label))
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

        self._counts[(stroke, label)] += 1
        self._update_counts()
        self.discard_btn.config(state="normal")
        self._draw_trace(devs, trigger_i)
        self._set_status(
            "Saved %s %s #%d (%d samples, %.2f g peak%s)" % (
                dict(STROKES)[stroke].lower(), dict(LABELS)[label].lower(),
                self._counts[(stroke, label)], len(accel),
                max(devs), ", gyro" if gyro else ""),
            GOOD,
        )

    def _scan_existing(self):
        """Count previously recorded swings so totals carry across runs."""
        try:
            with open(REC_FILE, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    key = (rec.get("stroke"), rec.get("label"))
                    if key in self._counts:
                        self._counts[key] += 1
        except OSError:
            pass

    def _discard_last(self):
        if not self._saved:
            return
        offset, stroke, label = self._saved.pop()
        with open(REC_FILE, "a", encoding="utf-8") as f:
            f.truncate(offset)
        key = (stroke, label)
        self._counts[key] = max(0, self._counts[key] - 1)
        self._update_counts()
        self._set_status(
            "Discarded last %s %s" % (dict(STROKES)[stroke].lower(),
                                      dict(LABELS)[label].lower()),
            WARN,
        )
        if not self._saved:
            self.discard_btn.config(state="disabled")

    # ---- UI ----

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both")

        ttk.Label(
            main, wraplength=TRACE_W + 40, justify="left",
            foreground="#444444", font=("Segoe UI", 9),
            text="Pick the stroke and placement you are about to practice, "
                 "then hold the remote's A button while you swing — "
                 "recording starts when A goes down and is saved when you "
                 "release it (A doesn't type its key while this window is "
                 "open). Repeat for as many samples per stroke as you "
                 "want. Recorded swings append to recordings\\swings.jsonl.",
        ).pack(anchor="w", pady=(0, 8))

        selectors = ttk.Frame(main)
        selectors.pack(anchor="w", fill="x")
        self._stroke_var = tk.StringVar(value=STROKES[0][0])
        self._label_var = tk.StringVar(value=LABELS[0][0])
        self._stroke_vars = {}
        self._count_vars = {}

        strokes = ttk.Labelframe(selectors, text="Stroke", padding=6)
        strokes.pack(side="left", anchor="n")
        for i, (key, text) in enumerate(STROKES):
            var = tk.StringVar()
            self._stroke_vars[key] = var
            ttk.Radiobutton(
                strokes, textvariable=var, variable=self._stroke_var,
                value=key,
            ).grid(row=i % 3, column=i // 3, sticky="w",
                   padx=(0 if i < 3 else 16, 0), pady=2)

        placements = ttk.Labelframe(selectors, text="Placement", padding=6)
        placements.pack(side="left", anchor="n", padx=(10, 0))
        for i, (key, text) in enumerate(LABELS):
            var = tk.StringVar()
            self._count_vars[key] = var
            ttk.Radiobutton(
                placements, textvariable=var, variable=self._label_var,
                value=key,
            ).grid(row=i % 3, column=i // 3, sticky="w",
                   padx=(0 if i < 3 else 16, 0), pady=2)

        # Placement counts show the selected stroke's tally, so refresh
        # them whenever the stroke class changes.
        self._stroke_var.trace_add("write",
                                   lambda *_: self._update_counts())
        self._update_counts()

        controls = ttk.Frame(main)
        controls.pack(fill="x", pady=(10, 6))
        self.discard_btn = ttk.Button(
            controls, text="Discard last", command=self._discard_last,
            state="disabled",
        )
        self.discard_btn.pack(side="left")
        self.status_label = ttk.Label(
            controls, text=IDLE_TEXT, foreground=MUTED
        )
        self.status_label.pack(side="left", padx=(14, 0))

        ttk.Label(main, text="Last capture (deviation from gravity, g):",
                  foreground=MUTED, font=("Segoe UI", 9)).pack(anchor="w")
        self.canvas = tk.Canvas(
            main, width=TRACE_W, height=TRACE_H, bg="white",
            highlightthickness=1, highlightbackground="#cccccc",
        )
        self.canvas.pack(pady=(2, 0))

    def _update_counts(self):
        """Stroke radios show totals; placement radios show the tally
        for the currently selected stroke class."""
        for skey, text in STROKES:
            total = sum(self._counts[(skey, l)] for l, _ in LABELS)
            self._stroke_vars[skey].set("%s  (%d)" % (text, total))
        stroke = self._stroke_var.get()
        for lkey, text in LABELS:
            self._count_vars[lkey].set(
                "%s  (%d)" % (text, self._counts[(stroke, lkey)])
            )

    def _set_status(self, text, color):
        self.status_label.config(text=text, foreground=color)

    def _draw_trace(self, devs, trigger_i):
        c = self.canvas
        c.delete("all")
        top = max(3.0, max(devs))
        pad = 6

        def sy(v):
            return TRACE_H - pad - (TRACE_H - 2 * pad) * (v / top)

        step = (TRACE_W - 2 * pad) / max(len(devs) - 1, 1)
        ty = sy(START)
        c.create_line(0, ty, TRACE_W, ty, fill=THRESH, dash=(3, 3))
        c.create_text(TRACE_W - 4, ty - 7, text="trigger", anchor="e",
                      fill=THRESH, font=("Segoe UI", 8))
        tx = pad + trigger_i * step
        c.create_line(tx, pad, tx, TRACE_H - pad, fill=MUTED, dash=(2, 4))
        pts = []
        for i, d in enumerate(devs):
            pts.extend((pad + i * step, sy(d)))
        c.create_line(*pts, fill=TRACE, width=2)

    # ---- lifecycle ----

    def _close(self):
        if self._on_close_cb is not None:
            self._on_close_cb()
        self.destroy()
