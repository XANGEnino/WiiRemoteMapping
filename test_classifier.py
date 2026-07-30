"""Replay every recorded swing through the LIVE capture + classify path.

Proves train == serve: each record's raw samples are fed one at a time
into motion.StrokeCapture (exactly as tennis.py will feed them), and the
emitted window must classify to the same stroke, aim and speed that
trainer.py stored in recordings/predictions.json when it exported the
model.  Also proves the runtime stays pure stdlib and fast enough.

Run:  py test_classifier.py
"""

import json
import os
import sys
import time
import unittest

import classifier
from motion import StrokeCapture

HERE = os.path.dirname(os.path.abspath(__file__))
SWINGS_PATH = os.path.join(HERE, "recordings", "swings.jsonl")
PRED_PATH = os.path.join(HERE, "recordings", "predictions.json")


def load_data():
    recs = []
    with open(SWINGS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    with open(PRED_PATH, "r", encoding="utf-8") as f:
        preds = json.load(f)
    return recs, preds


class TestRuntimeIsStdlibOnly(unittest.TestCase):
    def test_no_numpy_loaded(self):
        # importing classifier/motion must not have pulled in numpy —
        # the game has to run on a machine without it.
        self.assertNotIn("numpy", sys.modules)


class TestLiveReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs, cls.preds = load_data()
        cls.sc = classifier.StrokeClassifier()
        if not cls.sc.ok:
            raise unittest.SkipTest("recordings/model.json missing — "
                                    "run: py trainer.py --export")

    def test_replay_matches_predictions(self):
        agree_label = 0
        elapsed = 0.0
        for rec, pred in zip(self.recs, self.preds):
            cap = StrokeCapture()
            emitted = None
            for s in rec["accel"]:
                out = cap.feed(s[0], s[1], s[2])
                if out is not None and emitted is None:
                    emitted = out
            self.assertIsNotNone(
                emitted, "no capture for %s/%s" % (rec["label"],
                                                   rec["stroke"]))
            window, t0 = emitted
            # live t0 counts the (up to PRE) ring samples before the
            # trigger; the trainer's t0 indexes the whole record.
            self.assertEqual(t0, min(classifier.PRE, pred["t0"]))

            start = time.perf_counter()
            res = self.sc.classify(window, t0)
            elapsed += time.perf_counter() - start

            cls12 = res["side"] + ":" + res["stroke"]
            self.assertEqual(cls12, pred["pred"],
                             "record %d classified %s, trainer said %s"
                             % (pred["i"], cls12, pred["pred"]))
            self.assertAlmostEqual(res["aim"], pred["aim"], delta=1e-3)
            self.assertAlmostEqual(res["speed"], pred["speed"], delta=1e-3)
            agree_label += cls12 == \
                rec["label"].split("_")[0] + ":" + rec["stroke"]

        agreement = agree_label / len(self.recs)
        self.assertGreaterEqual(
            agreement, 0.99,
            "only %.1f%% of replays match their recorded label" %
            (agreement * 100))

        per_swing_ms = elapsed / len(self.recs) * 1000
        self.assertLess(per_swing_ms, 5.0,
                        "classify too slow: %.2f ms" % per_swing_ms)

    def test_no_trigger_at_rest(self):
        # an hour of holding still (any orientation) must never fire
        cap = StrokeCapture()
        for _ in range(3000):
            self.assertIsNone(cap.feed(-0.7, -0.1, 0.6))

    def test_rearm_after_capture(self):
        rec = self.recs[0]
        cap = StrokeCapture()
        fired = 0
        for _ in range(3):  # same swing three times with rests between
            for s in rec["accel"]:
                if cap.feed(s[0], s[1], s[2]) is not None:
                    fired += 1
            for _ in range(200):
                cap.feed(0.0, 0.0, 1.0)
        self.assertEqual(fired, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
