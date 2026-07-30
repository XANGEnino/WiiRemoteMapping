"""Offline analysis and trainer for the tennis stroke classifier.

Reads recordings/swings.jsonl, evaluates classifier candidates with
stratified 5-fold cross-validation, analyses how separable the aim
placements (cross / center / straight) are, and — with --export — writes
recordings/model.json (loaded by classifier.StrokeClassifier, pure
stdlib) plus recordings/predictions.json (per-record expected outputs
used by test_classifier.py to prove the live path matches training).

Preprocessing is imported from classifier.py so training and serving
share one implementation.

Usage:
    py trainer.py             analysis report only
    py trainer.py --export    also fit on all data and write the model
    py trainer.py --full      include the slow extras (DTW, sweeps)

Needs numpy (offline only; the game itself never imports this module).
"""

import argparse
import json
import os
import sys
from datetime import datetime

try:
    import numpy as np
except ImportError:
    sys.exit("trainer.py needs numpy:  py -m pip install numpy")

import classifier as clf

HERE = os.path.dirname(os.path.abspath(__file__))
SWINGS_PATH = os.path.join(HERE, "recordings", "swings.jsonl")
MODEL_PATH = clf.MODEL_PATH
PRED_PATH = os.path.join(HERE, "recordings", "predictions.json")

STROKES = ["topspin", "backspin", "normal", "smash", "dropshot", "lob"]
SIDES = ["forehand", "backhand"]
PLACEMENTS = ["cross", "center", "straight"]

N_FOLDS = 5
STD_FLOOR = 1e-6


# ---- data loading -------------------------------------------------------

def load_records():
    recs = []
    with open(SWINGS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    for r in recs:
        side, placement = r["label"].split("_")
        r["side"], r["placement"] = side, placement
        r["cls12"] = side + ":" + r["stroke"]
        t0 = clf.find_trigger(r["accel"])
        if t0 is None:
            sys.exit("record without trigger — data assumption broken")
        r["t0"] = t0
    return recs


def build_features(recs, post=clf.POST, subtract=True):
    X = np.array([clf.feature_vector(r["accel"], r["t0"], post=post,
                                     subtract_gravity=subtract)
                  for r in recs])
    return X


def fold_ids(recs):
    """Deterministic stratified folds: deal each 36-way group round-robin."""
    order = {}
    ids = np.empty(len(recs), dtype=int)
    for i, r in enumerate(recs):
        key = (r["label"], r["stroke"])
        ids[i] = order.get(key, 0) % N_FOLDS
        order[key] = order.get(key, 0) + 1
    return ids


def standardize(Xtr, X):
    mean = Xtr.mean(axis=0)
    std = np.maximum(Xtr.std(axis=0), STD_FLOOR)
    return (X - mean) / std, mean, std


# ---- classifier candidates ---------------------------------------------

def centroid_fit_predict(Xtr, ytr, Xte, n_cls):
    cen = np.stack([Xtr[ytr == c].mean(axis=0) for c in range(n_cls)])
    d2 = ((Xte[:, None, :] - cen[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def logreg_fit(Xtr, ytr, n_cls, iters=400, lr=0.5, l2=1e-3):
    n, d = Xtr.shape
    W = np.zeros((n_cls, d))
    b = np.zeros(n_cls)
    Y = np.eye(n_cls)[ytr]
    for _ in range(iters):
        z = Xtr @ W.T + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - Y) / n
        W -= lr * (g.T @ Xtr + l2 * W)
        b -= lr * g.sum(axis=0)
    return W, b


def knn_predict(Xtr, ytr, Xte, k):
    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(axis=2)
    idx = np.argsort(d2, axis=1)[:, :k]
    votes = ytr[idx]
    out = np.empty(len(Xte), dtype=int)
    for i in range(len(Xte)):
        out[i] = np.bincount(votes[i]).argmax()
    return out


def dtw_pairwise(seq):
    """Full pairwise DTW matrix over (n, T, 3) sequences (slow, ~30 s)."""
    n, T, _ = seq.shape
    D = np.zeros((n, n))
    inf = np.inf
    for i in range(n):
        cost = np.sqrt(((seq[i][None, :, None, :] -
                         seq[:, None, :, :]) ** 2).sum(-1))  # (n, T, T)
        acc = np.full((n, T, T), inf)
        acc[:, 0, 0] = cost[:, 0, 0]
        for a in range(T):
            for b in range(T):
                if a == 0 and b == 0:
                    continue
                best = np.full(n, inf)
                if a > 0:
                    best = np.minimum(best, acc[:, a - 1, b])
                if b > 0:
                    best = np.minimum(best, acc[:, a, b - 1])
                if a > 0 and b > 0:
                    best = np.minimum(best, acc[:, a - 1, b - 1])
                acc[:, a, b] = cost[:, a, b] + best
        D[i] = acc[:, T - 1, T - 1]
        if i % 120 == 0:
            print("  dtw row %d/%d" % (i, n), flush=True)
    return D


# ---- evaluation ---------------------------------------------------------

def crossval(X, recs, cls_names, with_dtw=False, dtw_D=None):
    """CV accuracy per candidate; returns dict name -> (pred12 array)."""
    y = np.array([cls_names.index(r["cls12"]) for r in recs])
    ids = fold_ids(recs)
    n_cls = len(cls_names)
    preds = {name: np.empty(len(recs), dtype=int)
             for name in ["centroid", "logreg", "knn1", "knn3", "knn5"]
             + (["dtw1nn"] if with_dtw else [])}
    for f in range(N_FOLDS):
        tr, te = ids != f, ids == f
        Xs, mean, std = standardize(X[tr], X)
        Xtr, Xte = Xs[tr], Xs[te]
        ytr = y[tr]
        preds["centroid"][te] = centroid_fit_predict(Xtr, ytr, Xte, n_cls)
        W, b = logreg_fit(Xtr, ytr, n_cls)
        preds["logreg"][te] = (Xte @ W.T + b).argmax(axis=1)
        for k in (1, 3, 5):
            preds["knn%d" % k][te] = knn_predict(Xtr, ytr, Xte, k)
        if with_dtw:
            sub = dtw_D[np.ix_(te, tr)]
            preds["dtw1nn"][te] = ytr[sub.argmin(axis=1)]
    return y, preds


def acc_report(y, pred, recs, cls_names):
    """12-class, 6-stroke and side accuracies for one prediction array."""
    ok12 = (pred == y).mean()
    true_stroke = np.array([STROKES.index(r["stroke"]) for r in recs])
    pred_stroke = np.array([STROKES.index(cls_names[p].split(":")[1])
                            for p in pred])
    true_side = np.array([SIDES.index(r["side"]) for r in recs])
    pred_side = np.array([SIDES.index(cls_names[p].split(":")[0])
                          for p in pred])
    return ok12, (pred_stroke == true_stroke).mean(), \
        (pred_side == true_side).mean(), true_stroke, pred_stroke


def stroke_confusion(true_stroke, pred_stroke):
    cm = np.zeros((6, 6), dtype=int)
    for t, p in zip(true_stroke, pred_stroke):
        cm[t][p] += 1
    return cm


def print_confusion(cm):
    head = "            " + "".join("%9s" % s[:8] for s in STROKES)
    print(head)
    for i, s in enumerate(STROKES):
        print("  %-10s" % s + "".join("%9d" % cm[i][j] for j in range(6)))


def pairwise_acc(cm, a, b):
    ia, ib = STROKES.index(a), STROKES.index(b)
    n = cm[ia].sum() + cm[ib].sum()
    hits = n - cm[ia][ib] - cm[ib][ia]
    return hits / n


# ---- direction (aim) analysis ------------------------------------------

def fisher_axis(Xs, groups):
    """Regularized Fisher axis between the cross and straight groups.

    Xs: standardized features of one side; groups: placement label per
    row.  Returns (w, knots dict) with w oriented so straight > cross.
    """
    d = Xs.shape[1]
    Sw = np.zeros((d, d))
    mus = {}
    for pl in PLACEMENTS:
        G = Xs[groups == pl]
        mu = G.mean(axis=0)
        mus[pl] = mu
        C = G - mu
        Sw += C.T @ C
    Sw += 0.1 * (np.trace(Sw) / d) * np.eye(d)
    w = np.linalg.solve(Sw, mus["straight"] - mus["cross"])
    w /= np.linalg.norm(w)
    if w @ mus["straight"] < w @ mus["cross"]:
        w = -w
    knots = {pl: float(w @ mus[pl]) for pl in PLACEMENTS}
    return w, knots


def misorder_rate(P, groups, knots):
    """3-way error of nearest-knot classification along the axis."""
    wrong = 0
    for p, g in zip(P, groups):
        nearest = min(PLACEMENTS, key=lambda pl: abs(p - knots[pl]))
        wrong += nearest != g
    return wrong / len(P)


def direction_analysis(X, recs, per_stroke=False):
    """CV misorder per side (optionally per side+stroke axes)."""
    ids = fold_ids(recs)
    side_arr = np.array([r["side"] for r in recs])
    stroke_arr = np.array([r["stroke"] for r in recs])
    place_arr = np.array([r["placement"] for r in recs])
    results = {}
    for side in SIDES:
        keys = [(side, s) for s in STROKES] if per_stroke else [(side,)]
        wrong = total = 0
        for key in keys:
            sel = side_arr == side
            if per_stroke:
                sel &= stroke_arr == key[1]
            for f in range(N_FOLDS):
                tr = sel & (ids != f)
                te = sel & (ids == f)
                Xs, mean, std = standardize(X[ids != f], X)
                w, knots = fisher_axis(Xs[tr], place_arr[tr])
                P = Xs[te] @ w
                wrong += misorder_rate(P, place_arr[te], knots) * te.sum()
                total += te.sum()
        results[side] = wrong / total
    return results


def direction_fullfit(X, recs):
    """Full-data per-side axes + per-placement projection stats."""
    Xs, mean, std = standardize(X, X)
    side_arr = np.array([r["side"] for r in recs])
    place_arr = np.array([r["placement"] for r in recs])
    out = {}
    for side in SIDES:
        sel = side_arr == side
        w, knots = fisher_axis(Xs[sel], place_arr[sel])
        stats = {}
        for pl in PLACEMENTS:
            P = Xs[sel & (place_arr == pl)] @ w
            stats[pl] = (float(P.mean()), float(P.std()))
        out[side] = {"w": w, "knots": knots, "stats": stats,
                     "misorder": misorder_rate(Xs[sel] @ w, place_arr[sel],
                                               knots)}
    return out


# ---- the exported model: 36-class logreg -------------------------------

def classes36():
    return [s + ":" + k + ":" + p
            for s in SIDES for k in STROKES for p in PLACEMENTS]


def eval36(X, recs):
    """CV the 36-class side*stroke*placement logreg — the export shape.

    Aim is the softmax expectation over the winning group's placements
    (cross -1, center 0, straight +1), exactly as StrokeClassifier
    computes it.
    """
    names = classes36()
    y36 = np.array([names.index(r["side"] + ":" + r["stroke"] + ":" +
                                r["placement"]) for r in recs])
    ids = fold_ids(recs)
    P = np.zeros((len(recs), len(names)))
    for f in range(N_FOLDS):
        tr, te = ids != f, ids == f
        Xs, _, _ = standardize(X[tr], X)
        W, b = logreg_fit(Xs[tr], y36[tr], len(names), iters=600)
        z = Xs[te] @ W.T + b
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        P[te] = e / e.sum(axis=1, keepdims=True)

    val = {"cross": -1.0, "center": 0.0, "straight": 1.0}
    group_p = P.reshape(len(recs), 12, 3).sum(axis=2)
    best12 = group_p.argmax(axis=1)
    pred_side = np.array([names[3 * g].split(":")[0] for g in best12])
    pred_stroke = np.array([names[3 * g].split(":")[1] for g in best12])
    aims = np.empty(len(recs))
    pred_place = np.empty(len(recs), dtype=object)
    for i, g in enumerate(best12):
        sub = P[i, 3 * g:3 * g + 3]
        sub = sub / (sub.sum() or 1.0)
        pls = [names[3 * g + j].split(":")[2] for j in range(3)]
        aims[i] = sum(sub[j] * val[pls[j]] for j in range(3))
        pred_place[i] = pls[int(sub.argmax())]

    side_arr = np.array([r["side"] for r in recs])
    stroke_arr = np.array([r["stroke"] for r in recs])
    place_arr = np.array([r["placement"] for r in recs])
    out = {"names": names, "y36": y36,
           "acc_side": (pred_side == side_arr).mean(),
           "acc_stroke": (pred_stroke == stroke_arr).mean(),
           "acc_12": ((pred_side == side_arr) &
                      (pred_stroke == stroke_arr)).mean(),
           "pred_stroke": pred_stroke, "aims": aims, "sides": {}}
    for side in SIDES:
        sel = side_arr == side
        flips = (((place_arr == "cross") & (pred_place == "straight")) |
                 ((place_arr == "straight") & (pred_place == "cross")))
        stats = {pl: (float(aims[sel & (place_arr == pl)].mean()),
                      float(aims[sel & (place_arr == pl)].std()))
                 for pl in PLACEMENTS}
        out["sides"][side] = {
            "place_err": (pred_place[sel] != place_arr[sel]).mean(),
            "flips": flips[sel].mean(), "aim_stats": stats}
    return out


# ---- report -------------------------------------------------------------

def run_analysis(recs, full=False):
    print("== dataset ==")
    print("  %d records, %d strokes x %d placements" %
          (len(recs), len(STROKES), 6))

    cls_names = [s + ":" + k for s in SIDES for k in STROKES]
    X = build_features(recs)

    dtw_D = None
    if full:
        print("== computing pairwise DTW (slow) ==", flush=True)
        seq = np.array([clf.feature_vector(r["accel"], r["t0"])
                        for r in recs]).reshape(len(recs), clf.FEAT_T, 3)
        dtw_D = dtw_pairwise(seq)

    print("== 5-fold CV, default config (POST=%d, gravity subtracted) =="
          % clf.POST)
    y, preds = crossval(X, recs, cls_names, with_dtw=full, dtw_D=dtw_D)
    best_name, best_acc, best_pred = None, -1.0, None
    for name, pred in preds.items():
        a12, a6, aside, ts, ps = acc_report(y, pred, recs, cls_names)
        print("  %-9s 12-class %5.1f%%   stroke %5.1f%%   side %5.1f%%"
              % (name, a12 * 100, a6 * 100, aside * 100))
        if name in ("centroid", "logreg") and a12 > best_acc:
            best_name, best_acc, best_pred = name, a12, pred
    a12, a6, aside, ts, ps = acc_report(y, best_pred, recs, cls_names)
    cm = stroke_confusion(ts, ps)
    print("== stroke confusion (%s, CV) ==" % best_name)
    print_confusion(cm)
    print("  dropshot<->backspin pairwise: %5.1f%%"
          % (pairwise_acc(cm, "dropshot", "backspin") * 100))
    print("  lob<->topspin pairwise:       %5.1f%%"
          % (pairwise_acc(cm, "lob", "topspin") * 100))

    if full:
        print("== A/B: no gravity subtraction ==")
        Xn = build_features(recs, subtract=False)
        yn, pn = crossval(Xn, recs, cls_names)
        for name in ("centroid", "logreg"):
            a12, a6, aside, _, _ = acc_report(yn, pn[name], recs, cls_names)
            print("  %-9s 12-class %5.1f%%   stroke %5.1f%%"
                  % (name, a12 * 100, a6 * 100))
        print("== POST sweep ==")
        for post in (35, 45, 50):
            Xp = build_features(recs, post=post)
            yp, pp = crossval(Xp, recs, cls_names)
            a12, a6, _, _, _ = acc_report(yp, pp["centroid"], recs,
                                          cls_names)
            print("  POST=%d  centroid 12-class %5.1f%%  stroke %5.1f%%"
                  % (post, a12 * 100, a6 * 100))

    if full:
        print("== Fisher-axis direction (reference) ==")
        mis = direction_analysis(X, recs)
        mis_ps = direction_analysis(X, recs, per_stroke=True)
        for side in SIDES:
            print("  %-9s pooled CV misorder %5.1f%%   per-stroke %5.1f%%"
                  % (side, mis[side] * 100, mis_ps[side] * 100))

    print("== 36-class model (side x stroke x placement, CV) ==")
    e36 = eval36(X, recs)
    print("  side %5.1f%%   stroke %5.1f%%   side*stroke %5.1f%%"
          % (e36["acc_side"] * 100, e36["acc_stroke"] * 100,
             e36["acc_12"] * 100))
    for side in SIDES:
        s = e36["sides"][side]
        print("  %-9s placement err %5.1f%%   cross<->straight flips %4.1f%%"
              % (side, s["place_err"] * 100, s["flips"] * 100))
        print("            aim  " +
              "  ".join("%s %+.2f+-%.2f" % (pl, s["aim_stats"][pl][0],
                                            s["aim_stats"][pl][1])
                        for pl in PLACEMENTS))

    worst_flips = max(e36["sides"][s]["flips"] for s in SIDES)
    aim_mode = ("continuous" if worst_flips < 0.05 else
                "buckets" if worst_flips < 0.15 else "kinematic")
    print("== gates ==")
    print("  side accuracy   %5.1f%%  (gate >= 97%%)"
          % (e36["acc_side"] * 100))
    print("  stroke accuracy %5.1f%%  (gate >= 85%%, target 90%%)"
          % (e36["acc_stroke"] * 100))
    print("  aim mode: %s (worst cross<->straight flip rate %.1f%%)"
          % (aim_mode, worst_flips * 100))
    return {"X": X, "e36": e36, "aim_mode": aim_mode}


# ---- export -------------------------------------------------------------

def export(recs, analysis):
    X, e36 = analysis["X"], analysis["e36"]
    Xs, mean, std = standardize(X, X)
    W, b = logreg_fit(Xs, e36["y36"], len(e36["names"]), iters=600)

    model = {
        "version": 1,
        "type": "logreg36",
        "config": {"e_trig": clf.E_TRIG, "e_end": clf.E_END,
                   "pre": clf.PRE, "post": clf.POST,
                   "feat_t": clf.FEAT_T, "subtract_gravity": True},
        "classes": e36["names"],
        "feat_mean": [round(v, 6) for v in mean],
        "feat_std": [round(v, 6) for v in std],
        "logreg": {"W": [[round(v, 6) for v in row] for row in W],
                   "b": [round(v, 6) for v in b]},
        "meta": {"trained_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                 "n_samples": len(recs),
                 "cv_acc_side": round(float(e36["acc_side"]), 4),
                 "cv_acc_stroke": round(float(e36["acc_stroke"]), 4),
                 "cv_acc_12": round(float(e36["acc_12"]), 4),
                 "cv_flips": {s: round(float(e36["sides"][s]["flips"]), 4)
                              for s in SIDES},
                 "aim_mode": analysis["aim_mode"]},
    }

    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model, f)
    print("wrote %s" % MODEL_PATH)

    # per-record expected outputs for test_classifier.py, computed through
    # the runtime classifier itself so the replay harness compares live
    # capture against exactly what serving will produce.
    sc = clf.StrokeClassifier(MODEL_PATH)
    preds = []
    agree = 0
    for i, r in enumerate(recs):
        res = sc.classify(r["accel"], r["t0"])
        cls = res["side"] + ":" + res["stroke"]
        agree += cls == r["cls12"]
        preds.append({"i": i, "label": r["label"], "stroke": r["stroke"],
                      "t0": r["t0"], "pred": cls,
                      "aim": None if res["aim"] is None
                      else round(res["aim"], 4),
                      "speed": round(res["speed"], 4)})
    with open(PRED_PATH, "w", encoding="utf-8") as f:
        json.dump(preds, f)
    print("wrote %s  (training-set agreement %.1f%%)"
          % (PRED_PATH, 100.0 * agree / len(recs)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", action="store_true",
                    help="fit on all data and write model.json")
    ap.add_argument("--full", action="store_true",
                    help="include slow extras (DTW, A/B, POST sweep)")
    args = ap.parse_args()

    recs = load_records()
    analysis = run_analysis(recs, full=args.full)
    if args.export:
        export(recs, analysis)


if __name__ == "__main__":
    main()
