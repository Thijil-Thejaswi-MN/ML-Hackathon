"""Classifier-only vs classifier+HMM, per trip + overall, failure cases -> results/hmm/comparison.csv

OWNER: M3.  Run after 06_tune_decoder.py (uses artifacts/hmm_config.json):
    python scripts/07_eval_hmm.py --model lightgbm_tuned
Decodes the student_test trips once with the tuned decoder and compares with the classifier-only argmax:
Top-1 (all / reliable observations), route continuity (raw and after stitching), arc-set F1 against the reference
route, decode time. Failure cases = trips where the HMM loses accuracy, with the longest wrong stretches.
Outputs: results/hmm/comparison.csv, comparison_summary.json, failure_cases.csv, figures/hmm/*.png
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from hmm_eval_common import ROOT, TripCache, evaluate_decoder, load_probs, pooled


def wrong_stretches(pred, truth, min_len=5):
    """Contiguous runs of wrong observations (start, length)."""
    bad = np.r_[False, pred != truth, False]
    d = np.diff(bad.astype(int))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends) if e - s >= min_len]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="student_test")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "artifacts/hmm_config.json").read_text())
    t0 = time.time()
    cache = TripCache(args.split, load_probs(args.split, args.model))
    df = evaluate_decoder(cache, cfg, keep_arcs=True)
    summ = pooled(df)
    out, fig_dir = ROOT / "results/hmm", ROOT / "figures/hmm"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fails = []
    for it, (_, r) in zip(cache.items, df.iterrows()):
        for s, n in sorted(wrong_stretches(r["_hmm"], it["truth"]), key=lambda x: -x[1])[:3]:
            fails.append({"trip_id": r.trip_id, "start_obs": s, "length": n,
                          "classifier_correct_in_stretch": float((it["cls"][s:s + n] == it["truth"][s:s + n]).mean()),
                          "hmm_minus_cls_trip_top1": r.hmm_top1 - r.cls_top1})
    fail = pd.DataFrame(fails).sort_values("length", ascending=False)
    fail.to_csv(out / "failure_cases.csv", index=False)

    table = df.drop(columns=["_hmm", "_route_arcs"])
    table.insert(1, "partition", args.split)
    table.to_csv(out / "comparison.csv", index=False)
    (out / "comparison_summary.json").write_text(json.dumps({
        "model_probs": args.model, "split": args.split, "decoder_config": cfg, "overall": summ,
        "trips_improved": int((df.hmm_top1 > df.cls_top1).sum()), "trips_worse": int((df.hmm_top1 < df.cls_top1).sum()),
        "wall_clock_s": round(time.time() - t0, 1)}, indent=1))

    x = np.arange(len(df))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    w = 0.38
    for ax, (a, b, title) in zip(axs, [("cls_top1", "hmm_top1", "Top-1 accuracy per trip"),
                                       ("cls_continuity", "hmm_continuity", "Route continuity per trip (strict)")]):
        ax.bar(x - w / 2, df[a], w, label="classifier only", color="#f28e1c")
        ax.bar(x + w / 2, df[b], w, label="classifier + HMM", color="#d62728")
        ax.set(xticks=x, xticklabels=[t[-2:] for t in df.trip_id], xlabel="trip id (last 2 digits)",
               title=f"{title}  [{args.split}]", ylim=(max(0, min(df[a].min(), df[b].min()) - 0.1), 1.0))
        ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(fig_dir / "01_hmm_vs_classifier_per_trip.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Top-1\n(all obs)", "Top-1\n(reliable)", "Continuity\n(raw)", "Arc-set F1\nvs reference"]
    cls = [summ["cls_top1"], summ["cls_top1_reliable"], summ["cls_continuity"], summ["cls_arc_f1"]]
    hmm = [summ["hmm_top1"], summ["hmm_top1_reliable"], summ["hmm_continuity"], summ["hmm_arc_f1"]]
    xx = np.arange(4)
    b1 = ax.bar(xx - w / 2, cls, w, label="classifier only", color="#f28e1c")
    b2 = ax.bar(xx + w / 2, hmm, w, label="classifier + HMM", color="#d62728")
    ax.bar_label(b1, fmt="%.3f", fontsize=7); ax.bar_label(b2, fmt="%.3f", fontsize=7)
    ax.set(xticks=xx, xticklabels=labels, ylim=(0.5, 1.02), title=f"Overall: classifier-only vs HMM  [{args.split}]")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(fig_dir / "02_hmm_vs_classifier_overall.png", dpi=130); plt.close(fig)

    print(table[["trip_id", "n_obs", "cls_top1", "hmm_top1", "cls_continuity", "hmm_continuity",
                 "stitched_continuity", "decode_seconds"]].round(4).to_string(index=False))
    print(f"[overall {args.split}] top1 {summ['cls_top1']:.4f} -> {summ['hmm_top1']:.4f} | reliable "
          f"{summ['cls_top1_reliable']:.4f} -> {summ['hmm_top1_reliable']:.4f} | continuity {summ['cls_continuity']:.3f} -> "
          f"{summ['hmm_continuity']:.3f} (stitched {summ['stitched_continuity']:.3f}) | decode {summ['decode_ms_per_1000_obs']:.0f} ms/1000 obs")


if __name__ == "__main__":
    main()
