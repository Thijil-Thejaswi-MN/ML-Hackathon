"""Export final model + scaler + thresholds + configs + manifest.json -> artifacts/

OWNER: M2.  Run once the final model is chosen (docs/DECISIONS.md):
    python scripts/09_export_artifacts.py --tag lightgbm_tuned
Copies results/models/<tag>/model.joblib -> artifacts/model.joblib (what api/ loads), writes artifacts/threshold.json
(tau chosen on train OOF) and artifacts/manifest.json (git commit, library versions, feature-list hash, metric
snapshot). artifacts/scaler.joblib, features.json and candidate_config.json come from 02_build_dataset.py and are
checked, not rewritten. hmm_config.json is written by M3's decoder tuning.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import joblib
    import numpy as np
    from pathsnap.training import _versions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="results/models/<tag> to export, e.g. lightgbm_tuned")
    args = ap.parse_args()
    src, art = ROOT / "results/models" / args.tag, ROOT / "artifacts"
    for f in ("model.joblib", "cv.json", "test.json", "config.yaml"):
        if not (src / f).exists():
            sys.exit(f"missing {src / f} - train it first with scripts/03_train_model.py")
    for f in ("scaler.joblib", "features.json", "candidate_config.json"):
        if not (art / f).exists():
            sys.exit(f"missing artifacts/{f} - run scripts/02_build_dataset.py")

    shutil.copy2(src / "model.joblib", art / "model.joblib")
    cv = json.loads((src / "cv.json").read_text())
    te = json.loads((src / "test.json").read_text())
    (art / "threshold.json").write_text(json.dumps({"threshold": cv["threshold"],
                                                    "chosen_on": "train OOF probabilities, max macro-F1 (frozen group folds)"},
                                                   indent=1))

    # smoke test: the exported model loads and returns one probability per row on the API's feature order
    features = json.loads((art / "features.json").read_text())
    model = joblib.load(art / "model.joblib")
    scaler = joblib.load(art / "scaler.joblib")
    p = np.asarray(model.predict_proba(scaler.transform(np.zeros((3, len(features))))))
    assert p.shape[0] == 3, f"predict_proba returned shape {p.shape}"

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    manifest = {
        "model_tag": args.tag, "model_file": "model.joblib",
        "model_class": type(model).__name__, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "versions": _versions(),
        "training_data": "80 train trips (label_reliable observations); student_test used only for evaluation",
        "features_sha256": hashlib.sha256((art / "features.json").read_bytes()).hexdigest(),
        "n_features": len(features), "threshold": cv["threshold"],
        "metrics_snapshot": {"cv_top1_mean": cv["top1_mean"], "cv_macro_f1_mean": cv["macro_f1_mean"],
                             "student_test_top1": te["all"]["top1"], "student_test_macro_f1": te["reliable"]["macro_f1"],
                             "student_test_log_loss": te["reliable"]["log_loss"]},
        "model_size_mb": round((art / "model.joblib").stat().st_size / 1e6, 2),
    }
    (art / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    if manifest["model_size_mb"] > 95:
        print("[warn] model.joblib is > 95 MB: GitHub rejects files > 100 MB. Use Git LFS or a smaller model.")


if __name__ == "__main__":
    main()
