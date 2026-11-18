"""Train ONE model by name: group-CV on frozen folds, OOF probs, full fit, student-test eval, timing -> results/models/<name>/, results/oof, results/preds

OWNER: M2.  Run AFTER scripts/02_build_dataset.py:
    python scripts/03_train_model.py --model logreg
    python scripts/03_train_model.py --model knn
    python scripts/03_train_model.py --model lightgbm          (or --model hgb if LightGBM is unavailable)
    python scripts/03_train_model.py --model extratrees
    python scripts/03_train_model.py --model rf
    python scripts/03_train_model.py --model mlp
    python scripts/03_train_model.py --model logreg --params '{"class_weight": "balanced"}' --tag logreg_balanced
Then: python scripts/04_compare_models.py

Hyperparameters come from configs/models/<model>.yaml; --params overrides them (JSON). Every run records its
config, seed, library versions and wall-clock time next to its results (README R7).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import yaml
    from pathsnap.training import run_experiment

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="registry name, e.g. logreg knn lightgbm hgb extratrees rf mlp")
    ap.add_argument("--config", help="YAML of hyperparameters (default configs/models/<model>.yaml)")
    ap.add_argument("--params", default="{}", help="JSON overrides, e.g. '{\"C\": 0.1}'")
    ap.add_argument("--tag", help="output folder name under results/models (default = model name)")
    ap.add_argument("--train-obs-frac", type=float, default=1.0,
                    help="fraction of training observations to fit on (whole observations; for slow models)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-save-model", action="store_true")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else ROOT / "configs/models" / f"{args.model}.yaml"
    params = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    params = {**(params or {}), **json.loads(args.params)}
    summary = run_experiment(args.model, params, tag=args.tag, train_obs_frac=args.train_obs_frac, seed=args.seed,
                             save_model=not args.no_save_model, notes=args.notes)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
