"""Train ONE model by name: group-CV on frozen folds, OOF probs, full fit, student-test eval, timing -> results/models/<name>/, results/oof, results/preds

OWNER: M2.  See README Part 3 for the contract and Part 4 for how to run.
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model")
    args = ap.parse_args()
    raise NotImplementedError("M2: implement; write outputs + config + seed + wall-clock next to results")


if __name__ == "__main__":
    main()
