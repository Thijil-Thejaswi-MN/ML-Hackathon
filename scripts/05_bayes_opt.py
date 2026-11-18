"""Optuna TPE on the selected model, group-CV on train only; save every trial; tuned-vs-untuned table -> results/bo/

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
