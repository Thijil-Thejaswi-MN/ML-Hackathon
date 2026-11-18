"""Tune HMM parameters on train OOF probs, confirm on student-test -> results/hmm/decoder_tuning.csv, artifacts/hmm_config.json

OWNER: M3.  See README Part 3 for the contract and Part 4 for how to run.
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model")
    args = ap.parse_args()
    raise NotImplementedError("M3: implement; write outputs + config + seed + wall-clock next to results")


if __name__ == "__main__":
    main()
