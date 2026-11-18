"""Export final model + scaler + thresholds + configs + manifest.json -> artifacts/

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
