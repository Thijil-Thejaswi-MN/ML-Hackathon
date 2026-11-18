"""Build results/model_comparison.csv + all required comparison figures (README 3.7, 5.2)

OWNER: M2.  See README Part 3 for the contract and Part 4 for how to run.
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results")
    args = ap.parse_args()
    raise NotImplementedError("M2: implement; write outputs + config + seed + wall-clock next to results")


if __name__ == "__main__":
    main()
