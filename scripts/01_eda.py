"""EDA plots, missing-value analysis, Folium EDA map -> figures/eda, maps/ (state the partition in every title)

OWNER: M1.  See README Part 3 for the contract and Part 4 for how to run.
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root")
    args = ap.parse_args()
    raise NotImplementedError("M1: implement; write outputs + config + seed + wall-clock next to results")


if __name__ == "__main__":
    main()
