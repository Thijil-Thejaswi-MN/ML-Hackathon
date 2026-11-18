"""Generate Folium HTML maps for chosen trips -> maps/

OWNER: M3.  See README Part 3 for the contract and Part 4 for how to run.
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trip")
    args = ap.parse_args()
    raise NotImplementedError("M3: implement; write outputs + config + seed + wall-clock next to results")


if __name__ == "__main__":
    main()
