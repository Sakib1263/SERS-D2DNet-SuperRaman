"""Run from any directory after installation; supply a fresh output location."""
import argparse
from sersd2d_superraman.cli import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(["demo", "--output", args.output])
