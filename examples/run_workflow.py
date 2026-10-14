"""Run the same persistent workflow used by the notebook control center."""

import argparse
from pathlib import Path
from sersd2d_superraman.workflow import Workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/workflow_demo.json")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run = Workflow(project, project / args.config)
    run.prepare()
    run.train_translations()
    run.train_classifier()
    run.build_bundle()
    run.evaluate()
    print(run.status())


if __name__ == "__main__":
    main()
