"""Command line entry point: python -m psng {train,validate,all} [--settings PATH] [--no-resume]"""

from __future__ import annotations

import argparse

from psng.pipeline import Pipeline
from psng.settings import Settings

STAGES = {"train": ["training"], "validate": ["validation"], "all": ["training", "validation"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="psng", description="Cluster polysomnography signal windows.")
    parser.add_argument("stage", choices=STAGES, help="which stage(s) to run")
    parser.add_argument("--settings", default="configs/settings.yaml", help="path to the settings YAML")
    parser.add_argument("--no-resume", action="store_true", help="ignore cached features and fitted models")
    args = parser.parse_args(argv)

    settings = Settings.load(args.settings)
    if args.no_resume:
        settings.run.resume = False
    for stage in STAGES[args.stage]:
        Pipeline(settings, stage).run()


if __name__ == "__main__":
    main()
