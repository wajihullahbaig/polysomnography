"""Command line entry point.

    python -m psng {train,validate,all,report,sweep} [--settings PATH] [--no-resume] [--docs DIR] [--model NAME] [--k-min N] [--k-max N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from psng.settings import Settings

STAGES = {"train": ["training"], "validate": ["validation"], "all": ["training", "validation"], "report": [], "sweep": [], "compare": []}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="psng", description="Cluster polysomnography signal windows.")
    parser.add_argument("stage", choices=STAGES, help="train / validate / all run the pipeline; report compares the stages; sweep scans the number of clusters; "
                             "compare runs every model in PCA / raw / autoencoder feature spaces plus a supervised reference")
    parser.add_argument("--settings", default="configs/settings.yaml", help="path to the settings YAML")
    parser.add_argument("--no-resume", action="store_true", help="ignore cached features and fitted models")
    parser.add_argument("--docs", default=None, help="report: also copy the figures into DIR/figures (for the README)")
    parser.add_argument("--model", default="kmeans", help="sweep: kmeans | minibatch | bisecting | gmm")
    parser.add_argument("--k-min", type=int, default=2, help="sweep: smallest number of clusters")
    parser.add_argument("--k-max", type=int, default=12, help="sweep: largest number of clusters")
    args = parser.parse_args(argv)

    settings = Settings.load(args.settings)
    if args.no_resume:
        settings.run.resume = False

    from psng.pipeline import Pipeline
    from psng.report import build_report, compare_spaces, copy_figures, sweep

    for stage in STAGES[args.stage]:
        Pipeline(settings, stage).run()
    if args.stage == "sweep":
        print(sweep(settings, args.model, list(range(args.k_min, args.k_max + 1))).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.stage in ("all", "report"):
        print(build_report(settings.run.output_dir, args.docs, settings))
    if args.stage == "compare":
        print(compare_spaces(settings)[1])
        if args.docs:
            copy_figures(settings.run.output_dir, Path(args.docs), [])


if __name__ == "__main__":
    main()
