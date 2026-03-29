#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import run_benchmark as rb

# Add new trackers here as they become supported in run_benchmark.py.
CONFIGURED_TRACKERS: List[str] = ["bytetrack", "ocsort", "mcbyte"]

# Metrics shown in the comparison table (rows).
TABLE_METRICS: List[str] = ["HOTA", "IDF1", "MOTA", "DetRe", "DetPr", "IDSW", "Frag"]

# Lower-is-better metrics; all others are treated as higher-is-better.
LOWER_IS_BETTER = {"IDSW", "Frag"}


def _run(cmd: List[str]) -> None:
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _read_summary(summary_file: Path) -> Dict[str, str]:
    if not summary_file.is_file():
        raise FileNotFoundError(f"Missing TrackEval summary: {summary_file}")
    lines = [line.strip() for line in summary_file.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Malformed TrackEval summary (expected header+values): {summary_file}")
    headers = lines[0].split()
    values = lines[1].split()
    return dict(zip(headers, values))


def _summary_file_for_tracker(
    trackers_root: Path,
    benchmark: str,
    split: str,
    tracker: str,
    detection_source: str,
) -> Path:
    tracker_output_name = rb._default_tracker_output_name(tracker, detection_source)
    return trackers_root / f"{benchmark}-{split}" / tracker_output_name / "pedestrian_summary.txt"


def _best_trackers(metric: str, per_tracker: Dict[str, str]) -> str:
    values = []
    for tracker_name, raw in per_tracker.items():
        try:
            values.append((tracker_name, float(raw)))
        except (TypeError, ValueError):
            continue

    if not values:
        return "NA"

    if metric in LOWER_IS_BETTER:
        best_value = min(v for _, v in values)
    else:
        best_value = max(v for _, v in values)

    winners = [name for name, value in values if value == best_value]
    return ",".join(winners)


def _print_comparison_table(metrics_by_tracker: Dict[str, Dict[str, str]], trackers: List[str]) -> None:
    header = ["Metric", *trackers, "Best"]
    rows: List[List[str]] = []
    for metric in TABLE_METRICS:
        row_values = {tracker: metrics_by_tracker.get(tracker, {}).get(metric, "NA") for tracker in trackers}
        rows.append([metric, *[row_values[t] for t in trackers], _best_trackers(metric, row_values)])

    widths = [len(col) for col in header]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    print()
    print("[COMPARISON] Tracker metrics (higher is better except IDSW/Frag)")
    print(fmt_row(header))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all configured trackers on a dataset and print a metric comparison table."
    )
    parser.add_argument(
        "dataset",
        help=(
            "Dataset name or path (e.g. 'dancetrack', 'mot17', 'mot20', "
            "'sportsmot', 'teamtrack', 'basketballmot', or a dataset root/split path)."
        ),
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=CONFIGURED_TRACKERS,
        help=f"Tracker list to compare (default: {' '.join(CONFIGURED_TRACKERS)}).",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running benchmarks and only read existing TrackEval summaries.",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Force re-running trackers even if existing TrackEval summaries are present.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for child script calls (default: current interpreter).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trackers = args.trackers
    if not trackers:
        raise RuntimeError("No trackers provided.")

    dataset_root, split = rb._resolve_dataset_root_and_split(dataset_path=args.dataset, split_override=None)
    benchmark = rb._infer_benchmark_from_dataset_root(dataset_root)
    gt_root = rb._default_trackeval_gt_root()
    trackers_root = rb._default_trackers_root()
    seqs = rb._read_seqs(gt_root / "seqmaps" / f"{benchmark}-{split}.txt")
    detection_source = rb._infer_detection_source(dataset_root=dataset_root, split=split, seqs=seqs)

    print(
        f"[INFO] Compare config: benchmark={benchmark} dataset_root={dataset_root} split={split} "
        f"detection_source={detection_source} trackers={','.join(trackers)}"
    )

    if not args.skip_run:
        for tracker in trackers:
            summary_file = _summary_file_for_tracker(
                trackers_root=trackers_root,
                benchmark=benchmark,
                split=split,
                tracker=tracker,
                detection_source=detection_source,
            )
            if summary_file.is_file() and not args.force_run:
                print(f"[INFO] Reusing cached results for tracker={tracker} -> {summary_file}")
                continue
            cmd = [
                args.python,
                str(Path(__file__).resolve().parent / "run_benchmark.py"),
                "simple",
                tracker,
                args.dataset,
                "--benchmark",
                benchmark,
            ]
            _run(cmd)

    metrics_by_tracker: Dict[str, Dict[str, str]] = {}
    for tracker in trackers:
        summary_file = _summary_file_for_tracker(
            trackers_root=trackers_root,
            benchmark=benchmark,
            split=split,
            tracker=tracker,
            detection_source=detection_source,
        )
        metrics_by_tracker[tracker] = _read_summary(summary_file)

    _print_comparison_table(metrics_by_tracker=metrics_by_tracker, trackers=trackers)


if __name__ == "__main__":
    main()
