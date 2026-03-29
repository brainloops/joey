#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import run_benchmark as rb

CONFIGURED_TRACKERS: List[str] = ["bytetrack", "ocsort", "mcbyte"]
TABLE_METRICS: List[str] = ["HOTA", "IDF1", "MOTA", "DetRe", "DetPr", "IDSW", "Frag"]
LOWER_IS_BETTER = {"IDSW", "Frag"}


def _run(cmd: List[str]) -> None:
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _read_summary(summary_file: Path) -> Dict[str, str]:
    if not summary_file.is_file():
        return {}
    lines = [line.strip() for line in summary_file.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return {}
    headers = lines[0].split()
    values = lines[1].split()
    return dict(zip(headers, values))


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


def _summary_file_for_tracker(
    trackers_root: Path,
    benchmark: str,
    split: str,
    tracker: str,
    detection_source: str,
) -> Path:
    tracker_output_name = rb._default_tracker_output_name(tracker, detection_source)
    return trackers_root / f"{benchmark}-{split}" / tracker_output_name / "pedestrian_summary.txt"


def _canonical_datasets(raw: List[str]) -> List[str]:
    if raw:
        return raw
    return ["dancetrack", "mot17", "mot20", "sportsmot", "teamtrack"]


def _print_cross_dataset_table(
    rows: List[Tuple[str, str, str, Dict[str, str]]],
    trackers: List[str],
) -> None:
    header = ["Dataset", "Split", "Metric", *trackers, "Best"]
    table_rows: List[List[str]] = []
    for dataset_label, split, metric, row_values in rows:
        best = _best_trackers(metric, row_values)
        table_rows.append([dataset_label, split, metric, *[row_values.get(t, "NA") for t in trackers], best])

    widths = [len(col) for col in header]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    print()
    print("[CROSS-DATASET] Tracker metrics (higher is better except IDSW/Frag)")
    print(fmt_row(header))
    print("-+-".join("-" * w for w in widths))
    for row in table_rows:
        print(fmt_row(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read TrackEval summaries across multiple datasets and print a combined comparison table."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help=(
            "Dataset names/paths. Defaults to: dancetrack mot17 mot20 sportsmot teamtrack. "
            "Examples: dancetrack mot17 mot20 sportsmot teamtrack"
        ),
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=CONFIGURED_TRACKERS,
        help=f"Tracker list to compare (default: {' '.join(CONFIGURED_TRACKERS)}).",
    )
    parser.add_argument(
        "--run-missing",
        action="store_true",
        help="Run missing tracker outputs before reading summaries.",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Re-run tracker outputs for all datasets/trackers (implies --run-missing).",
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

    datasets = _canonical_datasets(args.datasets)
    gt_root = rb._default_trackeval_gt_root()
    trackers_root = rb._default_trackers_root()

    collected_rows: List[Tuple[str, str, str, Dict[str, str]]] = []
    for dataset in datasets:
        dataset_root, split = rb._resolve_dataset_root_and_split(dataset_path=dataset, split_override=None)
        benchmark = rb._infer_benchmark_from_dataset_root(dataset_root)
        seqmap_file = gt_root / "seqmaps" / f"{benchmark}-{split}.txt"
        seqs = rb._read_seqs(seqmap_file)
        detection_source = rb._infer_detection_source(dataset_root=dataset_root, split=split, seqs=seqs)

        print(
            f"[INFO] Dataset config: dataset={dataset} benchmark={benchmark} split={split} "
            f"detection_source={detection_source}"
        )

        should_run = args.run_missing or args.force_run
        if should_run:
            for tracker in trackers:
                summary_file = _summary_file_for_tracker(
                    trackers_root=trackers_root,
                    benchmark=benchmark,
                    split=split,
                    tracker=tracker,
                    detection_source=detection_source,
                )
                if summary_file.is_file() and not args.force_run:
                    print(f"[INFO] Reusing cached results for {dataset}:{tracker} -> {summary_file}")
                    continue

                cmd = [
                    args.python,
                    str(Path(__file__).resolve().parent / "run_benchmark.py"),
                    "simple",
                    tracker,
                    dataset,
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

        dataset_label = f"{benchmark}"
        for metric in TABLE_METRICS:
            row_values = {tracker: metrics_by_tracker.get(tracker, {}).get(metric, "NA") for tracker in trackers}
            collected_rows.append((dataset_label, split, metric, row_values))

    _print_cross_dataset_table(rows=collected_rows, trackers=trackers)


if __name__ == "__main__":
    main()
