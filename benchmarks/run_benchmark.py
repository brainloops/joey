#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _benchmarks_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_trackeval_gt_root() -> Path:
    return _benchmarks_dir() / "trackeval_data" / "gt" / "mot_challenge"


def _default_trackers_root() -> Path:
    return _benchmarks_dir() / "trackeval_data" / "trackers" / "mot_challenge"


def _resolve_dataset_identifier(dataset: str) -> Path:
    # Accept a short dataset key for zero-config usage.
    key = dataset.strip().lower()
    if key in {"dancetrack", "dance-track", "dt"}:
        return (_benchmarks_dir() / "datasets" / "DanceTrack").resolve()
    return Path(dataset).expanduser().resolve()


def _run(cmd: List[str]) -> None:
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _format_compact_row(data: Dict[str, str]) -> str:
    return (
        f"{data['HOTA']:>7}  {data['IDF1']:>7}  {data['MOTA']:>7}  "
        f"{data['DetRe']:>7}  {data['DetPr']:>7}  {data['IDSW']:>7}  {data['Frag']:>7}"
    )


def _print_trackeval_compact_summary(trackers_root: Path, split: str, tracker_name: str) -> None:
    tracker_dir = trackers_root / f"DanceTrack-{split}" / tracker_name
    summary_file = tracker_dir / "pedestrian_summary.txt"
    detailed_file = tracker_dir / "pedestrian_detailed.csv"
    if not summary_file.is_file():
        print(f"[WARN] Compact summary skipped: missing {summary_file}")
        return

    lines = [ln.strip() for ln in summary_file.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        print(f"[WARN] Compact summary skipped: malformed {summary_file}")
        return
    headers = lines[0].split()
    values = lines[1].split()
    metrics = dict(zip(headers, values))

    wanted = ["HOTA", "IDF1", "MOTA", "DetRe", "DetPr", "IDSW", "Frag", "GT_Dets", "GT_IDs"]
    compact = {k: metrics.get(k, "NA") for k in wanted}

    print()
    print(f"[COMPACT] DanceTrack-{split} | tracker={tracker_name}")
    print("   HOTA     IDF1     MOTA    DetRe    DetPr     IDSW     Frag")
    print(_format_compact_row(compact))

    # Normalized helper metrics for easier intuition.
    try:
        idsw = float(compact["IDSW"])
        frag = float(compact["Frag"])
        gt_dets = float(compact["GT_Dets"])
        gt_ids = float(compact["GT_IDs"])
        idsw_per_gt_id = (idsw / gt_ids) if gt_ids > 0 else float("nan")
        idsw_per_1k_gtdet = (idsw * 1000.0 / gt_dets) if gt_dets > 0 else float("nan")
        frag_per_gt_id = (frag / gt_ids) if gt_ids > 0 else float("nan")
        print(
            f"   IDSW/GT_ID={idsw_per_gt_id:.2f}  "
            f"IDSW/1k_GTDet={idsw_per_1k_gtdet:.2f}  "
            f"Frag/GT_ID={frag_per_gt_id:.2f}"
        )
    except (TypeError, ValueError):
        pass

    print("Metric quick meanings:")
    print("  HOTA  : overall detection+association quality")
    print("  IDF1  : identity consistency F1")
    print("  MOTA  : error-focused MOT summary")
    print("  DetRe : GT objects found (recall)")
    print("  DetPr : predictions that are correct")
    print("  IDSW  : identity switches (lower better)")
    print("  Frag  : track continuity breaks")

    if not detailed_file.is_file():
        return

    rows = []
    with detailed_file.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get("seq", "")
            if not seq or seq == "COMBINED":
                continue
            try:
                hota = float(row.get("HOTA___AUC", "nan")) * 100.0
                idf1 = float(row.get("IDF1", "nan")) * 100.0
            except ValueError:
                continue
            rows.append((seq, hota, idf1))
    if not rows:
        return

    rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
    top = rows_sorted[:3]
    bottom = sorted(rows, key=lambda x: x[1])[:3]

    print("Top HOTA seqs (HOTA% / IDF1%):")
    for seq, hota, idf1 in top:
        print(f"  {seq:<14} {hota:6.2f} / {idf1:6.2f}")
    print("Bottom HOTA seqs (HOTA% / IDF1%):")
    for seq, hota, idf1 in bottom:
        print(f"  {seq:<14} {hota:6.2f} / {idf1:6.2f}")


def _read_seqs(seqmap_file: Path) -> List[str]:
    if not seqmap_file.is_file():
        raise FileNotFoundError(f"Seqmap file not found: {seqmap_file}")
    seqs: List[str] = []
    for raw in seqmap_file.read_text().splitlines():
        line = raw.strip()
        if not line or line == "name":
            continue
        seqs.append(line)
    if not seqs:
        raise RuntimeError(f"No sequences found in seqmap: {seqmap_file}")
    return seqs


def _validate_trackeval_inputs(gt_root: Path, trackers_root: Path, split: str, tracker_name: str) -> None:
    gt_split_dir = gt_root / f"DanceTrack-{split}"
    seqmap_file = gt_root / "seqmaps" / f"DanceTrack-{split}.txt"
    tracker_data_dir = trackers_root / f"DanceTrack-{split}" / tracker_name / "data"

    if not gt_split_dir.is_dir():
        raise FileNotFoundError(
            f"GT split folder missing: {gt_split_dir}\n"
            "Run: bash benchmarks/prepare_dancetrack_for_trackeval.sh"
        )
    if not seqmap_file.is_file():
        raise FileNotFoundError(
            f"Seqmap file missing: {seqmap_file}\n"
            "Run: bash benchmarks/prepare_dancetrack_for_trackeval.sh"
        )
    if not tracker_data_dir.is_dir():
        raise FileNotFoundError(
            f"Tracker data folder missing: {tracker_data_dir}\n"
            "Create tracker output files first."
        )

    seqs = _read_seqs(seqmap_file)
    missing = []
    for seq in seqs:
        pred = tracker_data_dir / f"{seq}.txt"
        if not pred.is_file():
            missing.append(pred)
    if missing:
        print(f"[ERROR] Found {len(missing)} missing sequence file(s) for tracker '{tracker_name}'.")
        for p in missing:
            print(f"[WARN] Missing prediction file: {p}")
        raise RuntimeError("Populate all required <seq>.txt files and rerun.")


def _resolve_dancetrack_root_and_split(dataset_path: str, split_override: Optional[str]) -> Tuple[Path, str]:
    path = _resolve_dataset_identifier(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    split_names = ("train", "val", "test")
    inferred_split: Optional[str] = None

    if path.name in split_names and path.is_dir():
        dataset_root = path.parent
        inferred_split = path.name
    else:
        dataset_root = path

    split = split_override or inferred_split
    if split is None:
        for candidate in ("val", "train", "test"):
            if (dataset_root / candidate).is_dir():
                split = candidate
                break

    if split is None:
        raise RuntimeError(
            f"Could not infer split from dataset path: {path}\n"
            "Expected either a split folder (.../train, .../val, .../test) "
            "or a root containing one of those folders."
        )

    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Inferred split folder not found: {split_dir}")
    return dataset_root, split


def _infer_detection_source(dataset_root: Path, split: str, seqs: List[str]) -> str:
    det_missing: List[Path] = []
    gt_missing: List[Path] = []
    for seq in seqs:
        seq_root = dataset_root / split / seq
        det_file = seq_root / "det" / "det.txt"
        gt_file = seq_root / "gt" / "gt.txt"
        if not det_file.is_file():
            det_missing.append(det_file)
        if not gt_file.is_file():
            gt_missing.append(gt_file)

    if not det_missing:
        return "det"
    if not gt_missing:
        if len(det_missing) != len(seqs):
            raise RuntimeError(
                "Found partial detector outputs: some sequences have det/det.txt and some do not.\n"
                "Standardize detector outputs across the split (or remove partial files) and rerun."
            )
        return "gt"

    preview = "\n".join(f"  - {p}" for p in det_missing[:3])
    raise RuntimeError(
        "Could not infer detection source because required MOT files are missing.\n"
        f"Missing det files (sample):\n{preview}\n"
        "Expected either all det/det.txt files (det mode) or all gt/gt.txt files (gt mode)."
    )


def _default_tracker_output_name(tracker: str, detection_source: str) -> str:
    return f"{tracker}_{detection_source}"


def cmd_detect(args: argparse.Namespace) -> None:
    script = _benchmarks_dir() / "generate_dancetrack_detections_rfdetr.py"
    cmd = [
        args.python,
        str(script),
        "--split",
        args.split,
        "--dataset-root",
        args.dataset_root,
        "--model-size",
        args.model_size,
        "--threshold",
        str(args.threshold),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.max_seqs is not None:
        cmd.extend(["--max-seqs", str(args.max_seqs)])
    _run(cmd)


def cmd_track(args: argparse.Namespace) -> None:
    script = _benchmarks_dir() / "run_supervision_dancetrack_baselines.py"
    cmd = [
        args.python,
        str(script),
        "--tracker",
        args.tracker,
        "--split",
        args.split,
        "--detection-source",
        args.detection_source,
        "--dataset-root",
        args.dataset_root,
        "--trackeval-gt-root",
        args.trackeval_gt_root,
        "--trackers-root",
        args.trackers_root,
        "--bytetrack-name",
        args.bytetrack_name,
        "--ocsort-name",
        args.ocsort_name,
        "--mcbyte-name",
        args.mcbyte_name,
        "--min-det-score",
        str(args.min_det_score),
    ]
    _run(cmd)


def _run_eval_for_tracker(
    python_bin: str,
    tracker_name: str,
    split: str,
    cores: int,
    do_preproc: str,
    metrics: List[str],
    gt_root: Path,
    trackers_root: Path,
    trackeval_verbose: bool,
) -> None:
    trackeval_runner = _root_dir() / "benchmarks/trackeval_mot_challenge_compat.py"
    if not trackeval_runner.is_file():
        raise FileNotFoundError(
            f"TrackEval runner not found: {trackeval_runner}\nExpected TrackEval under benchmarks/repos/TrackEval"
        )
    _validate_trackeval_inputs(gt_root=gt_root, trackers_root=trackers_root, split=split, tracker_name=tracker_name)
    cmd = [
        python_bin,
        str(trackeval_runner),
        "--BENCHMARK",
        "DanceTrack",
        "--SPLIT_TO_EVAL",
        split,
        "--GT_FOLDER",
        str(gt_root),
        "--TRACKERS_FOLDER",
        str(trackers_root),
        "--TRACKERS_TO_EVAL",
        tracker_name,
        "--METRICS",
        *metrics,
        "--USE_PARALLEL",
        "True",
        "--NUM_PARALLEL_CORES",
        str(cores),
        "--DO_PREPROC",
        do_preproc,
    ]
    if not trackeval_verbose:
        cmd.extend(
            [
                "--PRINT_CONFIG",
                "False",
                "--PRINT_RESULTS",
                "False",
                "--PRINT_ONLY_COMBINED",
                "True",
                "--TIME_PROGRESS",
                "False",
            ]
        )
    _run(cmd)


def cmd_eval(args: argparse.Namespace) -> None:
    gt_root = Path(args.trackeval_gt_root).resolve()
    trackers_root = Path(args.trackers_root).resolve()
    _run_eval_for_tracker(
        python_bin=args.python,
        tracker_name=args.tracker_name,
        split=args.split,
        cores=args.cores,
        do_preproc=args.do_preproc,
        metrics=args.metrics,
        gt_root=gt_root,
        trackers_root=trackers_root,
        trackeval_verbose=args.trackeval_verbose,
    )
    _print_trackeval_compact_summary(
        trackers_root=trackers_root,
        split=args.split,
        tracker_name=args.tracker_name,
    )


def cmd_run(args: argparse.Namespace) -> None:
    # 1) Track generation
    track_ns = argparse.Namespace(**vars(args))
    cmd_track(track_ns)

    # 2) TrackEval
    gt_root = Path(args.trackeval_gt_root).resolve()
    trackers_root = Path(args.trackers_root).resolve()
    if args.tracker in ("bytetrack", "both"):
        _run_eval_for_tracker(
            python_bin=args.python,
            tracker_name=args.bytetrack_name,
            split=args.split,
            cores=args.cores,
            do_preproc=args.do_preproc,
            metrics=args.metrics,
            gt_root=gt_root,
            trackers_root=trackers_root,
            trackeval_verbose=args.trackeval_verbose,
        )
        _print_trackeval_compact_summary(
            trackers_root=trackers_root,
            split=args.split,
            tracker_name=args.bytetrack_name,
        )
    if args.tracker in ("ocsort", "both"):
        _run_eval_for_tracker(
            python_bin=args.python,
            tracker_name=args.ocsort_name,
            split=args.split,
            cores=args.cores,
            do_preproc=args.do_preproc,
            metrics=args.metrics,
            gt_root=gt_root,
            trackers_root=trackers_root,
            trackeval_verbose=args.trackeval_verbose,
        )
        _print_trackeval_compact_summary(
            trackers_root=trackers_root,
            split=args.split,
            tracker_name=args.ocsort_name,
        )
    if args.tracker in ("mcbyte", "all"):
        _run_eval_for_tracker(
            python_bin=args.python,
            tracker_name=args.mcbyte_name,
            split=args.split,
            cores=args.cores,
            do_preproc=args.do_preproc,
            metrics=args.metrics,
            gt_root=gt_root,
            trackers_root=trackers_root,
            trackeval_verbose=args.trackeval_verbose,
        )
        _print_trackeval_compact_summary(
            trackers_root=trackers_root,
            split=args.split,
            tracker_name=args.mcbyte_name,
        )


def cmd_simple(args: argparse.Namespace) -> None:
    dataset_root, split = _resolve_dancetrack_root_and_split(
        dataset_path=args.dataset,
        split_override=args.split,
    )
    gt_root = Path(args.trackeval_gt_root).expanduser().resolve()
    trackers_root = Path(args.trackers_root).expanduser().resolve()
    seqmap_file = gt_root / "seqmaps" / f"DanceTrack-{split}.txt"
    seqs = _read_seqs(seqmap_file)
    detection_source = _infer_detection_source(dataset_root=dataset_root, split=split, seqs=seqs)

    run_args = argparse.Namespace(
        python=args.python,
        tracker=args.tracker,
        split=split,
        detection_source=detection_source,
        dataset_root=str(dataset_root),
        trackeval_gt_root=str(gt_root),
        trackers_root=str(trackers_root),
        bytetrack_name=args.bytetrack_name or _default_tracker_output_name("bytetrack", detection_source),
        ocsort_name=args.ocsort_name or _default_tracker_output_name("ocsort", detection_source),
        mcbyte_name=args.mcbyte_name or _default_tracker_output_name("mcbyte", detection_source),
        min_det_score=args.min_det_score,
        cores=8,
        metrics=["HOTA", "CLEAR", "Identity"],
        do_preproc="False",
        trackeval_verbose=False,
    )

    print(
        f"[INFO] Simple mode config: tracker={run_args.tracker} "
        f"dataset_root={run_args.dataset_root} split={run_args.split} "
        f"detection_source={run_args.detection_source}"
    )
    cmd_run(run_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Central benchmark runner (Python-only) for DanceTrack detection/tracking/evaluation."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for child scripts (default: current interpreter).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Generate MOT detections (det/det.txt) with RF-DETR.")
    detect.add_argument("--split", default="val", choices=["train", "val", "test"])
    detect.add_argument("--dataset-root", default="benchmarks/datasets/DanceTrack")
    detect.add_argument("--model-size", default="small", choices=["nano", "small", "medium", "large"])
    detect.add_argument("--threshold", type=float, default=0.25)
    detect.add_argument("--batch-size", type=int, default=8)
    detect.add_argument("--overwrite", action="store_true")
    detect.add_argument("--max-seqs", type=int, default=None)
    detect.set_defaults(func=cmd_detect)

    def add_track_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tracker", choices=["bytetrack", "ocsort", "mcbyte", "both", "all"], default="both")
        p.add_argument("--split", default="val")
        p.add_argument("--detection-source", choices=["gt", "det"], default="gt")
        p.add_argument("--dataset-root", default="benchmarks/datasets/DanceTrack")
        p.add_argument("--trackeval-gt-root", default="benchmarks/trackeval_data/gt/mot_challenge")
        p.add_argument("--trackers-root", default="benchmarks/trackeval_data/trackers/mot_challenge")
        p.add_argument("--bytetrack-name", default="bytetrack_baseline")
        p.add_argument("--ocsort-name", default="ocsort_baseline")
        p.add_argument("--mcbyte-name", default="mcbyte_baseline")
        p.add_argument("--min-det-score", type=float, default=0.0)

    def add_eval_args(
        p: argparse.ArgumentParser,
        include_split: bool = True,
        include_paths: bool = True,
    ) -> None:
        if include_split:
            p.add_argument("--split", default="val")
        if include_paths:
            p.add_argument("--trackeval-gt-root", default="benchmarks/trackeval_data/gt/mot_challenge")
            p.add_argument("--trackers-root", default="benchmarks/trackeval_data/trackers/mot_challenge")
        p.add_argument("--cores", type=int, default=8)
        p.add_argument("--metrics", nargs="+", default=["HOTA", "CLEAR", "Identity"])
        p.add_argument("--do-preproc", choices=["True", "False"], default="False")
        p.add_argument(
            "--trackeval-verbose",
            action="store_true",
            help="Show full native TrackEval tables/logging.",
        )

    track = subparsers.add_parser("track", help="Run trackers ByteTrack/OCSORT and write TrackEval-format outputs.")
    add_track_args(track)
    track.set_defaults(func=cmd_track)

    eval_p = subparsers.add_parser("eval", help="Run TrackEval for one tracker output folder.")
    add_eval_args(eval_p)
    eval_p.add_argument("--tracker-name", required=True, help="Tracker folder name in TrackEval trackers root.")
    eval_p.set_defaults(func=cmd_eval)

    run = subparsers.add_parser("run", help="Run tracking then TrackEval (equivalent to old shell wrapper).")
    add_track_args(run)
    add_eval_args(run, include_split=False, include_paths=False)
    run.set_defaults(func=cmd_run)

    simple = subparsers.add_parser(
        "simple",
        help="Minimal interface: pass only tracker + dataset (name or path), infer the rest.",
    )
    simple.add_argument("tracker", choices=["bytetrack", "ocsort", "mcbyte", "both", "all"])
    simple.add_argument(
        "dataset",
        help=(
            "Dataset name or path. Use 'dancetrack' for defaults, or pass a path to "
            "DanceTrack root/split folder."
        ),
    )
    simple.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Optional split override. If omitted, inferred from path naming conventions.",
    )
    simple.add_argument("--trackeval-gt-root", default=str(_default_trackeval_gt_root()))
    simple.add_argument("--trackers-root", default=str(_default_trackers_root()))
    simple.add_argument("--bytetrack-name", default=None)
    simple.add_argument("--ocsort-name", default=None)
    simple.add_argument("--mcbyte-name", default=None)
    simple.add_argument("--min-det-score", type=float, default=0.0)
    simple.set_defaults(func=cmd_simple)

    return parser


def main() -> None:
    parser = build_parser()
    argv = sys.argv[1:]

    # Convenience alias:
    #   python benchmarks/run_benchmark.py <tracker> <dataset>
    # behaves like:
    #   python benchmarks/run_benchmark.py simple <tracker> <dataset>
    if argv and argv[0] in {"bytetrack", "ocsort", "mcbyte", "both", "all"}:
        argv = ["simple", *argv]

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
