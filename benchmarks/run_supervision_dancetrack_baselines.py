#!/usr/bin/env python3

from __future__ import annotations

import argparse
import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

np = None
sv = None
tk = None
mcbyte_adapter = None


@dataclass
class DetectionRow:
    frame: int
    x: float
    y: float
    w: float
    h: float
    conf: float
    class_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run trackers ByteTrack/OCSORT on DanceTrack and export TrackEval-format results."
    )
    parser.add_argument(
        "--tracker",
        choices=["bytetrack", "ocsort", "mcbyte", "both", "all"],
        default="both",
        help="Which tracker(s) to run.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="DanceTrack split to run (train, val, test).",
    )
    parser.add_argument(
        "--detection-source",
        choices=["gt", "det"],
        default="gt",
        help="Input detections source per sequence.",
    )
    parser.add_argument(
        "--dataset-root",
        default="benchmarks/datasets/DanceTrack",
        help="Root path to DanceTrack dataset.",
    )
    parser.add_argument(
        "--trackeval-gt-root",
        default="benchmarks/trackeval_data/gt/mot_challenge",
        help="TrackEval GT root containing seqmaps.",
    )
    parser.add_argument(
        "--trackers-root",
        default="benchmarks/trackeval_data/trackers/mot_challenge",
        help="TrackEval trackers root for output files.",
    )
    parser.add_argument(
        "--bytetrack-name",
        default="bytetrack_baseline",
        help="Output tracker folder name for ByteTrack.",
    )
    parser.add_argument(
        "--ocsort-name",
        default="ocsort_baseline",
        help="Output tracker folder name for OCSort.",
    )
    parser.add_argument(
        "--mcbyte-name",
        default="mcbyte_baseline",
        help="Output tracker folder name for McByte.",
    )
    parser.add_argument(
        "--min-det-score",
        type=float,
        default=0.0,
        help="Filter detections by confidence before tracking.",
    )
    return parser.parse_args()


def read_seqmap(seqmap_file: Path) -> List[str]:
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


def read_seq_info(seqinfo_file: Path) -> Tuple[int, int, int]:
    if not seqinfo_file.is_file():
        raise FileNotFoundError(f"Missing seqinfo.ini: {seqinfo_file}")
    cfg = configparser.ConfigParser()
    cfg.read(seqinfo_file)
    seq_cfg = cfg["Sequence"]
    seq_len = int(seq_cfg["seqLength"])
    width = int(seq_cfg.get("imWidth", 0))
    height = int(seq_cfg.get("imHeight", 0))
    return seq_len, width, height


def parse_mot_rows(path: Path, source: str, min_score: float) -> Dict[int, List[DetectionRow]]:
    if not path.is_file():
        raise FileNotFoundError(f"Detection file not found: {path}")

    by_frame: Dict[int, List[DetectionRow]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) < 6:
            continue

        frame = int(float(fields[0]))
        x = float(fields[2])
        y = float(fields[3])
        w = float(fields[4])
        h = float(fields[5])

        if source == "gt":
            keep_flag = int(float(fields[6])) if len(fields) > 6 else 1
            class_id = int(float(fields[7])) if len(fields) > 7 else 1
            if keep_flag == 0 or class_id != 1:
                continue
            conf = 1.0
        else:
            conf = float(fields[6]) if len(fields) > 6 else 1.0
            class_id = int(float(fields[7])) if len(fields) > 7 else 1
            if class_id != 1:
                continue

        if conf < min_score:
            continue

        row = DetectionRow(frame=frame, x=x, y=y, w=w, h=h, conf=conf, class_id=1)
        by_frame.setdefault(frame, []).append(row)
    return by_frame


def to_detections(rows: Iterable[DetectionRow]) -> sv.Detections:
    rows_list = list(rows)
    if not rows_list:
        return sv.Detections.empty()

    xyxy = np.asarray(
        [[r.x, r.y, r.x + r.w, r.y + r.h] for r in rows_list],
        dtype=np.float32,
    )
    confidence = np.asarray([r.conf for r in rows_list], dtype=np.float32)
    class_id = np.asarray([r.class_id for r in rows_list], dtype=np.int32)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def create_tracker(name: str):
    if name == "bytetrack":
        if tk is None:
            raise RuntimeError(
                "Missing dependency 'trackers'. Install with:\n"
                "  pip install trackers"
            )
        for attr in ("ByteTrackTracker", "BYTESORTTracker", "BYTETrackTracker"):
            if hasattr(tk, attr):
                return getattr(tk, attr)()
        raise AttributeError(
            "trackers package does not expose a ByteTrack tracker class in this version."
        )

    if name == "ocsort":
        if tk is None:
            raise RuntimeError(
                "Missing dependency 'trackers'. Install with:\n"
                "  pip install trackers"
            )
        for attr in ("OCSORTTracker", "OCSortTracker", "OcSortTracker"):
            if hasattr(tk, attr):
                return getattr(tk, attr)()
        raise AttributeError(
            "trackers package does not expose an OCSORT tracker class in this version."
        )

    if name == "mcbyte":
        if mcbyte_adapter is None:
            raise RuntimeError(
                "Missing dependency 'yolox' (McByte package). Install McByte first, e.g.:\n"
                "  python -m pip install -e /home/david/projects/McByte --no-build-isolation"
            )
        cfg = mcbyte_adapter.McByteRfDetrConfig()
        return mcbyte_adapter.McByteRfDetrAdapter(config=cfg, save_folder=".")

    raise ValueError(f"Unsupported tracker name: {name}")


def update_tracker(tracker, detections: sv.Detections) -> sv.Detections:
    if hasattr(tracker, "update_with_detections"):
        return tracker.update_with_detections(detections)
    if hasattr(tracker, "update"):
        return tracker.update(detections)
    raise AttributeError("Tracker object has no update method compatible with supervision Detections.")


def extract_tracker_ids(detections: sv.Detections) -> np.ndarray:
    tracker_ids = getattr(detections, "tracker_id", None)
    if tracker_ids is None and hasattr(detections, "data"):
        data = detections.data if detections.data is not None else {}
        tracker_ids = data.get("tracker_id")
    if tracker_ids is None:
        return np.asarray([], dtype=np.int64)
    return np.asarray(tracker_ids)


def export_sequence_results(
    out_file: Path,
    tracker_name: str,
    seq_name: str,
    seq_length: int,
    detections_by_frame: Dict[int, List[DetectionRow]],
) -> Tuple[int, int]:
    tracker = create_tracker(tracker_name)
    lines: List[str] = []
    total_input_dets = 0

    for frame_idx in range(1, seq_length + 1):
        frame_rows = detections_by_frame.get(frame_idx, [])
        total_input_dets += len(frame_rows)
        frame_dets = to_detections(frame_rows)
        tracked = update_tracker(tracker, frame_dets)
        ids = extract_tracker_ids(tracked)
        if len(ids) == 0 or len(tracked.xyxy) == 0:
            continue

        # TrackEval requires unique, valid tracker IDs per frame.
        # Some tracker backends may return -1 for unassigned detections.
        best_by_track_id = {}
        for i, track_id_raw in enumerate(ids):
            if track_id_raw is None:
                continue
            try:
                track_id = int(track_id_raw)
            except Exception:
                continue
            if track_id < 0:
                continue

            xyxy = tracked.xyxy[i]
            x = float(xyxy[0])
            y = float(xyxy[1])
            w = float(xyxy[2] - xyxy[0])
            h = float(xyxy[3] - xyxy[1])
            score = 1.0
            if getattr(tracked, "confidence", None) is not None and len(tracked.confidence) > i:
                try:
                    score = float(tracked.confidence[i])
                except Exception:
                    score = 1.0

            prev = best_by_track_id.get(track_id)
            if prev is None or score > prev["score"]:
                best_by_track_id[track_id] = {"x": x, "y": y, "w": w, "h": h, "score": score}

        for track_id in sorted(best_by_track_id.keys()):
            box = best_by_track_id[track_id]
            lines.append(
                f"{frame_idx},{track_id},{box['x']:.6f},{box['y']:.6f},{box['w']:.6f},{box['h']:.6f},1,-1,-1,-1"
            )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(
        f"[DONE] tracker={tracker_name} seq={seq_name} inputs={total_input_dets} outputs={len(lines)} -> {out_file}"
    )
    return total_input_dets, len(lines)


def _rows_to_xyxy_conf(rows: Iterable[DetectionRow]) -> np.ndarray:
    rows_list = list(rows)
    if not rows_list:
        return np.zeros((0, 5), dtype=np.float32)
    out = np.zeros((len(rows_list), 5), dtype=np.float32)
    for i, r in enumerate(rows_list):
        out[i, 0] = float(r.x)
        out[i, 1] = float(r.y)
        out[i, 2] = float(r.x + r.w)
        out[i, 3] = float(r.y + r.h)
        out[i, 4] = float(r.conf)
    return out


def _read_frame_or_blank(seq_root: Path, frame_idx: int, width: int, height: int):
    import cv2

    candidate_exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    img_dir = seq_root / "img1"
    for ext in candidate_exts:
        p = img_dir / f"{frame_idx:08d}{ext}"
        if p.is_file():
            frame = cv2.imread(str(p))
            if frame is not None:
                return frame
    h = max(1, height)
    w = max(1, width)
    return np.zeros((h, w, 3), dtype=np.uint8)


def export_sequence_results_mcbyte(
    out_file: Path,
    seq_root: Path,
    seq_name: str,
    seq_length: int,
    seq_width: int,
    seq_height: int,
    detections_by_frame: Dict[int, List[DetectionRow]],
) -> Tuple[int, int]:
    tracker = create_tracker("mcbyte")
    lines: List[str] = []
    total_input_dets = 0

    for frame_idx in range(1, seq_length + 1):
        frame_rows = detections_by_frame.get(frame_idx, [])
        total_input_dets += len(frame_rows)
        frame_img = _read_frame_or_blank(seq_root=seq_root, frame_idx=frame_idx, width=seq_width, height=seq_height)
        dets_xyxy_conf = _rows_to_xyxy_conf(frame_rows)
        tracked = tracker.step(frame_img, dets_xyxy_conf)
        for tr in tracked:
            track_id = int(tr["track_id"])
            x, y, w, h = tr["tlwh"]
            score = float(tr["score"])
            if track_id < 0:
                continue
            lines.append(
                f"{frame_idx},{track_id},{x:.6f},{y:.6f},{w:.6f},{h:.6f},{score:.6f},-1,-1,-1"
            )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(
        f"[DONE] tracker=mcbyte seq={seq_name} inputs={total_input_dets} outputs={len(lines)} -> {out_file}"
    )
    return total_input_dets, len(lines)


def run_tracker_over_split(
    tracker_name: str,
    tracker_folder_name: str,
    split: str,
    seqs: List[str],
    dataset_root: Path,
    trackers_root: Path,
    detection_source: str,
    min_det_score: float,
) -> None:
    out_dir = trackers_root / f"DanceTrack-{split}" / tracker_folder_name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Running {tracker_name} on split={split}, output={out_dir}")

    total_in = 0
    total_out = 0
    for seq in seqs:
        seq_root = dataset_root / split / seq
        seqinfo = seq_root / "seqinfo.ini"
        seq_len, seq_w, seq_h = read_seq_info(seqinfo)
        source_file = seq_root / ("gt/gt.txt" if detection_source == "gt" else "det/det.txt")
        dets_by_frame = parse_mot_rows(source_file, detection_source, min_det_score)
        seq_out = out_dir / f"{seq}.txt"
        if tracker_name == "mcbyte":
            in_count, out_count = export_sequence_results_mcbyte(
                out_file=seq_out,
                seq_root=seq_root,
                seq_name=seq,
                seq_length=seq_len,
                seq_width=seq_w,
                seq_height=seq_h,
                detections_by_frame=dets_by_frame,
            )
        else:
            in_count, out_count = export_sequence_results(
                out_file=seq_out,
                tracker_name=tracker_name,
                seq_name=seq,
                seq_length=seq_len,
                detections_by_frame=dets_by_frame,
            )
        total_in += in_count
        total_out += out_count

    print(
        f"[SUMMARY] tracker={tracker_name} split={split} sequences={len(seqs)} "
        f"input_detections={total_in} output_rows={total_out}"
    )


def main() -> None:
    args = parse_args()
    global np
    global sv
    global tk
    global mcbyte_adapter

    try:
        import numpy as np_module
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'numpy'. Install with:\n"
            "  pip install numpy"
        ) from exc
    np = np_module

    trackers_needed = args.tracker in ("bytetrack", "ocsort", "both", "all")
    if trackers_needed:
        try:
            import trackers as tk_module
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency 'trackers'. Install with:\n"
                "  pip install trackers"
            ) from exc
    else:
        tk_module = None

    # trackers works with supervision.Detections objects for detector/tracker interchange.
    if trackers_needed:
        try:
            import supervision as sv_module
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency 'supervision'. Install with:\n"
                "  pip install supervision"
            ) from exc
    else:
        sv_module = None

    mcbyte_needed = args.tracker in ("mcbyte", "all")
    if mcbyte_needed:
        try:
            from yolox.tracker import rfdetr_adapter as mcbyte_adapter_module
        except ModuleNotFoundError as exc:
            missing_pkg = exc.name or "unknown package"
            raise RuntimeError(
                "McByte import failed due to a missing dependency.\n"
                f"Missing package: {missing_pkg}\n"
                "Install McByte runtime dependencies, for example:\n"
                "  python -m pip install -r /home/david/projects/McByte/requirements.txt"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "Missing McByte adapter (yolox.tracker.rfdetr_adapter). "
                "Install McByte in this environment first."
            ) from exc
    else:
        mcbyte_adapter_module = None

    sv = sv_module
    tk = tk_module
    mcbyte_adapter = mcbyte_adapter_module

    dataset_root = Path(args.dataset_root).resolve()
    trackeval_gt_root = Path(args.trackeval_gt_root).resolve()
    trackers_root = Path(args.trackers_root).resolve()

    seqmap = trackeval_gt_root / "seqmaps" / f"DanceTrack-{args.split}.txt"
    seqs = read_seqmap(seqmap)

    trackers_to_run: List[Tuple[str, str]]
    if args.tracker == "all":
        trackers_to_run = [
            ("bytetrack", args.bytetrack_name),
            ("ocsort", args.ocsort_name),
            ("mcbyte", args.mcbyte_name),
        ]
    elif args.tracker == "both":
        trackers_to_run = [("bytetrack", args.bytetrack_name), ("ocsort", args.ocsort_name)]
    elif args.tracker == "bytetrack":
        trackers_to_run = [("bytetrack", args.bytetrack_name)]
    elif args.tracker == "mcbyte":
        trackers_to_run = [("mcbyte", args.mcbyte_name)]
    else:
        trackers_to_run = [("ocsort", args.ocsort_name)]

    print(f"[INFO] Detection source: {args.detection_source}")
    print(f"[INFO] Sequence count from seqmap: {len(seqs)}")
    if args.detection_source == "gt":
        print("[WARN] Using GT boxes as detector input. This is useful for pipeline validation, not fair detector-tracker benchmarking.")

    for tracker_name, out_name in trackers_to_run:
        run_tracker_over_split(
            tracker_name=tracker_name,
            tracker_folder_name=out_name,
            split=args.split,
            seqs=seqs,
            dataset_root=dataset_root,
            trackers_root=trackers_root,
            detection_source=args.detection_source,
            min_det_score=args.min_det_score,
        )


if __name__ == "__main__":
    main()
