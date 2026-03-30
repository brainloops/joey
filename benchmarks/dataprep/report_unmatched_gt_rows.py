#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0.0 else 0.0


def _parse_det_file(path: Path) -> Dict[int, List[BBox]]:
    by_frame: Dict[int, List[BBox]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) < 6:
            continue
        frame = int(float(fields[0]))
        box = BBox(
            x=float(fields[2]),
            y=float(fields[3]),
            w=float(fields[4]),
            h=float(fields[5]),
        )
        by_frame.setdefault(frame, []).append(box)
    return by_frame


def _gt_rows(path: Path) -> List[Tuple[int, int, BBox]]:
    rows: List[Tuple[int, int, BBox]] = []
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return rows
    for raw in txt.splitlines():
        fields = raw.split(",")
        if len(fields) < 8:
            continue
        frame = int(float(fields[0]))
        track_id = int(float(fields[1]))
        class_id = int(float(fields[7]))
        if class_id != 1:
            continue
        rows.append(
            (
                frame,
                track_id,
                BBox(
                    x=float(fields[2]),
                    y=float(fields[3]),
                    w=float(fields[4]),
                    h=float(fields[5]),
                ),
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report GT rows that no longer match detections after filtering."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("benchmarks/datasets/BasketballMOT"),
        help="BasketballMOT root folder.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to scan.",
    )
    parser.add_argument(
        "--det-filename",
        default="det.txt",
        help="Detection filename under each sequence det/ folder.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum IoU for a GT row to be considered matched.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="Show up to N unmatched row samples per sequence.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete unmatched GT rows in-place (report-only by default).",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Backup suffix for gt.txt when --apply is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    split_root = dataset_root / args.split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split folder not found: {split_root}")

    seq_dirs = sorted([p for p in split_root.iterdir() if p.is_dir()])
    mode = "APPLY" if args.apply else "REPORT_ONLY"
    print(f"MODE,{mode}")
    print("SEQ,GT_ROWS,UNMATCHED,UNMATCHED_PCT,KEPT_AFTER")

    total_gt = 0
    total_unmatched = 0

    for seq_dir in seq_dirs:
        gt_file = seq_dir / "gt" / "gt.txt"
        det_file = seq_dir / "det" / args.det_filename
        if not gt_file.is_file() or not det_file.is_file():
            continue

        det_by_frame = _parse_det_file(det_file)
        gt_rows = _gt_rows(gt_file)
        unmatched_rows: List[Tuple[int, int, float]] = []
        matched_lines: List[str] = []
        unmatched_lines = 0

        gt_text = gt_file.read_text(encoding="utf-8")
        gt_lines = [ln for ln in gt_text.splitlines() if ln.strip()]

        for raw in gt_lines:
            fields = raw.split(",")
            if len(fields) < 8:
                matched_lines.append(raw)
                continue
            frame = int(float(fields[0]))
            track_id = int(float(fields[1]))
            class_id = int(float(fields[7]))
            if class_id != 1:
                matched_lines.append(raw)
                continue
            gt_box = BBox(
                x=float(fields[2]),
                y=float(fields[3]),
                w=float(fields[4]),
                h=float(fields[5]),
            )
            best_iou = 0.0
            for det_box in det_by_frame.get(frame, []):
                best_iou = max(best_iou, _bbox_iou(gt_box, det_box))
            if best_iou < float(args.iou_threshold):
                unmatched_rows.append((frame, track_id, best_iou))
                unmatched_lines += 1
                continue
            matched_lines.append(raw)

        gt_count = len(gt_rows)
        unmatched = len(unmatched_rows)
        kept_after = len(matched_lines)
        pct = (100.0 * unmatched / gt_count) if gt_count else 0.0
        print(f"{seq_dir.name},{gt_count},{unmatched},{pct:.2f},{kept_after}")

        if unmatched_rows and args.show_samples > 0:
            print(f"  samples ({min(args.show_samples, len(unmatched_rows))} shown):")
            for frame, track_id, best_iou in unmatched_rows[: args.show_samples]:
                print(f"    frame={frame} track={track_id} best_iou={best_iou:.3f}")

        if args.apply and unmatched_lines > 0:
            backup_path = gt_file.with_name(gt_file.name + args.backup_suffix)
            backup_path.write_text(gt_text, encoding="utf-8")
            payload = ("\n".join(matched_lines) + "\n") if matched_lines else ""
            gt_file.write_text(payload, encoding="utf-8")
            print(f"  applied: removed={unmatched_lines} backup={backup_path.name}")

        total_gt += gt_count
        total_unmatched += unmatched

    total_pct = (100.0 * total_unmatched / total_gt) if total_gt else 0.0
    print(f"TOTAL,{total_gt},{total_unmatched},{total_pct:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
