#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MOT-format detections for MOT-style benchmarks using RF-DETR "
            "(COCO person class only)."
        )
    )
    parser.add_argument(
        "--benchmark",
        default="MOT17",
        choices=["MOT17", "MOT20", "SportsMOT", "TeamTrack", "BasketballMOT"],
        help="MOT benchmark dataset family.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset root override. Defaults to benchmarks/datasets/<benchmark>.",
    )
    parser.add_argument(
        "--detector",
        default="FRCNN",
        choices=["FRCNN", "DPM", "SDP", "all"],
        help=(
            "MOT17 sequence subset selector. Ignored for MOT20. "
            "Ignored for SportsMOT/TeamTrack/BasketballMOT. "
            "Use one detector subset to avoid triplicate scene variants."
        ),
    )
    parser.add_argument(
        "--model-size",
        default="small",
        choices=["nano", "small", "medium", "large"],
        help="RF-DETR model size. Default uses RF-DETR Small.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="RF-DETR confidence threshold.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of frames per forward pass.",
    )
    parser.add_argument(
        "--det-filename",
        default="det_rfdetr.txt",
        help=(
            "Detection file name written under each sequence det/ folder "
            "(default: det_rfdetr.txt)."
        ),
    )
    parser.add_argument(
        "--write-det-txt",
        action="store_true",
        help=(
            "Also write det/det.txt for compatibility with scripts expecting that filename. "
            "Use with --overwrite if det/det.txt already exists."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-seqs", type=int, default=None)
    return parser.parse_args()


def list_sequences(split_dir: Path, benchmark: str, detector: str) -> List[Path]:
    seqs = [p for p in split_dir.iterdir() if p.is_dir()]
    seqs.sort(key=lambda p: p.name)
    if benchmark == "MOT17" and detector.lower() != "all":
        suffix = f"-{detector.upper()}"
        seqs = [p for p in seqs if p.name.endswith(suffix)]
    return seqs


def list_images(img_dir: Path) -> List[Path]:
    images = [
        p
        for p in img_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and p.stem.isdigit()
    ]
    images.sort(key=lambda p: p.name)
    return images


def build_model(size: str, batch_size: int):
    try:
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'rfdetr'. Install with:\n"
            "  pip install rfdetr"
        ) from exc

    mapping = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    model = mapping[size]()
    if hasattr(model, "optimize_for_inference"):
        # Keep compile disabled to support variable final batch sizes safely.
        model.optimize_for_inference(compile=False, batch_size=batch_size)
    return model


def resolve_person_class_id() -> int:
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
    except Exception:
        return 1

    if isinstance(COCO_CLASSES, dict):
        for class_id, name in COCO_CLASSES.items():
            if str(name).lower() == "person":
                return int(class_id)
    return 1


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    if args.dataset_root:
        return Path(args.dataset_root).expanduser().resolve()
    return (Path(__file__).resolve().parent / "datasets" / args.benchmark).resolve()


def parse_frame_number(image_path: Path) -> int:
    # MOT uses zero-padded numeric filenames (e.g. 000001.jpg).
    return int(image_path.stem)


def main() -> None:
    args = parse_args()

    dataset_root = resolve_dataset_root(args)
    split_dir = dataset_root / args.split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    model = build_model(args.model_size, batch_size=args.batch_size)
    person_class_id = resolve_person_class_id()
    seqs = list_sequences(split_dir, benchmark=args.benchmark, detector=args.detector)
    if args.max_seqs is not None:
        seqs = seqs[: args.max_seqs]

    print(
        f"[INFO] benchmark={args.benchmark} split={args.split} sequences={len(seqs)} "
        f"detector_subset={args.detector} model=rf-detr-{args.model_size} "
        f"threshold={args.threshold} batch_size={args.batch_size} "
        f"person_class_id={person_class_id}"
    )
    total_rows = 0
    total_images = 0
    total_seconds = 0.0

    for seq_dir in seqs:
        seq_start = time.perf_counter()
        img_dir = seq_dir / "img1"
        if not img_dir.is_dir():
            print(f"[WARN] Missing img1 folder, skipping: {img_dir}")
            continue

        images = list_images(img_dir)
        if not images:
            print(f"[WARN] No images found, skipping: {img_dir}")
            continue

        det_dir = seq_dir / "det"
        det_dir.mkdir(parents=True, exist_ok=True)

        det_file = det_dir / args.det_filename
        det_txt_file = det_dir / "det.txt"

        if det_file.exists() and not args.overwrite:
            print(f"[SKIP] Existing detections: {det_file}")
            continue
        if args.write_det_txt and det_txt_file.exists() and not args.overwrite:
            print(f"[SKIP] Existing detections: {det_txt_file} (use --overwrite to replace)")
            continue

        lines: List[str] = []
        for start in range(0, len(images), args.batch_size):
            batch_paths = images[start : start + args.batch_size]
            batch_frames = []
            batch_frame_nums = []

            for image_path in batch_paths:
                frame_num = parse_frame_number(image_path)
                frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    print(f"[WARN] Failed to read image: {image_path}")
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                batch_frames.append(frame_rgb)
                batch_frame_nums.append(frame_num)

            if not batch_frames:
                continue

            detections_batch = model.predict(batch_frames, threshold=args.threshold)
            if not isinstance(detections_batch, list):
                detections_batch = [detections_batch]

            for frame_num, detections in zip(batch_frame_nums, detections_batch):
                if detections is None or len(detections.xyxy) == 0:
                    continue

                class_ids = detections.class_id if detections.class_id is not None else []
                confs = detections.confidence if detections.confidence is not None else []

                for i, xyxy in enumerate(detections.xyxy):
                    class_id = int(class_ids[i]) if len(class_ids) > i else -1
                    # Keep only COCO person detections for MOT.
                    if class_id != person_class_id:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in xyxy.tolist()]
                    w = x2 - x1
                    h = y2 - y1
                    score = float(confs[i]) if len(confs) > i else 1.0
                    lines.append(f"{frame_num},-1,{x1:.3f},{y1:.3f},{w:.3f},{h:.3f},{score:.6f},1,-1,-1")

        payload = "\n".join(lines) + ("\n" if lines else "")
        det_file.write_text(payload)
        if args.write_det_txt:
            det_txt_file.write_text(payload)

        total_rows += len(lines)
        total_images += len(images)
        seq_seconds = time.perf_counter() - seq_start
        total_seconds += seq_seconds
        images_per_sec = (len(images) / seq_seconds) if seq_seconds > 0 else 0.0
        if args.write_det_txt:
            target_desc = f"{det_file} and {det_txt_file}"
        else:
            target_desc = str(det_file)
        print(
            f"[DONE] {seq_dir.name}: images={len(images)} det_rows={len(lines)} "
            f"time={seq_seconds:.2f}s ({seq_seconds * 1000:.0f}ms) ips={images_per_sec:.2f} -> {target_desc}"
        )

    avg_ips = (total_images / total_seconds) if total_seconds > 0 else 0.0
    print(
        f"[SUMMARY] benchmark={args.benchmark} split={args.split} sequences={len(seqs)} images={total_images} "
        f"detection_rows={total_rows} total_time={total_seconds:.2f}s avg_ips={avg_ips:.2f}"
    )


if __name__ == "__main__":
    main()
