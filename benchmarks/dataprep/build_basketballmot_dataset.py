#!/usr/bin/env python3
"""
Build BasketballMOT dataset layout from extracted clip files.

Input clips:
  benchmarks/dataprep/clips/<clip_id>/clip.mp4

Output dataset:
  benchmarks/datasets/BasketballMOT/<split>/<clip_id>/
    - img1/%06d.jpg
    - seqinfo.ini
    - gt/gt.txt            (empty placeholder)
    - det/                 (empty directory)

TrackEval prep:
  benchmarks/trackeval_data/gt/mot_challenge/BasketballMOT-<split>/<clip_id> -> symlink
  benchmarks/trackeval_data/gt/mot_challenge/seqmaps/BasketballMOT-<split>.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return data


def _probe_video(path: Path) -> tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(out.stdout or "{}")
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found: {path}")
    s0 = streams[0]
    width = int(s0.get("width", 0))
    height = int(s0.get("height", 0))
    fps_raw = str(s0.get("r_frame_rate", "0/1"))
    if "/" in fps_raw:
        num, den = fps_raw.split("/", 1)
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    else:
        fps = float(fps_raw)
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(f"Invalid stream metadata for {path}: {s0}")
    return width, height, fps


def _extract_frames(clip_path: Path, out_img_dir: Path, overwrite: bool, target_fps: float | None) -> int:
    if overwrite and out_img_dir.exists():
        shutil.rmtree(out_img_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(clip_path),
    ]
    if target_fps is not None and target_fps > 0:
        cmd.extend(["-vf", f"fps={target_fps:g}"])
    cmd.extend(
        [
        "-start_number",
        "1",
        str(out_img_dir / "%06d.jpg"),
        ]
    )
    subprocess.run(cmd, check=True)
    frames = [p for p in out_img_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"]
    return len(frames)


def _write_seqinfo(path: Path, name: str, seq_len: int, width: int, height: int, fps: float) -> None:
    text = (
        "[Sequence]\n"
        f"name={name}\n"
        "imDir=img1\n"
        f"frameRate={fps}\n"
        f"seqLength={seq_len}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n"
    )
    path.write_text(text, encoding="utf-8")


def _safe_symlink(target: Path, link: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        current = link.resolve()
        if current == target:
            return
        link.unlink()
    elif link.exists():
        raise ValueError(f"Cannot create symlink, path exists and is not symlink: {link}")
    link.symlink_to(target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BasketballMOT from extracted clips.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/dataprep/video_manifest.json"),
        help="Manifest path used to enumerate clip IDs",
    )
    parser.add_argument(
        "--clips-root",
        type=Path,
        default=Path("benchmarks/dataprep/clips"),
        help="Root folder containing extracted clip.mp4 files",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("benchmarks/datasets/BasketballMOT"),
        help="BasketballMOT dataset root",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to write",
    )
    parser.add_argument(
        "--trackeval-gt-root",
        type=Path,
        default=Path("benchmarks/trackeval_data/gt/mot_challenge"),
        help="TrackEval GT root",
    )
    parser.add_argument(
        "--overwrite-frames",
        action="store_true",
        help="Re-extract frames if img1 already exists",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Target frame sampling rate for img1 extraction. Use <=0 to preserve source FPS.",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    clips = manifest.get("clips", [])
    if not isinstance(clips, list):
        raise ValueError("'clips' in manifest must be an array")

    split_root = args.dataset_root / args.split
    split_root.mkdir(parents=True, exist_ok=True)

    seq_names: list[str] = []
    built = 0
    skipped = 0
    for entry in clips:
        if not isinstance(entry, dict):
            continue
        clip_id = str(entry.get("clip_id", "")).strip()
        if not clip_id:
            continue
        seq_names.append(clip_id)

        clip_path = args.clips_root / clip_id / "clip.mp4"
        if not clip_path.is_file():
            print(f"[WARN] Missing clip for clip_id={clip_id}: {clip_path}; skipping.")
            skipped += 1
            continue

        seq_root = split_root / clip_id
        img_dir = seq_root / "img1"
        gt_dir = seq_root / "gt"
        det_dir = seq_root / "det"
        seq_root.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        det_dir.mkdir(parents=True, exist_ok=True)
        (gt_dir / "gt.txt").touch(exist_ok=True)

        width, height, fps = _probe_video(clip_path)
        effective_fps = fps if args.target_fps <= 0 else float(args.target_fps)
        if img_dir.is_dir() and any(img_dir.iterdir()) and not args.overwrite_frames:
            jpgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"]
            frame_count = len(jpgs)
            print(f"[INFO] Reusing existing frames for {clip_id}: {frame_count}")
        else:
            frame_count = _extract_frames(
                clip_path=clip_path,
                out_img_dir=img_dir,
                overwrite=args.overwrite_frames,
                target_fps=(None if args.target_fps <= 0 else float(args.target_fps)),
            )
            print(f"[INFO] Extracted {frame_count} frames for {clip_id} at fps={effective_fps:g}")

        if frame_count <= 0:
            print(f"[WARN] No frames extracted for {clip_id}; skipping seqinfo.")
            skipped += 1
            continue
        _write_seqinfo(
            path=seq_root / "seqinfo.ini",
            name=clip_id,
            seq_len=frame_count,
            width=width,
            height=height,
            fps=effective_fps,
        )
        built += 1

    # TrackEval GT structure + seqmap
    te_split_root = args.trackeval_gt_root / f"BasketballMOT-{args.split}"
    te_seqmap_dir = args.trackeval_gt_root / "seqmaps"
    te_split_root.mkdir(parents=True, exist_ok=True)
    te_seqmap_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    for seq in seq_names:
        src = split_root / seq
        if not src.is_dir():
            continue
        dst = te_split_root / seq
        _safe_symlink(src, dst)
        linked += 1

    seqmap_path = te_seqmap_dir / f"BasketballMOT-{args.split}.txt"
    seqmap_text = "name\n" + "".join(f"{s}\n" for s in seq_names if (split_root / s).is_dir())
    seqmap_path.write_text(seqmap_text, encoding="utf-8")

    print(
        f"[DONE] BasketballMOT {args.split}: built={built}, skipped={skipped}, "
        f"trackeval_links={linked}, seqmap={seqmap_path}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
