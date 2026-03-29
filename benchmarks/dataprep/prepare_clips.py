#!/usr/bin/env python3
"""
Prepare 30-second clip files from a manifest.

Output layout:
  benchmarks/dataprep/clips/<clip_id>/clip.mp4

Behavior:
- Validate clip IDs are unique.
- Probe source video resolution with ffprobe.
- Skip clip extraction (with warning) when source is below 1080p.
- Extract the selected time window and resize/downscale to 1920x1080 when needed.

Default behavior is dry-run (print commands and validations only).
Use --run to execute ffmpeg and write clip files.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import Counter
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


def _require(obj: dict[str, Any], key: str, clip_id: str) -> Any:
    if key not in obj:
        raise ValueError(f"Clip '{clip_id}' is missing required field '{key}'")
    return obj[key]


def _probe_resolution(source_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(source_path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(out.stdout or "{}")
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found: {source_path}")
    stream0 = streams[0]
    width = int(stream0.get("width", 0))
    height = int(stream0.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read resolution for: {source_path}")
    return width, height


def _ffmpeg_trim_resize_cmd(
    source_path: Path,
    start_sec: float,
    end_sec: float,
    out_clip_path: Path,
    width: int,
    height: int,
    needs_resize: bool,
) -> list[str]:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-to",
        str(end_sec),
        "-i",
        str(source_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-an",
        str(out_clip_path),
    ]
    if needs_resize:
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        )
        cmd[8:8] = ["-vf", vf]
    return cmd


def _run(cmd: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare clips from manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/dataprep/video_manifest.json"),
        help="Path to manifest JSON",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute ffmpeg commands (default: dry-run print only)",
    )
    parser.add_argument(
        "--clips-root",
        type=Path,
        default=Path("benchmarks/dataprep/clips"),
        help="Root output directory for extracted clips",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)

    defaults = manifest.get("processing_defaults", {})
    clips = manifest.get("clips", [])

    if not isinstance(defaults, dict):
        raise ValueError("'processing_defaults' must be an object")
    if not isinstance(clips, list):
        raise ValueError("'clips' must be an array")

    width = int(defaults.get("resize_width", 1920))
    height = int(defaults.get("resize_height", 1080))
    clips_root = args.clips_root

    clip_ids: list[str] = []
    for idx, clip in enumerate(clips):
        if not isinstance(clip, dict):
            raise ValueError(f"Clip index {idx} must be an object")
        clip_ids.append(str(_require(clip, "clip_id", f"index_{idx}")))
    duplicates = [cid for cid, n in Counter(clip_ids).items() if n > 1]
    if duplicates:
        dupes = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate clip_id values found: {dupes}")

    extracted_count = 0
    skipped_count = 0

    for idx, clip in enumerate(clips):
        clip_id = clip_ids[idx]
        source_path_str = str(_require(clip, "source_path", clip_id))
        start_sec = float(_require(clip, "start_sec", clip_id))
        end_sec = float(_require(clip, "end_sec", clip_id))

        if end_sec <= start_sec:
            raise ValueError(f"Clip '{clip_id}' has end_sec <= start_sec")

        source_path = Path(source_path_str)
        if not source_path.exists():
            raise ValueError(f"Clip '{clip_id}' source does not exist: {source_path}")

        src_w, src_h = _probe_resolution(source_path)
        if src_w < width or src_h < height:
            print(
                f"[WARN] clip_id={clip_id} source={source_path} "
                f"resolution={src_w}x{src_h} is below {width}x{height}; skipping."
            )
            skipped_count += 1
            continue

        needs_resize = (src_w, src_h) != (width, height)
        out_dir = clips_root / clip_id
        out_clip_path = out_dir / "clip.mp4"

        if args.run:
            out_dir.mkdir(parents=True, exist_ok=True)

        trim_cmd = _ffmpeg_trim_resize_cmd(
            source_path=source_path,
            start_sec=start_sec,
            end_sec=end_sec,
            out_clip_path=out_clip_path,
            width=width,
            height=height,
            needs_resize=needs_resize,
        )

        print(
            f"[INFO] clip_id={clip_id} source_res={src_w}x{src_h} "
            f"target={width}x{height} resize={'yes' if needs_resize else 'no'}"
        )
        _run(trim_cmd, dry_run=not args.run)
        extracted_count += 1

    print(
        f"Processed {len(clips)} clips: extracted={extracted_count}, skipped={skipped_count}, "
        f"output_root={clips_root}"
        + (" (dry-run)" if not args.run else "")
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
