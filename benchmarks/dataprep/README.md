# Data Prep for Custom Benchmark Videos

This folder holds planning/configuration files and helper scripts for building
custom benchmark datasets from source videos in your vault, without duplicating
large source files.

## Goal

- Select short clips (about 30 seconds) from source videos
- Extract clips into `benchmarks/dataprep/clips/<clip_id>/clip.mp4`
- Keep clip extraction as a dataprep stage before frame decode
- Keep `benchmarks/datasets/` as the canonical final dataset location

## Vault alignment

Use the same source-key convention used in SAM3 vault trim config:

- `source_key = "source_root_name/relative/path/to/video.mp4"`
- Examples:
  - `personal_film/AAU/My Team/Game1.mp4`
  - `public_film/highschool/ny/Some Video.mp4`

This avoids copying source videos into the repo while still giving stable,
portable references.

## Manifest

Start from `video_manifest.json` and fill one entry per selected clip.

Each clip entry includes:

- `clip_id`: stable ID used for output naming
- `source_key`: vault-style source video key
- `source_path`: absolute local source path (optional but convenient)
- `origin`: `personal` or `public`
- `level_tag`: basketball level tag (for example: `hs_jv`, `hs_varsity`, `hs_modified`, `aau`, `college`, `nba`)
- `start_sec` / `end_sec`: clip range in source video

## Clip extraction stage

Run from repo root:

```bash
# Validate and print commands only
python benchmarks/dataprep/prepare_clips.py

# Execute extraction
python benchmarks/dataprep/prepare_clips.py --run
```

Script behavior:

- validates `clip_id` values are unique
- validates source paths exist
- probes source resolution with `ffprobe`
- skips clips below 1080p with a warning
- trims each clip to `[start_sec, end_sec]`
- writes clips to `benchmarks/dataprep/clips/<clip_id>/clip.mp4`
- downscales to 1920x1080 when source is larger than 1080p
