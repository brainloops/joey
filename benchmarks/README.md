# Benchmark Datasets Workspace

This folder is the local staging area for MOT benchmark datasets and related download scripts.

## Layout

- `datasets/` - downloaded benchmark data
- `download_benchmarks.sh` - helper script to fetch benchmarks and clone reference repos

## Tracker Baseline Policy

- For baseline tracker runs in this project, use Roboflow `trackers` implementations:
  - Roboflow `trackers` package `ByteTrackTracker`
  - Roboflow `trackers` package `OCSORTTracker`
- Keep this consistent across benchmarks so tracker comparisons are implementation-consistent.
- DanceTrack currently ships with GT annotations but no detector `det/det.txt` in this workspace, so:
  - use `benchmarks/run_supervision_dancetrack_baselines.py` to generate tracker outputs
  - choose detection source explicitly (`gt` for pipeline/proxy runs, `det` when you have detector outputs)

## Real Eval Flow (DanceTrack)

1. Generate detections in MOT format (`det/det.txt`) per sequence:

```bash
conda activate joey
pip install rfdetr
python benchmarks/run_benchmark.py detect --split val --model-size small --batch-size 8
```

2. Run tracker + TrackEval using generated detections:

```bash
conda activate joey
pip install numpy trackers supervision
python benchmarks/run_benchmark.py bytetrack dancetrack
```

Notes:
- DanceTrack `test` has no public GT, so local TrackEval is for `train`/`val`.
- You can still generate detections and tracking outputs for `test` for eventual benchmark submission workflows.
- `run_benchmark.py` now has a simplified interface: pass only tracker (`bytetrack`, `ocsort`, `mcbyte`, `both`, `all`) and dataset.
  - Fastest form: `python benchmarks/run_benchmark.py ocsort dancetrack`
  - McByte is also supported: `python benchmarks/run_benchmark.py mcbyte dancetrack`
  - You can still pass a path instead of the name if needed.
  - If dataset path points to `.../DanceTrack/val` (or `train`/`test`), split is inferred from that folder name.
  - If dataset path points to `.../DanceTrack`, split defaults to `val` when present, then `train`, then `test`.
  - Detection source is inferred from files: uses `det` when all `det/det.txt` files exist; otherwise falls back to `gt` when complete.

3. Compare all configured trackers with one command:

```bash
python benchmarks/compare_trackers.py dancetrack
```

Notes:
- This runs all trackers listed in `CONFIGURED_TRACKERS` inside `benchmarks/compare_trackers.py`.
- Output is a compact table:
  - rows = metrics (`HOTA`, `IDF1`, `MOTA`, `DetRe`, `DetPr`, `IDSW`, `Frag`)
  - columns = trackers
  - final `Best` column shows the winning tracker per metric

## Quick Start

From repo root:

```bash
bash benchmarks/download_benchmarks.sh
```

This will:

1. Create benchmark dataset folders
2. Clone benchmark reference repos
3. Attempt dataset downloads where possible
4. Print clear next steps for datasets that require auth/registration

## Notes

- Some datasets require credentials or terms acceptance:
  - `TeamTrack` (Kaggle API + accepted terms)
  - `MOT17/MOT20` (MOTChallenge account/terms)
- `DanceTrack` and `SportsMOT` are attempted through Hugging Face dataset snapshots when supported tooling is available.
