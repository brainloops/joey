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

MOT17 is supported in the same flow (after preparing TrackEval GT layout):

```bash
python benchmarks/compare_trackers.py mot17
```

SportsMOT is also supported:

```bash
python benchmarks/compare_trackers.py sportsmot
```

TeamTrack is supported after TeamTrack prep:

```bash
python benchmarks/compare_trackers.py teamtrack
```

Cross-dataset summary from cached results (default: DanceTrack + MOT17):

```bash
python benchmarks/cross_dataset_summary.py
```

You can specify datasets explicitly:

```bash
python benchmarks/cross_dataset_summary.py dancetrack mot17
```

Notes:
- This runs all trackers listed in `CONFIGURED_TRACKERS` inside `benchmarks/compare_trackers.py`.
- Output is a compact table:
  - rows = metrics (`HOTA`, `IDF1`, `MOTA`, `DetRe`, `DetPr`, `IDSW`, `Frag`)
  - columns = trackers
  - final `Best` column shows the winning tracker per metric

## MOT17/MOT20 TrackEval Prep

Use the MOTChallenge prep helper to stage GT links + seqmaps under:
`benchmarks/trackeval_data/gt/mot_challenge`

```bash
# MOT17 (default detector subset is FRCNN)
bash benchmarks/prepare_motchallenge_for_trackeval.sh

# MOT17 with another detector subset
bash benchmarks/prepare_motchallenge_for_trackeval.sh --benchmark MOT17 --detector SDP

# MOT20
bash benchmarks/prepare_motchallenge_for_trackeval.sh --benchmark MOT20
```

Notes:
- For MOT17, choose one detector subset (`FRCNN`, `DPM`, or `SDP`) so sequence names
  and detector source stay consistent.
- Local TrackEval is typically `train` split for MOT17/MOT20 because test GT is not public.

## MOT17/MOT20 RF-DETR Detections

Generate your own detections (COCO person class) with:

```bash
# MOT17 train, FRCNN sequence subset, keep original det.txt untouched
python benchmarks/generate_motchallenge_detections_rfdetr.py \
  --benchmark MOT17 \
  --split train \
  --detector FRCNN \
  --model-size small \
  --batch-size 8 \
  --threshold 0.4
```

By default this writes `det/det_rfdetr.txt` in each sequence.
If you want compatibility with scripts expecting `det/det.txt`, add:

```bash
python benchmarks/generate_motchallenge_detections_rfdetr.py \
  --benchmark MOT17 \
  --split train \
  --detector FRCNN \
  --write-det-txt \
  --overwrite
```

SportsMOT detections use the same script:

```bash
python benchmarks/generate_motchallenge_detections_rfdetr.py \
  --benchmark SportsMOT \
  --split val \
  --model-size small \
  --batch-size 8 \
  --threshold 0.4 \
  --write-det-txt \
  --overwrite
```

## SportsMOT TrackEval Prep

Extract and stage SportsMOT into TrackEval GT/seqmaps:

```bash
bash benchmarks/prepare_sportsmot_for_trackeval.sh
```

This creates:
- `benchmarks/datasets/SportsMOT/{train,val,test}` (from tar archives when needed)
- `benchmarks/trackeval_data/gt/mot_challenge/SportsMOT-{split}`
- `benchmarks/trackeval_data/gt/mot_challenge/seqmaps/SportsMOT-{split}.txt`

## TeamTrack TrackEval Prep

Prepare TeamTrack MOT archive, normalize split views, and build TrackEval layout:

```bash
bash benchmarks/prepare_teamtrack_for_trackeval.sh
```

By default, this script:
- extracts `benchmarks/datasets/TeamTrack/teamtrack-mot-002.zip` (if needed)
- builds normalized split links under `benchmarks/datasets/TeamTrack/{train,val,test}`
- decodes `img1.mp4` into `img1/*.jpg` when numeric frames are missing
- writes `TeamTrack-{split}` GT links + seqmaps under TrackEval GT root

TeamTrack RF-DETR detections:

```bash
python benchmarks/generate_motchallenge_detections_rfdetr.py \
  --benchmark TeamTrack \
  --split val \
  --model-size small \
  --batch-size 8 \
  --threshold 0.4 \
  --write-det-txt \
  --overwrite
```

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
