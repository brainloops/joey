# BasketballMOT Labeler

DearPyGui MVP labeler for in-place BasketballMOT annotation.

## Scope

- Label source: `benchmarks/datasets/BasketballMOT/<split>/<sequence>`
- Reads:
  - `img1/%06d.jpg`
  - `det/det.txt`
  - `seqinfo.ini`
- Writes:
  - `gt/gt.txt` (MOT GT format)

## Policy

- Track all people relevant to game flow, including referees.
- Exclude non-person objects (ball, net, hoop). The current detector pipeline is person-only.

## Run

From repo root:

```bash
python labeling/basketball-mot-labeler.py --min-score 0.25
```

Optional:

```bash
python labeling/basketball-mot-labeler.py --dataset-root benchmarks/datasets/BasketballMOT --split train
```

## Tracker-Assisted Mode (ByteTrack)

The labeler supports a ByteTrack-assisted workflow for faster pass-and-correct labeling.

### Dependencies

Install tracker dependencies in the same environment as the labeler:

```bash
pip install trackers supervision
```

If these packages are missing, the tracker buttons are disabled and the app stays in manual mode.

### Workflow

1. Create/select an active track (`E` or `Use` in track table).
2. On the current frame, click the target player's detection box to seed/correct.
3. Keep **Tracking enabled** ON (default), then press `Space` to seed ByteTrack from the current box and autoplay.
4. Press `Space` to pause when occlusions/overlaps happen.
5. Correct by clicking the right box on the paused frame.
6. Use `Right` to advance one frame at a time with autosave, or `Space` to resume fast autoplay.
7. Turn **Tracking enabled** OFF for review-only playback/navigation.

Notes:
- Tracking uses existing `det/det.txt` detections (no drawing tool in this MVP).
- Tracking writes directly into `gt/gt.txt` and autosaves state files.

## Core Workflow (Single-Person Passes)

1. Start with no active track.
2. Click a detection to seed a new track pass.
3. Advance frame-by-frame selecting that same person.
4. Assigned detections are omitted from candidate pools by default.
5. End track, then start next unmatched person pass.

## Hotkeys

- `Space`: play/pause active-track playback
- `A` / `Left`: previous frame
- `Right`: advance one frame (autosaves current track step; uses tracker if active)
- `W`: skip forward 5 frames
- `Q`: end current track pass
- `E`: create/select next track id
- `R`: undo last assignment
- `S`: save `gt/gt.txt`
- `T`: track from current frame (same behavior as tracking-on + play)
- `Y`: alias for track from current frame

