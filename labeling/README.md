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
python labeling/basketball-mot-labeler.py --split train --min-score 0.25
```

Optional:

```bash
python labeling/basketball-mot-labeler.py --dataset-root benchmarks/datasets/BasketballMOT --split train
```

## Core Workflow (Single-Person Passes)

1. Start with no active track.
2. Click a detection to seed a new track pass.
3. Advance frame-by-frame selecting that same person.
4. Assigned detections are omitted from candidate pools by default.
5. End track, then start next unmatched person pass.

## Hotkeys

- `Space`: assign top candidate for current active pass, then advance one frame
- `A` / `D`: previous / next frame
- `W`: skip forward 5 frames
- `Q`: end current track pass
- `E`: create/select next track id
- `R`: undo last assignment
- `S`: save `gt/gt.txt`

