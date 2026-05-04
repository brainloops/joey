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

## Tracker Editor Mode

The labeler runs in a single-pass tracker-editor workflow with three runtime-selectable trackers:
- `ByteTrack`
- `OCSort`
- `McByte` (Track-all pipeline)

### Dependencies

Install tracker dependencies in the same environment as the labeler:

```bash
pip install trackers supervision
```

If these packages are missing, tracker actions are disabled.

### Workflow

Use **Track all** to run detector outputs through the selected tracker across the full sequence, then auto-populate the right-side track table with generated track IDs.

Notes:
- Uses existing `det/det.txt` detections as detector input.
- Writes generated tracks to `gt/gt.txt` (plus session files) via autosave.
- Click a track row or click a box on canvas to focus the active track.
- Proposed tracks can be marked **Keep** to move them into a separate kept list with metadata.
- Keep metadata fields: `team` (`home`/`away`), `jersey_number` (integer), `name` (string).
- Use `Split (S)` at the current frame to split the active track into a new track from that frame forward.
- Split lineage is tracked with `group` + `segment` so split siblings stay visually grouped in table order.
- Use `Merge` (or `M`) to merge a **proposed** track into a **kept** track; target choices show track ID, jersey, name, and team.
- Merge keeps the kept track ID as target, preserves non-contiguous segments, and skips overlapping target frames.

## Hotkeys

- `Space`: play/pause playback
- `A` / `Left`: previous frame
- `Right`: next frame
- `S`: split active track at current frame
- `M`: merge active track into another track
- `T`: run **Track all**

