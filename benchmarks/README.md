# Benchmark Datasets Workspace

This folder is the local staging area for MOT benchmark datasets and related download scripts.

## Layout

- `datasets/` - downloaded benchmark data
- `download_benchmarks.sh` - helper script to fetch benchmarks and clone reference repos

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
