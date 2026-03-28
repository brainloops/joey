#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARK_DIR="${ROOT_DIR}/benchmarks"
DATA_DIR="${BENCHMARK_DIR}/datasets"
REPO_DIR="${BENCHMARK_DIR}/repos"

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
err() { echo "[ERROR] $*" >&2; }

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

should_skip_dataset_dir() {
  local dir="$1"
  if [[ "${FORCE_DATASETS}" -eq 1 ]]; then
    return 1
  fi
  if [[ -d "${dir}" ]] && [[ -n "$(ls -A "${dir}" 2>/dev/null)" ]]; then
    return 0
  fi
  return 1
}

clone_or_update_repo() {
  local name="$1"
  local url="$2"
  local dst="${REPO_DIR}/${name}"

  if [[ -d "${dst}/.git" ]]; then
    log "Repo ${name} already exists; pulling latest."
    git -C "${dst}" pull --ff-only || warn "Could not fast-forward ${name}; leaving as-is."
  else
    log "Cloning ${name} -> ${dst}"
    git clone "${url}" "${dst}"
  fi
}

download_hf_dataset() {
  local repo_id="$1"
  local out_dir="$2"

  mkdir -p "${out_dir}"
  if should_skip_dataset_dir "${out_dir}"; then
    log "${repo_id} appears present at ${out_dir}; skipping download."
    return 0
  fi

  if has_cmd huggingface-cli; then
    log "Downloading Hugging Face dataset ${repo_id} via huggingface-cli"
    huggingface-cli download "${repo_id}" --repo-type dataset --local-dir "${out_dir}" || {
      warn "huggingface-cli failed for ${repo_id}."
      return 1
    }
    return 0
  fi

  if has_cmd python3; then
    log "Trying Python snapshot download for ${repo_id}"
    python3 - <<PY || return 1
from pathlib import Path
import sys

repo_id = "${repo_id}"
out_dir = Path("${out_dir}")

try:
    from huggingface_hub import snapshot_download
except Exception:
    print(f"[WARN] huggingface_hub not installed. Install with: pip install huggingface_hub", file=sys.stderr)
    raise SystemExit(1)

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(out_dir),
    local_dir_use_symlinks=False,
)
print(f"[INFO] Downloaded {repo_id} into {out_dir}")
PY
    return 0
  fi

  warn "Neither huggingface-cli nor python3 is available for ${repo_id}."
  return 1
}

download_teamtrack_kaggle() {
  local out_dir="${DATA_DIR}/TeamTrack"
  mkdir -p "${out_dir}"
  if should_skip_dataset_dir "${out_dir}"; then
    log "TeamTrack appears present at ${out_dir}; skipping download."
    return 0
  fi

  if ! has_cmd kaggle; then
    warn "Kaggle CLI not found. Install with: pip install kaggle"
    return 1
  fi

  if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
    warn "Kaggle credentials missing at ~/.kaggle/kaggle.json"
    warn "Create API token at https://www.kaggle.com/settings and place it there."
    return 1
  fi

  log "Downloading TeamTrack from Kaggle into ${out_dir}"
  kaggle datasets download -d atomscott/teamtrack -p "${out_dir}" --unzip || {
    warn "Kaggle download for TeamTrack failed (possibly terms not accepted yet)."
    return 1
  }

  return 0
}

download_teamtrack_drive() {
  local out_dir="${DATA_DIR}/TeamTrack"
  local teamtrack_drive_url="https://drive.google.com/drive/folders/1D3jxrEWgWke0l1TWC_052OhYVs2IwDVZ"

  mkdir -p "${out_dir}"
  if should_skip_dataset_dir "${out_dir}"; then
    log "TeamTrack appears present at ${out_dir}; skipping download."
    return 0
  fi

  if has_cmd gdown; then
    log "Attempting TeamTrack download via Google Drive folder (gdown)"
    gdown --folder --remaining-ok "${teamtrack_drive_url}" -O "${out_dir}" || {
      warn "gdown folder download failed for TeamTrack."
      return 1
    }
    if [[ -z "$(ls -A "${out_dir}" 2>/dev/null)" ]]; then
      warn "TeamTrack Google Drive step completed but ${out_dir} is empty."
      warn "Likely cause: gdown folder limitations/auth/quota constraints."
      warn "Recommended: use Kaggle dataset path for complete TeamTrack download."
      return 1
    fi
    return 0
  fi

  if has_cmd python3; then
    log "Attempting TeamTrack download via Python gdown fallback"
    python3 - <<PY || return 1
from pathlib import Path
import sys

url = "https://drive.google.com/drive/folders/1D3jxrEWgWke0l1TWC_052OhYVs2IwDVZ"
out_dir = Path("${out_dir}")
out_dir.mkdir(parents=True, exist_ok=True)

try:
    import gdown
except Exception:
    print("[WARN] gdown not installed. Install with: pip install gdown", file=sys.stderr)
    raise SystemExit(1)

gdown.download_folder(
    url=url,
    output=str(out_dir),
    quiet=False,
    use_cookies=False,
    remaining_ok=True,
)
print(f"[INFO] TeamTrack downloaded into {out_dir}")
PY
    if [[ -z "$(ls -A "${out_dir}" 2>/dev/null)" ]]; then
      warn "TeamTrack Google Drive step completed but ${out_dir} is empty."
      warn "Likely cause: gdown folder limitations/auth/quota constraints."
      warn "Recommended: use Kaggle dataset path for complete TeamTrack download."
      return 1
    fi
    return 0
  fi

  warn "Neither gdown nor python3 is available for TeamTrack Google Drive download."
  return 1
}

download_motchallenge_zip() {
  local dataset_name="$1"   # MOT17 / MOT20
  local zip_url="$2"
  local out_root="${DATA_DIR}"
  local zip_path="${out_root}/${dataset_name}.zip"
  local expected_dir="${out_root}/${dataset_name}"

  mkdir -p "${out_root}"

  if should_skip_dataset_dir "${expected_dir}"; then
    log "${dataset_name} already appears downloaded at ${expected_dir}; skipping."
    return 0
  fi

  if ! has_cmd unzip; then
    warn "unzip is required to extract ${dataset_name}. Install with: sudo apt install unzip"
    return 1
  fi

  log "Downloading ${dataset_name} from ${zip_url}"
  if has_cmd curl; then
    curl -L --fail --retry 3 -o "${zip_path}" "${zip_url}" || {
      warn "curl download failed for ${dataset_name}"
      return 1
    }
  elif has_cmd wget; then
    wget -O "${zip_path}" "${zip_url}" || {
      warn "wget download failed for ${dataset_name}"
      return 1
    }
  elif has_cmd python3; then
    python3 - <<PY || return 1
import urllib.request
url = "${zip_url}"
dst = "${zip_path}"
print(f"[INFO] Downloading {url} -> {dst}")
urllib.request.urlretrieve(url, dst)
print("[INFO] Download complete")
PY
  else
    warn "Need curl, wget, or python3 to download ${dataset_name}"
    return 1
  fi

  log "Extracting ${dataset_name} into ${out_root}"
  unzip -q -o "${zip_path}" -d "${out_root}" || {
    warn "Failed to extract ${zip_path}"
    return 1
  }

  if [[ -d "${expected_dir}" ]]; then
    log "${dataset_name} downloaded and extracted successfully."
  else
    warn "${dataset_name} extracted, but expected folder ${expected_dir} was not found."
  fi

  return 0
}

print_motchallenge_instructions() {
  cat <<'EOF'

[NEXT STEP] MOT17 / MOT20 require MOTChallenge access:
  1) Create/login account at https://motchallenge.net/
  2) Accept terms and navigate to Downloads
  3) Download MOT17 and MOT20 zips
  4) Extract into:
     benchmarks/datasets/MOT17
     benchmarks/datasets/MOT20

EOF
}

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/download_benchmarks.sh [options]

Options:
  --no-repos          Skip cloning benchmark repos
  --no-datasets       Skip dataset downloads (only create folders + repos)
  --repos-only        Clone repos only
  --datasets-only     Download datasets only
  --force-datasets    Re-download datasets even if target folders are non-empty
  -h, --help          Show this help
EOF
}

DO_REPOS=1
DO_DATASETS=1
FORCE_DATASETS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-repos)
      DO_REPOS=0
      shift
      ;;
    --no-datasets)
      DO_DATASETS=0
      shift
      ;;
    --repos-only)
      DO_REPOS=1
      DO_DATASETS=0
      shift
      ;;
    --datasets-only)
      DO_REPOS=0
      DO_DATASETS=1
      shift
      ;;
    --force-datasets)
      FORCE_DATASETS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

log "Preparing benchmark workspace under ${BENCHMARK_DIR}"
mkdir -p "${DATA_DIR}" "${REPO_DIR}"
mkdir -p \
  "${DATA_DIR}/DanceTrack" \
  "${DATA_DIR}/TeamTrack" \
  "${DATA_DIR}/SportsMOT" \
  "${DATA_DIR}/MOT17" \
  "${DATA_DIR}/MOT20"

if [[ "${DO_REPOS}" -eq 1 ]]; then
  log "Cloning/updating benchmark reference repos"
  clone_or_update_repo "DanceTrack" "https://github.com/DanceTrack/DanceTrack.git"
  clone_or_update_repo "TeamTrack" "https://github.com/AtomScott/TeamTrack.git"
  clone_or_update_repo "SportsMOT" "https://github.com/MCG-NJU/SportsMOT.git"
  clone_or_update_repo "TrackEval" "https://github.com/JonathonLuiten/TrackEval.git"
fi

if [[ "${DO_DATASETS}" -eq 1 ]]; then
  log "Attempting dataset downloads"

  if ! download_hf_dataset "noahcao/dancetrack" "${DATA_DIR}/DanceTrack"; then
    warn "DanceTrack auto-download did not complete."
    warn "Manual source: https://dancetrack.github.io/"
  fi

  if ! download_hf_dataset "MCG-NJU/SportsMOT" "${DATA_DIR}/SportsMOT"; then
    warn "SportsMOT auto-download did not complete."
    warn "Manual source: https://github.com/MCG-NJU/SportsMOT"
  fi

  if ! download_teamtrack_kaggle; then
    warn "TeamTrack Kaggle download did not complete; trying Google Drive fallback."
    if ! download_teamtrack_drive; then
      warn "TeamTrack auto-download did not complete."
      warn "Manual sources: https://atomscott.github.io/TeamTrack/ and https://www.kaggle.com/datasets/atomscott/teamtrack"
    fi
  fi

  if ! download_motchallenge_zip "MOT17" "http://www.motchallenge.net/data/MOT17.zip"; then
    warn "MOT17 auto-download did not complete."
    warn "You can manually download from: http://www.motchallenge.net/data/MOT17/"
  fi

  if ! download_motchallenge_zip "MOT20" "http://www.motchallenge.net/data/MOT20.zip"; then
    warn "MOT20 auto-download did not complete."
    warn "You can manually download from: http://www.motchallenge.net/data/MOT20/"
  fi

  print_motchallenge_instructions
fi

cat <<EOF
[DONE] Benchmark workspace prepared.

Paths:
  Benchmarks root: ${BENCHMARK_DIR}
  Datasets:        ${DATA_DIR}
  Repositories:    ${REPO_DIR}

Run again anytime:
  bash benchmarks/download_benchmarks.sh
EOF
