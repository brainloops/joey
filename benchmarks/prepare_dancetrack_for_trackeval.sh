#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DANCE_ROOT="${ROOT_DIR}/benchmarks/datasets/DanceTrack"
TRACKEVAL_GT_ROOT="${ROOT_DIR}/benchmarks/trackeval_data/gt/mot_challenge"
TRACKEVAL_SEQMAP_DIR="${TRACKEVAL_GT_ROOT}/seqmaps"

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
err() { echo "[ERROR] $*" >&2; }

ensure_tools() {
  command -v unzip >/dev/null 2>&1 || {
    err "unzip is required. Install with: sudo apt install unzip"
    exit 1
  }
}

extract_zip_if_needed() {
  local zip_name="$1"
  local marker_dir="$2"
  local zip_path="${DANCE_ROOT}/${zip_name}"
  local marker_path="${DANCE_ROOT}/${marker_dir}"

  if [[ ! -f "${zip_path}" ]]; then
    warn "Missing ${zip_path}; skipping."
    return
  fi

  if [[ -d "${marker_path}" ]] && [[ -n "$(ls -A "${marker_path}" 2>/dev/null)" ]]; then
    log "${zip_name} already extracted -> ${marker_dir}/"
    return
  fi

  log "Extracting ${zip_name}..."
  unzip -q -n "${zip_path}" -d "${DANCE_ROOT}"
}

link_split_sequences() {
  local split_name="$1"
  shift
  local src_roots=("$@")
  local dst_root="${DANCE_ROOT}/${split_name}"

  mkdir -p "${dst_root}"
  for src in "${src_roots[@]}"; do
    local src_dir="${DANCE_ROOT}/${src}"
    if [[ ! -d "${src_dir}" ]]; then
      warn "Source split directory not found: ${src_dir}"
      continue
    fi
    for seq_dir in "${src_dir}"/*; do
      [[ -d "${seq_dir}" ]] || continue
      local seq_name
      seq_name="$(basename "${seq_dir}")"
      local link_path="${dst_root}/${seq_name}"
      if [[ -e "${link_path}" ]]; then
        continue
      fi
      ln -s "${seq_dir}" "${link_path}"
    done
  done
}

build_trackeval_gt_split() {
  local split_name="$1"
  local src_split_dir="${DANCE_ROOT}/${split_name}"
  local dst_split_dir="${TRACKEVAL_GT_ROOT}/DanceTrack-${split_name}"
  local seqmap_file="${TRACKEVAL_SEQMAP_DIR}/DanceTrack-${split_name}.txt"

  mkdir -p "${dst_split_dir}" "${TRACKEVAL_SEQMAP_DIR}"

  local seq_names=()
  for seq_link in "${src_split_dir}"/*; do
    [[ -d "${seq_link}" ]] || continue
    local seq_name
    seq_name="$(basename "${seq_link}")"
    local gt_file="${seq_link}/gt/gt.txt"
    local seqinfo_file="${seq_link}/seqinfo.ini"
    if [[ -f "${gt_file}" && -f "${seqinfo_file}" ]]; then
      if [[ ! -e "${dst_split_dir}/${seq_name}" ]]; then
        ln -s "${seq_link}" "${dst_split_dir}/${seq_name}"
      fi
      seq_names+=("${seq_name}")
    fi
  done

  {
    echo "name"
    for seq in "${seq_names[@]}"; do
      echo "${seq}"
    done
  } > "${seqmap_file}"

  log "Prepared TrackEval GT split DanceTrack-${split_name} with ${#seq_names[@]} sequence(s)"
}

main() {
  ensure_tools

  if [[ ! -d "${DANCE_ROOT}" ]]; then
    err "DanceTrack dataset root not found: ${DANCE_ROOT}"
    exit 1
  fi

  extract_zip_if_needed "train1.zip" "train1"
  extract_zip_if_needed "train2.zip" "train2"
  extract_zip_if_needed "val.zip" "val"
  extract_zip_if_needed "test1.zip" "test1"
  extract_zip_if_needed "test2.zip" "test2"

  log "Creating normalized split views in ${DANCE_ROOT}/{train,val,test}"
  link_split_sequences "train" "train1" "train2"
  link_split_sequences "val" "val"
  link_split_sequences "test" "test1" "test2"

  log "Creating TrackEval GT layout and seqmaps"
  build_trackeval_gt_split "train"
  build_trackeval_gt_split "val"
  build_trackeval_gt_split "test"

  cat <<EOF
[DONE] DanceTrack prepared for TrackEval.

Prepared locations:
  Source dataset root: ${DANCE_ROOT}
  TrackEval GT root:   ${TRACKEVAL_GT_ROOT}
  Seqmaps:             ${TRACKEVAL_SEQMAP_DIR}

Next step example (evaluate a tracker named bytetrack_baseline on val split):
  python3 benchmarks/repos/TrackEval/scripts/run_mot_challenge.py \\
    --BENCHMARK DanceTrack \\
    --SPLIT_TO_EVAL val \\
    --GT_FOLDER "${TRACKEVAL_GT_ROOT}" \\
    --TRACKERS_FOLDER "${ROOT_DIR}/benchmarks/trackeval_data/trackers/mot_challenge" \\
    --TRACKERS_TO_EVAL bytetrack_baseline \\
    --METRICS HOTA CLEAR Identity \\
    --USE_PARALLEL True \\
    --NUM_PARALLEL_CORES 8 \\
    --DO_PREPROC False
EOF
}

main "$@"

