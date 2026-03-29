#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SPORTS_ROOT="${ROOT_DIR}/benchmarks/datasets/SportsMOT"
SPORTS_DATA_ARCHIVE_DIR="${SPORTS_ROOT}/dataset"
SPORTS_SPLITS_DIR="${SPORTS_ROOT}/splits_txt"
TRACKEVAL_GT_ROOT="${ROOT_DIR}/benchmarks/trackeval_data/gt/mot_challenge"
TRACKEVAL_SEQMAP_DIR="${TRACKEVAL_GT_ROOT}/seqmaps"

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
err() { echo "[ERROR] $*" >&2; }

ensure_tools() {
  command -v tar >/dev/null 2>&1 || {
    err "tar is required."
    exit 1
  }
}

extract_split_if_needed() {
  local split_name="$1"
  local tar_path="${SPORTS_DATA_ARCHIVE_DIR}/${split_name}.tar"
  local split_dir="${SPORTS_ROOT}/${split_name}"

  if [[ -d "${split_dir}" ]] && [[ -n "$(ls -A "${split_dir}" 2>/dev/null)" ]]; then
    log "SportsMOT ${split_name}/ already extracted."
    return
  fi
  if [[ ! -f "${tar_path}" ]]; then
    warn "Missing archive ${tar_path}; skipping extraction for ${split_name}."
    return
  fi

  log "Extracting ${tar_path} -> ${SPORTS_ROOT}"
  tar -xf "${tar_path}" -C "${SPORTS_ROOT}"
}

build_trackeval_gt_split() {
  local split_name="$1"
  local src_split_dir="${SPORTS_ROOT}/${split_name}"
  local split_txt="${SPORTS_SPLITS_DIR}/${split_name}.txt"
  local dst_split_dir="${TRACKEVAL_GT_ROOT}/SportsMOT-${split_name}"
  local seqmap_file="${TRACKEVAL_SEQMAP_DIR}/SportsMOT-${split_name}.txt"

  mkdir -p "${dst_split_dir}" "${TRACKEVAL_SEQMAP_DIR}"

  if [[ ! -f "${split_txt}" ]]; then
    warn "Missing split file: ${split_txt}"
    return
  fi
  if [[ ! -d "${src_split_dir}" ]]; then
    warn "Missing split directory: ${src_split_dir}"
    return
  fi

  local seq_names=()
  while IFS= read -r seq_name; do
    seq_name="$(echo "${seq_name}" | xargs)"
    [[ -n "${seq_name}" ]] || continue
    local seq_root="${src_split_dir}/${seq_name}"
    local gt_file="${seq_root}/gt/gt.txt"
    local seqinfo_file="${seq_root}/seqinfo.ini"
    if [[ ! -d "${seq_root}" ]]; then
      warn "Missing sequence folder: ${seq_root}"
      continue
    fi
    if [[ ! -f "${seqinfo_file}" ]]; then
      warn "Skipping ${seq_name}: missing seqinfo.ini"
      continue
    fi
    if [[ ! -f "${gt_file}" ]]; then
      warn "Skipping ${seq_name}: missing gt/gt.txt (cannot evaluate this split locally)"
      continue
    fi
    if [[ ! -e "${dst_split_dir}/${seq_name}" ]]; then
      ln -s "${seq_root}" "${dst_split_dir}/${seq_name}"
    fi
    seq_names+=("${seq_name}")
  done < "${split_txt}"

  if [[ "${#seq_names[@]}" -eq 0 ]]; then
    warn "No GT-ready sequences found for SportsMOT-${split_name}; not writing seqmap."
    return
  fi

  {
    echo "name"
    for seq in "${seq_names[@]}"; do
      echo "${seq}"
    done
  } > "${seqmap_file}"

  log "Prepared TrackEval GT split SportsMOT-${split_name} with ${#seq_names[@]} sequence(s)"
}

main() {
  ensure_tools

  if [[ ! -d "${SPORTS_ROOT}" ]]; then
    err "SportsMOT root not found: ${SPORTS_ROOT}"
    exit 1
  fi
  if [[ ! -d "${SPORTS_DATA_ARCHIVE_DIR}" ]]; then
    err "SportsMOT archive folder not found: ${SPORTS_DATA_ARCHIVE_DIR}"
    exit 1
  fi

  extract_split_if_needed "train"
  extract_split_if_needed "val"
  extract_split_if_needed "test"

  build_trackeval_gt_split "train"
  build_trackeval_gt_split "val"
  build_trackeval_gt_split "test"

  cat <<EOF
[DONE] SportsMOT prepared for TrackEval.

Prepared locations:
  Source dataset root: ${SPORTS_ROOT}
  TrackEval GT root:   ${TRACKEVAL_GT_ROOT}
  Seqmaps:             ${TRACKEVAL_SEQMAP_DIR}

Local eval note:
  SportsMOT test may not include public GT, so local TrackEval is usually train/val.
EOF
}

main "$@"
