#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BENCHMARK="MOT17"
DETECTOR="FRCNN"  # MOT17 only: FRCNN|DPM|SDP|all
DATASET_ROOT_OVERRIDE=""
TRACKEVAL_GT_ROOT="${ROOT_DIR}/benchmarks/trackeval_data/gt/mot_challenge"
TRACKEVAL_SEQMAP_DIR="${TRACKEVAL_GT_ROOT}/seqmaps"

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
err() { echo "[ERROR] $*" >&2; }

usage() {
  cat <<'EOF'
Prepare MOTChallenge datasets into TrackEval GT/seqmap layout.

Usage:
  bash benchmarks/prepare_motchallenge_for_trackeval.sh [options]

Options:
  --benchmark NAME       Dataset benchmark name: MOT17 or MOT20 (default: MOT17)
  --detector NAME        MOT17 detector subset: FRCNN|DPM|SDP|all (default: FRCNN)
                         Ignored for MOT20.
  --dataset-root PATH    Override dataset root (default: benchmarks/datasets/<benchmark>)
  --trackeval-gt-root    Override TrackEval GT root
  -h, --help             Show this help

Examples:
  bash benchmarks/prepare_motchallenge_for_trackeval.sh
  bash benchmarks/prepare_motchallenge_for_trackeval.sh --benchmark MOT17 --detector SDP
  bash benchmarks/prepare_motchallenge_for_trackeval.sh --benchmark MOT20
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
      BENCHMARK="${2:-}"
      shift 2
      ;;
    --detector)
      DETECTOR="${2:-}"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT_OVERRIDE="${2:-}"
      shift 2
      ;;
    --trackeval-gt-root)
      TRACKEVAL_GT_ROOT="${2:-}"
      TRACKEVAL_SEQMAP_DIR="${TRACKEVAL_GT_ROOT}/seqmaps"
      shift 2
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

BENCHMARK_UPPER="$(echo "${BENCHMARK}" | tr '[:lower:]' '[:upper:]')"
if [[ "${BENCHMARK_UPPER}" != "MOT17" && "${BENCHMARK_UPPER}" != "MOT20" ]]; then
  err "--benchmark must be MOT17 or MOT20"
  exit 1
fi
BENCHMARK="${BENCHMARK_UPPER}"

DETECTOR_UPPER="$(echo "${DETECTOR}" | tr '[:lower:]' '[:upper:]')"
if [[ "${BENCHMARK}" == "MOT17" ]]; then
  if [[ "${DETECTOR_UPPER}" != "FRCNN" && "${DETECTOR_UPPER}" != "DPM" && "${DETECTOR_UPPER}" != "SDP" && "${DETECTOR_UPPER}" != "ALL" ]]; then
    err "--detector for MOT17 must be FRCNN, DPM, SDP, or all"
    exit 1
  fi
else
  if [[ "${DETECTOR_UPPER}" != "FRCNN" ]]; then
    warn "--detector is ignored for MOT20."
  fi
fi

if [[ -n "${DATASET_ROOT_OVERRIDE}" ]]; then
  DATASET_ROOT="${DATASET_ROOT_OVERRIDE}"
else
  DATASET_ROOT="${ROOT_DIR}/benchmarks/datasets/${BENCHMARK}"
fi

should_include_sequence() {
  local seq_name="$1"
  if [[ "${BENCHMARK}" == "MOT17" && "${DETECTOR_UPPER}" != "ALL" ]]; then
    [[ "${seq_name}" == *"-${DETECTOR_UPPER}" ]]
    return
  fi
  return 0
}

build_trackeval_gt_split() {
  local split_name="$1"
  local src_split_dir="${DATASET_ROOT}/${split_name}"
  local dst_split_dir="${TRACKEVAL_GT_ROOT}/${BENCHMARK}-${split_name}"
  local seqmap_file="${TRACKEVAL_SEQMAP_DIR}/${BENCHMARK}-${split_name}.txt"

  mkdir -p "${dst_split_dir}" "${TRACKEVAL_SEQMAP_DIR}"

  local seq_names=()
  for seq_dir in "${src_split_dir}"/*; do
    [[ -d "${seq_dir}" ]] || continue
    local seq_name
    seq_name="$(basename "${seq_dir}")"

    if ! should_include_sequence "${seq_name}"; then
      continue
    fi

    local gt_file="${seq_dir}/gt/gt.txt"
    local seqinfo_file="${seq_dir}/seqinfo.ini"

    if [[ ! -f "${seqinfo_file}" ]]; then
      warn "Skipping ${seq_name}: missing seqinfo.ini"
      continue
    fi
    if [[ ! -f "${gt_file}" ]]; then
      warn "Skipping ${seq_name}: missing gt/gt.txt (cannot evaluate this split locally)"
      continue
    fi

    if [[ ! -e "${dst_split_dir}/${seq_name}" ]]; then
      ln -s "${seq_dir}" "${dst_split_dir}/${seq_name}"
    fi
    seq_names+=("${seq_name}")
  done

  if [[ "${#seq_names[@]}" -eq 0 ]]; then
    warn "No GT-ready sequences found for ${BENCHMARK}-${split_name}; not writing seqmap."
    return
  fi

  {
    echo "name"
    for seq in "${seq_names[@]}"; do
      echo "${seq}"
    done
  } > "${seqmap_file}"

  log "Prepared TrackEval GT split ${BENCHMARK}-${split_name} with ${#seq_names[@]} sequence(s)"
}

main() {
  if [[ ! -d "${DATASET_ROOT}" ]]; then
    err "Dataset root not found: ${DATASET_ROOT}"
    exit 1
  fi

  if [[ ! -d "${DATASET_ROOT}/train" ]]; then
    err "Expected split folder missing: ${DATASET_ROOT}/train"
    exit 1
  fi

  log "Preparing ${BENCHMARK} for TrackEval"
  log "Dataset root: ${DATASET_ROOT}"
  log "TrackEval GT root: ${TRACKEVAL_GT_ROOT}"
  if [[ "${BENCHMARK}" == "MOT17" ]]; then
    log "Detector subset: ${DETECTOR_UPPER}"
  fi

  build_trackeval_gt_split "train"

  cat <<EOF
[DONE] ${BENCHMARK} prepared for TrackEval (train split).

Prepared locations:
  Source dataset root: ${DATASET_ROOT}
  TrackEval GT root:   ${TRACKEVAL_GT_ROOT}
  Seqmaps:             ${TRACKEVAL_SEQMAP_DIR}

Local eval note:
  MOTChallenge train has GT; test GT is not public, so local TrackEval is typically train-only.
EOF
}

main "$@"
