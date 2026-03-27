#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${ROOT_DIR}/benchmarks/datasets"

TOTAL_ERRORS=0
TOTAL_WARNINGS=0
TOTAL_PASSES=0

log() { echo "[INFO] $*"; }
pass() { echo "[PASS] $*"; TOTAL_PASSES=$((TOTAL_PASSES + 1)); }
warn() { echo "[WARN] $*"; TOTAL_WARNINGS=$((TOTAL_WARNINGS + 1)); }
fail() { echo "[FAIL] $*"; TOTAL_ERRORS=$((TOTAL_ERRORS + 1)); }

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/verify_layout.sh [options]

Options:
  --strict         Treat warnings as errors for exit code
  --dataset NAME   Verify only one dataset folder (DanceTrack|TeamTrack|SportsMOT|MOT17|MOT20)
  -h, --help       Show this help
EOF
}

STRICT=0
DATASET_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict)
      STRICT=1
      shift
      ;;
    --dataset)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --dataset" >&2
        usage
        exit 1
      fi
      DATASET_FILTER="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

dataset_enabled() {
  local name="$1"
  if [[ -z "${DATASET_FILTER}" ]]; then
    return 0
  fi
  [[ "${DATASET_FILTER}" == "${name}" ]]
}

check_required_dir() {
  local path="$1"
  local label="$2"
  if [[ -d "${path}" ]]; then
    pass "${label} exists: ${path}"
  else
    fail "${label} missing: ${path}"
  fi
}

check_required_file() {
  local path="$1"
  local label="$2"
  if [[ -f "${path}" ]]; then
    pass "${label} exists: ${path}"
  else
    fail "${label} missing: ${path}"
  fi
}

check_seqinfo_keys() {
  local seqinfo="$1"
  local seq_name="$2"
  local missing=0

  local keys=("name=" "imDir=" "frameRate=" "seqLength=" "imWidth=" "imHeight=" "imExt=")
  for k in "${keys[@]}"; do
    if rg -q "^\s*${k}" "${seqinfo}"; then
      :
    else
      warn "${seq_name}: seqinfo.ini missing key '${k%?}'"
      missing=1
    fi
  done

  if [[ "${missing}" -eq 0 ]]; then
    pass "${seq_name}: seqinfo.ini has expected keys"
  fi
}

verify_mot_sequence() {
  local seq_dir="$1"
  local seq_name
  seq_name="$(basename "${seq_dir}")"

  local img_dir="${seq_dir}/img1"
  local gt_file="${seq_dir}/gt/gt.txt"
  local seqinfo="${seq_dir}/seqinfo.ini"

  if [[ -d "${img_dir}" ]]; then
    pass "${seq_name}: img1/ exists"
  else
    fail "${seq_name}: missing img1/"
  fi

  if [[ -f "${gt_file}" ]]; then
    pass "${seq_name}: gt/gt.txt exists"
  else
    fail "${seq_name}: missing gt/gt.txt"
  fi

  if [[ -f "${seqinfo}" ]]; then
    pass "${seq_name}: seqinfo.ini exists"
    check_seqinfo_keys "${seqinfo}" "${seq_name}"
  else
    fail "${seq_name}: missing seqinfo.ini"
  fi

  if [[ -f "${gt_file}" ]]; then
    local line_count
    line_count="$(rg -n "." "${gt_file}" --count || true)"
    if [[ "${line_count}" == "0" ]]; then
      warn "${seq_name}: gt.txt is empty"
    else
      pass "${seq_name}: gt.txt has at least one annotation line"
    fi
  fi

  if [[ -d "${img_dir}" ]]; then
    local frame_count
    frame_count="$(rg --files "${img_dir}" | rg -c "\.(jpg|jpeg|png)$" || true)"
    if [[ "${frame_count}" == "0" ]]; then
      warn "${seq_name}: img1 has no image frames (*.jpg|*.jpeg|*.png)"
    else
      pass "${seq_name}: img1 has ${frame_count} frame files"
    fi
  fi
}

verify_dataset_root() {
  local dataset_name="$1"
  local dataset_path="${DATA_DIR}/${dataset_name}"
  local mot_style_hint="$2"

  log "Verifying dataset: ${dataset_name}"
  check_required_dir "${dataset_path}" "${dataset_name} root"

  if [[ ! -d "${dataset_path}" ]]; then
    return
  fi

  local seq_candidates=()
  while IFS= read -r line; do
    seq_candidates+=("${line}")
  done < <(rg --files -g "**/seqinfo.ini" "${dataset_path}" | sed 's#/seqinfo.ini$##' || true)

  if [[ "${#seq_candidates[@]}" -eq 0 ]]; then
    warn "${dataset_name}: no sequences with seqinfo.ini found under ${dataset_path}"
    if [[ "${mot_style_hint}" == "yes" ]]; then
      warn "${dataset_name}: expected MOT-style sequence folders with seqinfo.ini"
    fi
    return
  fi

  pass "${dataset_name}: found ${#seq_candidates[@]} sequence(s) with seqinfo.ini"

  local seq
  for seq in "${seq_candidates[@]}"; do
    verify_mot_sequence "${seq}"
  done
}

log "Benchmark layout verification"
check_required_dir "${DATA_DIR}" "datasets root"

if dataset_enabled "DanceTrack"; then
  verify_dataset_root "DanceTrack" "yes"
fi
if dataset_enabled "TeamTrack"; then
  verify_dataset_root "TeamTrack" "yes"
fi
if dataset_enabled "SportsMOT"; then
  verify_dataset_root "SportsMOT" "yes"
fi
if dataset_enabled "MOT17"; then
  verify_dataset_root "MOT17" "yes"
fi
if dataset_enabled "MOT20"; then
  verify_dataset_root "MOT20" "yes"
fi

echo
echo "========== Verification Summary =========="
echo "Passes:    ${TOTAL_PASSES}"
echo "Warnings:  ${TOTAL_WARNINGS}"
echo "Failures:  ${TOTAL_ERRORS}"
echo "=========================================="

if [[ "${TOTAL_ERRORS}" -gt 0 ]]; then
  exit 1
fi

if [[ "${STRICT}" -eq 1 && "${TOTAL_WARNINGS}" -gt 0 ]]; then
  exit 2
fi

exit 0
