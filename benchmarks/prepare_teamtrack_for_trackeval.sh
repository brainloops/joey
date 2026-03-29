#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEAMTRACK_ROOT="${ROOT_DIR}/benchmarks/datasets/TeamTrack"
TEAMTRACK_MOT_ZIP="${TEAMTRACK_ROOT}/teamtrack-mot-002.zip"
TEAMTRACK_MOT_ROOT="${TEAMTRACK_ROOT}/teamtrack-mot"
TEAMTRACK_RAW_ZIP="${TEAMTRACK_ROOT}/teamtrack-001.zip"
TEAMTRACK_RAW_ROOT="${TEAMTRACK_ROOT}/teamtrack"
TRACKEVAL_GT_ROOT="${ROOT_DIR}/benchmarks/trackeval_data/gt/mot_challenge"
TRACKEVAL_SEQMAP_DIR="${TRACKEVAL_GT_ROOT}/seqmaps"

SKIP_FRAME_EXTRACT=0
MAX_SEQS=0
SPLITS_CSV="train,val,test"

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
err() { echo "[ERROR] $*" >&2; }

usage() {
  cat <<'EOF'
Prepare TeamTrack MOT data into normalized split + TrackEval layouts.

Usage:
  bash benchmarks/prepare_teamtrack_for_trackeval.sh [options]

Options:
  --teamtrack-root PATH       TeamTrack dataset root (default: benchmarks/datasets/TeamTrack)
  --trackeval-gt-root PATH    TrackEval GT root (default: benchmarks/trackeval_data/gt/mot_challenge)
  --splits CSV                Comma-separated splits to prepare (default: train,val,test)
  --skip-frame-extract        Do not decode img1.mp4 into img1/*.jpg
  --max-seqs N                Process at most N sequences per split (0 = all)
  -h, --help                  Show this help

Notes:
  - The script expects TeamTrack MOT archive at:
      <teamtrack-root>/teamtrack-mot-002.zip
  - If sequence folders already have numeric img1/*.jpg frames, decoding is skipped.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --teamtrack-root)
      TEAMTRACK_ROOT="${2:-}"
      TEAMTRACK_MOT_ZIP="${TEAMTRACK_ROOT}/teamtrack-mot-002.zip"
      TEAMTRACK_MOT_ROOT="${TEAMTRACK_ROOT}/teamtrack-mot"
      shift 2
      ;;
    --trackeval-gt-root)
      TRACKEVAL_GT_ROOT="${2:-}"
      TRACKEVAL_SEQMAP_DIR="${TRACKEVAL_GT_ROOT}/seqmaps"
      shift 2
      ;;
    --splits)
      SPLITS_CSV="${2:-}"
      shift 2
      ;;
    --skip-frame-extract)
      SKIP_FRAME_EXTRACT=1
      shift
      ;;
    --max-seqs)
      MAX_SEQS="${2:-0}"
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

has_numeric_frames() {
  local img_dir="$1"
  if [[ ! -d "${img_dir}" ]]; then
    return 1
  fi
  shopt -s nullglob
  local f
  for f in "${img_dir}"/*.jpg "${img_dir}"/*.jpeg "${img_dir}"/*.png; do
    local stem="${f##*/}"
    stem="${stem%.*}"
    if [[ "${stem}" =~ ^[0-9]+$ ]]; then
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

decode_video_to_frames() {
  local video_path="$1"
  local out_dir="$2"

  mkdir -p "${out_dir}"
  if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -nostdin -hide_banner -loglevel error -y -i "${video_path}" "${out_dir}/%06d.jpg"
    return
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    err "Neither ffmpeg nor python3 is available to decode ${video_path}"
    exit 1
  fi

  python3 - <<PY
import cv2
from pathlib import Path

video = Path(r"${video_path}")
out_dir = Path(r"${out_dir}")
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {video}")

idx = 1
while True:
    ok, frame = cap.read()
    if not ok:
        break
    out_path = out_dir / f"{idx:06d}.jpg"
    cv2.imwrite(str(out_path), frame)
    idx += 1
cap.release()
print(f"[INFO] Decoded {idx - 1} frames from {video} -> {out_dir}")
PY
}

extract_member_from_zip() {
  local zip_path="$1"
  local member_path="$2"
  local dst_path="$3"

  if ! command -v python3 >/dev/null 2>&1; then
    return 2
  fi

  python3 - "$zip_path" "$member_path" "$dst_path" <<'PY'
import shutil
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
member_path = sys.argv[2]
dst_path = Path(sys.argv[3])
dst_path.parent.mkdir(parents=True, exist_ok=True)

if not zip_path.is_file():
    raise SystemExit(2)

with zipfile.ZipFile(zip_path) as zf:
    try:
        with zf.open(member_path) as src, dst_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    except KeyError:
        raise SystemExit(3)
PY
}

recover_missing_mp4() {
  local sport_name="$1"
  local split_name="$2"
  local seq_name="$3"
  local target_mp4="$4"

  local raw_member="teamtrack/${sport_name}/${split_name}/videos/${seq_name}.mp4"

  if [[ -f "${TEAMTRACK_RAW_ROOT}/${sport_name}/${split_name}/videos/${seq_name}.mp4" ]]; then
    ln -sf "${TEAMTRACK_RAW_ROOT}/${sport_name}/${split_name}/videos/${seq_name}.mp4" "${target_mp4}"
    return 0
  fi

  if [[ -f "${TEAMTRACK_RAW_ZIP}" ]]; then
    if extract_member_from_zip "${TEAMTRACK_RAW_ZIP}" "${raw_member}" "${target_mp4}"; then
      return 0
    fi
  fi

  return 1
}

extract_zip_if_needed() {
  if [[ -d "${TEAMTRACK_MOT_ROOT}" ]] && [[ -n "$(ls -A "${TEAMTRACK_MOT_ROOT}" 2>/dev/null)" ]]; then
    log "MOT archive already extracted: ${TEAMTRACK_MOT_ROOT}"
    return
  fi
  if [[ ! -f "${TEAMTRACK_MOT_ZIP}" ]]; then
    err "Missing TeamTrack MOT zip: ${TEAMTRACK_MOT_ZIP}"
    exit 1
  fi
  if ! command -v unzip >/dev/null 2>&1; then
    err "unzip is required."
    exit 1
  fi
  log "Extracting ${TEAMTRACK_MOT_ZIP} -> ${TEAMTRACK_ROOT}"
  unzip -q -n "${TEAMTRACK_MOT_ZIP}" -d "${TEAMTRACK_ROOT}"
}

prepare_split() {
  local split_name="$1"
  local normalized_split_root="${TEAMTRACK_ROOT}/${split_name}"
  local trackeval_split_root="${TRACKEVAL_GT_ROOT}/TeamTrack-${split_name}"
  local seqmap_file="${TRACKEVAL_SEQMAP_DIR}/TeamTrack-${split_name}.txt"

  mkdir -p "${normalized_split_root}" "${trackeval_split_root}" "${TRACKEVAL_SEQMAP_DIR}"

  local count=0
  local recovered_mp4_count=0
  local missing_media_count=0
  local seq_names=()
  local sport_dir
  for sport_dir in "${TEAMTRACK_MOT_ROOT}"/*; do
    [[ -d "${sport_dir}" ]] || continue
    local sport_name
    sport_name="$(basename "${sport_dir}")"
    local split_dir="${sport_dir}/${split_name}"
    [[ -d "${split_dir}" ]] || continue

    local seq_dir
    for seq_dir in "${split_dir}"/*; do
      [[ -d "${seq_dir}" ]] || continue

      local seq_name
      seq_name="$(basename "${seq_dir}")"
      local normalized_seq="${normalized_split_root}/${seq_name}"
      if [[ -e "${normalized_seq}" && ! -L "${normalized_seq}" ]]; then
        warn "Normalized path exists but is not symlink: ${normalized_seq}; skipping"
        continue
      fi
      if [[ -L "${normalized_seq}" ]]; then
        local target
        target="$(readlink -f "${normalized_seq}")"
        local current
        current="$(readlink -f "${seq_dir}")"
        if [[ "${target}" != "${current}" ]]; then
          warn "Sequence name collision at ${normalized_seq}; keeping existing target ${target}"
          continue
        fi
      else
        ln -s "${seq_dir}" "${normalized_seq}"
      fi

      local seqinfo="${seq_dir}/seqinfo.ini"
      local gt="${seq_dir}/gt/gt.txt"
      if [[ ! -f "${seqinfo}" ]]; then
        warn "Skipping ${sport_name}/${split_name}/${seq_name}: missing seqinfo.ini"
        continue
      fi
      if [[ ! -f "${gt}" ]]; then
        warn "Skipping ${sport_name}/${split_name}/${seq_name}: missing gt/gt.txt"
        continue
      fi

      local img_dir="${seq_dir}/img1"
      local mp4="${seq_dir}/img1.mp4"
      if [[ ! -f "${mp4}" ]]; then
        if recover_missing_mp4 "${sport_name}" "${split_name}" "${seq_name}" "${mp4}"; then
          recovered_mp4_count=$((recovered_mp4_count + 1))
        fi
      fi
      if [[ "${SKIP_FRAME_EXTRACT}" -eq 0 ]]; then
        if has_numeric_frames "${img_dir}"; then
          :
        elif [[ -f "${mp4}" ]]; then
          log "Decoding frames for ${split_name}/${seq_name}"
          decode_video_to_frames "${mp4}" "${img_dir}"
        else
          warn "No numeric img1 frames and no img1.mp4 found: ${seq_dir}"
          missing_media_count=$((missing_media_count + 1))
        fi
      else
        if [[ ! -f "${mp4}" ]] && ! has_numeric_frames "${img_dir}"; then
          missing_media_count=$((missing_media_count + 1))
        fi
      fi

      if [[ ! -e "${trackeval_split_root}/${seq_name}" ]]; then
        ln -s "${normalized_seq}" "${trackeval_split_root}/${seq_name}"
      fi
      seq_names+=("${seq_name}")
      count=$((count + 1))
      if [[ "${MAX_SEQS}" -gt 0 && "${count}" -ge "${MAX_SEQS}" ]]; then
        break 2
      fi
    done
  done

  if [[ "${#seq_names[@]}" -eq 0 ]]; then
    warn "No sequences prepared for TeamTrack-${split_name}"
    return
  fi

  {
    echo "name"
    for seq in "${seq_names[@]}"; do
      echo "${seq}"
    done
  } > "${seqmap_file}"
  log "Prepared TeamTrack-${split_name} with ${#seq_names[@]} sequence(s) | recovered_mp4=${recovered_mp4_count} missing_media=${missing_media_count}"
}

main() {
  if [[ ! -d "${TEAMTRACK_ROOT}" ]]; then
    err "TeamTrack root not found: ${TEAMTRACK_ROOT}"
    exit 1
  fi

  extract_zip_if_needed

  IFS=',' read -r -a splits <<< "${SPLITS_CSV}"
  local split
  for split in "${splits[@]}"; do
    split="$(echo "${split}" | xargs)"
    [[ -n "${split}" ]] || continue
    prepare_split "${split}"
  done

  cat <<EOF
[DONE] TeamTrack prepared for TrackEval.

Prepared locations:
  Source dataset root: ${TEAMTRACK_ROOT}
  Normalized splits:   ${TEAMTRACK_ROOT}/{train,val,test}
  TrackEval GT root:   ${TRACKEVAL_GT_ROOT}
  Seqmaps:             ${TRACKEVAL_SEQMAP_DIR}
EOF
}

main "$@"
