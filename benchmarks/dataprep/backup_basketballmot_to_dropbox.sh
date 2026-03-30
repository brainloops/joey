#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="/home/david/projects/joey/benchmarks/datasets/BasketballMOT"
DROPBOX_ROOT="/media/david/beast/Dropbox/basketball-dataset"
DEST_DIR="${DROPBOX_ROOT}/BasketballMOT"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "ERROR: Source directory does not exist: ${SOURCE_DIR}" >&2
  exit 1
fi

if [[ ! -d "${DROPBOX_ROOT}" ]]; then
  echo "ERROR: Dropbox root directory does not exist: ${DROPBOX_ROOT}" >&2
  exit 1
fi

echo "Backing up BasketballMOT dataset..."
echo "  Source: ${SOURCE_DIR}"
echo "  Destination: ${DEST_DIR}"

# Manual workflow: fully overwrite destination each run.
rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}"
rsync -a --delete "${SOURCE_DIR}/" "${DEST_DIR}/"

echo "Backup complete."
