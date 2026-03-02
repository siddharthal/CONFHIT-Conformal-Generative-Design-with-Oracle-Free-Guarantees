#!/usr/bin/env bash
# Copy required data and model files from a source repo (e.g. sequential_testing).
# Usage: ./scripts/copy_required_data.sh [SOURCE_DIR]
# Default SOURCE_DIR: ../sequential_testing (sibling of this repo).

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
BASE="${1:-$(cd "$REPO_ROOT/.." && pwd)/sequential_testing}"

if [[ ! -d "$BASE" ]]; then
  echo "Source not found: $BASE"
  echo "Usage: $0 [SOURCE_DIR]"
  echo "Example: $0 /path/to/sequential_testing"
  exit 1
fi

mkdir -p data/calib data/generated_samples trained_models

copy_if() {
  local src="$1" dest="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dest"
    echo "  $dest"
  else
    echo "  MISSING: $src" >&2
    return 1
  fi
}

echo "Copying from $BASE"
echo "Calibration data:"
copy_if "$BASE/validation_datasets/qed_valid.csv" data/calib/qed_calib.csv
# Drop last column (e.g. "train") from dock calibration file
if [[ -f "$BASE/validation_datasets/dock_valid_valid_no_train.csv" ]]; then
  python3 -c "
import pandas as pd
import sys
df = pd.read_csv(sys.argv[1])
df.iloc[:, :-1].to_csv('data/calib/dock_calib.csv', index=False)
" "$BASE/validation_datasets/dock_valid_valid_no_train.csv" && echo "  data/calib/dock_calib.csv"
else
  echo "  MISSING: $BASE/validation_datasets/dock_valid_valid_no_train.csv" >&2
fi
echo "Generated samples:"
copy_if "$BASE/generated_samples/qed_hgraph.csv" data/generated_samples/qed_hgraph.csv
copy_if "$BASE/generated_samples/dock_test_molcraft.csv" data/generated_samples/dock_test_molcraft.csv
echo "Model (for QED; optional for Molcraft if using precomputed features):"
if [[ -f "$BASE/trained_models/binary_qed.ckpt" ]]; then
  copy_if "$BASE/trained_models/binary_qed.ckpt" trained_models/binary_qed.ckpt
else
  echo "  SKIP: $BASE/trained_models/binary_qed.ckpt not found (QED design will need it)"
fi
echo "Done."
