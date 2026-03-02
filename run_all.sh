#!/usr/bin/env bash
# CONFHIT: Conformal Generative Design with Oracle-Free Guarantees
# Official repo: https://openreview.net/pdf?id=IruPup3KnX
# Run design, certification, and budget analysis for QED+Hgraph (N=7) and Molcraft (N=10).

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV="${CONDA_ENV:-chemprop}"

echo "=== CONFHIT Design ==="
conda run -n "$ENV" python src/design_main.py --config config/qed_hgraph_N7.json
conda run -n "$ENV" python src/design_main.py --config config/molcraft_N10.json

echo "=== Certification ==="
conda run -n "$ENV" python src/certification_main.py --config config/qed_hgraph_N7.json
conda run -n "$ENV" python src/certification_main.py --config config/molcraft_N10.json

echo "=== Budget analysis ==="
conda run -n "$ENV" python src/budget_analysis.py --config config/qed_hgraph_N7_budget.json
conda run -n "$ENV" python src/budget_analysis.py --config config/molcraft_N10_budget.json

echo "Done."
