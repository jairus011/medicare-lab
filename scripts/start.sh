#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python -m healthcare.verify
exec python -m streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-10000}" \
  --server.headless=true \
  --server.fileWatcherType=none \
  --browser.gatherUsageStats=false
