#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.streamlit"
if ! grep -q "gatherUsageStats" "$HOME/.streamlit/config.toml" 2>/dev/null; then
  cat >> "$HOME/.streamlit/config.toml" <<'EOF'
[browser]
gatherUsageStats = false
EOF
fi

python -m pip install -r requirements.txt
python run_app.py
