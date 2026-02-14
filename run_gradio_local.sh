#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
python - <<'PY'
from gradio_app import build_demo

demo = build_demo()
demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
PY
