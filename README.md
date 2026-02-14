# IoT Healthcare Risk & Energy-Aware Stratified MLP

This project provides a **Python + Streamlit** implementation for:

- Stratified multilayer perceptron (MLP) risk prediction for healthcare data.
- Explainable AI outputs (global permutation importance + local feature drivers).
- Energy-aware edge/cloud allocation simulation for IoT healthcare workflows.

## Open-source dataset
The app uses the open-source **Breast Cancer Wisconsin** dataset bundled with scikit-learn.

## Quick run (copy-paste)
```bash
cd /workspace/test
./quick_run.sh
```
Then open your browser at: `http://localhost:8501`

## Quick run (recommended)
```bash
pip install -r requirements.txt
python run_app.py
```
Then open your browser at: `http://localhost:8501`

## Alternative run
```bash
streamlit run app.py
```

## What the UI includes
- Model controls: hidden layers, learning rate, iterations, stratified CV folds.
- Metrics: ROC-AUC, accuracy, CV performance.
- Explainability:
  - Global feature importance with permutation importance.
  - Local explanation by top normalized feature magnitudes for a selected patient.
- Energy-aware operations:
  - Edge/cloud split by confidence threshold.
  - Energy accounting for edge power, cloud power, and network overhead.

## Troubleshooting
- If `streamlit` is not found, run: `pip install -r requirements.txt` again.
- If port `8501` is busy, run with another port:
  `streamlit run app.py --server.port 8502`

## Notes
This implementation is intended as a practical reference prototype for research-style workflows in IoT healthcare analytics.


## Browser access
- Start with `./quick_run.sh` (or `python run_app.py`).
- Open: `http://localhost:8501` in Chrome/Firefox/Edge/Safari.
- Keep the terminal running while using the UI; press `Ctrl+C` to stop.
- If Streamlit prompts for telemetry on first run, just press Enter once.
