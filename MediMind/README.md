# MediMind

Lightweight symptom-to-diagnosis assistant.

Quick start

1. Create a Python virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train the model (this creates `model.pkl` and `meta.pkl`):

```powershell
python train_model.py
```

3. Run the app:

```powershell
python app.py
```

Notes

- `train_model.py` now trains with a train/test split, uses class balancing, optionally calibrates probabilities, prints evaluation metrics, and saves metadata to `meta.pkl`.
- `app.py` loads `meta.pkl` (falls back to `columns.pkl`) and uses fuzzy matching (via `rapidfuzz`) for symptom extraction, returns top-3 diagnoses, and asks focused follow-ups using feature importances saved in `meta.pkl`.
- For explainability, `shap` is included in `requirements.txt`; computing SHAP explanations in real time may be slow—consider precomputing or enabling it conditionally.
Things to add in the future:
1. Quick follow-up for yes/no questions
2. Server integration and login page
3. Store previous results and diagnosis for quick pull up
4. Database to securely store the data
