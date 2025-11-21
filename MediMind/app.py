from flask import Flask, render_template, request, jsonify, session
import pickle
import numpy as np
import pandas as pd
import secrets
from rapidfuzz import process, fuzz
from datetime import datetime

app = Flask(__name__)
# This key allows the AI to remember conversation history
app.secret_key = secrets.token_hex(16)

# --- LOAD AI & METADATA ---
model = None
data_columns = []
meta = {}
df = None
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    # metadata (columns, classes, feature_importances, metrics)
    try:
        with open('meta.pkl', 'rb') as f:
            meta = pickle.load(f)
            data_columns = meta.get('columns', [])
    except Exception:
        # fallback to old columns file
        with open("columns.pkl", "rb") as f:
            data_columns = pickle.load(f)
            meta = {'columns': data_columns}

    # Load data for symptom logic
    df = pd.read_csv('training_data.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print("✅ AI Brain Loaded.")
except Exception as e:
    print(f"❌ Error: Run 'python train_model.py' first! ({e})")
    model = None

# --- SYMPTOM DICTIONARY (synonyms / keywords) ---
SYMPTOM_DICT = {
    "fever": ["high_fever", "mild_fever"],
    "cold": ["continuous_sneezing", "chills", "runny_nose"],
    "cough": ["cough", "phlegm", "breathlessness"],
    "headache": ["headache", "dizziness"],
    "stomach": ["stomach_pain", "acidity", "vomiting"],
    "skin": ["skin_rash", "itching"],
    "chest": ["chest_pain"]
}


def extract_symptoms(text, score_cutoff=75):
    """Extract symptoms from free text using exact and fuzzy matching.
    Returns a list of symptom column names.
    """
    if not text:
        return []
    text = text.lower()
    found = set()

    # 1) Check dictionary keywords (explicit words -> mapped columns)
    for key, values in SYMPTOM_DICT.items():
        if key in text:
            found.update(values)

    # 2) Exact column name matches (allow underscores mapped to spaces)
    for col in data_columns:
        clean_col = col.replace('_', ' ')
        if clean_col in text:
            found.add(col)

    # 3) Fuzzy match tokens/phrases against known symptom columns
    # Build a list of human-friendly names for matching
    choices = {col: col.replace('_', ' ') for col in data_columns}
    # also include symptom dictionary values and keys
    for key, vals in SYMPTOM_DICT.items():
        choices[key] = key
        for v in vals:
            choices[v] = v.replace('_', ' ')

    # Use rapidfuzz to find close matches; split text into candidate phrases
    tokens = [t.strip() for t in text.replace('/', ' ').split() if t.strip()]
    # Try whole text first
    if choices:
        matches = process.extract(
            text, choices, scorer=fuzz.token_sort_ratio, limit=10
        )
        for match, score, _ in matches:
            if score >= score_cutoff:
                # find original key for this friendly name
                # match is the key from choices mapping
                found.add(match)

        # Try tokens individually for short user inputs
        for tok in tokens:
            matches = process.extract(tok, choices, scorer=fuzz.partial_ratio, limit=5)
            for match, score, _ in matches:
                if score >= score_cutoff:
                    found.add(match)

    return list(found)


@app.route('/')
def home():
    session.clear()  # Reset memory on refresh
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({'error': 'Model not loaded. Run training first.'}), 500

    data = request.get_json()
    user_text = data.get('symptoms', '').lower().strip()

    # Initialize memory if needed
    if 'symptoms' not in session:
        session['symptoms'] = []
    if 'last_question' not in session:
        session['last_question'] = None

    # --- HANDLE CONVERSATION ---
    # Did user answer a previous question?
    if session.get('last_question') and user_text in ['yes', 'y', 'yeah', 'sure']:
        session['symptoms'].append(session['last_question'])
        session['last_question'] = None
    elif session.get('last_question') and user_text in ['no', 'n', 'nope']:
        session['last_question'] = None  # Ignore that symptom

    # Find new symptoms in text with fuzzy matching
    new_syms = extract_symptoms(user_text)
    for s in new_syms:
        if s not in session['symptoms']:
            session['symptoms'].append(s)

    current_symptoms = session['symptoms']

    if not current_symptoms:
        return jsonify({'diagnosis': [{'condition': "I'm listening...", 'probability': "", 'recommendation': "Please describe your symptoms (e.g. 'I have a headache')."}]})

    # --- PREDICT (build input vector) ---
    input_vector = np.zeros(len(data_columns), dtype=int)
    for s in current_symptoms:
        if s in data_columns:
            input_vector[data_columns.index(s)] = 1

    # Safe predict_proba
    try:
        probs = model.predict_proba([input_vector])[0]
    except Exception:
        # fallback to predict -> uniform low-confidence
        preds = model.predict([input_vector])
        probs = np.zeros(len(getattr(model, 'classes_', [preds[0]])))
        probs[0] = 1.0

    # Get top-3 diagnoses
    top_indices = np.argsort(probs)[::-1][:3]
    diagnoses = []
    for idx in top_indices:
        cond = model.classes_[idx]
        prob = float(probs[idx]) * 100
        # Reason: count matching symptoms and list a couple supporting symptoms
        reason = ''
        try:
            disease_rows = df[df['prognosis'] == cond]
            # aggregate symptom columns that are present for this disease
            supporting = []
            for col in data_columns:
                if disease_rows[col].sum() > 0 and col in current_symptoms:
                    supporting.append(col.replace('_', ' '))
            if supporting:
                reason = 'matching symptoms: ' + ', '.join(supporting)
            else:
                # pick top 2 associated symptoms for the disease
                assoc = [c for c in data_columns if disease_rows[c].sum() > 0]
                reason = 'common symptoms: ' + ', '.join([c.replace('_', ' ') for c in assoc[:2]]) if assoc else ''
        except Exception:
            reason = ''

        # Provide a human-friendly recommendation field (fallback if reason empty)
        recommendation = reason if reason else 'Please consult a doctor for confirmation.'
        diagnoses.append({'condition': cond, 'probability': f"{prob:.1f}%", 'reason': reason, 'recommendation': recommendation})

    # --- ACTIVE QUESTIONING: if top-1 confidence low ask about the most informative unseen symptom ---
    top_prob = float(probs[top_indices[0]]) * 100
    if top_prob < 60:
        top_disease = model.classes_[top_indices[0]]
        disease_rows = df[df['prognosis'] == top_disease]
        # candidate unseen symptoms
        potential_symptoms = [col for col in data_columns if disease_rows[col].sum() > 0 and col not in current_symptoms]

        next_question = None
        # Use saved feature importances if available
        fi = meta.get('feature_importances')
        if fi:
            # choose symptom with highest importance
            ranked = sorted(potential_symptoms, key=lambda s: fi[data_columns.index(s)] if s in data_columns and data_columns.index(s) < len(fi) else 0, reverse=True)
            if ranked:
                next_question = ranked[0]
        else:
            # fallback: choose most frequently associated symptom
            ranked = sorted(potential_symptoms, key=lambda s: disease_rows[s].sum(), reverse=True)
            if ranked:
                next_question = ranked[0]

        if next_question:
            session['last_question'] = next_question
            clean_name = next_question.replace('_', ' ')
            return jsonify({'diagnosis': [{'condition': "analyzing...", 'probability': f"{top_prob:.1f}%", 'recommendation': f"Do you also have {clean_name}?"}], 'candidates': diagnoses})

    # final output: return top-k diagnoses with reasons
    # Ensure top-level recommendation is present for UI consumers
    top_level_reco = 'This is a probabilistic assistant — consult a doctor for confirmation.'
    return jsonify({'diagnosis': diagnoses, 'recommendation': top_level_reco})


if __name__ == '__main__':
    app.run(debug=True, port=5000)