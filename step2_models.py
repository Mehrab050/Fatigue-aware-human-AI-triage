####################################################################################################
# 02 · Two prediction models
#
# HOW TO USE THIS FILE IN GOOGLE COLAB
#   Each block between two "CELL" lines is one Colab cell.
#   1. In Colab click  + Code  to make a new empty cell.
#   2. Copy one block (from its CELL line down to the next CELL line) and paste it in.
#   3. Press Shift+Enter to run it. Wait until it finishes, then do the next block.
#   Lines starting with # are explanations; Python ignores them.
####################################################################################################
# To decide who should review a case we need, for every case, the expected cost of giving it to each decider:
#
# * AI model: probability that the case is fraud. The AI flags a case when that is the cheaper decision.
# * Analyst outcome model: for a given case *and* a given analyst, the probability of each outcome:
#   correct clear (TN), false alarm (FP), miss (FN), correct flag (TP).
#
# Expected cost = 0.057 × P(false alarm) + 1 × P(miss).
#
# Both models are gradient-boosted trees trained only on the training months (3–6).
# The test month (7) is used only for evaluation.

# ============================== CELL 1 ==============================
# --- Paths (same cell in every notebook) ---
# On Colab, files are kept in your Google Drive so that every notebook can read
# what the previous notebook saved. Off Colab, a local folder is used instead.
import os
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/fatigue_triage'
except ImportError:
    BASE = os.environ.get('PROJECT_DIR', 'fatigue_triage')

DATA = os.path.join(BASE, 'data')        # input data copied from the public benchmark
OUT = os.path.join(BASE, 'outputs')      # model outputs and result tables
FIG = os.path.join(BASE, 'figures')      # saved figures
for folder in (DATA, OUT, FIG):
    os.makedirs(folder, exist_ok=True)

LAM = 0.057    # cost of one false alarm, relative to a missed fraud (missed fraud costs 1)
ANALYSTS = [f'standard#{i}' for i in range(9)]   # analyst ids used in the dataset
SCENARIOS = ['homogenous', 'variable-1', 'variable-2', 'variable-3', 'variable-4']
import pandas as pd
pd.set_option('display.max_columns', 50)   # show whole tables, not "..."
pd.set_option('display.width', 200)
print('Project folder:', BASE)

# ============================== CELL 2 ==============================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

alerts = pd.read_parquet(f'{DATA}/alerts.parquet')
train = pd.read_parquet(f'{DATA}/train.parquet')
train['model_score'] = alerts.loc[train.index, 'model_score']   # fix found in notebook 01
test = pd.read_parquet(f'{DATA}/test.parquet')
test_dec = pd.read_parquet(f'{DATA}/test_decisions.parquet').loc[test.index]

FEATURES = [c for c in test.columns if c not in ('fraud_label', 'month')]
y_train = train['fraud_bool'].astype(int).values
y_test = test['fraud_label'].astype(int).values
print(len(FEATURES), 'features')

# ============================== CELL 3 ==============================
# 1. AI model
#
# Decision rule: flag when expected cost of flagging is lower than expected cost of clearing,
# i.e. when 0.057 × (1 − p) < p, which gives the threshold p ≥ 0.057 / 1.057 ≈ 0.054.

ai_model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          categorical_features='from_dtype', random_state=0)
ai_model.fit(train[FEATURES], y_train)

p_fraud = ai_model.predict_proba(test[FEATURES])[:, 1]
THRESHOLD = LAM / (1 + LAM)
ai_decision = (p_fraud >= THRESHOLD).astype(int)

fp = int(((ai_decision == 1) & (y_test == 0)).sum())
fn = int(((ai_decision == 0) & (y_test == 1)).sum())
print(f'Test AUC: {roc_auc_score(y_test, p_fraud):.3f}')
print(f'AI alone on all test cases: {fp} false alarms, {fn} misses, cost = {LAM * fp + fn:.1f}')

# ============================== CELL 4 ==============================
# Are the probabilities honest? Points near the diagonal = well calibrated.
obs, pred = calibration_curve(y_test, p_fraud, n_bins=10, strategy='quantile')
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot([0, 1], [0, 1], color='grey', ls='--', lw=0.8)
ax.plot(pred, obs, marker='o')
ax.set_xlabel('predicted fraud probability'); ax.set_ylabel('observed fraud rate')
ax.set_title('AI model calibration (test month)')
lim = max(pred.max(), obs.max()) * 1.1; ax.set_xlim(0, lim); ax.set_ylim(0, lim)
fig.tight_layout(); fig.savefig(f'{FIG}/02_ai_calibration.png', dpi=150); plt.show()

# ============================== CELL 5 ==============================
# 2. Analyst outcome model
#
# Training data: each case with the analyst who reviewed it. Target: one of four outcomes.
# The analyst id is an input, so one model learns how each analyst behaves on each kind of case.

OUTCOMES = ['TN', 'FP', 'FN', 'TP']
decision = train['decision'].astype(int).values
outcome = 2 * y_train + decision          # 0 = TN, 1 = FP, 2 = FN, 3 = TP

X_train = train[FEATURES].copy()
X_train['analyst'] = pd.Categorical(train['assignment'], categories=ANALYSTS)

analyst_model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                               categorical_features='from_dtype', random_state=0)
analyst_model.fit(X_train, outcome)

# Predict the four outcome probabilities for every test case and every analyst
probs = np.zeros((len(test), len(ANALYSTS), 4))
for k, a in enumerate(ANALYSTS):
    X = test[FEATURES].copy()
    X['analyst'] = pd.Categorical([a] * len(test), categories=ANALYSTS)
    probs[:, k, :] = analyst_model.predict_proba(X)
print('probs shape (cases, analysts, outcomes):', probs.shape)

# ============================== CELL 6 ==============================
# Check: does the model get each analyst right?
#
# On the test month we know what every analyst actually decided, so we can compare the model's predicted
# false-alarm and miss counts with the real ones.

rows = []
for k, a in enumerate(ANALYSTS):
    d = test_dec[a].values
    rows.append(dict(analyst=a,
                     predicted_FP=probs[:, k, 1].sum(), actual_FP=((d == 1) & (y_test == 0)).sum(),
                     predicted_FN=probs[:, k, 2].sum(), actual_FN=((d == 0) & (y_test == 1)).sum()))
check = pd.DataFrame(rows).set_index('analyst').round(0)
print(check)

# ============================== CELL 7 ==============================
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, kind in zip(axes, ['FP', 'FN']):
    ax.scatter(check[f'actual_{kind}'], check[f'predicted_{kind}'])
    m = max(check[f'actual_{kind}'].max(), check[f'predicted_{kind}'].max()) * 1.1
    ax.plot([0, m], [0, m], color='grey', ls='--', lw=0.8)
    ax.set_xlabel(f'actual {kind}'); ax.set_ylabel(f'predicted {kind}')
    ax.set_title({'FP': 'False alarms per analyst', 'FN': 'Misses per analyst'}[kind])
fig.tight_layout(); fig.savefig(f'{FIG}/02_analyst_model_check.png', dpi=150); plt.show()

# ============================== CELL 8 ==============================
# 3. Save everything the next notebooks need

np.savez(f'{OUT}/predictions.npz', probs=probs, p_fraud=p_fraud, ai_decision=ai_decision,
         y_test=y_test, human_decisions=test_dec[ANALYSTS].values.astype(int), case_id=test.index.values)
print('Saved', f'{OUT}/predictions.npz')

# Open 03_assignment.ipynb next.
