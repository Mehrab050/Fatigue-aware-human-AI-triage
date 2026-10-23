####################################################################################################
# 01 · Explore the data
#
# HOW TO USE THIS FILE IN GOOGLE COLAB
#   Each block between two "CELL" lines is one Colab cell.
#   1. In Colab click  + Code  to make a new empty cell.
#   2. Copy one block (from its CELL line down to the next CELL line) and paste it in.
#   3. Press Shift+Enter to run it. Wait until it finishes, then do the next block.
#   Lines starting with # are explanations; Python ignores them.
####################################################################################################
# Questions for this notebook:
# 1. What does one case look like, and how common is fraud among the alerts?
# 2. How do the 9 analysts differ? We describe each one with signal detection theory:
#    *sensitivity* (d′, how well they separate fraud from legitimate cases) and *criterion* (c, how willing
#    they are to flag a case).
# 3. How much work can each analyst take (capacity)?

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
from scipy.stats import norm

alerts = pd.read_parquet(f'{DATA}/alerts.parquet')
train = pd.read_parquet(f'{DATA}/train.parquet')
test = pd.read_parquet(f'{DATA}/test.parquet')
test_dec = pd.read_parquet(f'{DATA}/test_decisions.parquet').loc[test.index]

print('Training cases:', len(train), '| months', sorted(train['month'].unique()))
print('Test cases:    ', len(test), '| month', sorted(test['month'].unique()))
print(f"Fraud rate: train {train['fraud_bool'].mean():.1%}, test {test['fraud_label'].mean():.1%}")

# ============================== CELL 3 ==============================
# 1. What we observe for each case
#
# * Training data: each case was reviewed by one analyst. We see who reviewed it (assignment),
#   what they decided (decision, 1 = flag as fraud) and the true label. This is realistic: in practice a case
#   is never reviewed by everyone.
# * Test data: for evaluation only, the benchmark gives the decision every analyst would have made on
#   every test case. This lets us score any assignment rule fairly.

print(train[['assignment', 'decision', 'fraud_bool']].head())

# ============================== CELL 4 ==============================
print(test_dec.head())   # one column per analyst (0/1). The extra column model#0 belongs to the original study; we do not use it.

# ============================== CELL 5 ==============================
# 2. A data check: the model_score column
#
# model_score is the score of the system that raised the alert. Its average is very different in the training
# file than in the test file. In the original alerts file the same cases have a consistent score, so we take the
# column from there. Without this fix, a model trained on the training file performs close to chance on the test set.

print('model_score mean, train file :', round(train['model_score'].mean(), 3))
print('model_score mean, test file  :', round(test['model_score'].mean(), 3))
print('model_score mean, alerts file (training cases):', round(alerts.loc[train.index, 'model_score'].mean(), 3))
print('Test scores identical in both files:', np.allclose(alerts.loc[test.index, 'model_score'], test['model_score']))

# ============================== CELL 6 ==============================
# 3. Analyst profiles with signal detection theory
#
# For each analyst:
# * hit rate H = share of frauds they flag; false-alarm rate F = share of legitimate cases they flag
# * sensitivity d′ = z(H) − z(F)  (higher = better at telling fraud apart)
# * criterion c = −(z(H) + z(F)) / 2  (negative = flags easily, positive = cautious about flagging)
#
# We compute these from the training data (one review per case), which is what a real team would have.

def sdt(decision, truth):
    H = decision[truth == 1].mean()
    F = decision[truth == 0].mean()
    H, F = np.clip([H, F], 0.001, 0.999)   # avoid infinite z-scores
    return H, F, norm.ppf(H) - norm.ppf(F), -(norm.ppf(H) + norm.ppf(F)) / 2

rows = []
for a in ANALYSTS:
    part = train[train['assignment'] == a]
    H, F, dprime, c = sdt(part['decision'].values, part['fraud_bool'].values)
    cost = (LAM * ((part['decision'] == 1) & (part['fraud_bool'] == 0)) +
            ((part['decision'] == 0) & (part['fraud_bool'] == 1))).mean()
    rows.append(dict(analyst=a, cases=len(part), hit_rate=H, false_alarm_rate=F,
                     d_prime=dprime, criterion=c, cost_per_case=cost))
profiles = pd.DataFrame(rows).set_index('analyst')
profiles.to_csv(f'{OUT}/analyst_profiles.csv')
print(profiles.round(3))

# ============================== CELL 7 ==============================
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(profiles['criterion'], profiles['d_prime'], s=60)
for a, r in profiles.iterrows():
    ax.annotate(a.split('#')[1], (r['criterion'], r['d_prime']), xytext=(5, 3), textcoords='offset points')
ax.axvline(0, color='grey', lw=0.8, ls='--')
ax.set_xlabel('criterion c  (← flags easily      cautious →)')
ax.set_ylabel("sensitivity d'")
ax.set_title('Analysts differ in skill and in willingness to flag')
fig.tight_layout(); fig.savefig(f'{FIG}/01_analyst_profiles.png', dpi=150); plt.show()

# ============================== CELL 8 ==============================
# Reading the plot. Analysts differ more in *criterion* than in *sensitivity*: some flag almost everything
# (many false alarms, few misses), others flag rarely. Because a missed fraud costs about 17 times a false alarm
# (1 / 0.057), who reviews which case matters.
#
# 4. Workload limits (capacity)
#
# Each scenario fixes how many test cases each analyst can review. The AI decides whatever is left.

cap_rows = []
for s in SCENARIOS:
    c = pd.read_csv(f'{DATA}/capacity_{s}.csv').iloc[0]
    row = {a: int(c[a]) for a in ANALYSTS}
    row['AI'] = int(c['batch_size']) - sum(row.values())
    cap_rows.append(pd.Series(row, name=s))
capacity = pd.DataFrame(cap_rows)
capacity.to_csv(f'{OUT}/capacity.csv')
print(capacity)

# Open 02_models.ipynb next.
