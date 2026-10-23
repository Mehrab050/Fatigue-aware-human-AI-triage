####################################################################################################
# 04 · Extension: analysts get tired
#
# HOW TO USE THIS FILE IN GOOGLE COLAB
#   Each block between two "CELL" lines is one Colab cell.
#   1. In Colab click  + Code  to make a new empty cell.
#   2. Copy one block (from its CELL line down to the next CELL line) and paste it in.
#   3. Press Shift+Enter to run it. Wait until it finishes, then do the next block.
#   Lines starting with # are explanations; Python ignores them.
####################################################################################################
# The benchmark's analysts never tire. Real reviewers do: in long monitoring tasks, people become more likely to
# let a case pass as time on task grows (the *vigilance decrement*).
#
# Fatigue model (an assumption, stated openly)
#
# * Each analyst works through their queue in order. The queue is split into 4 equal blocks (think: 4 parts of a shift).
# * In block b = 0,1,2,3, with probability f_b = s · b / 3 the analyst clears a case they would otherwise
#   have flagged. So block 1 has no fatigue and block 4 has lapse rate s.
# * s is the fatigue strength. We do not know its true value, so we test s = 0 to 0.4.
#
# With lapse probability f, the outcome probabilities change to:
# false alarm (1-f) P_FP, miss P_FN + f P_TP. Expected cost: 0.057(1-f)P_FP + P_FN + f P_TP.
#
# Rules we compare
#
#   Rule  ·  Assignment  ·  Order inside each queue
#   Fatigue-blind  ·  optimal from notebook 03  ·  random
#   Fatigue-blind + risk-first  ·  optimal from notebook 03  ·  highest fraud probability first
#   Fatigue-aware  ·  optimal over (analyst, block) seats  ·  decided by the optimization
#
# Fatigue-aware is still an assignment problem: each analyst now has 4 groups of seats, one per block, each with its
# own cost. The AI never tires.
#
# Because lapses are random, we compute the expected real cost exactly instead of simulating:
# for a case with real decision d and lapse probability f, expected cost = (1-f)·cost(d) + f·cost(0).

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
import time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

pred = np.load(f'{OUT}/predictions.npz')
probs, p_fraud, ai_decision = pred['probs'], pred['p_fraud'], pred['ai_decision']
y, human_decisions = pred['y_test'], pred['human_decisions']
capacity = pd.read_csv(f'{OUT}/capacity.csv', index_col=0)
n = len(y)
DECIDERS = ANALYSTS + ['AI']
AI = len(ANALYSTS)
BLOCKS = 4
all_decisions = np.column_stack([human_decisions, ai_decision])
P_TN, P_FP, P_FN, P_TP = (probs[:, :, k] for k in range(4))
cost_ai = np.where(ai_decision == 1, LAM * (1 - p_fraud), p_fraud)


def solve_assignment(cost, cap):
    seats = np.repeat(np.arange(cost.shape[1]), cap)
    rows, cols = linear_sum_assignment(cost[:, seats])
    chosen = np.empty(cost.shape[0], dtype=int)
    chosen[rows] = seats[cols]
    return chosen


def case_costs(decisions):
    return np.where((decisions == 1) & (y == 0), LAM, 0.0) + np.where((decisions == 0) & (y == 1), 1.0, 0.0)


def expected_cost(decisions, lapse):
    """Exact expected real cost per case when each case has lapse probability `lapse`."""
    return (1 - lapse) * case_costs(decisions) + lapse * case_costs(np.zeros(n, dtype=int))


def block_sizes(total):
    """Split a queue of `total` cases into BLOCKS nearly equal parts."""
    return [total // BLOCKS + (1 if b < total % BLOCKS else 0) for b in range(BLOCKS)]

# ============================== CELL 3 ==============================
cap = capacity.loc['homogenous', DECIDERS].values.astype(int)
blind = np.load(f'{OUT}/optimal_assignment_homogenous.npy')      # fatigue-blind optimal rule
blind_decisions = all_decisions[np.arange(n), blind]

rows, per_case = [], {}
for s in [0.0, 0.1, 0.2, 0.3, 0.4]:
    f = np.array([s * b / (BLOCKS - 1) for b in range(BLOCKS)])   # lapse rate per block

    # 1. Fatigue-blind, random order: each case is equally likely to fall in any block
    lapse1 = np.where(blind < AI, f.mean(), 0.0)
    c1 = expected_cost(blind_decisions, lapse1)

    # 2. Fatigue-blind, risk-first order inside each analyst's queue
    lapse2 = np.zeros(n)
    for j in range(AI):
        queue = np.where(blind == j)[0]
        queue = queue[np.argsort(-p_fraud[queue])]                 # riskiest first
        lapse2[queue] = f[np.repeat(np.arange(BLOCKS), block_sizes(len(queue)))]
    c2 = expected_cost(blind_decisions, lapse2)

    # 3. Fatigue-aware: seats are (analyst, block) pairs, plus the AI
    columns, seat_lapse, seat_decider, seat_cap = [], [], [], []
    for j in range(AI):
        for b, size in enumerate(block_sizes(cap[j])):
            columns.append(LAM * (1 - f[b]) * P_FP[:, j] + P_FN[:, j] + f[b] * P_TP[:, j])
            seat_lapse.append(f[b]); seat_decider.append(j); seat_cap.append(size)
    columns.append(cost_ai); seat_lapse.append(0.0); seat_decider.append(AI); seat_cap.append(cap[AI])

    t = time.time()
    group = solve_assignment(np.column_stack(columns), np.array(seat_cap))
    aware_decider = np.array(seat_decider)[group]
    c3 = expected_cost(all_decisions[np.arange(n), aware_decider], np.array(seat_lapse)[group])

    for name, c in [('Fatigue-blind', c1), ('Fatigue-blind + risk-first', c2), ('Fatigue-aware', c3)]:
        rows.append(dict(fatigue_strength=s, rule=name, expected_cost=c.sum()))
        per_case[f'{s}|{name}'] = c
    print(f's = {s}: solved in {time.time() - t:.0f} s')

fatigue = pd.DataFrame(rows)
fatigue.to_csv(f'{OUT}/fatigue_results.csv', index=False)
np.savez(f'{OUT}/fatigue_per_case.npz', **per_case)
print(fatigue.pivot(index='fatigue_strength', columns='rule', values='expected_cost').round(1))

# ============================== CELL 4 ==============================
fig, ax = plt.subplots(figsize=(6, 4))
for name, g in fatigue.groupby('rule', sort=False):
    ax.plot(g['fatigue_strength'], g['expected_cost'], marker='o', label=name)
ax.set_xlabel('fatigue strength s (lapse rate in the last block)')
ax.set_ylabel('expected cost (test month)')
ax.set_title('Cost as analysts tire')
ax.legend(); fig.tight_layout(); fig.savefig(f'{FIG}/04_fatigue_sweep.png', dpi=150); plt.show()

# ============================== CELL 5 ==============================
# What to look for. Ignoring fatigue costs more and more as fatigue grows. Compare the two rules that
# plan for it: if *risk-first* ordering is about as good as the full fatigue-aware optimization, most of the benefit
# comes from when a case is reviewed, not who reviews it. That would be a practical result, because reordering a
# queue costs nothing to implement.
#
# Why does it help? Look at where the risky cases go
#
# For s = 0.2 we check, for the fatigue-aware rule, the average fraud probability of cases placed in each block.
# If the optimizer works as intended, risky cases are reviewed early (fresh) or sent to the AI.

s = 0.2
f = np.array([s * b / (BLOCKS - 1) for b in range(BLOCKS)])
columns, seat_block, seat_decider, seat_cap = [], [], [], []
for j in range(AI):
    for b, size in enumerate(block_sizes(cap[j])):
        columns.append(LAM * (1 - f[b]) * P_FP[:, j] + P_FN[:, j] + f[b] * P_TP[:, j])
        seat_block.append(b); seat_decider.append(j); seat_cap.append(size)
columns.append(cost_ai); seat_block.append(-1); seat_decider.append(AI); seat_cap.append(cap[AI])
group = solve_assignment(np.column_stack(columns), np.array(seat_cap))
block = np.array(seat_block)[group]

summary = pd.DataFrame({'where': ['AI'] + [f'analyst block {b + 1}' for b in range(BLOCKS)],
                        'cases': [(block == -1).sum()] + [(block == b).sum() for b in range(BLOCKS)],
                        'mean fraud probability': [p_fraud[block == -1].mean()] +
                                                  [p_fraud[block == b].mean() for b in range(BLOCKS)]})
print(summary.round(3))

# Open 05_results.ipynb next.
