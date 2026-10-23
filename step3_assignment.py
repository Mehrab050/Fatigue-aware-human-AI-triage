####################################################################################################
# 03 · Who should review each case? (optimal assignment)
#
# HOW TO USE THIS FILE IN GOOGLE COLAB
#   Each block between two "CELL" lines is one Colab cell.
#   1. In Colab click  + Code  to make a new empty cell.
#   2. Copy one block (from its CELL line down to the next CELL line) and paste it in.
#   3. Press Shift+Enter to run it. Wait until it finishes, then do the next block.
#   Lines starting with # are explanations; Python ignores them.
####################################################################################################
# The optimization model
#
# * Cases i = 1..n (here n = 4457), deciders j = 9 analysts + the AI.
# * c_ij = expected cost if decider j handles case i (from notebook 02).
# * K_j = number of cases decider j must handle (capacity).
# * x_ij = 1 if case i goes to decider j.
#
# min sum_i sum_j c_ij x_ij   subject to   sum_j x_ij = 1  for all i,   sum_i x_ij = K_j  for all j,   x_ij in {0,1}
#
# This is a transportation problem. Its constraint matrix is totally unimodular, so the integer optimum can be
# found with an assignment algorithm: give decider j exactly K_j "seats", then match cases to seats with the
# Hungarian algorithm (scipy.optimize.linear_sum_assignment). Below we also check on a small sample that
# an integer-programming solver (PuLP/CBC) gives the same answer.
#
# Rules we compare
#
#   Rule  ·  How cases are given out
#   Random  ·  random cases to each decider, respecting capacity
#   AI-confidence  ·  AI keeps the cases it is most sure about; the rest go to analysts at random (common in practice)
#   Optimal  ·  the model above
#   AI only  ·  AI decides everything (ignores capacity; reference only)
#
# All rules are scored with the real decisions in the test month: cost = 0.057 × false alarms + misses.

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
# Install the PuLP optimization library (version 4 changed its interface; this code uses version 3)
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pulp<4'], check=True)

# ============================== CELL 3 ==============================
import time, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

pred = np.load(f'{OUT}/predictions.npz')
probs, p_fraud, ai_decision = pred['probs'], pred['p_fraud'], pred['ai_decision']
y, human_decisions = pred['y_test'], pred['human_decisions']
capacity = pd.read_csv(f'{OUT}/capacity.csv', index_col=0)
n = len(y)
DECIDERS = ANALYSTS + ['AI']
AI = len(ANALYSTS)                     # column index of the AI

# Real decision of every decider on every case: shape (cases, 10)
all_decisions = np.column_stack([human_decisions, ai_decision])

# Expected cost matrix: shape (cases, 10)
cost_analysts = LAM * probs[:, :, 1] + probs[:, :, 2]          # 0.057*P(FP) + P(FN)
cost_ai = np.where(ai_decision == 1, LAM * (1 - p_fraud), p_fraud)
cost_matrix = np.column_stack([cost_analysts, cost_ai])

# ============================== CELL 4 ==============================
def solve_assignment(cost, cap):
    """Exact optimal assignment. cost: (cases, deciders); cap: cases per decider (sums to number of cases)."""
    seats = np.repeat(np.arange(cost.shape[1]), cap)   # decider index of each seat
    rows, cols = linear_sum_assignment(cost[:, seats])
    chosen = np.empty(cost.shape[0], dtype=int)
    chosen[rows] = seats[cols]
    return chosen


def case_costs(decisions):
    """Real cost of each case given final decisions (0/1)."""
    return np.where((decisions == 1) & (y == 0), LAM, 0.0) + np.where((decisions == 0) & (y == 1), 1.0, 0.0)


def score(chosen):
    d = all_decisions[np.arange(n), chosen]
    return dict(cost=case_costs(d).sum(), false_alarms=int(((d == 1) & (y == 0)).sum()),
                misses=int(((d == 0) & (y == 1)).sum()))

# ============================== CELL 5 ==============================
# 1. Check the solver against integer programming (small sample)

import pulp

rng = np.random.default_rng(1)
sample = rng.choice(n, 300, replace=False)
small_cost = cost_matrix[sample]
small_cap = np.full(10, 30)            # 300 cases, 30 per decider

prob = pulp.LpProblem('assignment', pulp.LpMinimize)
x = {(i, j): pulp.LpVariable(f'x_{i}_{j}', cat='Binary') for i in range(300) for j in range(10)}
prob += pulp.lpSum(small_cost[i, j] * x[i, j] for i in range(300) for j in range(10))
for i in range(300):
    prob += pulp.lpSum(x[i, j] for j in range(10)) == 1
for j in range(10):
    prob += pulp.lpSum(x[i, j] for i in range(300)) == small_cap[j]
prob.solve(pulp.PULP_CBC_CMD(msg=False))

hungarian = solve_assignment(small_cost, small_cap)
print('Integer program objective:', round(pulp.value(prob.objective), 6))
print('Hungarian objective:      ', round(small_cost[np.arange(300), hungarian].sum(), 6))

# ============================== CELL 6 ==============================
# 2. Compare the rules on all 4,457 test cases, for each capacity scenario

results, per_case = [], {}
for s in SCENARIOS:
    cap = capacity.loc[s, DECIDERS].values.astype(int)
    rng = np.random.default_rng(0)

    # Random
    random_rule = np.repeat(np.arange(10), cap); rng.shuffle(random_rule)

    # AI-confidence: the AI keeps the cases farthest from its threshold
    confidence = np.abs(p_fraud - LAM / (1 + LAM))
    ai_keeps = np.argsort(-confidence)[:cap[AI]]
    conf_rule = np.full(n, AI)
    others = np.setdiff1d(np.arange(n), ai_keeps)
    seats = np.repeat(np.arange(len(ANALYSTS)), cap[:AI]); rng.shuffle(seats)
    conf_rule[others] = seats

    # Optimal
    t = time.time()
    optimal_rule = solve_assignment(cost_matrix, cap)
    seconds = time.time() - t

    rules = {'Random': random_rule, 'AI-confidence': conf_rule,
             'Optimal': optimal_rule, 'AI only': np.full(n, AI)}
    for name, chosen in rules.items():
        results.append(dict(scenario=s, rule=name, **score(chosen)))
        per_case[f'{s}|{name}'] = case_costs(all_decisions[np.arange(n), chosen])
    # Published result of the original method on the same cases (reference only)
    ref = pd.read_parquet(f'{DATA}/reference_{s}.parquet')['prediction'].reindex(pred['case_id']).values
    results.append(dict(scenario=s, rule='Published method (reference)', cost=case_costs(ref).sum(),
                        false_alarms=int(((ref == 1) & (y == 0)).sum()), misses=int(((ref == 0) & (y == 1)).sum())))
    per_case[f'{s}|Published method (reference)'] = case_costs(ref)
    print(f'{s}: optimal assignment solved in {seconds:.1f} s')

results = pd.DataFrame(results)
results.to_csv(f'{OUT}/assignment_results.csv', index=False)
np.savez(f'{OUT}/assignment_per_case.npz', **per_case)
print(np.save(f'{OUT}/optimal_assignment_homogenous.npy', solve_assignment(cost_matrix, capacity.loc['homogenous', DECIDERS].values.astype(int))))

# ============================== CELL 7 ==============================
table = results.pivot(index='rule', columns='scenario', values='cost')[SCENARIOS]
table['mean'] = table.mean(axis=1)
print(table.sort_values('mean').round(1))

# ============================== CELL 8 ==============================
# What to look for. Optimal should beat Random and AI only in every scenario. Expect it to be close to
# AI-confidence: that rule already sends the hardest cases to people, which is most of the benefit. The published
# method should be slightly better than ours because its analyst model is more elaborate.
#
# How to read this. Lower is better. Differences of a few units on one test month can be noise; notebook 05
# puts confidence intervals on the key differences. The published method is shown only to check that this simpler
# pipeline is in the same range; it uses a different analyst model.

# Where does the optimal rule send the cases? Average AI fraud probability of the cases each decider gets.
chosen = np.load(f'{OUT}/optimal_assignment_homogenous.npy')
print(pd.DataFrame({'cases': pd.Series(chosen).value_counts().sort_index().values,
              'mean fraud probability': [p_fraud[chosen == j].mean() for j in range(10)]},
             index=DECIDERS).round(3))

# Open 04_fatigue.ipynb next.
