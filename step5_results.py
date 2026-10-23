####################################################################################################
# 05 · Results with uncertainty
#
# HOW TO USE THIS FILE IN GOOGLE COLAB
#   Each block between two "CELL" lines is one Colab cell.
#   1. In Colab click  + Code  to make a new empty cell.
#   2. Copy one block (from its CELL line down to the next CELL line) and paste it in.
#   3. Press Shift+Enter to run it. Wait until it finishes, then do the next block.
#   Lines starting with # are explanations; Python ignores them.
####################################################################################################
# One test month is one sample. To see whether a difference between two rules is real, we bootstrap:
# resample the 4,457 test cases with replacement 2,000 times and recompute the cost difference each time.
# Because both rules are scored on the same resampled cases, this is a paired comparison.
# If the 95% interval excludes 0, the difference is unlikely to be noise.

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

assign = np.load(f'{OUT}/assignment_per_case.npz')
fat = np.load(f'{OUT}/fatigue_per_case.npz')
assignment_results = pd.read_csv(f'{OUT}/assignment_results.csv')
fatigue_results = pd.read_csv(f'{OUT}/fatigue_results.csv')
n = len(assign[assign.files[0]])
rng = np.random.default_rng(0)
boot_idx = rng.integers(0, n, size=(2000, n))     # same resamples for every comparison


def paired_ci(a, b):
    """Total cost of rule a minus rule b: point estimate and 95% bootstrap interval."""
    diffs = (a - b)[boot_idx].sum(axis=1)
    return (a - b).sum(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

# ============================== CELL 3 ==============================
# 1. Optimal assignment versus the simple rules (no fatigue)

rows = []
for s in ['homogenous', 'variable-1', 'variable-2', 'variable-3', 'variable-4']:
    for other in ['Random', 'AI-confidence', 'AI only']:
        d, lo, hi = paired_ci(assign[f'{s}|Optimal'], assign[f'{s}|{other}'])
        rows.append(dict(scenario=s, comparison=f'Optimal − {other}', difference=d, ci_low=lo, ci_high=hi))
ci_assign = pd.DataFrame(rows)
ci_assign.to_csv(f'{OUT}/ci_assignment.csv', index=False)
print(ci_assign.round(1))

# ============================== CELL 4 ==============================
# Pooled over the 5 capacity scenarios
#
# Each scenario alone has wide intervals. Summing the differences over the five scenarios (same resampled cases
# in each) gives a tighter estimate. The scenarios share the same test cases, so this is not five independent samples.

rows = []
SC5 = ['homogenous', 'variable-1', 'variable-2', 'variable-3', 'variable-4']
for other in ['Random', 'AI-confidence', 'AI only']:
    a = sum(assign[f'{s}|Optimal'] for s in SC5)
    b = sum(assign[f'{s}|{other}'] for s in SC5)
    d, lo, hi = paired_ci(a, b)
    rows.append(dict(comparison=f'Optimal − {other} (sum of 5 scenarios)', difference=d, ci_low=lo, ci_high=hi,
                     percent_change=100 * d / b.sum()))
pooled = pd.DataFrame(rows)
pooled.to_csv(f'{OUT}/ci_assignment_pooled.csv', index=False)
print(pooled.round(1))

# ============================== CELL 5 ==============================
# 2. Fatigue-aware versus fatigue-blind rules

rows = []
for s in [0.1, 0.2, 0.3, 0.4]:
    for other in ['Fatigue-blind', 'Fatigue-blind + risk-first']:
        d, lo, hi = paired_ci(fat[f'{s}|Fatigue-aware'], fat[f'{s}|{other}'])
        base = fat[f'{s}|{other}'].sum()
        rows.append(dict(fatigue_strength=s, comparison=f'Fatigue-aware − {other}', difference=d,
                         ci_low=lo, ci_high=hi, percent_change=100 * d / base))
ci_fatigue = pd.DataFrame(rows)
ci_fatigue.to_csv(f'{OUT}/ci_fatigue.csv', index=False)
print(ci_fatigue.round(2))

# ============================== CELL 6 ==============================
# 3. Figures

order = ['AI only', 'Random', 'AI-confidence', 'Optimal', 'Published method (reference)']
mean_cost = assignment_results.groupby('rule')['cost'].mean().reindex(order)
spread = assignment_results.groupby('rule')['cost'].agg(['min', 'max']).reindex(order)

fig, ax = plt.subplots(figsize=(7, 3.8))
colors = ['#2f6db5' if r == 'Optimal' else '#9aa5b1' for r in order[::-1]]
ax.barh(order[::-1], mean_cost.values[::-1], color=colors)
ax.errorbar(mean_cost.values[::-1], np.arange(len(order)),
            xerr=[mean_cost.values[::-1] - spread['min'].values[::-1], spread['max'].values[::-1] - mean_cost.values[::-1]],
            fmt='none', color='black', capsize=3, lw=1)
ax.set_xlabel('cost on test month (mean over 5 capacity scenarios; bars = min–max)')
ax.set_title('Who should review each case?')
fig.tight_layout(); fig.savefig(f'{FIG}/05_assignment_rules.png', dpi=150); plt.show()

# ============================== CELL 7 ==============================
fig, ax = plt.subplots(figsize=(6.5, 4))
sub = ci_fatigue[ci_fatigue['comparison'] == 'Fatigue-aware − Fatigue-blind + risk-first']
ax.errorbar(sub['fatigue_strength'] - 0.004, sub['difference'],
            yerr=[sub['difference'] - sub['ci_low'], sub['ci_high'] - sub['difference']],
            fmt='o', capsize=3, label='vs fatigue-blind + risk-first')
sub = ci_fatigue[ci_fatigue['comparison'] == 'Fatigue-aware − Fatigue-blind']
ax.errorbar(sub['fatigue_strength'] + 0.004, sub['difference'],
            yerr=[sub['difference'] - sub['ci_low'], sub['ci_high'] - sub['difference']],
            fmt='s', capsize=3, label='vs fatigue-blind')
ax.axhline(0, color='grey', lw=0.8, ls='--')
ax.set_xlabel('fatigue strength s'); ax.set_ylabel('cost difference (negative = fatigue-aware better)')
ax.set_title('Gain from planning for fatigue (95% bootstrap CI)')
ax.legend(); fig.tight_layout(); fig.savefig(f'{FIG}/05_fatigue_gain.png', dpi=150); plt.show()

# ============================== CELL 8 ==============================
# 4. Summary numbers for the write-up

hom = assignment_results[assignment_results['scenario'] == 'homogenous'].set_index('rule')['cost']
s2 = fatigue_results[fatigue_results['fatigue_strength'] == 0.2].set_index('rule')['expected_cost']
print(f"No fatigue, equal capacities: optimal {hom['Optimal']:.1f} vs random {hom['Random']:.1f} "
      f"vs AI-confidence {hom['AI-confidence']:.1f} vs AI only {hom['AI only']:.1f} "
      f"(published method {hom['Published method (reference)']:.1f})")
print(f"Fatigue s = 0.2: fatigue-aware {s2['Fatigue-aware']:.1f} vs blind {s2['Fatigue-blind']:.1f} "
      f"vs blind + risk-first {s2['Fatigue-blind + risk-first']:.1f}")

# 5. What the numbers say (check them against your own run)
#
# * Optimal assignment vs. everything to the AI: clearly better in every scenario.
# * Optimal vs. random assignment: lower cost in every scenario (about 9% pooled), but the 95% intervals,
#   including the pooled one, still include zero. One test month is not enough to rule out noise.
# * Optimal vs. AI-confidence: no clear difference. A simple, common rule is already strong here.
# * Fatigue: ignoring it is costly, and the gap grows with fatigue strength. Ordering each queue riskiest-first
#   recovers almost all of the gain; the full fatigue-aware optimization is not clearly better than that free rule.
#
# 6. Limitations (state these first when you present the project)
#
# 1. The analysts are synthetic. Their decisions come from a simulator, not from real fraud investigators.
# 2. Fatigue is assumed, not measured. The benchmark analysts do not tire; the lapse model and its strength
#    are assumptions, which is why we sweep the strength instead of claiming one value.
# 3. One domain, one test month. Results may not transfer to other review tasks (for example radiology worklists),
#    and the bootstrap captures sampling noise only, not model mis-specification.
# 4. Fixed capacities. Real teams can change staffing, add breaks or rotate tasks; those decisions are outside this model.
# 5. Expected cost needs well-calibrated models. If the analyst model is wrong about an analyst, the optimizer
#    will trust that analyst too much or too little.
