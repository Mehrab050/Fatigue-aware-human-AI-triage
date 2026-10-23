# Fatigue-Aware Human–AI Triage Under Workload Limits

An AI scores every incoming case, but human reviewers have limited hours. Which cases should go to which
reviewer, and which should the AI decide alone? And does the answer change when reviewers get tired over a shift?

This project answers both questions on a public fraud-review benchmark with 9 simulated analysts, using an
optimization model and a fatigue model taken from human-factors research on vigilance.

The same structure appears in radiology worklists, insurance claims review, and content moderation.

## Questions

1. Does assigning cases with an optimization model beat simple rules (random, AI-confidence, AI-only)?
2. If analysts make more lapses later in a shift, how much does ignoring fatigue cost, and what fixes it?

## Method in one page

| Step | What | Notebook |
|---|---|---|
| Data | 26,165 training alerts (one analyst decision each) and 4,457 test alerts with every analyst's decision, plus capacity limits | 00, 01 |
| Analysts | Signal detection theory: sensitivity d′ and criterion c per analyst | 01 |
| AI model | Gradient-boosted trees predict fraud probability; flag when that is the cheaper decision | 02 |
| Analyst model | Predicts each outcome (correct clear, false alarm, miss, correct flag) for any case–analyst pair | 02 |
| Assignment | Minimize expected cost subject to "each case once" and "each decider exactly their capacity". A transportation problem, solved exactly with the Hungarian algorithm and checked against an integer program (PuLP/CBC) | 03 |
| Fatigue | Shift split into 4 blocks; in block b an analyst lets a flagged case pass with probability s·b/3. Fatigue strength s swept from 0 to 0.4 | 04 |
| Uncertainty | Paired bootstrap (2,000 resamples of test cases), 95% intervals | 05 |

Cost of a decision = 1 per missed fraud + 0.057 per false alarm (the benchmark's cost ratio).

## Results (from a full run; small differences across package versions are possible)

**Assignment, no fatigue** (cost on the test month, mean over 5 capacity scenarios; lower is better):

| Rule | Cost |
|---|---|
| AI only | 215.2 |
| Random | 170.4 |
| AI-confidence | 159.0 |
| **Optimal (this project)** | **155.7** |
| Published method on the same cases (reference) | 151.0 |

* Optimal beats AI-only clearly (−28%, 95% CI excludes zero).
* Optimal is lower than random in all 5 scenarios (−9% pooled), but the interval still includes zero.
* Optimal and AI-confidence are not distinguishable.
* This simpler pipeline lands within about 3% of the published method.

**Fatigue** (equal capacities, expected cost):

| Fatigue strength s | Fatigue-blind | Blind + risk-first order | Fatigue-aware optimization |
|---|---|---|---|
| 0.0 | 152.8 | 152.8 | 152.8 |
| 0.1 | 174.3 | 165.8 | 164.4 |
| 0.2 | 195.8 | 178.7 | 177.0 |
| 0.3 | 217.3 | 191.6 | 194.1 |
| 0.4 | 238.8 | 204.5 | 207.4 |

* Ignoring fatigue is costly: planning for it lowers cost by 6–13% (all intervals exclude zero).
* **A free rule, reviewing each queue riskiest-first, recovers almost all of that gain.** The full fatigue-aware
  optimization is not clearly better than it at any fatigue level.
* Mechanism: the fatigue-aware model moves high-risk cases into the first block of the shift (mean fraud
  probability 0.175 in block 1 vs 0.059 in block 4) and to the AI.

**Takeaway:** in this setting, *when* a case is reviewed matters about as much as *who* reviews it, and the
"when" part can be captured by a simple scheduling rule.

## Limitations (read these first)

1. **The analysts are synthetic.** Their decisions come from a simulator in the benchmark, not real investigators.
2. **Fatigue is assumed, not measured.** The lapse model and its strength are assumptions; that is why strength is swept.
3. **One domain, one test month.** The bootstrap covers sampling noise only, not model mis-specification.
4. **Fixed capacities.** Breaks, rotation and staffing are outside the model.
5. **The optimizer trusts the analyst model.** Where that model is wrong about an analyst, assignments suffer. This is
   the likely reason the full fatigue-aware model does not beat the simple ordering rule.

## Next steps

* Separate "what the case is" from "how the analyst behaves": use the AI's fraud probability for the case and model
  only the analyst's decision given the truth. The AI's probabilities are better calibrated than the analyst model's.
* Replace the assumed fatigue curve with one estimated from real time-on-task data (needs a human study and ethics approval).
* Add rest breaks as a decision variable.

## How to run (Google Colab)

1. Upload the `notebooks` folder to Google Drive, or open each notebook in Colab with **File → Upload notebook**.
2. Run the notebooks in order: `00` → `01` → `02` → `03` → `04` → `05`. Use **Runtime → Run all** in each one.
3. Each notebook asks to connect Google Drive the first time. Outputs are saved to `MyDrive/fatigue_triage/`,
   which is how the next notebook finds them.

Approximate run times on a free Colab machine: 00 about 3–5 min (400 MB download), 02 about 1–3 min,
03 and 04 about 3–6 min each, others under a minute.

`expected_results/` holds the tables and figures from a full run, so you can check your numbers against them.

## Data and credit

* Fraud alerts, simulated analyst decisions, capacity limits and the published reference results come from the
  public repository of *Cost-Sensitive Learning to Defer to Multiple Experts with Workload Constraints*
  (Transactions on Machine Learning Research, 2024): https://github.com/feedzai/deccaf
* The applications come from the Bank Account Fraud dataset (NeurIPS 2022): https://github.com/feedzai/bank-account-fraud
* The fatigue extension, the signal-detection analysis, the baselines, the bootstrap analysis and all code in this
  repository are new. The original method's code is not used.

**Data note.** In the benchmark's training file, the `model_score` column is on a different scale than in the test
file. Taking it from the original alerts file for both fixes this; without the fix, a model trained on the training
file performs close to chance on the test month (see notebook 01).
