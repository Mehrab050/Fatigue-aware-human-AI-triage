####################################################################################################
# 00 · Setup and data
#
# HOW TO USE THIS FILE IN GOOGLE COLAB
#   Each block between two "CELL" lines is one Colab cell.
#   1. In Colab click  + Code  to make a new empty cell.
#   2. Copy one block (from its CELL line down to the next CELL line) and paste it in.
#   3. Press Shift+Enter to run it. Wait until it finishes, then do the next block.
#   Lines starting with # are explanations; Python ignores them.
####################################################################################################
# Project: Fatigue-aware human–AI triage under workload limits.
#
# This notebook downloads the public fraud-review benchmark, checks the file is intact, and copies
# only the files this project needs into MyDrive/fatigue_triage/data.
#
# Run the notebooks in order: 00 → 01 → 02 → 03 → 04 → 05. Each one saves its outputs to
# Google Drive, so you can close a notebook and open the next one.
#
# Data source. Financial fraud alerts with decisions from 9 synthetic fraud analysts and their
# workload limits, released with the paper *Cost-Sensitive Learning to Defer to Multiple Experts with
# Workload Constraints* (TMLR 2024) and its public code repository. The applications themselves come from
# the public Bank Account Fraud dataset (NeurIPS 2022). The analysts are simulated, not real people.

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
# 1. Download the archive (about 400 MB, a few minutes)
#
# The archive goes to the temporary Colab disk (/content), not to Drive, because we only keep a few small
# files from it.

import hashlib, urllib.request

ZIP_URL = 'https://media.githubusercontent.com/media/feedzai/deccaf/main/Data_and_models.zip'
ZIP_PATH = os.environ.get('ZIP_PATH', '/content/Data_and_models.zip')
EXPECTED_SHA256 = '03361af4aa829905260b9eb6d0b2834718406423e506d6a72f795b9db66db7b2'

if not os.path.exists(ZIP_PATH):
    print('Downloading...')
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
print('Size (MB):', round(os.path.getsize(ZIP_PATH) / 1e6, 1))

# ============================== CELL 3 ==============================
# Check the download is complete and unchanged (compare its fingerprint with the published one)
sha = hashlib.sha256()
with open(ZIP_PATH, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        sha.update(chunk)
assert sha.hexdigest() == EXPECTED_SHA256, 'Download is incomplete or different. Delete the file and rerun.'
print('Checksum OK')

# ============================== CELL 4 ==============================
# 2. Copy the files we need
#
# We use one scenario of the benchmark: alerts flagged at a 5% false-positive rate, with a false alarm costing
# 0.057 of a missed fraud. Files are renamed to plain names.

import zipfile

SC = 'alert_0.05-data_0.05-l_0.057'
ROOT = 'Data_and_models/'
files = {
    'alerts.parquet': f'{ROOT}data/alerts/alert_0.05-data_0.05.parquet',
    'train.parquet': f'{ROOT}testbed/{SC}/train_alert/5k-1#regular/train.parquet',
    'test.parquet': f'{ROOT}testbed/{SC}/test/test.parquet',
    'test_decisions.parquet': f'{ROOT}testbed/{SC}/test/test_expert_pred.parquet',
    'analyst_properties.parquet': f'{ROOT}experts/{SC}/expert_properties.parquet',
}
for s in SCENARIOS:
    files[f'capacity_{s}.csv'] = f'{ROOT}testbed/{SC}/test/testsize#{s}_fullteam/capacity.csv'
    # published results of the original method, used only as a reference point in notebook 03
    files[f'reference_{s}.parquet'] = (f'{ROOT}deferral/def_results_alert/{SC}/5k-1#regular/'
                                      f'DeCCaF/testsize#{s}_fullteam/results.parquet')

with zipfile.ZipFile(ZIP_PATH) as z:
    for new_name, inside_path in files.items():
        with z.open(inside_path) as src, open(os.path.join(DATA, new_name), 'wb') as dst:
            dst.write(src.read())

for name in sorted(os.listdir(DATA)):
    print(f'{name:38s} {os.path.getsize(os.path.join(DATA, name)) / 1e6:6.2f} MB')

# Done. Open 01_explore_data.ipynb next.
