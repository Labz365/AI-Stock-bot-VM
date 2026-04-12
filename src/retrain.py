"""
MANUAL RETRAIN SCRIPT
Run this to force a full data refresh and model retrain outside of the auto-schedule.
The daily bot (run_bot.py) handles automatic monthly retraining; use this for on-demand runs.
"""

import subprocess
import sys
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python = sys.executable

steps = [
    ('Download stock data',           [python, 'src/download_data.py']),
    ('Download alt + market context', [python, 'src/download_all_data.py']),
    ('Build features',                [python, 'src/build_features.py']),
    ('Train models',                  [python, 'src/train_model.py']),
]

for name, cmd in steps:
    print("\n" + "=" * 50)
    print(f"STEP: {name}")
    print("=" * 50)
    result = subprocess.run(cmd, cwd=base)
    if result.returncode != 0:
        print(f"\nERROR: '{name}' failed with exit code {result.returncode}. Aborting.")
        sys.exit(result.returncode)

# Update the auto-retrain marker so the bot doesn't retrain again immediately
marker = os.path.join(base, 'data', 'last_retrain.txt')
from datetime import date
with open(marker, 'w') as f:
    f.write(date.today().isoformat())

print("\n" + "=" * 50)
print("RETRAIN COMPLETE")
print("Models updated with latest market data.")
print(f"Next auto-retrain reset to today ({date.today()}).")
print("=" * 50)
