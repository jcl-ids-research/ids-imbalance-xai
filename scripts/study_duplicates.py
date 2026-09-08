"""Report how many distinct configurations the study has actually evaluated."""

from collections import Counter

import optuna

STORAGE = "sqlite:///studies/outer/outer.db"
STUDY = "rm-dmvt-outer"

study = optuna.load_study(study_name=STUDY, storage=STORAGE)
complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]

signatures = Counter(tuple(sorted(trial.params.items())) for trial in complete)
duplicates = {key: count for key, count in signatures.items() if count > 1}

print(f"complete_trials={len(complete)}")
print(f"distinct_configurations={len(signatures)}")
print(f"duplicated_configurations={len(duplicates)}")
print(f"wasted_trials={len(complete) - len(signatures)}")

for params, count in sorted(duplicates.items(), key=lambda item: -item[1]):
    layers = dict(params)["n_layers"]
    rate = dict(params)["learning_rate"]
    print(f"  x{count}  n_layers={layers} lr={rate:.6g}")
