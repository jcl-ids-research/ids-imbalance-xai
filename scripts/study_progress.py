"""Report progress of the narrow outer-holdout study."""

import optuna

STORAGE = "sqlite:///studies/outer/outer.db"
STUDY = "rm-dmvt-outer"

study = optuna.load_study(study_name=STUDY, storage=STORAGE)
complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
running = [trial for trial in study.trials if trial.state.name == "RUNNING"]
failed = [trial for trial in study.trials if trial.state.name == "FAIL"]

print(f"complete={len(complete)} running={len(running)} failed={len(failed)}")

if complete:
    ordered = sorted(complete, key=lambda trial: float(trial.value), reverse=True)
    print(f"best_value={study.best_value:.6f}")
    print("top5:")
    for trial in ordered[:5]:
        print(f"  trial {trial.number}: {float(trial.value) * 100:.3f}%  {trial.params}")
