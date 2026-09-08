#!/bin/bash
# Install the isolated RM-DMVT environment. Run once before evening tuning.
#
# The server already has CUDA PyTorch, NumPy, scikit-learn and XGBoost. The
# virtual environment inherits those system packages so bootstrap does not
# download another multi-gigabyte torch wheel or replace the working CUDA stack.
set -euo pipefail

PROJECT_ROOT=${1:-/opt/ids_revision/rm_dmvt}
UV_HOME=${UV_HOME:-/root/.local/bin}

mkdir -p "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$UV_HOME:$PATH"
fi

cd "$PROJECT_ROOT"
uv venv --python /usr/bin/python3 --system-site-packages .venv
uv pip install --python .venv/bin/python \
  "optuna>=3.4" \
  "polars==0.20.31" \
  "typer>=0.12" \
  "rich>=13.7" \
  "typing-extensions>=4.8"
# Optuna's unconstrained NumPy dependency may place NumPy 2.x inside the
# environment. Remove that overlay so the environment exposes the server's
# tested NumPy 1.26.4 alongside CUDA PyTorch 2.2.2.
uv pip uninstall --python .venv/bin/python numpy || true
uv pip install --python .venv/bin/python --no-deps -e .

.venv/bin/python - <<'PY'
import optuna
import polars
import torch
import typer

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("optuna", optuna.__version__)
print("polars", polars.__version__)
print("typer", typer.__version__)
PY
