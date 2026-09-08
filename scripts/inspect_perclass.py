"""Show the exact shape of the per-class block."""

from __future__ import annotations

import json
from pathlib import Path

path = Path("/opt/ids_revision/results/shift_experiment/unsw_keep005_seed42.json")
payload = json.loads(path.read_text(encoding="utf-8"))

for name, block in payload["variants"].items():
    print(f"--- {name}")
    print(json.dumps(block, indent=2)[:600])
    print()
    break

print("full dump of one variant's per_class if nested elsewhere:")
text = path.read_text(encoding="utf-8")
index = text.find("per_class")
print(text[max(0, index - 200) : index + 800])
