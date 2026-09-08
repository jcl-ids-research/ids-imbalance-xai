# Server experiment source archive

`legacy_paper/` is an immutable provenance snapshot of the code that produced
the manuscript's authoritative server results. It was copied from
`/opt/ids_revision` on 2026-09-08 after a read-only inventory.

The snapshot contains:

- `ablation/`: model definitions and the original ablation trainer;
- `deploy/`: corrected four-dataset runs, baselines, multiclass experiments,
  fidelity, robustness, attention and figure scripts;
- `distribution_validation/`: synthetic-distribution validation;
- `experiment_runner/`: the original experiment dispatcher;
- root `run_*.sh` launch receipts; and
- `SERVER_CODE_MANIFEST.json`: source path, server environment and SHA-256 for
  every archived file.

The source archive contained 118 entries and had SHA-256
`123e7f89f54d5c0733dabfcd1127f959fd15bb8382aa80136bb33ca222a9ec3d`.
Every one of the 117 source files in the extracted snapshot matches its server
manifest hash.

Do not edit files under `legacy_paper/`. The maintained orchestration and data
adapters live under `rm_dmvt/src/ids_diffusion/`; they invoke or validate this
snapshot without changing the historical algorithms, seeds or preprocessing.

Large result arrays, caches, checkpoints and model weights are not archived in
Git. They remain on the server and are represented by the light evidence
package and manifests under `evidence/`.
