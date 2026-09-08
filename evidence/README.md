# Paper result evidence

`paper_results/` contains the publication-safe portion of the authoritative
results from `/opt/ids_revision/deploy/results`.

Included families are the corrected main experiments, classical and deep
baselines, multiclass results, adversarial evaluation, attention analysis,
fidelity reports and rendered figures. Only JSON, concise logs and figures no
larger than 2 MiB are included. Model weights, predictions, synthetic rows,
training caches and databases remain on the server.

The server-generated evidence archive contained 283 files, was 16,833,972
bytes, and had SHA-256
`5a5ba7f52c4efe9dbf6ea363c08154924deb2c9b8fa28cb42159313e1282dfd3`.
The archive hash was verified after download before extraction.

`paper_results/scarcity/` was added later, in the same read-only way, and holds
the three-seed records behind the minority-recall experiment under training-side
scarcity, with its own manifest. The directory now holds 287 files in total.

The maintained verification command checks these files against the manuscript
claims and reports missing families or seed coverage before any experiment is
started.
