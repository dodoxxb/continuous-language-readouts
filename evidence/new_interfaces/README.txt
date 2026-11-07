Staging note, November 4, 2026: the paragraphs below describe the original
full evidence archive. This repository omits the CALM source-token caches and
keeps document-level equality aggregates instead. The selected coefficient
checkpoints are staged separately as release assets and are not tracked by Git.
Run `python tools/check_evidence.py --case new-interfaces` from the repository
root for the portable aggregate check. The original verifier referenced below
requires the omitted full token-level caches.

November 2, 2026 interface extension (original archive description)

This directory contains every retained primary/follow-up outcome, all fitting
curves, native/readout token predictions, paired document/source summaries,
protocols, upstream model metadata, local checksums, and the exact local code.
No negative condition or failed transfer was removed. The old evidence tree
is byte-identical to the October 23 archive.

Offline numerical reproduction (requires numpy):
  python code/verify_new_interfaces_20261102.py --root .
This verifies reported summaries and intervals from saved predictions. It does
not rerun pretrained models or reproduce optimization from saved scalar logs.

Pretrained weights, the OpenWebText source parquet, large fitted effective
vocabulary matrices, and fitting-state caches remain in the project workspace;
they are not bundled in this archive. Model/data identifiers and hashes are in
source/ and calm/cohort/manifest.json. Refitting requires those external assets.
The six small affine coefficient checkpoints for the matched fitting-population
control are included; their frozen vocabulary matrix is an external model asset.
The downloaded LLaCO files are indexed components, not the complete 8B model.
Their local hashes/HTTP ranges do not substitute for a full-checkpoint hash.

Model/code pins:
  CALM code: 0249935517df77466a0ef76e33e5a8f18e3fd90a
  CALM AE: cccczshao/CALM-Autoencoder @ 615a8d00f37f3aebb34e43271d7b61f171bd0c56
  CALM-M: cccczshao/CALM-M @ 391dbc2b40aa292e8dd6dd8506fd8de46d5e1af6
  LLaCO code: 994c750f4fe845f713d4c72e379e047c7524a8c7
  LLaCO: escapist413/LLaCO-8B @ 60cf1dff9154bdde70fa099ed8430b8edc67081e

Run the supplied scripts in the ELF project layout, with official repositories
under third_party/CALM and third_party/LLaCO, weights under models/, and the
pinned metadata under outputs/new_interfaces_20261102/source/. Use the released
CALM dependencies transformers4.43.0/tokenizers0.19.1 in an isolated target.
Actual native-state preparation originally used transformers4.57.3; exact
batch-shape parity under 4.43.0 gives identical checked encoder states and zero
native-label differences over all validation/test tokens. Full official-loader
weight and temperature1 sampling-path checks are retained separately.

Execution order:
  new_interfaces_download_20261102.py calm / llaco
  new_interfaces_calm_20261102.py cohort, prepare, fit --seed17/29/43, test
  calm_fit_control_20261102.py fit, fresh
  new_interfaces_calm_20261102.py generated
  calm_pinned_parity_20261102.py; calm_generation_checks_20261102.py
  calm_generated_fit_20261102.py
  new_interfaces_llaco_20261102.py; same script --competition-audit
  verify_new_interfaces_20261102.py --root outputs/new_interfaces_20261102
Use separate shell arguments for each mode/seed, not the punctuation above.

Scientific scope:
CALM's three primary optimization runs select the same ridge initialization.
The later fitting control was motivated by that result and uses fresh documents.
The matched fitting-population control reuses those held-out follow-up documents
for newly sampled prior states, with both arms selected on a common prior-state
validation pool. All six fits are still improving at the last checkpoint, so
their remaining gap is not an affine expressivity lower bound.
One-step CALM-M states use temperature1 and teacher prefixes, not the published
low-temperature/free-running evaluation. LLaCO is a geometry/adapter component
audit. The exact cache trades an extra4GiB for component latency; sparse mass
budgets fail to save work away from clean codewords. No full-model speedup,
task-quality improvement, or universal decoder theorem is claimed.
