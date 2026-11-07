# Offline evidence included in the staging repository

The saved records let a reader audit selected published calculations without a GPU, a network connection or pretrained model weights. They do not reproduce the original neural measurements. The full evidence archive in the manuscript workspace contains additional task, routing, context and mechanism analyses; it is not represented as part of this repository.

Run `python tools/check_evidence.py` from the repository root, or choose an individual case with `--case`.

| Case | Evidence | What is checked |
| :--- | :--- | :--- |
| `stability` | `calibration/` | Conditional pairwise AUROCs and the 27 observed below-proxy-radius flips; not an independent bootstrap rerun. |
| `fixed-mdp` | `calibration/` | Fixed-variant MDP aggregate and source-summary consistency. |
| `fixed-test` | `confirmatory/` | Fixed-split readout statistics, saved selections, paired intervals and endpoint scores. |
| `langflow-budget` | `confirmatory/` | The post-result LangFlow token-budget follow-up. |
| `prospective` | `prospective/` | Frozen ELF predictor refits, equal repair counts, paired effects and evaluator summaries. |
| `aurora` | `aurora/` | Native/affine output agreement, exact repair masks, frozen predictor refits and evaluator aggregates. |
| `new-interfaces` | `new_interfaces/`, `new_interfaces_aggregate/` | CALM document-level agreement and bootstrap calculations plus LLaCO geometry, approximation bounds and component timing. |

The CALM arrays in `new_interfaces_aggregate/` contain **per-document agreement rates**, not source token IDs or model states. `tools/export_calm_document_aggregates.py` records how they were derived, with hashes of the source caches and aggregate files in the manifest. Their agreement with the original caches was checked locally, but a user without those caches cannot independently replay the token-level comparisons. The original full verifier in `new_interfaces/code/` requires the omitted caches.

The AURORA bundle contains generated token IDs and masks used for the recorded output comparison. Original XSum articles, summaries and decoded task outputs are not included. Model licenses and dataset permissions must be considered separately from the source-code license; links and pins in the evidence record identify upstream sources.

The AURORA wrapper retains the original verifier and checksum-protected records. One portability correction concerns only deliberately rank-deficient duplicate-feature designs: their finite SVD condition-number estimates vary across BLAS implementations, so the wrapper checks the same deficient rank and a very large or undefined condition number instead of requiring identical numerical round-off. Coefficients, predictions and reported contrasts keep the original tolerances. See `tools/check_aurora.py`.
