# Local staging validation — November 4, 2026

- All seven supported CPU evidence checks passed from a clean export of the Git index, using Python 3.12 and NumPy 2.4.6. The AURORA rank-aware wrapper also passed with Python 3.10 and NumPy 1.26.4.
- The affine readout command reproduced 64 of 64 saved token decisions in each of three spot checks: ELF fresh-test states, AURORA S32 test states and LangFlow original-budget test states. These are saved-state checks, not new generation or full-test evaluations.
- All 42 staged checkpoint files match their source selection hashes. The five local bundle hashes and every checkpoint contained in those bundles passed verification.
- The CALM document-level aggregate verifier passed without access to the original cached token IDs. It recomputes only the documented saved comparisons; it does not recover token identities or independently rerun model inference.
- A scan of staged text files for common GitHub, Hugging Face and cloud access-key patterns returned no matches. This is a targeted scan, not a guarantee that every possible sensitive pattern is detectable.

The repository and weight bundles have not been published. No GPU model was trained or generated during this release-preparation pass.
