# Local staging validation — November 5, 2026

- All seven supported CPU evidence checks passed from a clean export of the Git index, using Python 3.12 and NumPy 2.4.6. These checks recompute stored measurements; they do not rerun generation, fitting or neural evaluation.
- All five local release bundles and all 42 checkpoints contained in them passed their recorded SHA-256 checks.
- On the preceding release pass, the affine readout command reproduced 64 of 64 saved decisions in each of three spot checks: ELF fresh-test states, AURORA S32 test states and LangFlow original-budget test states. The AURORA wrapper also passed with NumPy 1.26.4. See [the November 4 record](LOCAL_VALIDATION_20261104.md).
- The CALM check uses document-level equality aggregates, not the original token identities; it does not independently regenerate samples.

This record covers local checks performed on November 5, 2026. Publishing the repository, attaching the five Release assets and replacing the V3 manuscript are separate steps described in [PUBLISHING.md](PUBLISHING.md); verify the arXiv server preview after those steps.
