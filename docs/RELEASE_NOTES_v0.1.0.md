# v0.1.0 — Readout and evidence release

Initial research release accompanying *What Simple Readouts Miss in Continuous Language Diffusion* (arXiv:2606.08810). The repository contains portable CPU checks of selected retained measurements, code to apply fitted affine readouts to matching saved states, and archival experiment scripts and protocols.

The five checkpoint assets contain 12 fitted affine readouts (ELF, AURORA and LangFlow), 24 exploratory contextual/focal/shuffled residual controls, and six CALM coefficient snapshots. Their exact selection records and hashes are in `weights/manifest.json`; model-specific context and limitations are in `weights/README.md`. `weights/routing_gate.json` contains small task-fitted selection coefficients.

The fitted checkpoints imitate native decisions; they are not standalone generative models. Fit seeds are not independent generator-training seeds. Contextual controls include null and adverse outcomes. The CPU checks recompute saved results; they do not rerun generation, model training or neural evaluation. Original pretrained generators, tokenizers, datasets and large latent caches are not included.

The 42 fitted checkpoints and the five Release bundles are made available by their rights holders, Zhicheng Du and Lan Ma, under Apache-2.0 (see `WEIGHTS_NOTICE.txt` and `LICENSE`). Upstream pretrained generators, tokenizers, datasets and the frozen CALM vocabulary matrix are not included and remain subject to their publishers' terms.
