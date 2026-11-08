# Checkpoints and model cards

Checkpoints are distributed separately from the Git source through [GitHub Releases](https://github.com/dodoxxb/continuous-language-readouts/releases). The individual `.pt` files are excluded from Git because several exceed GitHub's 100 MB repository-file recommendation; Release assets allow up to 2 GiB each, so all five archives fit well within the limit. The table below lists the five `v0.1.0` Release assets with their direct download links, sizes and SHA-256 hashes.

| Release bundle | Checkpoints | Size | SHA-256 |
| :--- | ---: | ---: | :--- |
| [elf_readouts_20261104.tar](https://github.com/dodoxxb/continuous-language-readouts/releases/download/v0.1.0/elf_readouts_20261104.tar) | 3 | 188.47 MiB | `497d2d91e94bb9a4e5086b5243cf9d73f9be0a6b61cabac2be9a326eae7ea18b` |
| [aurora_readouts_20261104.tar](https://github.com/dodoxxb/continuous-language-readouts/releases/download/v0.1.0/aurora_readouts_20261104.tar) | 3 | 442.30 MiB | `990f0f39c918be507f3754b96dc93c7f016de8dd3139cfade17ff3f38365b9d3` |
| [langflow_readouts_20261104.tar](https://github.com/dodoxxb/continuous-language-readouts/releases/download/v0.1.0/langflow_readouts_20261104.tar) | 6 | 537.24 MiB | `6d5bc74a55206a950dd795e9b9c272e64590a6f5e05e8e74ce901eee22c4ab44` |
| [context_controls_20261104.tar](https://github.com/dodoxxb/continuous-language-readouts/releases/download/v0.1.0/context_controls_20261104.tar) | 24 | 34.61 MiB | `169078557e7e7fa8a6b9a6408372d31a4aaa2de09c493b5b2f601adc6abdff45` |
| [calm_coefficients_20261104.tar](https://github.com/dodoxxb/continuous-language-readouts/releases/download/v0.1.0/calm_coefficients_20261104.tar) | 6 | 6.07 MiB | `cf98b28af8e1c462b72abc0a9027ff1c1a6cf25e8797f2802190d4436ef32bbd` |

Use [`manifest.json`](manifest.json) and the corresponding [`provenance/`](provenance/) records for individual checkpoint hashes and validation selections.

| Asset | Count | Purpose and required context |
| :--- | ---: | :--- |
| `elf/seed{17,29,43}.pt` | 3 | Affine classifiers in ELF's 512-dimensional, 32,100-token interface; fitted on the 16,384-sequence training pool and selected by validation agreement. |
| `aurora/seed{17,29,43}.pt` | 3 | Affine classifiers in AURORA's 768-dimensional, 50,257-token interface; not weight transfers from ELF. |
| `langflow_1k/` and `langflow_8k/` | 6 | Three fits at each of the original and post-result enlarged LangFlow budgets. The 8k experiment is an explicitly motivated follow-up. |
| `context/` and `context_global/` | 24 | Equal-parameter rank-eight focal/context/shuffled controls across ELF and AURORA, two fit seeds and two normalization rounds. These require the original frozen seed-17 base readout and the correct contextual features. |
| `calm/` | 6 | Posterior/prior fitting-population coefficient snapshots, three fits each. The frozen CALM vocabulary matrix is an external asset; these coefficients are not complete decoders. |
| `routing_gate.json` | 1 file | Task-fitted ridge selection coefficients for ELF and AURORA; this is not a generative checkpoint or evidence of speedup. |

All fit seeds are probe/adapter seeds, not independent generator-training seeds. The context checkpoints include failures and controls, and should not be interpreted as endorsed improvements. The affine readouts imitate native decisions only on their specified interface states; valid-position and stopping rules must be supplied separately. In particular, native output is a fidelity target rather than semantic truth.

The 42 author-fitted checkpoints and the five Release bundles are released by their rights holders, Zhicheng Du and Lan Ma, under [Apache-2.0](../LICENSE); see [`WEIGHTS_NOTICE.txt`](../WEIGHTS_NOTICE.txt). Use, modification and redistribution, including commercial use, are permitted under that license, with this notice and `LICENSE` retained on redistribution. The underlying pretrained models, their tokenizers, the frozen CALM vocabulary matrix and the original datasets are not included in the archives and remain with their respective publishers under their own terms.

The five uncompressed release bundles group files without duplicate flat filenames; see [`BUNDLES.json`](BUNDLES.json). Each archive is below 2 GiB and can be extracted with `tar -xf bundle.tar -C weights/`. After downloading all five into `weights/bundles/`, `python tools/check_weight_bundles.py` verifies every bundle and checkpoint. To check only an extracted subset, use for example `python tools/check_weights.py --prefix elf/`. `python tools/build_weight_bundles.py` recreates the archives from locally staged assets.
