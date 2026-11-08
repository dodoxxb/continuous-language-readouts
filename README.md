<div align="center">

# What Simple Readouts Miss in Continuous Language Diffusion

**Zhicheng Du**<sup>1</sup> &nbsp; · &nbsp; **Lan Ma**<sup>1,†</sup>

<sup>1</sup>Tsinghua Shenzhen International Graduate School, Tsinghua University &nbsp;&nbsp; <sup>†</sup>Corresponding author

[![arXiv](https://img.shields.io/badge/arXiv-2606.08810-b31b1b.svg)](https://arxiv.org/abs/2606.08810)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements-audit.txt)
[![Code license](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)

<i>How much of a native decoder's decision can a simple token-wise readout preserve, and what is lost when it does?</i>

</div>

**TL;DR:** On the tested generated states, simple affine readouts reproduce most native lexical decisions in ELF and AURORA. The remaining disagreements have unequal consequences: with the same number of native repairs, repair policies can change external scores in opposite directions. Predicting those consequences on a new setting and turning them into a computational benefit are separate problems.

> The arXiv link points to the public preprint. The title and results described here are from the prepared V3 manuscript.

## Code and weights

| Material | Location |
| :--- | :--- |
| Code, reproducibility tools and selected evidence | This repository: [`tools/`](tools/), [`research/`](research/), and [`evidence/`](evidence/). |
| Fitted readouts, coefficients and contextual controls | Five [GitHub Release assets](https://github.com/dodoxxb/continuous-language-readouts/releases/tag/v0.1.0) on tag `v0.1.0`; file names, sizes and SHA-256 hashes are in [`weights/README.md`](weights/README.md). |
| Frozen routing coefficients | [`weights/routing_gate.json`](weights/routing_gate.json), a small JSON file in this repository. |
| Original ELF/AURORA/LangFlow/CALM/LLaCO generators, tokenizers and datasets | Their upstream publishers; model identifiers and limitations are listed below. |

Code and checkpoints are distributed in separate steps: source from this repository, fitted checkpoints from the `v0.1.0` Release assets. These checkpoints are readouts or adapters, not standalone generators.

The five archives total **1.18 GiB** (6.07, 34.61, 188.47, 442.30 and 537.24 MiB). The largest single asset is 537.24 MiB, well under GitHub's 2 GiB per-asset limit; GitHub places no cap on a release's total size or download bandwidth, so no external model host is required. Per-file sizes and SHA-256 hashes are in [`weights/README.md`](weights/README.md), and `python tools/check_weight_bundles.py` verifies downloaded bundles against [`weights/BUNDLES.json`](weights/BUNDLES.json).

## What this project investigates

This is an independent analysis built on [ELF: Embedded Language Flows](https://github.com/lillian039/ELF) and other authors' released models; it is not the official ELF repository. The readouts are trained to imitate native decisions on generated states. The native target is a **fidelity label**, not semantic ground truth.

| Question | Finding in the evaluated setting | Boundary |
| :--- | :--- | :--- |
| How much is readable? | Affine readouts match **97.17%** of ELF and **95.74%** of AURORA native lexical decisions. | Generated endpoint states already contain contextual computation; these percentages are not cross-model quality scores. |
| Do all misses matter equally? | Equal-count repairs have different evaluator consequences across both interfaces. Rare-first repair improves XSum reference similarity. | Native-oracle repair is not a deployable policy; source-consistency benefit on XSum remains unresolved. |
| Does an effect predict new outcomes? | A frozen composition-aware predictor improves on the tested ELF-B/M conditions. | It fails on the distilled student; its AURORA benefit is unresolved. |
| Can available signals guide a choice? | On fresh XSum articles, a task-fitted selector improves BERTScore by **0.261 points** over the same-budget random expectation at a 50% native-call budget. | All-native scores higher, and the measured overhead does not establish end-to-end speedup. |
| What changes with the interface? | CALM and LLaCO separate compressed-state fitting, directional identity and soft adapter input. | These are boundary checks with different state units, not matched replications or full-model speed benchmarks. |

The manuscript retains the controls, intervals and negative results behind these statements. Its supporting analysis also shows why linear MSE alone is insufficient and why an empirical first-order margin proxy is not a certified stability radius.

## Repository contents

| Path | Purpose |
| :--- | :--- |
| [`tools/readout.py`](tools/readout.py) | Apply an ELF, AURORA or LangFlow affine checkpoint to supplied endpoint states in bounded batches. |
| [`tools/check_evidence.py`](tools/check_evidence.py) | Recompute selected reported statistics from saved records on CPU. |
| [`evidence/`](evidence/README.md) | Scalar evidence and document-level aggregates for the supported offline checks. |
| [`research/`](research/README.md) | Archival experimental scripts and protocols; full neural reruns need the original layout and external assets. |
| [`weights/`](weights/README.md) | Model cards, hashes and provenance for staged readouts, CALM coefficients, contextual controls and routing coefficients. |
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | Dependencies, check scope, supported inputs and reproducibility limits. |

## Getting started

### Check reported measurements on CPU

Use Python 3.10 or later. This check does not download data or load a neural model.

```bash
python -m pip install -r requirements-audit.txt
python tools/check_evidence.py --list
python tools/check_evidence.py
```

To run a single check, for example:

```bash
python tools/check_evidence.py --case prospective
```

The checks recompute aggregate values, matched counts and specified intervals **from retained measurements**. They do not regenerate samples, fit probes, rerun evaluators or verify unavailable upstream models. CALM uses per-document equality aggregates instead of source token sequences. See [the evidence guide](evidence/README.md) for each check's scope.

### Apply a released affine readout to saved states

The 12 large affine checkpoints are staged separately from Git. See [the checkpoint guide](weights/README.md) for their status, file identities and expected layout. Install PyTorch for your CPU or CUDA platform and NumPy:

```bash
python -m pip install -r requirements-readout.txt
python tools/readout.py \
  --checkpoint weights/release_assets/elf/seed17.pt \
  --states /path/to/endpoint_states.npy \
  --output predicted_token_ids.npy \
  --device cpu --chunk-size 128
```

The input is a C-contiguous NumPy array of shape `(..., feature_dim)`, in the **same model's native interface coordinates**. The output has shape `(...)` and contains token IDs only. No tokenization, EOS boundary, native lexical mask, generator or factuality evaluation is supplied by this command. The model-specific conditions and vocabularies must match the checkpoint; see [model cards](weights/README.md).

## Models and evidence

The checkpoint inventory includes three ELF and three AURORA affine fits, six LangFlow budget-control fits, 24 exploratory contextual/focal/shuffled residual controls, and six CALM coefficient snapshots. Fit seeds are not independent generator-training seeds. The contextual adapters preserve null or adverse results and require their original base readout; they are not standalone generation models.

No third-party generator weights are included. Obtain them and their tokenizers from the original publishers, including [official ELF checkpoints](https://huggingface.co/embedded-language-flows), [AURORA GPT-2 OWT](https://huggingface.co/fyv587/aurora-gpt2-owt), [CALM](https://huggingface.co/cccczshao/CALM-M), and [LLaCO](https://huggingface.co/escapist413/LLaCO-8B). Check the exact revisions and dependencies recorded in [`evidence/`](evidence/README.md). Model and dataset terms are separate from this repository's source-code license.

The weight manifest records exact hashes and original selection records. Large weight files are excluded from Git and grouped into release assets described in [`weights/README.md`](weights/README.md).

## Citation

```bibtex
@article{du2026readouts,
  title   = {What Simple Readouts Miss in Continuous Language Diffusion},
  author  = {Du, Zhicheng and Ma, Lan},
  journal = {arXiv preprint arXiv:2606.08810},
  year    = {2026}
}
```

The title above is the prepared V3 title; the linked public arXiv version may still have the earlier title. Please also cite the original ELF paper and the relevant upstream model or dataset when using their assets.

## Acknowledgements and license

We thank the authors of ELF, AURORA, LangFlow, CALM, LLaCO and the datasets used for making their work available. Newly authored source code is provided under [Apache-2.0](LICENSE), with upstream notices in [NOTICE.txt](NOTICE.txt) and [`LICENSES/`](LICENSES/). The author-fitted checkpoints are released by their rights holders, Zhicheng Du and Lan Ma, under the same [Apache-2.0](LICENSE) terms; see [WEIGHTS_NOTICE.txt](WEIGHTS_NOTICE.txt). Upstream model weights, tokenizers and datasets are not bundled and retain their publishers' terms; no blanket commercial permission for those assets is implied.

Contact: Zhicheng Du (duzc21@mails.tsinghua.edu.cn) · Lan Ma (malan@sz.tsinghua.edu.cn)
