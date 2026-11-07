# Reproduction guide

There are three different operations:

1. **Check retained measurements.** `python tools/check_evidence.py` uses Python 3.10+ and NumPy, caps numerical library threads at one, and needs no network or pretrained models. A historical verifier writes a report, so the wrapper runs it on a temporary copy. The CALM check uses document-level match rates derived from the original cached comparisons; token identities and original text are not included.
2. **Apply a fitted affine probe.** `tools/readout.py` loads a selected `weight`/`bias` state dictionary with PyTorch's `weights_only=True` and applies it in float32 to saved, matching model-interface states. It predicts IDs only. Use the appropriate lexical mask, model vocabulary and stopping protocol when evaluating them.
3. **Repeat neural experiments.** The scripts under `research/` and the evidence source directories document the original procedures, but require pinned third-party repositories, their pretrained checkpoints, data, large intermediate states and the recorded original workspace structure. Full regeneration and fitting have not been revalidated in a clean environment for this release.

The default inference device is CPU with two PyTorch threads. Increase `--chunk-size` cautiously: the temporary logits have `chunk_size × vocabulary_size` elements. Use a CUDA device only after checking available memory. No code here reserves or isolates GPU memory from other processes.

Different model families use different vocabularies and state units. Check the provenance in [`weights/`](../weights/README.md) and protocol files before reusing a probe. A checkpoint selected for one model does not imply compatibility with another; fit-seed variation is not independent generator-training variation.
