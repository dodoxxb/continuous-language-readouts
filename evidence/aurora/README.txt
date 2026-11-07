AURORA independent-interface replication, 2026-10-22

Run: OPENBLAS_NUM_THREADS=1 python aurora/verify.py
Requires Python 3.10+ and NumPy. The verifier writes no files.

The bundle preserves labels, margins, exact repair masks, all per-sequence evaluator losses, calibration coefficients, all outcome signs and the frozen protocol. Token IDs are losslessly stored as uint16; every ID is below 50257. Repairs are reconstructed from the linear17 prediction, native label and packed binary edit mask. No edit information is discarded.

The original ready.json files and test freeze also refer to large latent arrays, fitted readout weights and original repository-relative paths outside this portable bundle. Their hashes document provenance; portable verification does not rerun GPU generation, readout fitting or evaluator forward passes. Source scripts retain their original repository layout assumptions and are included for inspection and reproduction in that layout. Pinned upstream checkpoints must be downloaded separately.

Main hypotheses use GPT-2-Large and prespecified 97.5% paired intervals. OPT-125M and frequency-normalized rarity are sensitivity analyses. Native decisions are an oracle reference, not factual truth; these experiments do not establish end-to-end speedup.
