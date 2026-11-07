# Prospective readout prediction and equal-error repair

Run `python verify.py` here with Python 3.10+ and NumPy. The verifier refits the calibration regressions, reproduces frozen predictions, paired bootstrap intervals and corpus PPL, and checks identical per-sequence repair counts. It does not rerun GPU measurements or certify generated content.

The calibration set was previously inspected. Before new generation, we froze the protocol, three readouts, regression coefficients and checkpoint hashes. `inference_amendment_before_generation.json` records a pre-generation correction preserving common-random-number dependence in bootstrap. `loader_dependency_fix.json` records a missing PD configuration dependency repaired without changing experimental sampling settings. The primary experiment script retains its originally frozen hash.

Tests comprise three checkpoint/sampling conditions × two generation seeds × 256 sequences = 1,536 trajectories. Three readouts give 4,608 prediction records. Ten text-score arms give 15,360 per-sequence measurements. Six repair arms use the prespecified readout seed 17, all correcting exactly floor(errors/2) positions. These are native-label oracle interventions, not a usable decoder or acceleration method.

`text_scores.csv` contains measured evaluator NLL sums and token counts. `predictions.csv` contains every fixed predictor input/output. `calibration.csv` and `predictors.json` permit refitting and prediction checks. `generation_manifest.json` records local latent/token cache hashes; checkpoints and large caches remain outside this package. The source scripts require the repository paths and local model installations, whereas `verify.py` is portable. Original generation uses PyTorch, Transformers and the released ELF port; analysis additionally uses SciPy; figures use Matplotlib.

Native-mask scoring is conditional; token agreement is not factual correctness. Different detokenizations have different evaluator lengths and truncation coverage. MAE and repair primary intervals are 97.5% Bonferroni intervals for two tests. Per-condition and other policy intervals are descriptive 95% intervals. Failed hypotheses and all prespecified arms are retained.

The OPT-125M secondary evaluation was added after the initial protocol and generation, before inspecting the primary scoring results. It evaluates the exact same 15,360 text arms, without fitting or selecting a predictor or repair policy. It is reported separately as evaluator sensitivity, not a third confirmatory primary hypothesis.
