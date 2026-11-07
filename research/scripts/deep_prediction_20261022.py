"""Exploratory prediction-transfer diagnosis, using saved arrays only.

Run with OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2.
All fitting uses the original calibration set. Old frozen test outcomes have
already been inspected; none of these new comparisons is confirmatory. The
oracle centered-error decomposition is descriptive and is never a predictor.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/deep_mechanism_20261022/prediction'
FEATURES = ['error_rate', 'log_length', 'native_nll', 'rare_error_rate',
            'numeric_error_rate', 'lowmargin_error_rate']
METHODS = ['original_baseline', 'original_tail', 'anchor_baseline', 'anchor_tail']
FITS = [17, 29, 43]
SOURCES = {
    'elf': (ROOT / 'outputs/prospective_tail_20261013', 'predictions.csv'),
    'aurora': (ROOT / 'outputs/aurora_replication_20261022',
               'analysis/gpt2-large/predictions.csv'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readcsv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def writecsv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def arrays(rows):
    return (np.array([[float(row[k]) for k in FEATURES] for row in rows]),
            np.array([float(row['target']) for row in rows]))


def raw_design(x, anchored, tail):
    if not anchored:
        return x if tail else x[:, :3]
    e = x[:, 0]
    ratio = np.divide(x[:, 3:], e[:, None], out=np.zeros((len(x), 3)),
                      where=e[:, None] > 0)
    return np.column_stack([x[:, 1:3], ratio]) if tail else x[:, 1:3]


def fit(x, target, anchored=False, tail=False):
    raw = raw_design(x, anchored, tail)
    mu = raw.mean(0)
    sd = raw.std(0)
    sd = np.where(sd > 1e-12, sd, 1.)
    design = np.column_stack([np.ones(len(x)), (raw - mu) / sd])
    exposure = x[:, 0] if anchored else np.ones(len(x))
    # Normalize mean squared exposure so alpha has the original scale.
    scale = float(np.sqrt(np.mean(exposure ** 2))) if anchored else 1.
    scale = max(scale, 1e-12)
    weighted = design * (exposure / scale)[:, None]
    penalty = np.eye(design.shape[1]) * 10.
    penalty[0, 0] = 0.
    beta = np.linalg.solve(weighted.T @ weighted + penalty,
                           weighted.T @ (target / scale))
    return dict(mean=mu.tolist(), std=sd.tolist(), beta=beta.tolist(),
                anchored=anchored, tail=tail, exposure_scale=scale)


def predict(x, model):
    raw = raw_design(x, model['anchored'], model['tail'])
    z = np.column_stack([np.ones(len(x)),
                         (raw - model['mean']) / model['std']])
    exposure = x[:, 0] if model['anchored'] else np.ones(len(x))
    return (z @ model['beta']) * exposure


def allfits(x, y):
    return {name: fit(x, y, name.startswith('anchor'), name.endswith('tail'))
            for name in METHODS}


def moments(y, prediction):
    error = prediction - y
    mse = np.mean(error ** 2)
    bias2 = np.mean(error) ** 2
    centered = np.mean((error - error.mean()) ** 2)
    assert np.isclose(mse, bias2 + centered)
    return dict(target_mean=float(y.mean()), prediction_mean=float(prediction.mean()),
                signed_bias=float(error.mean()), mae=float(np.mean(np.abs(error))),
                rmse=float(np.sqrt(mse)), mse=float(mse), bias_squared=float(bias2),
                centered_error_variance=float(centered),
                oracle_bias_removed_mae=float(np.mean(np.abs(error-error.mean()))),
                spearman=float(spearmanr(y, prediction).statistic),
                target_std=float(y.std()), prediction_std=float(prediction.std()))


def bootstrap(rows, key, conditions=None):
    rng = np.random.default_rng(61022731)
    groups = {}
    for row in rows:
        if conditions is None or row['condition'] in conditions:
            groups.setdefault((row['condition'], row['generation_seed']), []).append(row)
    draws = np.zeros(4000)
    point = []
    indices = {}
    for (condition, seed), group in sorted(groups.items()):
        group = sorted(group, key=lambda row: row['sequence'])
        value = np.array([row[key] for row in group])
        sequence = tuple(row['sequence'] for row in group)
        pairing_key = (seed, sequence)
        if pairing_key not in indices:
            indices[pairing_key] = rng.integers(len(group), size=(4000, len(group)))
        draws += value[indices[pairing_key]].mean(1) / len(groups)
        point.append(value.mean())
    return dict(mean=float(np.mean(point)), low=float(np.quantile(draws, .025)),
                high=float(np.quantile(draws, .975)))


def association(x):
    z = np.column_stack([np.ones(len(x)), x[:, :3]])
    residual = x[:, 3:] - z @ np.linalg.lstsq(z, x[:, 3:], rcond=1e-10)[0]
    output = {}
    for j, name in enumerate(FEATURES[3:]):
        var = float(np.var(x[:, j+3]))
        output[name] = dict(correlation_with_error_rate=float(np.corrcoef(x[:, 0], x[:, j+3])[0, 1]),
                            variance=float(var),
                            residual_variance_after_baseline=float(np.var(residual[:, j])),
                            residual_variance_fraction=float(np.var(residual[:, j]) / var)
                            if var > 0 else None)
    return output


def feature_shift(x, xt, y, yt, model):
    sd = x.std(0)
    sd = np.where(sd > 1e-12, sd, 1.)
    z = (x - x.mean(0)) / sd
    zt = (xt - x.mean(0)) / sd
    cov = z.T @ z / len(z) + .001 * np.eye(x.shape[1])
    inv = np.linalg.inv(cov)
    dist = np.einsum('ij,jk,ik->i', z, inv, z)
    testdist = np.einsum('ij,jk,ik->i', zt, inv, zt)
    threshold = float(np.quantile(dist, .99))
    low, high = np.quantile(x, [.01, .99], axis=0)
    features = {}
    coefficient = np.array(model['beta'][1:]) / np.array(model['std'])
    for j, name in enumerate(FEATURES):
        shift = float(xt[:, j].mean() - x[:, j].mean())
        features[name] = dict(calibration_mean=float(x[:, j].mean()),
                              test_mean=float(xt[:, j].mean()),
                              standardized_mean_shift=float(shift / sd[j]),
                              outside_calibration_q01_q99=float(np.mean((xt[:, j] < low[j]) | (xt[:, j] > high[j]))),
                              predicted_mean_shift_contribution=float(shift * coefficient[j]))
    predshift = sum(q['predicted_mean_shift_contribution'] for q in features.values())
    observedshift = float(yt.mean() - y.mean())
    assert np.isclose(predshift, predict(xt, model).mean() - predict(x, model).mean())
    return dict(features=features, mahalanobis_q99_calibration_threshold=threshold,
                outside_mahalanobis_q99=float(np.mean(testdist > threshold)),
                actual_target_mean_shift=observedshift, frozen_tail_predicted_mean_shift=predshift,
                unexplained_mean_shift=observedshift - predshift,
                association=association(xt),
                rare_fraction_of_errors=float(np.mean(np.divide(xt[:, 3], xt[:, 0],
                    out=np.zeros(len(xt)), where=xt[:, 0] > 0))))


def compute_family(family):
    root, testname = SOURCES[family]
    calrows = readcsv(root / 'calibration.csv')
    testrows = [r for r in readcsv(root / testname) if r.get('common_scorable', '1') == '1']
    conditions = sorted(set(row['condition'] for row in testrows))
    bysequence = {}
    fit_results = {}
    all_predictions = []
    cv_rows = []
    for fit_seed in FITS:
        cal = sorted([r for r in calrows if int(r['fit_seed']) == fit_seed], key=lambda r: int(r['sequence']))
        test = [r for r in testrows if int(r['fit_seed']) == fit_seed]
        x, y = arrays(cal)
        xt, yt = arrays(test)
        models = allfits(x, y)
        preds = {name: predict(xt, model) for name, model in models.items()}
        for name in ['original_baseline', 'original_tail']:
            saved_key = name.removeprefix('original_') + '_prediction'
            assert np.allclose(preds[name], [float(row[saved_key]) for row in test],
                               atol=1e-10, rtol=1e-9)
        # A zero-disagreement sequence must map to exactly zero in anchored models.
        null_x = xt.copy()
        null_x[:, [0, 3, 4, 5]] = 0
        for name in ['anchor_baseline', 'anchor_tail']:
            assert np.array_equal(predict(null_x, models[name]), np.zeros(len(xt)))
        # Deterministic 5-fold calibration CV, never using the frozen tests.
        # Identical sequence membership is shared by all readout fits.
        perm = np.random.default_rng(61022732).permutation(len(cal))
        fold = np.zeros(len(cal), dtype=int)
        fold[perm] = np.arange(len(cal)) % 5
        oof = {name: np.zeros(len(cal)) for name in METHODS}
        for k in range(5):
            train, val = fold != k, fold == k
            fitted = allfits(x[train], y[train])
            for name in METHODS:
                oof[name][val] = predict(x[val], fitted[name])
        for i, row in enumerate(cal):
            cv_rows.append(dict(family=family, fit_seed=fit_seed, sequence=int(row['sequence']),
                                fold=int(fold[i]), target=float(y[i]),
                                **{name: float(oof[name][i]) for name in METHODS}))
        details = {}
        for condition in conditions:
            mask = np.array([row['condition'] == condition for row in test])
            details[condition] = dict(
                n=int(mask.sum()),
                metrics={name: moments(yt[mask], val[mask]) for name, val in preds.items()},
                shift=feature_shift(x, xt[mask], y, yt[mask], models['original_tail']))
        fit_results[str(fit_seed)] = dict(models=models, calibration_association=association(x),
                                         calibration_cv={name: moments(y, pred) for name, pred in oof.items()},
                                         conditions=details)
        for i, row in enumerate(test):
            key = (row['condition'], int(row['generation_seed']), int(row['sequence']))
            item = dict(family=family, condition=key[0], generation_seed=key[1], sequence=key[2],
                        fit_seed=fit_seed, target=float(yt[i]),
                        **{name: float(value[i]) for name, value in preds.items()})
            all_predictions.append(item)
            bysequence.setdefault(key, []).append(item)
    records = []
    for (condition, seed, sequence), items in sorted(bysequence.items()):
        assert len(items) == 3
        ae = {name: float(np.mean([abs(item[name] - item['target']) for item in items])) for name in METHODS}
        record = dict(family=family, condition=condition, generation_seed=seed, sequence=sequence, **ae)
        record.update(original_tail_gain=ae['original_baseline'] - ae['original_tail'],
                      anchor_tail_gain=ae['anchor_baseline'] - ae['anchor_tail'],
                      anchor_tail_over_original_tail=ae['original_tail'] - ae['anchor_tail'],
                      anchor_tail_over_original_baseline=ae['original_baseline'] - ae['anchor_tail'])
        records.append(record)
    summary = []
    for condition in ['all'] + conditions:
        subset = [r for r in records if condition == 'all' or r['condition'] == condition]
        pred = [r for r in all_predictions if condition == 'all' or r['condition'] == condition]
        keys = ['original_tail_gain', 'anchor_tail_gain', 'anchor_tail_over_original_tail',
                'anchor_tail_over_original_baseline']
        # Each fit has equal weight; moments are averaged over fits to avoid a
        # pooled-mean bias removal concealing different per-readout intercepts.
        metrics = {}
        for name in METHODS:
            perfit = [moments(np.array([r['target'] for r in pred if r['fit_seed'] == seed]),
                              np.array([r[name] for r in pred if r['fit_seed'] == seed]))
                      for seed in FITS]
            metrics[name] = {key: float(np.mean([q[key] for q in perfit])) for key in perfit[0]}
        contrast = {key: bootstrap(records, key, None if condition == 'all' else [condition]) for key in keys}
        base = metrics['original_baseline']
        tail = metrics['original_tail']
        mse_decomposition = dict(total_tail_minus_baseline=tail['mse']-base['mse'],
                                 bias_squared_tail_minus_baseline=tail['bias_squared']-base['bias_squared'],
                                 centered_variance_tail_minus_baseline=tail['centered_error_variance']-base['centered_error_variance'])
        summary.append(dict(condition=condition, sequences=len(subset), metrics=metrics,
                            contrasts=contrast, mse_decomposition=mse_decomposition,
                            seed_metrics={str(seed): {
                                key: float(np.mean([r[key] for r in subset if r['generation_seed'] == seed]))
                                for key in METHODS+keys}
                                for seed in sorted(set(r['generation_seed'] for r in subset))}))
    cv_summary = {}
    for name in METHODS:
        perfit = [fit_results[str(seed)]['calibration_cv'][name] for seed in FITS]
        cv_summary[name] = {key: float(np.mean([q[key] for q in perfit])) for key in perfit[0]}
    return dict(family=family, fits=fit_results, calibration_cv=cv_summary, summary=summary), records, all_predictions, cv_rows


def plan():
    files = []
    for root, testname in SOURCES.values():
        files += [root / 'protocol.json', root / 'calibration.csv', root / testname, root / 'predictors.json']
    return dict(scope='Post hoc exploratory analysis of already observed frozen test outcomes. No new confirmatory claim.',
                plan_date='2026-10-22',
                target='Per-sequence affine minus native evaluator NLL under the existing native boundary.',
                diagnostics=['Exact MSE decomposition into squared signed bias and centered residual variance; oracle centering is not a method.',
                             'Calibration-standardized mean shifts, calibration q01/q99 and Mahalanobis-q99 support diagnostics.',
                             'Exact frozen-linear mean-shift accounting; unexplained shift is not proof of a causal mechanism or of population concept shift.',
                             'Composition-column redundancy after residualizing on baseline features.',
                             'Five-fold calibration-only OOF checks with shared sequence folds across readouts.'],
                new_models='Exactly two models: exposure-anchored baseline and tail. Predict excess NLL as error_rate times a ridge linear model of context (log_length, native_nll) and, for tail, rare/numeric/lowmargin fractions among errors. Calibration-only standardization; alpha=10; unpenalized intercept inside exposure model; average squared exposure normalization. No per-condition tuning, hyperparameter selection, clipping, or target-test adaptation.',
                anchor_identity='If native and affine token sequences match under a shared boundary, excess NLL is exactly zero. Both new models enforce this identity. It does not guarantee generalization or monotonicity.',
                inference='4000 paired sequence bootstrap draws; average absolute error over three readouts before bootstrap; shared resampling across conditions with the same generation seed. All 95% intervals descriptive; no multiplicity-adjusted discovery claim.',
                exclusions='Only original common-scorable exclusions; no outcome-based exclusions.',
                input_sha256={str(path.relative_to(ROOT)): digest(path) for path in files})


def compare(actual, saved):
    if isinstance(actual, dict):
        assert actual.keys() == saved.keys()
        for key in actual:
            compare(actual[key], saved[key])
    elif isinstance(actual, list):
        assert len(actual) == len(saved)
        for first, second in zip(actual, saved):
            compare(first, second)
    elif isinstance(actual, (float, int)) and not isinstance(actual, bool):
        assert np.isclose(actual, saved, atol=1e-10, rtol=1e-9), (actual, saved)
    else:
        assert actual == saved, (actual, saved)


def extended_descriptives(family):
    """No fitting on target tests; uncertainty for exact error decomposition."""
    root, testname = SOURCES[family]
    source = [r for r in readcsv(root / testname) if r.get('common_scorable', '1') == '1']
    result = {}
    for condition in sorted(set(r['condition'] for r in source)):
        rows = [r for r in source if r['condition'] == condition]
        seeds = sorted(set(int(r['generation_seed']) for r in rows))
        full = []
        seed_stats = {}
        for seed in seeds:
            arrays_by_fit = []
            for fit_seed in FITS:
                seq = sorted([r for r in rows if int(r['generation_seed']) == seed and int(r['fit_seed']) == fit_seed],
                             key=lambda r: int(r['sequence']))
                errors = np.array([[float(r['baseline_prediction'])-float(r['target']),
                                    float(r['tail_prediction'])-float(r['target'])] for r in seq])
                arrays_by_fit.append(errors)
            errors = np.stack(arrays_by_fit, axis=1)  # sequence, readout, method
            full.append(errors)
            means = errors.mean(0)
            seed_stats[str(seed)] = dict(
                signed_bias=dict(zip(['baseline', 'tail'], means.mean(0).tolist())),
                mse_tail_minus_baseline=float((errors[:, :, 1]**2-errors[:, :, 0]**2).mean()),
                bias_squared_tail_minus_baseline=float((means[:, 1]**2-means[:, 0]**2).mean()),
                centered_variance_tail_minus_baseline=float((errors[:, :, 1].var(0)-errors[:, :, 0].var(0)).mean()))
        errors = np.concatenate(full)
        rng = np.random.default_rng(61022733)
        sums, squares = [], []
        for block in full:
            indices = rng.integers(len(block), size=(4000, len(block)))
            sampled = block[indices]
            sums.append(sampled.mean(1))
            squares.append((sampled**2).mean(1))
        boot_mean = np.mean(sums, axis=0)
        boot_square = np.mean(squares, axis=0)
        boot_bias = boot_mean**2
        boot_centered = boot_square-boot_bias
        metrics = {}
        for name, val, actual in [
            ('mse_tail_minus_baseline', boot_square,
             np.mean(errors**2, axis=0)),
            ('bias_squared_tail_minus_baseline', boot_bias,
             np.mean(errors, axis=0)**2),
            ('centered_variance_tail_minus_baseline', boot_centered,
             np.var(errors, axis=0))]:
            draw = (val[:, :, 1]-val[:, :, 0]).mean(1)
            metrics[name] = dict(mean=float((actual[:, 1]-actual[:, 0]).mean()),
                                 low=float(np.quantile(draw, .025)), high=float(np.quantile(draw, .975)))
        result[condition] = dict(seed_moments=seed_stats, decomposition_descriptive_ci=metrics)
    if family == 'aurora':
        cal = readcsv(root / 'calibration.csv')
        saved = json.loads((root / 'predictors.json').read_text())
        native = np.load(root / 'data/calibration/y.npy')
        counts = np.load(root / 'counts.npy')
        freq = {}
        for fit_seed in FITS:
            rows = sorted([r for r in cal if int(r['fit_seed']) == fit_seed], key=lambda r: int(r['sequence']))
            x, y = arrays(rows)
            ids = np.array([int(r['sequence']) for r in rows])
            affine = np.load(root / f'data/calibration/linear{fit_seed}.npy')
            valid = native != 50256
            mismatch = valid & (native != affine)
            threshold = saved[str(fit_seed)]['frequency_matched_count_cutoff']
            rare = (mismatch & (counts[native] < threshold)).sum(1) / valid.sum(1).clip(1)
            x[:, 3] = rare[ids]
            model = fit(x, y, tail=True)
            for key in ['mean', 'std', 'beta']:
                assert np.allclose(model[key], saved[str(fit_seed)]['tail_frequency_matched'][key])
            test = [r for r in source if int(r['fit_seed']) == fit_seed]
            xt, _ = arrays(test)
            xt[:, 3] = [float(r['frequency_matched_rare_error_rate']) for r in test]
            assert np.allclose(predict(xt, model), [float(r['frequency_matched_prediction']) for r in test])
            freq[str(fit_seed)] = dict(cutoff=threshold, calibration_association=association(x),
                                      calibration_rare_fraction=float(np.mean(np.divide(x[:, 3], x[:, 0], out=np.zeros(len(x)), where=x[:, 0] > 0))),
                                      test_association=association(xt))
        result['existing_frequency_matched_control'] = freq
    return result


def narrative(results):
    elf, aurora = results['elf'], results['aurora']
    pd = next(q for q in elf['summary'] if q['condition'] == 'PD8')
    am = aurora['summary'][0]
    pb, pt = pd['metrics']['original_baseline'], pd['metrics']['original_tail']
    decomposition = elf['extended_descriptives']['PD8']['decomposition_descriptive_ci']
    lines = [
        'Exploratory diagnosis of frozen prediction transfer (2026-10-22)',
        '',
        'STATUS: All test outcomes were previously inspected. This is post hoc diagnosis, not a new confirmatory experiment. All new models were fit only on the original calibration sequences; no target-test fitting, target-label calibration, or condition-specific tuning was performed. Original frozen artifacts are unchanged.',
        '',
        'PD: a shift in level rather than simple disappearance of predictive ordering',
        f"Original MAE baseline {pb['mae']:.8f}; composition {pt['mae']:.8f}. Signed bias (prediction minus target) changes from {pb['signed_bias']:+.8f} to {pt['signed_bias']:+.8f} nats. Mean per-readout Spearman changes from {pb['spearman']:.4f} to {pt['spearman']:.4f}.",
        'Exact identity: E[(prediction-target)^2] = E[prediction-target]^2 + Var(prediction-target). Each component is computed separately per readout and then averaged. Conditional bootstrap uncertainty uses 4,000 sequence draws, stratified by generation seed.',
    ]
    for key, value in decomposition.items():
        lines.append(f"{key}: {value['mean']:+.9f}, descriptive 95% interval [{value['low']:+.9f}, {value['high']:+.9f}].")
    lines += [
        'The observed MSE deterioration is accounted for by increased squared bias, partly offset by a centered-variance reduction. The latter interval includes zero. Both generation seeds reproduce the direction of increased bias and the point direction of reduced centered variance. Oracle mean removal is diagnostic only; it uses target outcomes and is not a usable predictor.',
        'The PD mean disagreement rate is far outside the B calibration distribution. In fit 17 it shifts +3.09 calibration SD; 51.6% of PD sequences exceed the calibration 99th-percentile regularized Mahalanobis radius. This documents support stress, but does not establish population covariate shift as the unique causal explanation. Distillation, checkpoint, step count, and guidance change jointly.',
        '',
        'AURORA: a small incremental signal with substantial residual uncertainty',
        f"Pooled MAE baseline {am['metrics']['original_baseline']['mae']:.8f}; composition {am['metrics']['original_tail']['mae']:.8f}. Original prespecified 97.5% interval remains unresolved. Mean signed biases are already small: {am['metrics']['original_baseline']['signed_bias']:+.8f} and {am['metrics']['original_tail']['signed_bias']:+.8f} nats. A large PD-like intercept error is not evident.",
        'Under the original count<100 threshold, calibration rare-error burden has correlation about 0.995 with aggregate error burden and retains only about 1% of its variance after baseline residualization. This limits incremental information in that feature. The preexisting relative-frequency-matched cutoff (6.23884 counts) raises residual variance to 16-19%, yet its existing test gain also remains unresolved. Thus count-threshold saturation is only a partial descriptive limitation, not an explanation that rescues the hypothesis.',
        '',
        'Composition prediction is not merely a rare-word penalty',
        'In the three ELF calibration fits, standardized rare-error coefficients are +0.00015, -0.00310, and +0.00568 nats, whereas low-margin-error coefficients are consistently -0.0238 to -0.0266. These are conditional linear associations under correlated features, not causal effects, but they preclude describing the fitted predictor as a universal positive rare-error penalty.',
        '',
        'A simple structural predictor was tested and failed',
        'The excess-score identity at zero disagreements motivates y_hat=e*(a+context terms+composition-fraction terms), where e is disagreement rate. Exactly two new models, baseline and composition, were specified before computing their new test outcomes. Alpha=10 and all standardization/fitting use original calibration only. No tuning or model selection was performed.',
        f"ELF five-fold calibration OOF MAE: original tail {elf['calibration_cv']['original_tail']['mae']:.8f}, anchored tail {elf['calibration_cv']['anchor_tail']['mae']:.8f}. PD test MAE: original tail {pt['mae']:.8f}, anchored tail {pd['metrics']['anchor_tail']['mae']:.8f}.",
        f"AURORA five-fold calibration OOF MAE: original tail {aurora['calibration_cv']['original_tail']['mae']:.8f}, anchored tail {aurora['calibration_cv']['anchor_tail']['mae']:.8f}. Pooled test MAE: original tail {am['metrics']['original_tail']['mae']:.8f}, anchored tail {am['metrics']['anchor_tail']['mae']:.8f}.",
        'The candidate is not recommended; it already underperforms in calibration OOF evaluation and worsens transfer. A mathematically correct boundary constraint does not guarantee a good regression model. This negative result should not be presented as a new algorithmic contribution.',
        '',
        'Implication for the paper',
        'Separate rank/order, absolute calibration, and matched-intervention effects. A repair ranking can transfer without a calibrated sequence-level quality forecast. Report PD as a calibration-boundary example; retain AURORA as unresolved incremental prediction. The audit does not identify the full generation mechanism, establish why distillation causes the shift, or prove universal predictability.',
        '',
        'Reproduction: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /data/duzc/envs/conda/iclr-legacy-cpu-py310/bin/python scripts/deep_prediction_20261022.py --verify',
    ]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_plan = plan()
    if args.plan_only:
        if (OUT / 'analysis_plan.json').exists():
            compare(frozen_plan, json.loads((OUT / 'analysis_plan.json').read_text()))
        else:
            dump(OUT / 'analysis_plan.json', frozen_plan)
        print('Exploratory analysis plan and input hashes recorded.')
        return
    compare(frozen_plan, json.loads((OUT / 'analysis_plan.json').read_text()))
    extension = json.loads((OUT / 'descriptive_extension_plan.json').read_text())
    for path, expected in extension['input_sha256'].items():
        assert digest(ROOT / path) == expected, path
    results = {}
    for family in SOURCES:
        result, records, predictions, cv = compute_family(family)
        result['extended_descriptives'] = extended_descriptives(family)
        results[family] = result
        if args.verify:
            compare(result, json.loads((OUT / (family + '_summary.json')).read_text()))
        else:
            dump(OUT / (family + '_summary.json'), result)
            writecsv(OUT / (family + '_per_sequence.csv'), records)
            writecsv(OUT / (family + '_predictions.csv'), predictions)
            writecsv(OUT / (family + '_calibration_oof.csv'), cv)
        print(json.dumps(dict(family=family,
                              calibration_cv_mae={k: v['mae'] for k, v in result['calibration_cv'].items()},
                              tests=[dict(condition=r['condition'],
                                          mae={k: v['mae'] for k, v in r['metrics'].items()},
                                          contrasts=r['contrasts'],
                                          mse_decomposition=r['mse_decomposition'])
                                     for r in result['summary']]), indent=2))
    report = narrative(results)
    if args.verify:
        assert (OUT / 'REPORT.txt').read_text() == report
    else:
        (OUT / 'REPORT.txt').write_text(report)
    if args.verify:
        print('Independent recomputation matched every saved summary value.')


if __name__ == '__main__':
    main()
