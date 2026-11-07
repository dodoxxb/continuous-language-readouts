"""Recompute the AURORA scalar evidence from saved labels and evaluator losses.

Run as `python aurora/verify.py` after extracting the evidence supplement.
Only NumPy and the standard library are required. No output files are modified.
This does not rerun model fitting, latent generation or language-model scoring.
"""
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = [17, 29, 43]
POLICIES = ['random11', 'random23', 'random37', 'rare', 'numeric', 'lowmargin']
ARMS = ['native'] + [f'linear{s}' for s in SEEDS] + ['repair_' + p for p in POLICIES]
FEATURES = ['error_rate', 'log_length', 'native_nll', 'rare_error_rate',
            'numeric_error_rate', 'lowmargin_error_rate']


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def same(a, b):
    assert np.allclose(a, b, rtol=1e-8, atol=1e-9, equal_nan=True), (a, b)


def features(y, pred, margin, native_nll, counts, numeric, cutoff=100):
    valid = y != 50256
    wrong = (pred != y) & valid
    n = valid.sum(1).clip(1)
    return np.stack([wrong.sum(1)/n, np.log(n), native_nll,
                     (wrong & (counts[y] < cutoff)).sum(1)/n,
                     (wrong & numeric[y]).sum(1)/n,
                     (wrong & (margin < 2)).sum(1)/n], axis=1)


def ridge(x, target):
    mean = x.mean(0)
    std = x.std(0)
    std = np.where(std > 1e-12, std, 1.)
    design = np.column_stack([np.ones(len(x)), (x-mean)/std])
    penalty = np.eye(design.shape[1])*10.
    penalty[0, 0] = 0.
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return {'mean': mean, 'std': std, 'beta': beta}


def predict(x, fit):
    return np.column_stack([np.ones(len(x)), (x-fit['mean'])/fit['std']]) @ fit['beta']


def bootstrap(values, protocol, condition, key, level):
    conditions = list(protocol['test']['conditions']) if condition == 'all' else [condition]
    n = protocol['test']['sequences_per_seed']
    rng = np.random.default_rng(61022501)
    point, resampled = [], []
    for seed in protocol['test']['seeds']:
        indices = rng.integers(n, size=(4000, n))
        for cond in conditions:
            array = np.full(n, np.nan)
            for i, value in values[cond, seed].items():
                array[i] = value[key]
            assert np.isfinite(array).sum() >= 2
            point.append(np.nanmean(array))
            resampled.append(np.nanmean(array[indices], axis=1))
    sample = np.mean(resampled, axis=0)
    alpha = (1-level)/2
    return np.mean(point), *np.quantile(sample, [alpha, 1-alpha])


def load_pool(path, model):
    z = np.load(path/'tokens.npz')
    y = z['y'].astype(np.int32)
    valid = y != 50256
    arm = {'native': y, **{f'linear{s}': z[f'linear{s}'].astype(np.int32) for s in SEEDS}}
    for policy in POLICIES:
        changed = np.unpackbits(z['mask_'+policy], axis=1, count=y.shape[1]).astype(bool)
        arm['repair_'+policy] = np.where(changed, y, arm['linear17'])
    score, nll, eligible = {}, {}, {}
    for name in ARMS:
        rr = rows(path/f'scores_{model}_{name}.csv')
        assert [int(r['sequence']) for r in rr] == list(range(len(y)))
        assert all(r['method'] == name and r['evaluator'] == model for r in rr)
        token_count = np.array([int(r['tokens']) for r in rr])
        uncapped = np.array([int(r['uncapped_tokens']) for r in rr])
        eligible[name] = (uncapped >= 2) & (uncapped <= 1024)
        assert np.array_equal(eligible[name], [bool(int(r['eligible'])) for r in rr])
        assert np.array_equal(token_count, np.where(eligible[name], uncapped-1, 0))
        assert np.array_equal(valid.sum(1), [int(r['valid_tokens']) for r in rr])
        assert np.array_equal(((arm[name] == y) & valid).sum(1), [int(r['correct']) for r in rr])
        total = np.array([float(r['nll_sum']) for r in rr])
        assert np.isfinite(total).all()
        assert (total[~eligible[name]] == 0).all()
        nll[name] = np.divide(total, token_count, out=np.full(len(y), np.nan), where=token_count>0)
        score[name] = (total, token_count)
    good = np.logical_and.reduce(list(eligible.values()))
    return y, z['margin'], arm, nll, score, good


def main():
    manifest = read(ROOT/'manifest.json')
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    protocol = read(ROOT/'protocol.json')
    assert hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest() == read(ROOT/'protocol_frozen.json')['sha256']
    frozen_verified = 0
    for original, expected in read(ROOT/'test_frozen.json')['files'].items():
        if original.startswith('outputs/aurora_replication_20261022/'):
            name = original.removeprefix('outputs/aurora_replication_20261022/')
            if name.startswith('data/calibration/'):
                name = name.removeprefix('data/').replace('/ready.json', '/original_ready.json')
            candidate = ROOT/name
        elif original.startswith('scripts/'):
            candidate = ROOT/'experiment_code'/Path(original).name
        else:
            continue
        if candidate.is_file():
            assert hashlib.sha256(candidate.read_bytes()).hexdigest() == expected, original
            frozen_verified += 1
    counts = np.load(ROOT/'counts.npy')
    numeric = np.load(ROOT/'numeric.npy')
    train = np.load(ROOT/'training_labels.npz')['y']
    assert np.array_equal(counts, np.bincount(train[train != 50256], minlength=50257))
    fits = read(ROOT/'predictors.json')
    for seed in SEEDS:
        history = read(ROOT/f'fits/seed{seed}/validation_history.json')
        best = max(history, key=lambda r: r['lexical_agreement'])
        selected = read(ROOT/f'fits/seed{seed}/selected.json')
        assert best['step'] == selected['selected_step']
        same(best['lexical_agreement'], selected['validation_agreement'])
        for key in ['baseline', 'tail', 'tail_frequency_matched']:
            fits[str(seed)][key] = {k: np.array(v) for k, v in fits[str(seed)][key].items()}

    y, margin, arms, nll, score, good = load_pool(ROOT/'calibration', 'gpt2-large')
    eligibility = read(ROOT/'calibration_eligibility.json')
    assert eligibility['common_scorable'] == int(good.sum())
    assert eligibility['excluded_indices'] == np.flatnonzero(~good).tolist()
    for seed in SEEDS:
        fit = fits[str(seed)]
        cutoff = 100*counts.sum()/protocol['frequency_sensitivity']['elf_reference_tokens']
        same(cutoff, fit['frequency_matched_count_cutoff'])
        x = features(y, arms[f'linear{seed}'], margin, nll['native'], counts, numeric)
        xf = features(y, arms[f'linear{seed}'], margin, nll['native'], counts, numeric, cutoff)
        target = nll[f'linear{seed}']-nll['native']
        for name, xx in [('baseline', x[:, :3]), ('tail', x), ('tail_frequency_matched', xf)]:
            derived = ridge(xx[good], target[good])
            for key in ['mean', 'std', 'beta']:
                same(derived[key], fit[name][key])

    verified = {}
    for model in ['gpt2-large', 'opt-125m']:
        derived, scores, agreements, coverage = {}, {}, {}, {}
        summary = read(ROOT/'analysis'/model/'summary.json')
        for cond in protocol['test']['conditions']:
            for seed in protocol['test']['seeds']:
                path = ROOT/'test'/cond/str(seed)
                y, margin, arms, nll, score, good = load_pool(path, model)
                valid = y != 50256
                wrong = (arms['linear17'] != y) & valid
                errors = wrong.sum(1)
                for pi, policy in enumerate(POLICIES):
                    changed = arms['repair_'+policy] != arms['linear17']
                    assert not (changed & ~wrong).any()
                    assert np.array_equal(changed.sum(1), errors//2)
                    for i in range(len(y)):
                        idx = np.flatnonzero(wrong[i])
                        rng = np.random.default_rng(np.random.SeedSequence([seed, i, pi, 61022]))
                        tie = rng.random(len(idx))
                        if policy.startswith('random'):
                            chosen = rng.permutation(idx)[:len(idx)//2]
                        else:
                            priority = counts[y[i, idx]] if policy == 'rare' else -numeric[y[i, idx]].astype(int) if policy == 'numeric' else margin[i, idx]
                            chosen = idx[np.lexsort((tie, priority))[:len(idx)//2]]
                        assert np.array_equal(np.sort(chosen), np.flatnonzero(changed[i]))
                documented = next(r for r in summary['validity'] if r['condition'] == cond and r['generation_seed'] == seed)
                assert documented['common_scorable'] == int(good.sum())
                assert documented['excluded_indices'] == np.flatnonzero(~good).tolist()
                predictions = []
                for fs in SEEDS:
                    fit = fits[str(fs)]
                    x = features(y, arms[f'linear{fs}'], margin, nll['native'], counts, numeric)
                    xf = features(y, arms[f'linear{fs}'], margin, nll['native'], counts, numeric, fit['frequency_matched_count_cutoff'])
                    predictions.append((nll[f'linear{fs}']-nll['native'],
                                        predict(x[:, :3], fit['baseline']), predict(x, fit['tail']),
                                        predict(xf, fit['tail_frequency_matched'])))
                random = np.mean([nll['repair_'+p] for p in POLICIES[:3]], axis=0)
                records = {}
                for i in np.flatnonzero(good):
                    records[int(i)] = {p: random[i]-nll['repair_'+p][i] for p in POLICIES[3:]}
                    if model == 'gpt2-large':
                        b = np.mean([abs(t[i]-base[i]) for t, base, tail, freq in predictions])
                        h = np.mean([abs(t[i]-tail[i]) for t, base, tail, freq in predictions])
                        f = np.mean([abs(t[i]-freq[i]) for t, base, tail, freq in predictions])
                        records[int(i)].update(baseline=b, tail=h, gain=b-h, frequency=f, frequency_gain=b-f)
                derived[cond, seed] = records
                scores[cond, seed] = {a: (total[good].sum(), tokens[good].sum()) for a, (total, tokens) in score.items()}
                for a in ARMS:
                    agreements[cond, seed, a] = (((arms[a] == y) & valid).sum(), valid.sum(), (arms[a] == y).mean())
                    if a.startswith('linear'):
                        for group, mask in {'unseen': counts[y] == 0, 'seen_1_99': (counts[y] > 0) & (counts[y] < 100),
                                            'frequent_100plus': counts[y] >= 100, 'numeric': numeric[y], 'lowmargin': margin < 2}.items():
                            mask = mask & valid
                            coverage[cond, seed, a, group] = (mask.sum(), ((arms[a] == y) & mask).sum())
        for r in summary['repair']:
            same(bootstrap(derived, protocol, r['condition'], r['policy'], r['level']), [r['mean'], r['ci_low'], r['ci_high']])
        if model == 'gpt2-large':
            for r in summary['prediction']:
                c = r['condition']
                same(bootstrap(derived, protocol, c, 'gain', r['level']), [r['mean'], r['ci_low'], r['ci_high']])
                b = bootstrap(derived, protocol, c, 'baseline', .95)[0]
                h = bootstrap(derived, protocol, c, 'tail', .95)[0]
                same([b, h, (b-h)/b], [r['baseline_mae'], r['tail_mae'], r['relative_mae_reduction']])
                same(bootstrap(derived, protocol, c, 'frequency_gain', .95), [r['frequency_matched_gain'], r['frequency_matched_ci_low'], r['frequency_matched_ci_high']])
                same(bootstrap(derived, protocol, c, 'frequency', .95)[0], r['frequency_matched_mae'])
        for r in rows(ROOT/'analysis'/model/'ppl.csv'):
            totals = [value[r['method']] for (cond, seed), value in scores.items() if r['condition'] in ['all', cond]]
            loss, tokens = np.sum(totals, axis=0)
            same([tokens, loss/tokens, math.exp(loss/tokens)], [int(r['scored_tokens']), float(r['nll']), float(r['ppl'])])
        for r in rows(ROOT/'analysis'/model/'agreement.csv'):
            correct, count, raw = agreements[r['condition'], int(r['generation_seed']), r['method']]
            same([correct, count, correct/count, raw], [int(r['lexical_correct']), int(r['lexical_tokens']), float(r['lexical_agreement']), float(r['raw_all_position_agreement'])])
        for r in rows(ROOT/'analysis'/model/'coverage.csv'):
            count, correct = coverage[r['condition'], int(r['generation_seed']), r['method'], r['group']]
            same([count, correct], [int(r['tokens']), int(r['correct'])])
        verified[model] = sum(len(v) for v in derived.values())
    if (ROOT/'regression_controls.py').exists():
        for family in ['elf','aurora']:
            inputs=ROOT if family=='aurora' else ROOT/'regularization_controls/elf_inputs'
            subprocess.run([sys.executable,str(ROOT/'regression_controls.py'),'--family',family,'--input-root',str(inputs),
                            '--output-dir',str(ROOT/'regularization_controls'/family),'--verify'],check=True)
    print(json.dumps({'status': 'passed', 'common_scorable_sequences': verified, 'available_pretest_freeze_hashes_verified': frozen_verified,
                      'checks': ['artifact hashes', 'training counts', 'validation selection', 'calibration refit',
                                 'every exact edit including tie randomization', 'common scoring sets',
                                 'paired prediction and repair intervals', 'agreement', 'coverage', 'PPL'],
                      'scope': 'Recomputation from stored labels and losses; no model generation or scoring rerun'}, indent=2))


if __name__ == '__main__':
    main()
