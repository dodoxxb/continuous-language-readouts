"""CPU analysis of the exploratory, frozen-readout context adapter intervention.

Fits are averaged within each generation; they are not independent samples.
XSum bootstrap units are documents shared across families and generation seeds.
OWT indices are shared across conditions within each generation-seed stratum.
No fidelity result is interpreted as task quality or generation causality.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'outputs/deep_mechanism_20261022/context'
ARMS = ['base', 'focal', 'context', 'shuffled']
METRICS = ['agreement', 'rare_agreement', 'common_agreement',
           'fixed_per_1000_tokens', 'introduced_per_1000_tokens']
CONTRASTS = [('context', 'focal'), ('context', 'shuffled'),
             ('focal', 'base'), ('context', 'base'), ('shuffled', 'base')]
SEED = 61022951


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def ratio(a, b):
    return float(a) / b if b else np.nan


def finite_mean(a, axis=0):
    """Like nanmean without suppressing unrelated RuntimeWarnings."""
    a = np.asarray(a, dtype=np.float64)
    count = np.isfinite(a).sum(axis=axis)
    total = np.where(np.isfinite(a), a, 0.).sum(axis=axis)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count > 0)


def value_or_none(value):
    return float(value) if np.isfinite(value) else None


def load(root, seeds):
    rows = []
    sources = {}
    required = [(f, s) for f in ['elf', 'aurora'] for s in ['owt', 'xsum']]
    for family, scope in required:
        path = root / family / 'evaluation' / scope / 'per_sequence.csv'
        completed = path.with_name('complete.json')
        if not path.exists() or not completed.exists():
            raise FileNotFoundError('Evaluation not complete: ' + str(path))
        sources[str(path.relative_to(root))] = sha(path)
        sources[str(completed.relative_to(root))] = sha(completed)
        complete = json.loads(completed.read_text())
        file_rows = read_csv(path)
        assert complete['family'] == family and complete['scope'] == scope
        assert complete['rows'] == len(file_rows)
        seen = set()
        for row in file_rows:
            assert row['family'] == family and row['scope'] == scope
            for key in ['generation_seed', 'sequence', 'tokens', 'correct',
                        'rare_tokens', 'rare_correct', 'fixed', 'introduced']:
                row[key] = int(row[key])
            key = (family, scope, row['condition'], row['generation_seed'],
                   row['sequence'], row['arm'])
            assert key not in seen, ('Duplicate record', key)
            seen.add(key)
            assert 0 <= row['rare_correct'] <= row['rare_tokens'] <= row['tokens']
            assert row['rare_correct'] <= row['correct'] <= row['tokens']
            assert 0 <= row['correct'] - row['rare_correct'] <= row['tokens'] - row['rare_tokens']
            rows.append(row)
        cells = {}
        for row in file_rows:
            cells.setdefault((row['condition'], row['generation_seed']), set()).add(row['sequence'])
        expected_cells, expected_size = (6, 256) if scope == 'owt' else (2, 1024)
        assert len(cells) == expected_cells, (family, scope, len(cells))
        assert all(indices == set(range(expected_size)) for indices in cells.values()), (family, scope, 'Incomplete sequence indices')
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ['family', 'scope', 'condition', 'generation_seed', 'sequence'])
        groups.setdefault(key, {})[row['arm']] = row
    expected = {'base'} | {f'{arm}_{seed}' for arm in ARMS[1:] for seed in seeds}
    records = []
    values = {}
    for key, items in sorted(groups.items()):
        assert set(items) == expected, ('Missing fit/arm', key, set(items))
        baseline = items['base']
        for item in items.values():
            assert item['tokens'] == baseline['tokens']
            assert item['rare_tokens'] == baseline['rare_tokens']
            assert item['correct'] - baseline['correct'] == item['fixed'] - item['introduced']
            assert item['fixed'] <= baseline['tokens'] - baseline['correct']
            assert item['introduced'] <= baseline['correct']
        matrix = np.full((len(METRICS), len(ARMS)), np.nan)
        for ai, arm in enumerate(ARMS):
            rr = [baseline] if arm == 'base' else [items[f'{arm}_{seed}'] for seed in seeds]
            averaged = {name: float(np.mean([r[name] for r in rr]))
                        for name in ['correct', 'rare_correct', 'fixed', 'introduced']}
            n = baseline['tokens']; rare = baseline['rare_tokens']
            metric = [ratio(averaged['correct'], n), ratio(averaged['rare_correct'], rare),
                      ratio(averaged['correct'] - averaged['rare_correct'], n - rare),
                      1000 * ratio(averaged['fixed'], n), 1000 * ratio(averaged['introduced'], n)]
            matrix[:, ai] = metric
            records.append(dict(zip(['family', 'scope', 'condition', 'generation_seed', 'sequence'], key),
                                arm=arm, tokens=n, rare_tokens=rare,
                                **averaged, **{m: value_or_none(v) for m, v in zip(METRICS, metric)}))
        values[key] = matrix
    return records, values, sources


def strata(values, scope, family, condition):
    selected = {k: v for k, v in values.items()
                if k[1] == scope and (family == 'pooled' or k[0] == family)
                and (condition == 'all' or k[2] == condition)}
    assert selected
    grouped = {}
    for key, matrix in selected.items():
        # XSum: one cluster per source document, shared across all model/seed replicas.
        # OWT: sequence-index pairing across conditions is preserved within seed.
        cluster = (0, key[4]) if scope == 'xsum' else (key[3], key[4])
        grouped.setdefault(cluster, []).append(matrix)
    by_stratum = {}
    for (seed, index), matrices in sorted(grouped.items()):
        by_stratum.setdefault(seed, []).append(finite_mean(np.stack(matrices), axis=0))
    arrays = [np.stack(by_stratum[s]) for s in sorted(by_stratum)]
    return arrays, len(selected), len(grouped), selected


def analyze_group(values, scope, family, condition, draws):
    arrays, generations, clusters, selected = strata(values, scope, family, condition)
    point = finite_mean(np.stack([finite_mean(a, axis=0) for a in arrays]), axis=0)
    rng = np.random.default_rng(SEED)
    boot_strata = []
    for array in arrays:
        # Chunk bootstrap draws to limit temporary array memory on shared CPUs.
        chunks = []
        for start in range(0, draws, 100):
            indices = rng.integers(len(array), size=(min(100, draws-start), len(array)))
            chunks.append(finite_mean(array[indices], axis=1))
        boot_strata.append(np.concatenate(chunks, axis=0))
    samples = finite_mean(np.stack(boot_strata), axis=0)
    metric_results = {}
    for mi, metric in enumerate(METRICS):
        result = dict(arms={}, contrasts={})
        for ai, arm in enumerate(ARMS):
            b = samples[:, mi, ai]; valid = b[np.isfinite(b)]
            result['arms'][arm] = dict(mean=value_or_none(point[mi, ai]),
                                      ci_low=float(np.quantile(valid, .025)) if len(valid) else None,
                                      ci_high=float(np.quantile(valid, .975)) if len(valid) else None)
        for left, right in CONTRASTS:
            li, ri = ARMS.index(left), ARMS.index(right)
            b = samples[:, mi, li] - samples[:, mi, ri]; valid = b[np.isfinite(b)]
            result['contrasts'][left + '_minus_' + right] = dict(
                mean=value_or_none(point[mi, li] - point[mi, ri]),
                ci_low=float(np.quantile(valid, .025)) if len(valid) else None,
                ci_high=float(np.quantile(valid, .975)) if len(valid) else None,
                finite_bootstrap_draws=len(valid))
        metric_results[metric] = result
    counts = dict(generations=generations, clusters=clusters,
                  generation_seed_strata=len(arrays),
                  zero_lexical_generations=sum(not np.isfinite(a[0, 0]) for a in selected.values()),
                  zero_rare_generations=sum(not np.isfinite(a[1, 0]) for a in selected.values()),
                  zero_common_generations=sum(not np.isfinite(a[2, 0]) for a in selected.values()))
    return dict(scope=scope, family=family, condition=condition, counts=counts, metrics=metric_results)


def compute(root, draws):
    protocol = json.loads((root / 'protocol.json').read_text())
    seeds = protocol['training']['seeds']
    records, values, sources = load(root, seeds)
    results = []
    for family in ['elf', 'aurora']:
        conditions = sorted({key[2] for key in values if key[0] == family and key[1] == 'owt'})
        for condition in ['all'] + conditions:
            results.append(analyze_group(values, 'owt', family, condition, draws))
        results.append(analyze_group(values, 'xsum', family, 'all', draws))
    results.append(analyze_group(values, 'xsum', 'pooled', 'all', draws))
    summary = dict(
        status='complete', scope='Exploratory follow-up on previously examined evaluation cohorts',
        bootstrap_draws=draws, bootstrap_seed=SEED, confidence_level=.95,
        fit_seeds=seeds, fit_aggregation='Arithmetic mean within the same generated sequence before inference',
        estimand='Macro native lexical agreement. XSum averages replicas within document; OWT averages conditions within sequence index and then equal-weights generation-seed strata.',
        empty_rule='An empty denominator gives undefined accuracy; generation stays in coverage counts. No zero is imputed. Each contrast uses identical support across arms.',
        multiplicity='All intervals are exploratory descriptive 95% intervals; no confirmatory significance or familywise claim.',
        limitations=[
            'Native decoder targets are fidelity labels, not factual or semantic ground truth.',
            'A fitted readout intervention does not identify the causal mechanism of the diffusion trajectory.',
            'Four preceding full-canvas states and rank eight are a restricted context model.',
            'Focal residual remains affine in the focal latent; its extra parameters control further optimization, not a larger function class.',
            'Fidelity alone establishes neither output-quality improvement nor acceleration.',
            'Generation seeds and adapter fit seeds do not constitute independently trained generator checkpoints.',
            'XSum uses source-prefix latent context and the existing native output boundary; it is not an independently stopping generator.'
        ], source_sha256=sources, protocol_sha256=sha(root / 'protocol.json'), results=results)
    return summary, records


def compare(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual: compare(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for x, y in zip(actual, expected): compare(x, y)
    elif isinstance(actual, (float, int)) and not isinstance(actual, bool):
        assert np.isclose(actual, expected, rtol=1e-10, atol=1e-12), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, default=DEFAULT)
    parser.add_argument('--draws', type=int, default=4000)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    assert args.draws >= 100
    summary, records = compute(args.input_root, args.draws)
    dest = args.input_root / 'analysis'
    if args.verify:
        compare(summary, json.loads((dest / 'summary.json').read_text()))
        print('Context adapter paired analysis recomputation passed')
        return
    dump(dest / 'summary.json', summary)
    with (dest / 'per_sequence_fit_mean.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print(json.dumps([dict(scope=r['scope'], family=r['family'], condition=r['condition'],
                          context_minus_focal=r['metrics']['agreement']['contrasts']['context_minus_focal'])
                      for r in summary['results'] if r['condition'] == 'all'], indent=2))


if __name__ == '__main__':
    main()
