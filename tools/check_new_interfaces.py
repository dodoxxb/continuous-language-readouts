"""Recheck CALM document aggregates and the retained LLaCO component results."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'evidence/new_interfaces'
CALM = RAW / 'calm'
LLACO = RAW / 'llaco'
AGG = ROOT / 'evidence/new_interfaces_aggregate'


def same(a, b):
    assert np.allclose(a, b, atol=1e-12, rtol=1e-12), (a, b)


def ci(values, indices):
    return np.quantile(values[indices].mean(axis=1), [.025, .975])


def main():
    manifest = json.loads((AGG / 'manifest.json').read_text())
    for row in manifest['files'].values():
        assert hashlib.sha256((AGG / row['aggregate_file']).read_bytes()).hexdigest() == row['aggregate_sha256']
    boot = np.random.default_rng(61102009).integers(512, size=(4000, 512))
    for row in json.loads((CALM / 'summary.json').read_text()):
        with np.load(AGG / f"primary_{row['condition']}.npz") as data:
            keys = ['17', '29', '43'] if row['fit'] == 'three_fit_mean' else [row['fit']]
            a = np.stack([data['token_' + k] for k in keys]).mean(0)
            b = np.stack([data['chunk_' + k] for k in keys]).mean(0)
            same(a.mean(), row['token_agreement']); same(b.mean(), row['chunk_agreement'])
            same(ci(a, boot), row['token_ci']); same(ci(b, boot), row['chunk_ci'])
            same(data['token_reference'].mean(), row['native_reference'])
            same(data['token_native_base'].mean(), row['native_radial_preservation'])
    old = [json.loads(line)['document_sha256'] for line in (CALM / 'cohort/documents.jsonl').read_text().splitlines()]
    fresh = [json.loads(line)['document_sha256'] for line in (CALM / 'fit_control/fresh_documents.jsonl').read_text().splitlines()]
    assert len(old) == len(set(old)) == 4864 and len(fresh) == len(set(fresh)) == 512 and not set(old) & set(fresh)
    boot = np.random.default_rng(61102209).integers(512, size=(4000, 512))
    with np.load(AGG / 'fresh.npz') as data:
        base = data['token_ridge_17']
        for row in json.loads((CALM / 'fit_control/summary.json').read_text())['results']:
            seeds = [17] if row['family'] == 'ridge' else [17, 29, 43]
            a = np.stack([data[f"token_{row['family']}_{s}"] for s in seeds]).mean(0)
            same(a.mean(), row['agreement']); same(ci(a, boot), row['ci'])
            same((a - base).mean(), row['gain_over_ridge']); same(ci(a - base, boot), row['gain_ci'])
    boot = np.random.default_rng(61102022).integers(512, size=(4000, 512))
    with np.load(AGG / 'generated.npz') as data:
        for row in json.loads((CALM / 'generated_summary.json').read_text())['results']:
            keys = ['17', '29', '43'] if row['fit'] == 'three_fit_mean' else ['tied17', 'tied29', 'tied43'] if row['fit'] == 'tied_three_fit_mean' else [row['fit']]
            a = np.stack([data['token_' + k] for k in keys]).mean(0)
            same(a.mean(), row['agreement']); same(ci(a, boot), row['ci'])
            if 'gain_ci' in row:
                delta = a - data['token_ridge']
                same(delta.mean(), row['gain_over_ridge']); same(ci(delta, boot), row['gain_ci'])
    boot = np.random.default_rng(61102309).integers(512, size=(4000, 512))
    summary = json.loads((CALM / 'generated_fit/summary.json').read_text())
    means = {}
    with np.load(AGG / 'generated_fit.npz') as data:
        for row in summary['results']:
            family = row['family']
            a = np.stack([data[f'token_{family}_{s}'] for s in [17, 29, 43]]).mean(0)
            means[family] = a
            same(a.mean(), row['agreement']); same(ci(a, boot), row['ci'])
            for seed in [17, 29, 43]:
                same(data[f'token_{family}_{seed}'].mean(), row['individual_agreements'][str(seed)])
                folder = CALM / 'generated_fit' / f'{family}_seed{seed}'
                history = json.loads((folder / 'history.json').read_text())
                saved = json.loads((folder / 'manifest.json').read_text())
                assert saved['selected_step'] == max(history, key=lambda x: x['agreement'])['step']
                same(saved['validation_agreement'], max(row['agreement'] for row in history))
    delta = means['prior'] - means['posterior']
    same(delta.mean(), summary['gain']); same(ci(delta, boot), summary['gain_ci'])

    result = json.loads((LLACO / 'summary.json').read_text())
    rows = list(csv.DictReader((LLACO / 'per_source.csv').open()))
    for row in result['geometry']:
        selected = [v for v in rows if v['endpoint'] == row['endpoint'] and float(v['gamma']) == row['gamma'] and v['split'] == 'test']
        assert len(selected) == 768
        for key in ['cosine_correct', 'euclidean_correct', 'adapter_correct', 'adapter_cosine_disagree', 'source_rank', 'posterior_max']:
            same(np.mean([float(v[key]) for v in selected]), row[key])
    sparse = list(csv.DictReader((LLACO / 'sparse_per_source.csv').open()))
    for row in sparse:
        assert float(row['l2_error']) <= float(row['bound']) + 1e-4
        if int(row['sparse']):
            assert -1e-6 <= float(row['omitted_mass']) <= float(row['epsilon']) + 1e-6 and int(row['kept_k']) <= 64
        else:
            assert int(row['kept_k']) == 131072
    for row in result['sparse']:
        selected = [v for v in sparse if v['split'] == 'test' and float(v['epsilon']) == row['epsilon'] and float(v['gamma']) == row['gamma']]
        assert len(selected) == 768
        same(np.mean([int(v['sparse']) for v in selected]), row['sparse_fraction'])
    for row in result['runtime']:
        same(np.median(row['measurements']), row['median_ms'])
        assert len(row['measurements']) == 20
    print(json.dumps({'status': 'passed', 'checks': ['CALM primary and paired bootstrap', 'CALM fresh, generated and matched-population fits', 'LLaCO geometry, sparse bounds and component timing'], 'scope': 'CALM checks use exported document-level agreement rates, not the source token identities; no pretrained inference or fitting is rerun.'}, indent=2))


if __name__ == '__main__':
    main()
