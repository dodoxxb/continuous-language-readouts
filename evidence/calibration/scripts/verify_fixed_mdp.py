#!/usr/bin/env python3
"""Verify the fixed-variant MDP table from included original summary records.

Standard library only. Does not fit a model, decode tensors, or download data.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

BASE = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    meta = json.loads((BASE / 'data/mdp_saved_provenance.json').read_text())
    table_path = BASE / 'data/fixed_mdp.csv'
    assert sha(table_path.read_bytes()) == meta['fixed_table_sha256']
    table = {r['run']: r for r in csv.DictReader(table_path.open())}
    checked = []
    for run in meta['runs']:
        name = run['run']
        original = BASE / run['summary_file']
        assert sha(original.read_bytes()) == run['summary_sha256'], name
        rows = list(csv.DictReader(original.open()))
        fit = next(r for r in rows if r['readout'] == 'final_z')
        native = next(r for r in rows if r['readout'] == 'official_decoder')
        assert float(fit['noise_aug']) == 0.02
        total = 0
        for shard in run['generation_shard_summaries']:
            raw = shard['raw_text'].encode('utf-8')
            assert sha(raw) == shard['sha256']
            total += int(json.loads(raw)['num_samples'])
        num, den = run['train_fraction']
        assert num == 3 and den == 4 and total % den == 0
        train = total * num // den
        agreement_eval = total - train
        # Matches the saved 32k PPL cap; small-run CSVs omit that count.
        ppl_eval = min(agreement_eval, 4096)
        assert agreement_eval == run['exported_prediction_count_observed']
        expected_counts = {'n_samples': total, 'n_train': train,
                           'n_eval': agreement_eval, 'ppl_eval_n': ppl_eval}
        for key, val in expected_counts.items():
            if key in fit:
                assert int(fit[key]) == val, (name, key)
            if key in native:
                assert int(native[key]) == val, (name, key)
        target = table[name]
        for key, val in [('n_total', total), ('n_train', train),
                         ('n_agreement_eval', agreement_eval), ('n_ppl_eval', ppl_eval)]:
            assert int(target[key]) == val, (name, key)
        assert target['readout'] == 'final_z' and float(target['noise_aug']) == 0.02
        expected = {'token_agreement': float(fit['token_agreement']),
                    'ppl': float(fit['ppl_mean']),
                    'native_ppl': float(native['ppl_mean']),
                    'ppl_gap': float(fit['ppl_mean']) - float(native['ppl_mean']),
                    'readout_margin_p10': float(fit['margin_p10'])}
        for key, val in expected.items():
            assert math.isclose(float(target[key]), val, rel_tol=1e-13, abs_tol=1e-13), (name, key)
        assert json.loads(target['train_sequence_range']) == [0, train]
        assert json.loads(target['eval_sequence_range']) == [train, total]
        checked.append({'pool': total, 'train': train, 'agreement_eval': agreement_eval,
                        'ppl_eval': ppl_eval, 'agreement_percent': round(100 * expected['token_agreement'], 2),
                        'mdp_ppl': round(expected['ppl'], 2), 'native_ppl': round(expected['native_ppl'], 2)})
    assert len(checked) == 5 and sorted(r['pool'] for r in checked) == [1024, 4096, 8192, 16384, 32768]
    # Verify that source paths in the portable metadata are repository-relative.
    for run in meta['runs']:
        assert not Path(run['original_relative_path']).is_absolute()
        for shard in run['generation_shard_summaries']:
            assert not Path(shard['original_relative_path']).is_absolute()
    print(json.dumps({'verified': True, 'rows': checked,
                      'scope': 'Included saved aggregate records and split arithmetic; no training or model inference.'}, indent=2))


if __name__ == '__main__':
    main()
