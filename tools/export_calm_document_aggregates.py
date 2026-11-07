"""Export CALM correctness aggregates without redistributing token sequences.

Requires the original local evidence tree. The exported data retain one value
per document and condition; they cannot independently recover token identities
or rerun the upstream models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def export(src: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    files = [(f'primary_{p.stem}', p) for p in sorted((src / 'test').glob('*.npz'))]
    files += [('fresh', src / 'fit_control/fresh_predictions.npz'),
              ('generated', src / 'generated_predictions.npz'),
              ('generated_fit', src / 'generated_fit/test_predictions.npz')]
    assert len(files) == 8
    provenance = {}
    for name, path in files:
        with np.load(path, allow_pickle=False) as data:
            native = data['native']
            assert native.ndim == 2 and native.shape[1] == 4 and native.shape[0] % 512 == 0
            result = {}
            for key in data.files:
                if key in {'native', 'margin', 'x'}:
                    continue
                values = data[key]
                assert values.shape == native.shape, (path, key, values.shape)
                matches = values == native
                result[f'token_{key}'] = matches.reshape(512, -1).mean(axis=1)
                result[f'chunk_{key}'] = matches.all(axis=-1).reshape(512, -1).mean(axis=1)
            dest = target / f'{name}.npz'
            np.savez_compressed(dest, **result)
            provenance[name] = {'source_name': str(path.relative_to(src)), 'source_sha256': sha(path),
                                'aggregate_file': dest.name, 'aggregate_sha256': sha(dest),
                                'documents': 512, 'chunks': int(native.shape[0]), 'keys': sorted(result)}
    (target / 'manifest.json').write_text(json.dumps({'description': 'Per-document equality rates only; no token IDs or latent states.', 'files': provenance}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Original new_interfaces/calm directory.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
