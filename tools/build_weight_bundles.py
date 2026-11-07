"""Create uncompressed GitHub-release-sized weight bundles from local assets."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'weights'
GROUPS = {
    'elf_readouts': ['elf'],
    'aurora_readouts': ['aurora'],
    'langflow_readouts': ['langflow_1k', 'langflow_8k'],
    'context_controls': ['context', 'context_global'],
    'calm_coefficients': ['calm'],
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = json.loads((W / 'manifest.json').read_text())
    output = W / 'bundles'
    output.mkdir(exist_ok=True)
    result = {}
    for name, prefixes in GROUPS.items():
        rows = [row for row in manifest['weights'] if Path(row['path']).parts[2] in prefixes]
        assert rows, name
        bundle = output / f'{name}_20261104.tar'
        with tarfile.open(bundle, 'w', format=tarfile.PAX_FORMAT) as tar:
            for row in rows:
                src = ROOT / row['path']
                assert src.is_file() and src.stat().st_size == row['bytes']
                arcname = Path(row['path']).relative_to('weights').as_posix()
                info = tarfile.TarInfo(arcname)
                info.size = row['bytes']; info.mode = 0o644; info.mtime = 0
                with src.open('rb') as f:
                    tar.addfile(info, f)
            data = (json.dumps({'bundle': name, 'files': rows}, indent=2) + '\n').encode()
            info = tarfile.TarInfo(f'bundle_metadata/{name}.json')
            info.size = len(data); info.mode = 0o644; info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
        # GitHub release assets accept files up to 2 GiB; validate the local size.
        assert bundle.stat().st_size < 2 * 1024**3
        result[name] = {'file': bundle.name, 'bytes': bundle.stat().st_size, 'sha256': sha(bundle), 'weights': len(rows)}
        print(name, result[name], flush=True)
    assert sum(x['weights'] for x in result.values()) == len(manifest['weights'])
    (W / 'BUNDLES.json').write_text(json.dumps({'status': 'prepared', 'intended_release_tag': 'v0.1.0', 'extract_into': 'weights/', 'bundles': result}, indent=2) + '\n')


if __name__ == '__main__':
    main()
