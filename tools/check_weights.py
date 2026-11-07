"""Verify locally staged release-asset size and SHA-256, without loading it."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', help='Only check asset paths under this group, e.g. elf/ or context/elf/.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'weights/manifest.json').read_text())
    items = [item for item in manifest['weights'] if not args.prefix or item['path'].removeprefix('weights/release_assets/').startswith(args.prefix)]
    if not items:
        parser.error('no weights match that prefix')
    missing, mismatched = [], []
    for item in items:
        path = ROOT / item['path']
        if not path.is_file():
            missing.append(item['path'])
            continue
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
        if path.stat().st_size != item['bytes'] or h.hexdigest() != item['sha256']:
            mismatched.append(item['path'])
    print(json.dumps({'expected': len(items), 'missing': missing, 'mismatched': mismatched}, indent=2))
    if missing or mismatched:
        sys.exit(1)


if __name__ == '__main__':
    main()
