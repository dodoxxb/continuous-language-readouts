"""Check each staged bundle and every contained checkpoint without extraction."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'weights'


def sha_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 << 20), b''):
        h.update(block)
    return h.hexdigest()


def main():
    weights = json.loads((W / 'manifest.json').read_text())['weights']
    expected = {Path(x['path']).relative_to('weights').as_posix(): x for x in weights}
    bundles = json.loads((W / 'BUNDLES.json').read_text())['bundles']
    checked = set()
    for name, info in bundles.items():
        path = W / 'bundles' / info['file']
        assert path.stat().st_size == info['bytes']
        with path.open('rb') as stream:
            assert sha_stream(stream) == info['sha256']
        count = 0
        with tarfile.open(path, 'r:') as archive:
            for member in archive:
                assert member.isfile(), member.name
                if member.name == f'bundle_metadata/{name}.json':
                    continue
                assert member.name in expected and member.name not in checked, member.name
                item = expected[member.name]
                assert member.size == item['bytes']
                assert sha_stream(archive.extractfile(member)) == item['sha256']
                checked.add(member.name)
                count += 1
        assert count == info['weights'], name
        print(f'{name}: {count} checkpoint hashes passed')
    assert checked == set(expected)
    print(f'All {len(checked)} checkpoints and {len(bundles)} bundle hashes passed.')


if __name__ == '__main__':
    main()
