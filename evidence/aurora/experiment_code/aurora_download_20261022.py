"""Download only the two pinned research checkpoints; bounded network/IO load."""
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/aurora_replication_20261022'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    records = []
    for name, folder in [('aurora-gpt2-owt', 'owt'), ('aurora-text-autoencoder', 'autoencoder')]:
        meta = json.loads((OUT / 'source' / (name + '_metadata.json')).read_text())
        dest = ROOT / 'models/aurora-lm' / folder
        dest.mkdir(parents=True, exist_ok=True)
        for entry in meta['siblings']:
            filename = entry['rfilename']
            if filename not in ['config.yaml', 'README.md'] and not filename.endswith('.safetensors'):
                continue
            path = dest / filename
            expected = entry.get('lfs', {}).get('sha256')
            if not path.exists():
                tmp = path.with_name(path.name + '.part')
                url = f"https://huggingface.co/fyv587/{name}/resolve/{meta['sha']}/{filename}"
                expected_size=entry['size']
                failures=0
                while (tmp.stat().st_size if tmp.exists() else 0) < expected_size:
                    offset=tmp.stat().st_size if tmp.exists() else 0
                    end=min(offset+(16<<20),expected_size)-1
                    request=urllib.request.Request(url+f'?download=true&offset={offset}',headers={'Range':f'bytes={offset}-{end}'})
                    started=time.monotonic();total=0
                    try:
                        with urllib.request.urlopen(request,timeout=60) as response:
                            if response.status==206:
                                if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                                    raise RuntimeError('Wrong ranged response')
                            elif offset:
                                raise RuntimeError('Server ignored resume range')
                            with tmp.open('ab') as f:
                                remaining=end-offset+1
                                while remaining:
                                    block=response.read(min(1<<20,remaining))
                                    if not block:break
                                    f.write(block);total+=len(block);remaining-=len(block)
                                    delay=total/(12<<20)-(time.monotonic()-started)
                                    if delay>0:time.sleep(delay)
                                f.flush();os.fsync(f.fileno())
                        if total==0:raise RuntimeError('Empty download response')
                        print('Downloaded',filename,tmp.stat().st_size,'/',expected_size,flush=True)
                        failures=0
                    except Exception as e:
                        failures+=1;print('Retry',filename,type(e).__name__,str(e),flush=True)
                        if failures>=5:raise
                        time.sleep(2)
                if tmp.stat().st_size != expected_size:
                    raise RuntimeError('Size mismatch: '+filename)
                if expected and sha256(tmp) != expected:
                    raise RuntimeError('Checksum mismatch: ' + filename)
                tmp.replace(path)
            digest = sha256(path)
            if expected and digest != expected:
                raise RuntimeError('Checksum mismatch: ' + filename)
            records.append(dict(repo=meta['id'], revision=meta['sha'], path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest, upstream_lfs_sha256=expected))
            print('Verified', filename, path.stat().st_size, digest, flush=True)
    (OUT / 'source/download_manifest.json').write_text(json.dumps(records, indent=2) + '\n')

if __name__ == '__main__':
    main()
