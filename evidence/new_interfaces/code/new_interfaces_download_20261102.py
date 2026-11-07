"""Pinned, resumable HTTP range download; four streams and no global installs."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, time, urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/new_interfaces_20261102'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def fetch(repo, rev, remote, dest, size, offset=0, expected=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size==size and (not expected or sha(dest)==expected):
        return dict(path=str(dest.relative_to(ROOT)),bytes=size,sha256=sha(dest),reused=True)
    chunks=dest.with_name(dest.name+'.chunks'); chunks.mkdir(exist_ok=True)
    span=4<<20
    def part(start):
        length=min(span,size-start); p=chunks/f'{start:012d}'
        if p.exists() and p.stat().st_size==length:return
        for attempt in range(6):
            host='huggingface.co' if attempt%2==0 else 'hf-mirror.com'
            url=f'https://{host}/{repo}/resolve/{rev}/{remote}?download=true&range_start={offset+start}&attempt={attempt}'
            req=urllib.request.Request(url,headers={'Range':f'bytes={offset+start}-{offset+start+length-1}'})
            try:
                t=time.monotonic()
                with urllib.request.urlopen(req,timeout=25) as r:
                    if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset+start}-'):
                        raise RuntimeError('Server did not honor exact range')
                    data=r.read(length)
                if len(data)!=length: raise RuntimeError('Short range')
                tmp=p.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(p)
                # At most 8 MiB/s aggregate over four streams.
                time.sleep(max(0,length/(2<<20)-(time.monotonic()-t)))
                return
            except Exception as e:
                print('retry',dest.name,start,attempt,type(e).__name__,flush=True)
                if attempt==5:raise
                time.sleep(1)
    starts=list(range(0,size,span))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(part,starts)):
            if i%16==0:print(dest.name,min((i+1)*span,size),'/',size,flush=True)
    tmp=dest.with_name(dest.name+'.verified_partial')
    with tmp.open('wb') as f:
        for start in starts:f.write((chunks/f'{start:012d}').read_bytes())
    digest=sha(tmp)
    if expected and digest!=expected:raise RuntimeError('SHA256 mismatch: '+str(dest))
    tmp.replace(dest)
    for p in chunks.iterdir():p.unlink()
    chunks.rmdir()
    return dict(path=str(dest.relative_to(ROOT)),bytes=size,sha256=digest,upstream_sha256=expected,range_offset=offset)

def small(repo,rev,remote,dest,expected_size):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{rev}/{remote}',timeout=45) as r: data=r.read()
        if len(data)!=expected_size:raise RuntimeError(remote)
        dest.write_bytes(data)
    assert dest.stat().st_size==expected_size
    return dict(path=str(dest.relative_to(ROOT)),bytes=expected_size,sha256=sha(dest))

def main():
    a=argparse.ArgumentParser();a.add_argument('which',choices=['calm','llaco']);args=a.parse_args()
    records=[]
    repos=['cccczshao/CALM-Autoencoder','cccczshao/CALM-M'] if args.which=='calm' else ['escapist413/LLaCO-8B']
    for repo in repos:
        meta=json.loads((OUT/'source'/(repo.replace('/','_')+'.json')).read_text());rev=meta['sha']
        for e in meta['siblings']:
            name=e['rfilename'];dest=ROOT/'models'/repo.split('/')[-1]/name
            if name=='.gitattributes' or name=='training_args.bin' or name=='__0_0.distcp':continue
            if repo.endswith('Autoencoder') and name=='pytorch_model.bin':continue
            kwargs=dict(repo=repo,rev=rev,remote=name,dest=dest)
            if e['size']>4<<20:r=fetch(**kwargs,size=e['size'],expected=e.get('lfs',{}).get('sha256'))
            else:r=small(**kwargs,expected_size=e['size'])
            r.update(repo=repo,revision=rev);records.append(r)
        print('verified',repo,flush=True)
    if args.which=='llaco':
        # Exact individual torch-save records, indexed by upstream DCP metadata.
        # No full-model or checkpoint-level SHA verification is claimed for ranges.
        import pickle
        m=pickle.loads((ROOT/'models/LLaCO-8B/.metadata').read_bytes())
        for name in ['spherical_adapter.log_beta','spherical_adapter.bias','spherical_geometry.E_geometry']:
            si=[s for ix,s in m.storage_data.items() if ix.fqn==name]
            assert len(si)==1;s=si[0]
            r=fetch(repo,rev,s.relative_path,ROOT/'models/LLaCO-8B/components'/(name+'.pt'),s.length,s.offset)
            r.update(repo=repo,revision=rev,component=name,verification='HTTP Content-Range, record length, local SHA256; no upstream per-record hash')
            records.append(r)
    (OUT/'source'/f'{args.which}_download_manifest.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
