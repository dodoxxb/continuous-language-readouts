"""Score only independently stopped native/autonomous outputs of a fresh cohort."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
import task_readout_score_20261022 as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/native_free_routing_20261023'
PARENT=ROOT/'outputs/task_readout_20261022'


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--family',choices=['elf','aurora'],required=True)
    p.add_argument('--metric',choices=['rouge','bertscore','nll'],required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.device.startswith('cuda'):base.resources(a.device)
    else:torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if a.metric=='bertscore':
        for r in json.loads((PARENT/'source/roberta-large_download_verified.json').read_text()):
            assert base.sha(ROOT/r['path'])==r['sha256']
    metric=base.load_metric(a.metric,a.device)
    rows={r['index']:r for r in map(json.loads,(OUT/'cohort/cohort.jsonl').read_text().splitlines())}
    paths=sorted((OUT/'texts'/a.family).glob('*/[0-9][0-9][0-9][0-9].json'))
    assert len(paths)==512, f'Incomplete generation cohort: {len(paths)}'
    dest=OUT/'scores'/a.metric/a.family
    dest.mkdir(parents=True,exist_ok=True)
    stream=torch.cuda.Stream(device=a.device,priority=1) if a.device.startswith('cuda') else None
    if stream:stream.wait_stream(torch.cuda.current_stream(a.device))
    from contextlib import nullcontext
    started=time.monotonic();n=0;hits=0
    for path in paths:
        q=json.loads(path.read_text());row=rows[q['index']]
        assert set(q['texts'])=={'native','autonomous17'}
        assert q['source_sha256']==row['document_sha256']
        assert q['reference_sha256']==row['reference_sha256']
        for arm,text in q['texts'].items():
            target=dest/(base.key(q['index'],text)+'.json')
            if target.exists():
                old=json.loads(target.read_text())
                assert old['source_sha256']==row['document_sha256']
                assert old['reference_sha256']==row['reference_sha256']
                assert old['output_sha256']==q['text_sha256'][arm]
                hits+=1;continue
            with torch.cuda.stream(stream) if stream else nullcontext():
                value,trace=base.calculate(a.metric,metric,row['document'],row['reference'],text,a.device)
            if trace is not None:
                np.save(target.with_suffix('.npy'),trace)
                value['trace_sha256']=base.sha(target.with_suffix('.npy'))
            base.dump(target,dict(metric=a.metric,index=q['index'],output_sha256=q['text_sha256'][arm],
                source_sha256=row['document_sha256'],reference_sha256=row['reference_sha256'],value=value))
            n+=1
        if q['index']%32==31:
            print(a.metric,a.family,q['generation_seed'],q['index'],'new',n,'seconds',round(time.monotonic()-started,1),flush=True)
    record=dict(metric=a.metric,family=a.family,records=len(paths),new_scores=n,cache_hits=hits,
        elapsed=time.monotonic()-started,complete=True)
    base.dump(OUT/'scores'/f'complete_{a.metric}_{a.family}.json',record)
    print(record,flush=True)


if __name__=='__main__':main()
