"""Paired-document analysis of locally frozen exploratory margin policies."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'outputs/deep_mechanism_20261022/margin_policies'
PARENT=ROOT/'outputs/task_readout_20261022'
ARMS=['repair_highmargin','repair_pair_rare','repair_pair_common',
      'repair_lowmargin','repair_random11','repair_random23','repair_random37']
NEW=ARMS[:3]
SEEDS=[61022801,61024801]
FAMILIES=['elf','aurora']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def csvwrite(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def collect(out,parent):
    cohort={r['index']:r for r in map(json.loads,(out/'inputs/cohort.jsonl').read_text().splitlines())}
    rows=[];bundle={};missing=[]
    for family in FAMILIES:
        for seed in SEEDS:
            for index in range(1024):
                new=json.loads((out/f'texts/{family}/{seed}/{index:04}.json').read_text())
                old=json.loads((parent/f'texts/{family}/{seed}/{index:04}.json').read_text())
                for arm in ARMS:
                    q=new if arm in NEW else old
                    text=q['texts'][arm]
                    digest=hashlib.sha256(text.encode()).hexdigest()
                    assert digest==q['text_sha256'][arm]
                    key=f'{index:04}_'+digest
                    for metric in ['bertscore','nll']:
                        choices=[out/f'scores/{metric}/{family}/{key}.json',parent/f'scores/{metric}/{family}/{key}.json']
                        path=next((p for p in choices if p.exists()),None)
                        if path is None:
                            missing.append(dict(metric=metric,family=family,generation_seed=seed,index=index,arm=arm,key=key));continue
                        ident=f'{metric}/{family}/{key}'
                        if ident not in bundle:
                            record=json.loads(path.read_text())
                            assert record['metric']==metric and record['index']==index
                            assert record['output_sha256']==digest
                            for field in ['source_sha256','reference_sha256']:
                                assert record[field]==cohort[index]['document_sha256' if field=='source_sha256' else 'reference_sha256']
                            bundle[ident]=dict(key=ident,source_file=str(path),source_sha256=sha(path),record=record)
                        value=bundle[ident]['record']['value']
                        eligible=bool(value['eligible']) if metric=='nll' else True
                        score=float(value['nll_sum'])/int(value['tokens']) if metric=='nll' and eligible else float(value['f1']) if metric=='bertscore' else None
                        assert score is None or np.isfinite(score)
                        rows.append(dict(metric=metric,family=family,generation_seed=seed,index=index,arm=arm,
                                         eligible=int(eligible),score=score,input_key=ident,output_sha256=digest))
    if missing:
        dump(out/'metric_missing.json',dict(missing_rows=len(missing),first_examples=missing[:20],scores_complete=False))
        raise RuntimeError(f'{len(missing)} arm-metric rows still missing; no analysis emitted')
    return rows,bundle


def statistics(rows):
    values={}
    exclusions=[]
    for row in rows:
        key=(row['metric'],row['family'],row['generation_seed'],row['index'],row['arm'])
        assert key not in values
        values[key]=row['score']
        if row['metric']=='nll' and not row['eligible']:
            exclusions.append({k:row[k] for k in ['family','generation_seed','index','arm','input_key']})
    excluded=sorted(set(r['index'] for r in exclusions))
    summary=[];documents=[]
    for metric in ['bertscore','nll']:
        keep=np.array([i for i in range(1024) if metric=='bertscore' or i not in excluded],dtype=int)
        assert len(keep)>0
        for family in ['all']+FAMILIES:
            families=FAMILIES if family=='all' else [family]
            perarm={arm:np.array([np.mean([values[(metric,f,seed,int(i),arm)] for f in families for seed in SEEDS]) for i in keep]) for arm in ARMS}
            random=np.mean([perarm[f'repair_random{s}'] for s in [11,23,37]],axis=0)
            sign=100. if metric=='bertscore' else -1.
            contrasts={'paired_count':sign*(perarm['repair_pair_rare']-perarm['repair_pair_common']),
                       'high_vs_low':sign*(perarm['repair_highmargin']-perarm['repair_lowmargin']),
                       'high_vs_random':sign*(perarm['repair_highmargin']-random)}
            rng=np.random.default_rng(61022743)
            indices=rng.integers(len(keep),size=(4000,len(keep)))
            stats={}
            for name,val in contrasts.items():
                level=.975 if family=='all' and name=='paired_count' else .95
                alpha=(1-level)/2
                draw=val[indices].mean(1)
                stats[name]=dict(mean=float(val.mean()),ci_low=float(np.quantile(draw,alpha)),
                                 ci_high=float(np.quantile(draw,1-alpha)),interval_level=level,
                                 interpretation='Positive favors pair_rare' if name=='paired_count' else 'Positive favors highmargin')
            summary.append(dict(metric=metric,family=family,documents=len(keep),generations=len(keep)*len(families)*2,
                                units='BERTScore F1 points (x100)' if metric=='bertscore' else 'nats per evaluator token',
                                arm_means={k:float(v.mean()*(100. if metric=='bertscore' else 1.)) for k,v in perarm.items()},
                                contrasts=stats))
            for j,index in enumerate(keep):
                documents.append(dict(metric=metric,family=family,index=int(index),
                                      **{k:float(v[j]) for k,v in contrasts.items()},
                                      **{k:float(v[j]) for k,v in perarm.items()}))
    return dict(status='complete',scores_complete=True,scope='Exploratory locally frozen policies on previously observed generated states; not confirmatory.',
                resampling_unit='Common XSum document, paired across two seeds and both architectures',
                bootstrap_draws=4000,bootstrap_seed=61022743,
                bertscore_documents=1024,nll_common_documents=1024-len(excluded),
                nll_excluded_document_indices=excluded,nll_ineligible_arm_replicas=exclusions,
                nll_common_set='Intersection over all seven compared arms, two architectures and two generation seeds; same common document set for pooled and family-specific NLL contrasts.',
                summary=summary),documents


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input-root',type=Path,default=DEFAULT)
    parser.add_argument('--parent-root',type=Path,default=PARENT);args=parser.parse_args()
    out=args.input_root
    rows,bundle=collect(out,args.parent_root)
    result,documents=statistics(rows)
    result['protocol_sha256']=sha(out/'protocol.json')
    csvwrite(out/'metric_rows.csv',rows);csvwrite(out/'document_metrics.csv',documents)
    with (out/'metric_inputs.jsonl').open('w') as handle:
        for key in sorted(bundle):handle.write(json.dumps(bundle[key],allow_nan=False)+'\n')
    result['metric_rows_sha256']=sha(out/'metric_rows.csv')
    result['metric_inputs_sha256']=sha(out/'metric_inputs.jsonl')
    dump(out/'analysis_summary.json',result)
    (out/'analysis').mkdir(exist_ok=True)
    dump(out/'analysis/summary.json',result)
    dump(out/'analysis_ready.json',dict(status='scores_analyzed',scores_complete=True,
                                      analysis_sha256=sha(out/'analysis_summary.json'),portable_metric_verification_required=True))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
