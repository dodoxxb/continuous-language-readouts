"""Explicit post-result high-margin versus old rare-first comparison."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/deep_mechanism_20261022/margin_policies'
PARENT=ROOT/'outputs/task_readout_20261022'
FAMILIES=['elf','aurora'];SEEDS=[61022801,61024801]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def loadcsv(path):
    with path.open() as handle:return list(csv.DictReader(handle))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input-root',type=Path,default=OUT)
    parser.add_argument('--parent-root',type=Path,default=PARENT);args=parser.parse_args();out=args.input_root
    plan=json.loads((out/'descriptive_extension.json').read_text())
    for name,expected in plan['unchanged_primary_sha256'].items():assert sha(out/name)==expected
    oldsummary=json.loads((out/'analysis_summary.json').read_text())
    cohort={r['index']:r for r in map(json.loads,(out/'inputs/cohort.jsonl').read_text().splitlines())}
    high={(r['metric'],r['family'],int(r['generation_seed']),int(r['index'])):r for r in loadcsv(out/'metric_rows.csv') if r['arm']=='repair_highmargin'}
    rows=[];bundle={};excluded=set(oldsummary['nll_excluded_document_indices']);newbad=[]
    for family in FAMILIES:
        for seed in SEEDS:
            for index in range(1024):
                q=json.loads((args.parent_root/f'texts/{family}/{seed}/{index:04}.json').read_text())
                text=q['texts']['repair_rare'];digest=hashlib.sha256(text.encode()).hexdigest()
                assert digest==q['text_sha256']['repair_rare'];key=f'{index:04}_{digest}'
                for metric in ['bertscore','nll']:
                    identity=f'{metric}/{family}/{key}'
                    path=args.parent_root/f'scores/{metric}/{family}/{key}.json'
                    if identity not in bundle:
                        record=json.loads(path.read_text())
                        assert record['metric']==metric and record['index']==index and record['output_sha256']==digest
                        assert record['source_sha256']==cohort[index]['document_sha256']
                        assert record['reference_sha256']==cohort[index]['reference_sha256']
                        bundle[identity]=dict(key=identity,source_file=str(path),source_sha256=sha(path),record=record)
                    value=bundle[identity]['record']['value']
                    eligible=True if metric=='bertscore' else bool(value['eligible'])
                    rare=float(value['f1']) if metric=='bertscore' else float(value['nll_sum'])/int(value['tokens']) if eligible else None
                    h=high[(metric,family,seed,index)]
                    if metric=='nll' and not eligible:
                        excluded.add(index);newbad.append(dict(family=family,generation_seed=seed,index=index))
                    rows.append(dict(metric=metric,family=family,generation_seed=seed,index=index,
                                     highmargin=float(h['score']) if h['score']!='' else None,rare=rare,
                                     highmargin_eligible=int(h['eligible']),rare_eligible=int(eligible),
                                     highmargin_input_key=h['input_key'],rare_input_key=identity))
    stats=[]
    for metric in ['bertscore','nll']:
        keep=[i for i in range(1024) if metric=='bertscore' or i not in excluded]
        for family in ['all']+FAMILIES:
            selected=[r for r in rows if r['metric']==metric and r['index'] in keep and (family=='all' or r['family']==family)]
            byindex={i:[] for i in keep}
            for row in selected:byindex[row['index']].append((row['highmargin']-row['rare'])*(100 if metric=='bertscore' else -1))
            values=np.array([np.mean(byindex[i]) for i in keep])
            rng=np.random.default_rng(61022744);indices=rng.integers(len(keep),size=(4000,len(keep)))
            boot=values[indices].mean(1)
            stats.append(dict(metric=metric,family=family,documents=len(keep),mean=float(values.mean()),
                              ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),interval_level=.95,
                              positive='highmargin better than old rare-first',units='BERTScore points x100' if metric=='bertscore' else 'nats per evaluator token'))
    with (out/'high_vs_rare_rows.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'high_vs_rare_inputs.jsonl').open('w') as handle:
        for key in sorted(bundle):handle.write(json.dumps(bundle[key],allow_nan=False)+'\n')
    result=dict(status='complete',scope=plan['status'],plan_sha256=sha(out/'descriptive_extension.json'),
                primary_results_unchanged=True,nll_common_documents=1024-len(excluded),nll_excluded_document_indices=sorted(excluded),
                original_rare_ineligible_replicas=newbad,bootstrap_seed=61022744,bootstrap_draws=4000,summary=stats,
                rows_sha256=sha(out/'high_vs_rare_rows.csv'),inputs_sha256=sha(out/'high_vs_rare_inputs.jsonl'))
    (out/'high_vs_rare_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,expected in plan['unchanged_primary_sha256'].items():assert sha(out/name)==expected
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
