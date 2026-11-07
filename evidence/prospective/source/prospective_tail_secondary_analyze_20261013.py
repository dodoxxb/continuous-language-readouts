"""Secondary evaluator sensitivity; no changes to the primary regression test."""
import json,math
import numpy as np
from prospective_tail_20261013 import OUT,CONDITIONS,FITS,POLICIES,readcsv,writecsv,dump
from prospective_tail_analyze_20261013 import bootstrap

def main():
    scores=[];seq=[]
    for cond in CONDITIONS:
        for seed in json.loads((OUT/'protocol.json').read_text())['generation_seeds']:
            dest=OUT/'test'/cond/str(seed);methods=['native']+[f'linear_seed{s}' for s in FITS]+POLICIES;rows={m:readcsv(dest/f'opt_{m}.csv') for m in methods}
            scores.extend(r for rr in rows.values() for r in rr)
            for i in range(256):
                if any(int(rows[m][i]['scored_tokens'])<1 for m in methods):continue
                nll={m:float(rows[m][i]['nll_sum'])/int(rows[m][i]['scored_tokens']) for m in methods};rand=np.mean([nll[m] for m in POLICIES if m.startswith('repair_random')])
                seq.append(dict(condition=cond,generation_seed=seed,sequence=i,**{f'repair_{p}_gain':float(rand-nll['repair_'+p]) for p in ['rare','numeric','lowmargin']}))
    summary=[];ppl=[]
    for cond in ['all']+list(CONDITIONS):
        rr=[r for r in seq if cond=='all' or r['condition']==cond];strata=np.array([f"{r['condition']}_{r['generation_seed']}" for r in rr])
        for policy in ['rare','numeric','lowmargin']:
            mean,lo,hi=bootstrap(np.array([r[f'repair_{policy}_gain'] for r in rr]),strata)
            summary.append(dict(condition=cond,policy=policy,mean_random_minus_target_nll=mean,ci_low=lo,ci_high=hi,interval_level=.95,sequences=len(rr)))
    for cond in CONDITIONS:
        for method in methods:
            rr=[r for r in scores if r['condition']==cond and r['method']==method];ss=sum(float(r['nll_sum']) for r in rr);ns=sum(int(r['scored_tokens']) for r in rr);d=sum(int(r['valid_tokens']) for r in rr);cc=sum(int(r['correct']) for r in rr)
            ppl.append(dict(condition=cond,method=method,ppl=math.exp(ss/ns),agreement=cc/d,scored_tokens=ns,valid_tokens=d,correct=cc))
    for name,rr in [('secondary_text_scores.csv',scores),('secondary_per_sequence.csv',seq),('secondary_repair_summary.csv',summary),('secondary_ppl_summary.csv',ppl)]:writecsv(OUT/name,rr)
    dump(OUT/'secondary_analysis.json',dict(repair=summary,ppl=ppl,sequences=len(seq)));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
