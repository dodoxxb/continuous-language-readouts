"""Independent invariants over actual token edits and measurement records."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/prospective_tail_20261013'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    frozen=json.loads((OUT/'frozen.json').read_text())
    for name,h in frozen['files'].items():
        if name.endswith(('.py','.json','.csv','.npy')):assert sha(ROOT/name)==h,name
    amend=json.loads((OUT/'inference_amendment_before_generation.json').read_text())
    assert sha(ROOT/'scripts/prospective_tail_analyze_20261013.py')==amend['analysis_sha256']
    secondary=json.loads((OUT/'secondary_protocol.json').read_text())
    assert sha(ROOT/'scripts/prospective_tail_secondary_20261013.py')==secondary['code_sha256']
    report=[]
    for dest in sorted((OUT/'test').glob('*/*')):
        token=np.load(dest/'tokens.npz');rep=np.load(dest/'repairs.npz');y=token['native'];v=token['valid'];p=token['linear_seed17'];wrong=(p!=y)&v;n=wrong.sum(1)//2
        assert y.shape==(256,1024) and np.array_equal(v,y!=0)
        gm=json.loads((dest/'generation.json').read_text());assert gm['tokens_sha256']==sha(dest/'tokens.npz')
        for policy in rep.files:
            ids=rep[policy];changed=ids!=p
            assert not (changed&~wrong).any(),(dest,policy,'changed an originally correct or invalid position')
            assert np.array_equal(changed.sum(1),n),(dest,policy,'repair budget')
            assert np.array_equal(ids[changed],y[changed]),(dest,policy,'wrong repair label')
            assert np.array_equal(ids[~changed],p[~changed])
        for prefix in ['ppl','opt']:
            for path in dest.glob(prefix+'_*.csv'):
                rows=list(csv.DictReader(path.open()));assert len(rows)==256
                assert [int(r['sequence']) for r in rows]==list(range(256))
                assert all(int(r['scored_tokens'])>0 and np.isfinite(float(r['nll_sum'])) for r in rows)
                assert all(0<int(r['scored_tokens'])<=1023 for r in rows)
        report.append(dict(condition=dest.parent.name,generation_seed=int(dest.name),sequences=256,valid_tokens=int(v.sum()),original_errors=int(wrong.sum()),corrected_per_arm=int(n.sum()),repair_arms=len(rep.files),primary_score_files=len(list(dest.glob('ppl_*.csv'))),secondary_score_files=len(list(dest.glob('opt_*.csv')))))
    assert len(report)==6 and all(r['repair_arms']==6 and r['primary_score_files']==10 and r['secondary_score_files']==10 for r in report)
    result=dict(status='passed',frozen_code_and_inputs_unchanged=True,actual_edit_invariants_checked=True,common_scorable_sequences=1536,pools=report)
    (OUT/'integrity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
