"""Post hoc CPU-only accounting of cached matched readout repairs.

This does not estimate natural-language factuality or identify a generation
mechanism. Original confirmatory task analyses are untouched. The unchanged
token term is an aligned sequence-loss accounting term, not a separately
randomized mediator intervention.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from pathlib import Path
import csv, difflib, hashlib, json, time
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'outputs/task_readout_20261022'
OUT = ROOT / 'outputs/deep_mechanism_20261022/repair'
FAMILIES = ['elf', 'aurora']
SEEDS = [61022801, 61024801]
POLICIES = ['random11', 'random23', 'random37', 'rare', 'numeric', 'lowmargin']
COMPONENTS = ['mean_nll_gain', 'changed_token_gain', 'unchanged_token_gain',
              'length_normalization_gain', 'unchanged_prefix_gain',
              'unchanged_distance_1_4_gain', 'unchanged_distance_5_16_gain',
              'unchanged_distance_17plus_gain', 'reference_unigram_cost_gain',
              'nll_minus_reference_unigram_gain']
FEATURES = ['selected_log1p_native_count', 'selected_log1p_linear_count',
            'selected_log_count_native_minus_linear', 'selected_native_margin',
            'selected_relative_position', 'selected_native_in_source',
            'selected_linear_in_source', 'selected_native_only_in_source',
            'selected_native_in_reference', 'selected_linear_in_reference',
            'selected_native_only_in_reference', 'selected_numeric_native']

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def writecsv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def score_trace(family, index, text, tokenizer):
    key = f'{index:04}_' + hashlib.sha256(text.encode()).hexdigest()
    path = SRC / f'scores/nll/{family}/{key}.json'
    value = json.loads(path.read_text())['value']
    if not value['eligible']:
        return None
    trace = np.load(path.with_suffix('.npy')).astype(np.float64)
    ids = np.array(tokenizer.encode(text, add_special_tokens=False), dtype=np.int64)
    assert len(ids) == len(trace) == value['tokens']
    assert abs(trace.sum() - value['nll_sum']) < 1e-8
    return ids, trace

def decompose(base, repaired, unigram_cost):
    b, bl = base; r, rl = repaired
    nb, nr = len(b), len(r)
    changed_b = np.ones(nb, bool); changed_r = np.ones(nr, bool)
    matched = []
    opcodes = difflib.SequenceMatcher(None, b.tolist(), r.tolist(), autojunk=False).get_opcodes()
    last_b = last_r = -1
    bins = np.zeros(4, np.float64)
    for tag, b0, b1, r0, r1 in opcodes:
        if tag != 'equal':
            # Last changed boundary; a pure deletion/insertion counts as a
            # context change for following otherwise unchanged token targets.
            last_b, last_r = b1 - 1, r1 - 1
            continue
        changed_b[b0:b1] = False; changed_r[r0:r1] = False
        matched.append((b0, b1, r0, r1))
        d = bl[b0:b1] - rl[r0:r1]
        if last_b < 0 and last_r < 0 and b0 == r0 == 0:
            bins[0] += d.sum()
        else:
            distance = np.minimum(np.arange(b0, b1) - last_b,
                                  np.arange(r0, r1) - last_r)
            bins[1] += d[distance <= 4].sum()
            bins[2] += d[(distance > 4) & (distance <= 16)].sum()
            bins[3] += d[distance > 16].sum()
    direct = (bl[changed_b].sum() - rl[changed_r].sum()) / nb
    context = sum((bl[b0:b1] - rl[r0:r1]).sum() for b0,b1,r0,r1 in matched) / nb
    length = rl.sum() * (1 / nb - 1 / nr)
    total = bl.mean() - rl.mean()
    assert abs(direct + context + length - total) < 1e-10
    assert abs(bins.sum() / nb - context) < 1e-10
    unigram_gain = unigram_cost[b].mean() - unigram_cost[r].mean()
    fixed = np.full(3,np.nan)
    if nb == nr:
        changed = b != r
        fixed = np.array([total,(bl[changed]-rl[changed]).sum()/nb,(bl[~changed]-rl[~changed]).sum()/nb])
        assert abs(fixed[0]-fixed[1:].sum())<1e-10
    return np.array([total, direct, context, length, *(bins / nb), unigram_gain, total-unigram_gain]), dict(
        baseline_tokens=nb, repaired_tokens=nr, changed_baseline_tokens=int(changed_b.sum()),
        changed_repaired_tokens=int(changed_r.sum()), unchanged_aligned_tokens=int((~changed_b).sum())),fixed

def bootstrap(x, draws):
    x = np.asarray(x).reshape(1024, -1)
    mask = np.isfinite(x); sums = np.where(mask, x, 0).sum(1); nums = mask.sum(1)
    if not nums.sum():
        return dict(mean=None, lower=None, upper=None, documents=0, generations=0, level=.95)
    a = sums[draws].sum(1); b = nums[draws].sum(1); boot = a[b > 0] / b[b > 0]
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(mean=float(sums.sum()/nums.sum()), lower=float(lo), upper=float(hi),
                documents=int((nums>0).sum()), generations=int(nums.sum()), level=.95)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(status='post_hoc_diagnostic_not_confirmatory',
        motivation='Assess whether matched rare-first NLL gains reduce to changed-token self scores and how repair policies differ in observable selection properties.',
        data='All 4096 frozen XSum generations and all six existing repair policies; no outcome selection or new text generation.',
        alignment='Deterministic difflib.SequenceMatcher(autojunk=False) on the evaluator GPT-2 token sequences; monotone exact-token matches.',
        accounting='meanNLL(base)-meanNLL(repair) = changed-token sum difference / nbase + aligned unchanged-token loss difference / nbase + repair loss sum * (1/nbase-1/nrepair).',
        interpretation='The unchanged-token term shows scorer autoregressive context effects under output edits, not a causal mechanism of diffusion training/generation. Alignment can be ambiguous for repeated tokens and edits can change segmentation.',
        features='Native-tokenizer occurrence in supplied source prefix and reference, token counts from old OWT fit corpus, native margin, position, numeric flag. Token occurrence is not semantic grounding.',
        unigram_null='Fixed external reference prior from AURORA OWT readout-training GPT-2 token counts with additive alpha=1 across the full vocabulary; calculate the exact same mean-per-evaluator-token cost on each output. This is not the scoring model pretraining distribution. NLL minus unigram cost is descriptive arithmetic, not an identified unbiased quality metric.',
        alignment_sensitivity='Added after initial accounting to examine alignment ambiguity: restrict to generations where ALL six repaired texts and baseline have the same evaluator-token length; split loss changes by exact same-position token equality. This output-dependent subgroup is descriptive sensitivity, not a confirmatory causal estimate.',
        uncertainty='Original 4000 paired-document bootstrap draws, 95% descriptive intervals, no multiplicity correction. Pooled NLL retains the original 1023 common documents.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),SRC/'analysis/analysis_arrays.npz',SRC/'analysis/bootstrap_indices.npz',SRC/'cohort.jsonl',ROOT/'outputs/aurora_replication_20261022/counts.npy']})
    dump(OUT/'protocol.json', protocol)
    gpt = AutoTokenizer.from_pretrained(ROOT/'models/gpt2-large', local_files_only=True)
    toks = {f:AutoTokenizer.from_pretrained(ROOT/'models'/('t5-small' if f=='elf' else 'gpt2'), local_files_only=True) for f in FAMILIES}
    cohort = [json.loads(s) for s in (SRC/'cohort.jsonl').read_text().splitlines()]
    counts = {f:np.load(ROOT/('outputs/prospective_tail_20261013/token_counts.npy' if f=='elf' else 'outputs/aurora_replication_20261022/counts.npy')) for f in FAMILIES}
    unigram_counts = counts['aurora'].astype(np.float64)
    assert len(unigram_counts) == len(gpt)
    unigram_cost = -np.log((unigram_counts + 1) / (unigram_counts.sum() + len(unigram_counts)))
    np.save(OUT/'reference_unigram_cost.npy', unigram_cost)
    numeric = {f:np.load(ROOT/('outputs/prospective_tail_20261013/numeric.npy' if f=='elf' else 'outputs/aurora_replication_20261022/numeric.npy')) for f in FAMILIES}
    reference_ids = {f:[set(toks[f].encode(q['reference'], add_special_tokens=False)) for q in cohort] for f in FAMILIES}
    values = np.load(SRC/'analysis/analysis_arrays.npz')['values']
    comp = np.full((1024,2,2,6,len(COMPONENTS)),np.nan)
    feat = np.full((1024,2,2,6,len(FEATURES)),np.nan)
    budgets = np.zeros((1024,2,2),np.int32)
    lengths = np.zeros((1024,2,2,6,5),np.int32)
    fixed = np.full((1024,2,2,6,3),np.nan)
    rows = []; started = time.time()
    for fi,family in enumerate(FAMILIES):
        c = counts[family]
        for si,seed in enumerate(SEEDS):
            for i in range(1024):
                path = SRC/f'texts/{family}/{seed}/{i:04}.json'
                q = json.loads(path.read_text()); z = np.load(path.with_suffix('.npz'))
                y, linear, valid = z['native'], z['linear17'], z['valid']
                budgets[i,fi,si] = q['repair_budget']
                source_set = set(cohort[i][family+'_prefix_ids']); ref_set = reference_ids[family][i]
                base = score_trace(family, i, q['texts']['linear17'], gpt)
                traces = {'linear17':base}
                for pi,policy in enumerate(POLICIES):
                    arm = 'repair_'+policy
                    selected = np.flatnonzero(z[arm] != linear)
                    assert len(selected) == q['repair_budget'] and valid[selected].all()
                    assert (z[arm][selected] == y[selected]).all()
                    r = dict(family=family,seed=seed,index=i,policy=policy,budget=len(selected))
                    if len(selected):
                        ys,ls = y[selected],linear[selected]
                        ns = np.isin(ys,list(source_set)); ps = np.isin(ls,list(source_set))
                        nr = np.isin(ys,list(ref_set)); pr = np.isin(ls,list(ref_set))
                        features = np.array([np.log1p(c[ys]).mean(),np.log1p(c[ls]).mean(),
                            (np.log1p(c[ys])-np.log1p(c[ls])).mean(),z['margin'][selected].mean(),
                            ((selected-q['native_start'])/max(q['valid_tokens']-1,1)).mean(),
                            ns.mean(),ps.mean(),(ns&~ps).mean(),nr.mean(),pr.mean(),(nr&~pr).mean(),numeric[family][ys].mean()])
                        feat[i,fi,si,pi] = features
                        r.update({k:float(v) for k,v in zip(FEATURES,features)})
                    else:r.update({k:'' for k in FEATURES})
                    text = q['texts'][arm]
                    cachekey = q['text_sha256'][arm]
                    if cachekey not in traces:traces[cachekey] = score_trace(family,i,text,gpt)
                    repaired = traces[cachekey]
                    if base is not None and repaired is not None:
                        components,ld,fp = decompose(base,repaired,unigram_cost); comp[i,fi,si,pi] = components
                        fixed[i,fi,si,pi] = fp
                        lengths[i,fi,si,pi] = list(ld.values())
                        original = values[5,i,fi,si,1] - values[5,i,fi,si,4+pi]
                        if np.isfinite(original):assert abs(components[0]-original)<1e-10
                        r.update({k:float(v) for k,v in zip(COMPONENTS,components)});r.update(ld)
                    else:
                        r.update({k:'' for k in COMPONENTS})
                        r.update({k:'' for k in ['baseline_tokens','repaired_tokens','changed_baseline_tokens','changed_repaired_tokens','unchanged_aligned_tokens']})
                    rows.append(r)
                if i%256==255:print(family,seed,i+1,'seconds',round(time.time()-started,1),flush=True)
    writecsv(OUT/'per_generation_policy.csv',rows)
    np.savez_compressed(OUT/'arrays.npz',components=comp,features=feat,budgets=budgets,lengths=lengths,fixed_position=fixed)
    draws = np.load(SRC/'analysis/bootstrap_indices.npz')['indices']
    common = np.isfinite(values[5,:,:,:,:10]).all(axis=(1,2,3))
    nll_results = []; features_results = []; raw_results=[];fixed_results=[]
    for group in ['pooled']+FAMILIES:
        fidx = slice(None) if group=='pooled' else slice(FAMILIES.index(group),FAMILIES.index(group)+1)
        cc = comp[:,fidx].copy(); ff = feat[:,fidx].copy()
        if group=='pooled':cc[~common] = np.nan
        fx = fixed[:,fidx].copy()
        fx[~np.isfinite(fx).all(axis=(3,4))]=np.nan
        if group=='pooled':fx[~common]=np.nan
        for pi,policy in enumerate(POLICIES):
            for ki,key in enumerate(COMPONENTS):
                raw_results.append(dict(group=group,policy=policy,component=key,contrast='linear_minus_repaired',**bootstrap(cc[:,:,:,pi,ki],draws)))
            for ki,key in enumerate(FEATURES):
                features_results.append(dict(group=group,policy=policy,feature=key,contrast='selected_mean',**bootstrap(ff[:,:,:,pi,ki],draws)))
        for pi,policy in enumerate(POLICIES[3:],3):
            cd = cc[:,:,:,pi,:] - cc[:,:,:,:3,:].mean(3)
            fd = ff[:,:,:,pi,:] - ff[:,:,:,:3,:].mean(3)
            xd = fx[:,:,:,pi,:] - fx[:,:,:,:3,:].mean(3)
            for ki,key in enumerate(['mean_nll_gain','changed_position_gain','unchanged_position_gain']):
                fixed_results.append(dict(group=group,policy=policy,component=key,contrast='same_length_all_six_policies_targeted_minus_random',**bootstrap(xd[:,:,:,ki],draws)))
            for ki,key in enumerate(COMPONENTS):
                nll_results.append(dict(group=group,policy=policy,component=key,contrast='targeted_gain_minus_random_gain',**bootstrap(cd[:,:,:,ki],draws)))
            for ki,key in enumerate(FEATURES):
                features_results.append(dict(group=group,policy=policy,feature=key,contrast='targeted_minus_random',**bootstrap(fd[:,:,:,ki],draws)))
    # Full equality is checked per generation and arm above, not just after
    # averaging; report the maximum accumulated accounting residual as well.
    residual = comp[...,0] - comp[...,1:4].sum(-1)
    finite = np.isfinite(residual)
    summary = dict(protocol=protocol,documents=1024,generations=4096,pooled_nll_common_documents=int(common.sum()),
        components=COMPONENTS,features=FEATURES,nll_contrasts=nll_results,baseline_repair_components=raw_results,
        fixed_position_sensitivity=fixed_results,
        selected_features=features_results,verification=dict(max_accounting_residual=float(np.abs(residual[finite]).max()),
        all_policies_equal_native_edit_budget=True,all_4096_generations_retained=True),
        caveats=['Post hoc descriptive intervals are not new confirmatory hypothesis tests.',
                 'Scorer-token alignment does not isolate factual content or generation causality.',
                 'Token occurrence in source/reference is a surface overlap diagnostic, not factuality.',
                 'Equal native-token correction budgets need not equal evaluator-token edit counts.'])
    dump(OUT/'summary.json',summary)
    writecsv(OUT/'nll_contrasts.csv',nll_results);writecsv(OUT/'selected_features.csv',features_results)
    writecsv(OUT/'fixed_position_sensitivity.csv',fixed_results)
    print(json.dumps(dict(nll_contrasts=[q for q in nll_results if q['group']=='pooled'],verification=summary['verification']),indent=2))

if __name__=='__main__':main()
