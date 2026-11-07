"""Build frozen post hoc margin controls on saved XSum endpoints, CPU only."""
import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/task_readout_20261022'
OUT = ROOT / 'outputs/deep_mechanism_20261022/margin_policies'
ARMS = ['repair_highmargin', 'repair_pair_rare', 'repair_pair_common']
SEEDS = [61022801, 61024801]
FAMILIES = ['elf', 'aurora']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def csvwrite(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parent_inputs():
    paths = [PARENT/'cohort.jsonl', PARENT/'protocol.json', PARENT/'frozen.json',
             ROOT/'outputs/prospective_tail_20261013/token_counts.npy',
             ROOT/'outputs/aurora_replication_20261022/counts.npy']
    for family in FAMILIES:
        for seed in SEEDS:
            base = PARENT/'texts'/family/str(seed)
            for i in range(1024):
                paths += [base/f'{i:04}.json', base/f'{i:04}.npz']
    return paths


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        verify_parents()
        print('Existing margin protocol and all parent hashes verified.')
        return
    if (OUT/'texts').exists():
        raise RuntimeError('Freeze must precede new intervention outputs')
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in parent_inputs()}
    protocol = dict(
        date='2026-10-22', status='Post hoc diagnostic policy design on previously observed generations; new policies frozen before their metric outcomes are scored. Not confirmatory or preregistered.',
        cohorts='All 1024 existing common XSum documents, two generation seeds and two architectures: 4096 outputs. Fit 17 affine output is the shared baseline.',
        families=FAMILIES, generation_seeds=SEEDS, sequences_per_pool=1024, arms=ARMS,
        boundary='Reuse original native valid mask and native end exactly. No EOS changes or additional exclusions. AURORA leading EOS handling remains original.',
        budget='For e valid native/affine mismatches correct exactly floor(e/2) native token labels. Keep zero-edit and empty-output samples.',
        random_seed_components='NumPy default_rng(SeedSequence([generation_seed, document_index, family_index, 61022741, stream])). Stream 0: ascending margin ties; stream 1: descending margin ties; stream 2: equal-count pair choices.',
        highmargin='Order valid mismatches by descending native top1-top2 margin, with fixed independent random tie priorities; correct the first floor(e/2).',
        pairing='Order valid mismatches by ascending native margin with fixed random ties; form consecutive disjoint pairs. If e is odd leave the last (highest-margin, with seeded ties) unmatched in BOTH pair arms. Within each pair rare selects the lower native-token training count and common the higher; equal counts use a fixed random bit and opposite members. Every pair contributes one correction to each arm.',
        counts='Original interface-specific OWT native-token count arrays, unchanged. No new count threshold.',
        diagnostics='Export selected and remaining margin, raw and log1p native training count, relative position within native output span, native and affine token-ID presence in the actual source prefix. Pair confidence match is approximate: retain and report every absolute gap, maximum, quantiles, signed imbalance, count ties and source/position imbalance. No pair dropping or caliper.',
        interpretation='Separates confidence-ranked repair from count ranking inside adjacent-confidence pairs. Native oracle correction changes output tokens at fixed generated states. Does not manipulate lexical rarity causally, identify generation dynamics, or yield a deployable routing rule.',
        metrics=['BERTScore F1 using existing RoBERTa-large layer17 settings', 'GPT2-Large NLL using existing task boundary and scorer'],
        primary_diagnostic='pair_rare minus pair_common BERTScore; pair_common minus pair_rare NLL. Positive favors lower-count repair. Use 97.5% paired-document bootstrap intervals for these two metric contrasts, 4000 draws; resample the same 1024 document indices across both seeds and architectures.',
        secondary_diagnostic='highmargin vs original lowmargin and vs mean of original three random arms; family-specific and other contrasts descriptive 95% paired-document intervals. Retain all signs; no policy selection.',
        missingness='All outputs retained for BERTScore. NLL uses the common eligible document set across the compared arms and all 4 replicas; report every exclusion. No SummaC scoring, factuality claim, human-quality claim, or throughput claim.',
        execution='CPU-only construction, maximum two BLAS/OMP threads; root separately runs one GPU metric worker with the existing 5.5 GiB cap.',
        parent_input_sha256=inputs)
    dump(OUT/'protocol.json', protocol)
    dump(OUT/'frozen.json', dict(protocol_sha256=sha(OUT/'protocol.json'), intervention_outputs_created=False,
                                new_policy_outcomes_scored=False, script_sha256=sha(Path(__file__))))
    print('Frozen three margin policies and parent hashes before construction/scoring.')


def verify_parents():
    p = json.loads((OUT/'protocol.json').read_text())
    f = json.loads((OUT/'frozen.json').read_text())
    assert sha(OUT/'protocol.json') == f['protocol_sha256']
    for name, expected in p['parent_input_sha256'].items():
        assert sha(ROOT/name) == expected, name
    return p


def rng(seed, index, family, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, index, FAMILIES.index(family), 61022741, stream]))


def select(native, linear, margin, valid, counts, family, seed, index):
    errors = np.flatnonzero(valid & (native != linear))
    budget = len(errors)//2
    order = errors[np.lexsort((rng(seed,index,family,0).random(len(errors)), margin[errors]))]
    pairs = order[:2*budget].reshape(-1, 2)
    high = errors[np.lexsort((rng(seed,index,family,1).random(len(errors)), -margin[errors]))[:budget]]
    low_side = rng(seed,index,family,2).integers(2, size=budget)
    if budget:
        left, right = counts[native[pairs[:, 0]]], counts[native[pairs[:, 1]]]
        low_side = np.where(left < right, 0, np.where(right < left, 1, low_side))
    rare = pairs[np.arange(budget), low_side]
    common = pairs[np.arange(budget), 1-low_side]
    return dict(zip(ARMS, [high, rare, common])), pairs, int(order[-1]) if len(errors)%2 else -1


def means(indices, native, linear, margin, counts, source, start, end):
    if not len(indices):
        return {k: None for k in ['margin', 'native_count', 'log1p_native_count', 'relative_position',
                                  'native_source_overlap', 'affine_source_overlap']}
    return dict(margin=float(margin[indices].mean()), native_count=float(counts[native[indices]].mean()),
                log1p_native_count=float(np.log1p(counts[native[indices]]).mean()),
                relative_position=float(((indices-start)/max(end-start-1,1)).mean()),
                native_source_overlap=float(np.isin(native[indices], source).mean()),
                affine_source_overlap=float(np.isin(linear[indices], source).mean()))


def build():
    verify_parents()
    from transformers import AutoTokenizer
    cohort = {r['index']:r for r in map(json.loads,(PARENT/'cohort.jsonl').read_text().splitlines())}
    inp = OUT/'inputs'
    inp.mkdir(exist_ok=True)
    shutil.copy2(PARENT/'cohort.jsonl', inp/'cohort.jsonl')
    sequence_rows, pair_rows, input_manifest, output_manifest = [], [], {}, {}
    for family in FAMILIES:
        tokenizer = AutoTokenizer.from_pretrained(ROOT/'models'/('t5-small' if family=='elf' else 'gpt2'), local_files_only=True)
        cp = ROOT/('outputs/prospective_tail_20261013/token_counts.npy' if family=='elf' else 'outputs/aurora_replication_20261022/counts.npy')
        shutil.copy2(cp, inp/f'counts_{family}.npy')
        counts = np.load(cp)
        for seed in SEEDS:
            oldbase = PARENT/'texts'/family/str(seed)
            dest = OUT/'texts'/family/str(seed)
            dest.mkdir(parents=True, exist_ok=True)
            oldmeta = [json.loads((oldbase/f'{i:04}.json').read_text()) for i in range(1024)]
            width = max(q['canvas_length'] for q in oldmeta)
            inputs = {k:np.zeros((1024,width),dtype=dtype) for k,dtype in
                      [('native',np.int64),('linear17',np.int64),('margin',np.float64),('valid',bool)]}
            inputs.update({k:np.array([q[k] for q in oldmeta], dtype=np.int64)
                           for k in ['canvas_length','native_start','native_end']})
            for index, old in enumerate(oldmeta):
                assert old['index']==index and old['family']==family and old['generation_seed']==seed
                z=np.load(oldbase/f'{index:04}.npz')
                native,linear,margin,valid = z['native'],z['linear17'],z['margin'].astype(np.float64),z['valid']
                length=len(native)
                assert old['evidence_sha256']==sha(oldbase/f'{index:04}.npz')
                for k in ['native','linear17','margin','valid']:
                    inputs[k][index,:length] = locals().get({'linear17':'linear'}.get(k,k))
                selected,pairs,unpaired=select(native,linear,margin,valid,counts,family,seed,index)
                errors=np.flatnonzero(valid & (native!=linear));budget=len(errors)//2
                assert budget==old['repair_budget']
                source=np.array(cohort[index][family+'_prefix_ids'],dtype=np.int64)
                start,end=old['native_start'],old['native_end']
                fields={k:old[k] for k in ['index','family','generation_seed','native_start','native_end',
                        'canvas_length','valid_tokens','errors','repair_budget','editable','native_eos_present','states_sha256']}
                fields.update(texts={},text_sha256={},agreement={},readout_ends={},
                              parent_text_sha256=sha(oldbase/f'{index:04}.json'),
                              parent_evidence_sha256=sha(oldbase/f'{index:04}.npz'),
                              protocol_sha256=sha(OUT/'protocol.json'), unmatched_error=unpaired)
                evidence=dict(native=native,linear17=linear,margin=margin,valid=valid,pairs=pairs,
                              unmatched_error=np.array(unpaired,dtype=np.int64))
                # The original native/affine decode is independently replayed before
                # new text export; this catches boundary/tokenizer drift.
                for arm,ids in [('native',native),('linear17',linear)]:
                    text=tokenizer.decode(np.where(valid,ids,native)[:end],skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
                    assert text==old['texts'][arm], (family,seed,index,arm)
                for arm,choice in selected.items():
                    repaired=linear.copy();repaired[choice]=native[choice]
                    mask=np.zeros(length,dtype=bool);mask[choice]=True
                    assert int(mask.sum())==budget
                    assert int(((repaired!=native)&valid).sum())==len(errors)-budget
                    assert np.array_equal(np.flatnonzero(repaired!=linear),np.sort(choice))
                    text=tokenizer.decode(np.where(valid,repaired,native)[:end],skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
                    fields['texts'][arm]=text
                    fields['text_sha256'][arm]=hashlib.sha256(text.encode()).hexdigest()
                    fields['agreement'][arm]=dict(correct=int(((repaired==native)&valid).sum()),total=int(valid.sum()))
                    fields['readout_ends'][arm]=end
                    evidence[arm]=repaired;evidence[arm+'_selected']=mask
                    remaining=np.setdiff1d(errors,choice)
                    selected_stats=means(choice,native,linear,margin,counts,source,start,end)
                    remaining_stats=means(remaining,native,linear,margin,counts,source,start,end)
                    sequence_rows.append(dict(family=family,generation_seed=seed,index=index,arm=arm,
                        errors=len(errors),budget=budget,remaining_errors=len(remaining),
                        **{'selected_'+k:v for k,v in selected_stats.items()},
                        **{'remaining_'+k:v for k,v in remaining_stats.items()}))
                for pi,(left,right) in enumerate(pairs):
                    rare=int(selected['repair_pair_rare'][pi]);common=int(selected['repair_pair_common'][pi])
                    pair_rows.append(dict(family=family,generation_seed=seed,index=index,pair=pi,
                        lower_margin_position=int(left),higher_margin_position=int(right),
                        rare_position=rare,common_position=common,
                        rare_margin=float(margin[rare]),common_margin=float(margin[common]),
                        absolute_margin_gap=float(abs(margin[rare]-margin[common])),
                        rare_count=int(counts[native[rare]]),common_count=int(counts[native[common]]),
                        count_tie=int(counts[native[rare]]==counts[native[common]]),
                        rare_relative_position=float((rare-start)/max(end-start-1,1)),
                        common_relative_position=float((common-start)/max(end-start-1,1)),
                        rare_native_source_overlap=int(native[rare] in source),
                        common_native_source_overlap=int(native[common] in source),
                        rare_affine_source_overlap=int(linear[rare] in source),
                        common_affine_source_overlap=int(linear[common] in source)))
                target=dest/f'{index:04}.npz'
                np.savez_compressed(target,**evidence)
                fields['evidence_sha256']=sha(target)
                dump(target.with_suffix('.json'),fields)
                for path in [target,target.with_suffix('.json')]:
                    output_manifest[str(path.relative_to(OUT))]=sha(path)
            input_path=inp/f'{family}_{seed}.npz'
            np.savez_compressed(input_path,**inputs)
            input_manifest[str(input_path.relative_to(OUT))]=sha(input_path)
            print('Constructed',family,seed,'1024 outputs; no scoring performed',flush=True)
    for path in [inp/'cohort.jsonl']+[inp/f'counts_{family}.npy' for family in FAMILIES]:
        input_manifest[str(path.relative_to(OUT))]=sha(path)
    csvwrite(OUT/'per_sequence.csv',sequence_rows)
    csvwrite(OUT/'pairs.csv',pair_rows)
    groups={}
    for family in ['all']+FAMILIES:
        rows=[r for r in sequence_rows if family=='all' or r['family']==family]
        pairs=[r for r in pair_rows if family=='all' or r['family']==family]
        gaps=np.array([r['absolute_margin_gap'] for r in pairs])
        base=[r for r in rows if r['arm']==ARMS[0]]
        q=dict(outputs=len(base),zero_budget=sum(r['budget']==0 for r in base),
               total_errors=sum(r['errors'] for r in base),total_corrections_per_arm=sum(r['budget'] for r in base),
               pairs=len(pairs),count_tie_fraction=float(np.mean([r['count_tie'] for r in pairs])),
               absolute_margin_gap=dict(zip(['mean','median','q90','q95','max'],[float(gaps.mean()),*np.quantile(gaps,[.5,.9,.95,1.]).tolist()])),
               rare_minus_common={key:float(np.mean([r['rare_'+key]-r['common_'+key] for r in pairs]))
                                  for key in ['margin','count','relative_position','native_source_overlap','affine_source_overlap']})
        q['rare_minus_common']['log1p_count']=float(np.mean([np.log1p(r['rare_count'])-np.log1p(r['common_count']) for r in pairs]))
        q['arms']={}
        for arm in ARMS:
            rr=[r for r in rows if r['arm']==arm]
            stats={}
            for prefix,weight in [('selected','budget'),('remaining','remaining_errors')]:
                for key in ['margin','native_count','log1p_native_count','relative_position','native_source_overlap','affine_source_overlap']:
                    good=[r for r in rr if r[weight]>0]
                    stats[prefix+'_'+key+'_token_weighted']=float(sum(r[prefix+'_'+key]*r[weight] for r in good)/sum(r[weight] for r in good))
                    stats[prefix+'_'+key+'_sequence_mean']=float(np.mean([r[prefix+'_'+key] for r in good]))
            q['arms'][arm]=stats
        groups[family]=q
    dump(OUT/'construction_summary.json',dict(status='All token policies built; no new scores inspected',groups=groups))
    dump(OUT/'inputs_manifest.json',dict(protocol_sha256=sha(OUT/'protocol.json'),inputs=input_manifest,outputs=output_manifest))
    dump(OUT/'construction_ready.json',dict(status='construction_complete',outputs=4096,arms=ARMS,
         scores_complete=False,portable_verification_required=True,protocol_sha256=sha(OUT/'protocol.json')))
    print(json.dumps(groups,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','build']);args=parser.parse_args()
    freeze() if args.mode=='freeze' else build()
