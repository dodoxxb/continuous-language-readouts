"""Select paired XSum inputs without generation or outcome inspection."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/task_readout_20261022'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if (OUT/'cohort.jsonl').exists():raise RuntimeError('Cohort already selected; refusing overwrite')
    tok=AutoTokenizer.from_pretrained(ROOT/'models/gpt2',local_files_only=True)
    elf=AutoTokenizer.from_pretrained(ROOT/'models/t5-small',local_files_only=True)
    path=ROOT/'third_party/AURORA-LM/data/prepare_xsum_conditional.py'
    spec=importlib.util.spec_from_file_location('aurora_xsum_prepare',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    endings=mod.build_sentence_end_set(tok)
    raw={s:pq.read_table(ROOT/f'data/xsum/data/{s}-00000-of-00001.parquet').to_pylist() for s in ['train','validation','test']}
    trainids={r['id'] for r in raw['train']}
    trainhash={hashlib.sha256(r['document'].encode()).hexdigest() for r in raw['train']}
    counts={};selected={}
    for split,n,seed in [('validation',8,61022601),('test',1024,61022602)]:
        ids=np.random.default_rng(seed).permutation(len(raw[split]));rows=[];reject={}
        for i in ids:
            row=raw[split][int(i)];doc=row['document'];ref=row['summary']
            reason=None
            if row['id'] in trainids or hashlib.sha256(doc.encode()).hexdigest() in trainhash:reason='exact_train_overlap'
            src=tok(doc,add_special_tokens=False)['input_ids'];tgt=tok(ref,add_special_tokens=False)['input_ids']
            packed=mod.process_token_ids(src,tgt,block_size=1024,max_condition_len=960,min_target_len=16,eos_id=tok.eos_token_id,sentence_end_set=endings)
            if not doc.strip() or not ref.strip() or packed is None:reason='official_aurora_input_ineligible'
            if reason:reject[reason]=reject.get(reason,0)+1;continue
            prefix=packed['input_ids'][:packed['condition_len']]
            eprefix=elf(doc,add_special_tokens=False)['input_ids'][:1024]
            rows.append(dict(index=len(rows),raw_index=int(i),id=row['id'],document=doc,reference=ref,
                aurora_prefix_ids=prefix,elf_prefix_ids=eprefix,
                document_sha256=hashlib.sha256(doc.encode()).hexdigest(),reference_sha256=hashlib.sha256(ref.encode()).hexdigest()))
            if len(rows)==n:break
        assert len(rows)==n
        selected[split]=rows;counts[split]=dict(total_rows=len(raw[split]),selected=n,rejections_before_filling=reject,selection_seed=seed)
    assert not {r['id'] for r in selected['validation']}&{r['id'] for r in selected['test']}
    for name,split in [('pilot_cohort','validation'),('cohort','test')]:
        (OUT/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected[split]))
    manifest=dict(dataset='EdinburghNLP/xsum, local parquet snapshot',selection=counts,
        paired_across_architectures_and_generation_seeds=True,
        all_references_excluded_from_generation=True,
        dataset_sha256={s:sha(ROOT/f'data/xsum/data/{s}-00000-of-00001.parquet') for s in raw},
        cohort_sha256=sha(OUT/'cohort.jsonl'),pilot_sha256=sha(OUT/'pilot_cohort.jsonl'),
        upstream_preprocessing_sha256=sha(path),test_ids=[r['id'] for r in selected['test']])
    (OUT/'source/cohort_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
