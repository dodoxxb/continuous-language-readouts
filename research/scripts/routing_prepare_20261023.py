"""Freeze an independently generated, native-free budget-routing audit."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,time
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from routing_features_20261023 import ROOT,OUT,OLD,FEATURE_NAMES,assets,sha,dump

def plan():
 p=OUT/'protocol/protocol.json'
 if p.exists():return
 assert not (OUT/'generated').exists()
 obj=dict(date='2026-10-23',status='new prospective test after previous exploratory results',
  question='Can affordable affine-only endpoint features select whole outputs for native decoding better than random under exactly the same batch budget?',
  scope='Counterfactual budget-routing audit, not measured end-to-end speedup or human/factual quality. Existing lexical heads omit EOS training; autonomous outputs test these existing heads unmodified.',
  families=['elf','aurora'],fit_seed=17,secondary_ensemble_fit_seeds=[17,29,43],
  development=dict(source='outputs/task_readout_20261022',documents=1024,generation_seeds=[61022801,61024801],
   status='Previously examined XSum outputs now explicitly reused for task-specific risk calibration; no claim that the gate is zero-refit.',
   target='100*(native BERTScore F1 - autonomous17 BERTScore F1), each endpoint independently stopped at its own EOS'),
  test=dict(documents=256,generation_seeds=[61032301,61034301],effective_seed='generation base + new cohort index',
   cohort_seed=61032300,dataset='local XSum test parquet',
   exclusions='IDs or exact document hashes in original 1024-document test, eight-document validation pilot, or XSum train; also duplicate IDs/hashes within new cohort',
   eligibility='Same official AURORA packing/input eligibility and source truncation as original XSum study; references never supplied to generation',
   independence='New document IDs/hashes and new generation seeds, with frozen models and protocol before generation'),
  features=dict(names=FEATURE_NAMES,scope='Entire generated target canvas, including positions after affine EOS; no native-valid mask or native stopping boundary',
   formulas=dict(mean_log1p_gap='mean log1p(top1-top2 affine17 logit gap)',q10_log1p_gap='linear-interpolated 10th percentile of log1p gaps',
    mean_entropy='mean categorical affine17 entropy in natural-log nats',mean_log1p_predicted_count='mean log1p(original OWT training count of affine17 argmax)',
    predicted_numeric_fraction='fraction of affine17 argmax tokens with original digit-containing-token indicator',
    own_eos_present='any affine17 argmax equals interface EOS',own_stop_fraction='affine17 own stopping index divided by known target canvas length; original AURORA leading-EOS convention',
    canvas_length='known generated target canvas length'),
   excluded_inputs=['native labels','native margins','native error positions/rates','native masks/boundaries/lengths','native or affine external-LM scores','references'],
   affine_dtype='FP32, TF32 disabled',stop_semantics='Same autonomous17 convention as original task study; each output uses its own EOS, and falls back to canvas end if absent'),
  models=dict(primary='Separate per-family ridge, alpha=10, unpenalized intercept, development-only mean/std; zero SD replaced by one; all eight features',
   stopping_only='Same fixed ridge using only own_eos_present, own_stop_fraction, canvas_length; descriptive ablation',
   calibration_cv='Five document-grouped folds, seed61032319, descriptive only; no model or hyperparameter selection',
   no_hyperparameter_search=True),
  policies=dict(primary='Rank predicted native-minus-autonomous BERTScore gain; route largest 128 of 256 sequences separately within each family/generation seed',
   tie_rule='Ascending SHA256 of family|generation_seed|document_id, then cohort index',
   baselines=['threefit_disagreement: mean pairwise argmax disagreement across17/29/43 on full target canvas',
    'negative_mean_log1p_gap: smaller affine17 margin first','stopping_only ridge with same alpha and 50% budget'],
   comparisons='Primary full ridge versus exact same-budget random expectation. All other policies and full-ridge versus stopping-only are retained descriptively, without winner selection.',
   random='Expected score of uniformly selecting exactly128 of256 sequences; per-sequence marginal probability0.5, analytically averaged, no random-policy outcome selection',
   no_online_threshold_claim=True),
  endpoints=dict(primary='Pooled BERTScore routing gain versus same-budget random expectation in points on 100x scale',
   secondary=['per-family BERTScore effects','all-affine/all-native/routed scores','NLL benefit versus random, own stopping boundaries','output length, empty fraction, own-EOS/canvas-stop rates',
    'risk MAE and Spearman association on fresh outputs; never used for tuning'],
   nll='Require native and autonomous NLL scorable across all four family-seed replicas for a document; retain original full-cohort fixed routing decisions and comparator probability0.5 even if subset route fraction differs; disclose exact exclusions/coverage',
   metrics='Reuse fixed RoBERTa-large layer17 BERTScore and GPT2-Large NLL scorer settings of original task study; fresh score namespace, source/reference/output hashes checked'),
  inference=dict(primary_confidence=.95,bootstrap_draws=4000,bootstrap_seed=61032329,
   unit='Document, paired across both families and both generation seeds; average two seeds and equal-family means within document',
   conditioning='Calibration weights and selected routing decisions held fixed across resamples. Intervals do not cover risk fitting uncertainty or fresh batch re-ranking uncertainty.',
   multiplicity='One pooled primary contrast only; all other intervals descriptive95%',
   retain='Every new document, policy, sign, null/adverse result; do not adapt sample size after outcomes'),
  resources='At most two bounded GPU workers, low-priority CUDA stream, 5.5GiB allocator cap plus2GiB free headroom, two CPU threads; no downloads or retraining of generators')
 dump(p,obj);print('protocol',sha(p),flush=True)

def cohort():
 import numpy as np
 import pyarrow.parquet as pq
 from transformers import AutoTokenizer
 plan();dest=OUT/'cohort/cohort.jsonl'
 if dest.exists():raise RuntimeError('Refusing to replace selected cohort')
 oldrows=[json.loads(line) for p in [OLD/'cohort.jsonl',OLD/'pilot_cohort.jsonl'] for line in p.read_text().splitlines()]
 blocked_ids={r['id'] for r in oldrows};blocked_hash={r['document_sha256'] for r in oldrows}
 trainpath=ROOT/'data/xsum/data/train-00000-of-00001.parquet'
 for batch in pq.ParquetFile(trainpath).iter_batches(batch_size=4096,columns=['id','document']):
  for r in batch.to_pylist():blocked_ids.add(r['id']);blocked_hash.add(hashlib.sha256(r['document'].encode()).hexdigest())
 raw=pq.read_table(ROOT/'data/xsum/data/test-00000-of-00001.parquet').to_pylist()
 tok=AutoTokenizer.from_pretrained(ROOT/'models/gpt2',local_files_only=True)
 elf=AutoTokenizer.from_pretrained(ROOT/'models/t5-small',local_files_only=True)
 source=ROOT/'third_party/AURORA-LM/data/prepare_xsum_conditional.py'
 spec=importlib.util.spec_from_file_location('routing_xsum_prepare',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 endings=mod.build_sentence_end_set(tok);rows=[];rejections={};seen_ids=set();seen_hash=set()
 for i in np.random.default_rng(61032300).permutation(len(raw)):
  r=raw[int(i)];doc=r['document'];ref=r['summary'];dh=hashlib.sha256(doc.encode()).hexdigest()
  reason=None
  if r['id'] in blocked_ids or dh in blocked_hash:reason='old_or_train_overlap'
  elif r['id'] in seen_ids or dh in seen_hash:reason='duplicate_new_cohort'
  if reason:rejections[reason]=rejections.get(reason,0)+1;continue
  src=tok(doc,add_special_tokens=False)['input_ids'];tgt=tok(ref,add_special_tokens=False)['input_ids']
  packed=mod.process_token_ids(src,tgt,block_size=1024,max_condition_len=960,min_target_len=16,eos_id=tok.eos_token_id,sentence_end_set=endings)
  if not doc.strip() or not ref.strip() or packed is None:
   rejections['input_ineligible']=rejections.get('input_ineligible',0)+1;continue
  rows.append(dict(index=len(rows),raw_index=int(i),id=r['id'],document=doc,reference=ref,
   aurora_prefix_ids=packed['input_ids'][:packed['condition_len']],elf_prefix_ids=elf(doc,add_special_tokens=False)['input_ids'][:1024],
   document_sha256=dh,reference_sha256=hashlib.sha256(ref.encode()).hexdigest()))
  seen_ids.add(r['id']);seen_hash.add(dh)
  if len(rows)==256:break
 assert len(rows)==256 and not seen_ids.intersection(blocked_ids) and not seen_hash.intersection(blocked_hash)
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 identities(rows,oldrows)
 dump(OUT/'cohort/manifest.json',dict(status='selected_before_fresh_generation',documents=len(rows),selection_seed=61032300,
  exact_id_and_hash_exclusions_passed=True,rejections_before_filling=rejections,cohort_sha256=sha(dest),
  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [trainpath,ROOT/'data/xsum/data/test-00000-of-00001.parquet',OLD/'cohort.jsonl',OLD/'pilot_cohort.jsonl',source]},
  protocol_sha256=sha(OUT/'protocol/protocol.json')))
 print('cohort',sha(dest),flush=True)

def identities(rows=None,oldrows=None):
 if rows is None:rows=[json.loads(s) for s in (OUT/'cohort/cohort.jsonl').read_text().splitlines()]
 if oldrows is None:oldrows=[json.loads(s) for p in [OLD/'cohort.jsonl',OLD/'pilot_cohort.jsonl'] for s in p.read_text().splitlines()]
 dump(OUT/'cohort/identity.json',dict(new=[{k:r[k] for k in ['index','id','document_sha256','reference_sha256']} for r in rows],
  previously_observed=[{k:r[k] for k in ['id','document_sha256','reference_sha256']} for r in oldrows],
  train_overlap_note='Full train IDs/hashes checked during cohort construction against pinned local parquet; portable identities independently check old-test/pilot overlap, not the full omitted train corpus.'))

def freeze():
 if (OUT/'frozen.json').exists():raise RuntimeError('Refusing to replace freeze')
 assert not (OUT/'generated').exists(),'Fresh generation must follow freeze'
 weights=json.loads((OUT/'calibration/weights.json').read_text());assert set(weights['models'])=={'elf','aurora'}
 files=[OUT/'protocol/protocol.json',OUT/'protocol/runtime_preflight.json',OUT/'protocol/analysis_synthetic_check.json',OUT/'cohort/cohort.jsonl',OUT/'cohort/manifest.json',OUT/'cohort/identity.json',OUT/'calibration/weights.json',
  OUT/'calibration/fit_data.csv',OUT/'calibration/target_provenance.json']
 files+=list(OUT.glob('calibration/features_*.*'))
 files+=sorted(ROOT.glob('scripts/routing_*_20261023.py'))
 required=['routing_prepare_20261023.py','routing_features_20261023.py','routing_analyze_20261023.py','routing_runtime_20261023.py','routing_score_20261023.py','routing_verify_20261023.py']
 assert all((ROOT/'scripts'/p) in files for p in required),required
 files+=[ROOT/'scripts/task_readout_runtime_20261022.py',ROOT/'scripts/task_readout_score_20261022.py',ROOT/'scripts/task_readout_outputs_20261022.py']
 for f in ['elf','aurora']:
  mp,cp,npth,_=assets(f);files+=mp+[cp,npth]
 files+=[OLD/'frozen.json',OLD/'protocol.json',ROOT/'models/ELF-B-xsum-torch/config.yml',ROOT/'models/ELF-B-xsum-torch/checkpoint_39800',ROOT/'models/aurora-lm/xsum/300000.safetensors',ROOT/'models/aurora-lm/autoencoder/100000.safetensors']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
 dump(OUT/'frozen.json',dict(status='ready_for_fresh_generation',frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
  test_generated=False,files=hashes,scope='Observed development, prospectively frozen independent new-document test. Counterfactual routing only.'))
 print('FROZEN',sha(OUT/'frozen.json'),'protocol',sha(OUT/'protocol/protocol.json'),'weights',sha(OUT/'calibration/weights.json'),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','cohort','identities','freeze']);a=p.parse_args()
 {'plan':plan,'cohort':cohort,'identities':identities,'freeze':freeze}[a.mode]()

if __name__=='__main__':main()
