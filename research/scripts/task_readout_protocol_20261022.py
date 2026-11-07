"""Prospective protocol and mechanical freeze for paired XSum transfer."""
from pathlib import Path
import argparse,datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/task_readout_20261022'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    protocol=dict(version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      hypothesis='Do frozen OWT readouts and equal-count native-oracle repair priorities transfer to a conditional task, measured against human-written references and source articles?',
      cohort=dict(dataset='EdinburghNLP/xsum',test_documents=1024,validation_pilot_documents=8,
          selection='Fixed shuffled eligible test documents, excluding exact training ID/document matches; reference only determines released AURORA length eligibility, never sampling.',
          cohort_manifest_sha256=sha(OUT/'source/cohort_manifest.json')),
      families=['elf','aurora'],generation_seeds=[61022801,61024801],effective_seed='base + document index; bases separated by 2000',
      readout_fits=[17,29,43],readout_training='Previously frozen OWT affine fits; no XSum fitting or selection.',
      samplers=dict(elf='Released ELF-B-XSum EMA1, official 64-step logit-normal ODE, CFG 2, self-conditioning scale 1, source T5 prefix up to 1024, target 64, full canvas 1088.',
          aurora='Released AURORA GPT2-XSum checkpoint 300000, shared autoencoder, official conditional public sampler, 32 ODE steps per 16-token block, CFG 2 in sigma window 0.9--1.0, repaint, nominal 64 new tokens (block rounding may yield 64--79), source prefix up to 960.'),
      interventions=dict(primary_fit=17,policies=['random11','random23','random37','rare','numeric','lowmargin'],
          budget='Exactly floor(e/2) native-token substitutions per generation, e = lexical disagreements before native first EOS; ELF native PAD excluded. Keep generations with zero edits in primary intent-to-treat analysis.',
          ranking='Ascending original OWT native-token count; numeric first using original Unicode-digit labels; ascending native logit margin; three uniform random orders. Policy-specific seeded random tie breaks.',
          rng='NumPy SeedSequence([generation_base, document_index, policy_index, family_index, 61022])',
          boundaries='All main arms stop at native first EOS. Interior predicted special tokens are skipped when decoding, not used to truncate. Autonomous linear17 at its own first EOS is a separate boundary control.',
          truth='Native labels are oracle targets for this intervention, not ground truth language or factuality.'),
      metrics=dict(rouge='rouge-score ROUGE1/2/L F1, stemming; reference summary; empty output scored zero.',
          bertscore='bert-score 0.3.13 distribution (module reports 0.3.12), roberta-large layer 17, no IDF, no rescaling, FP32, official slow tokenizer. F1 primary; P/R retained.',
          summac='summac 0.0.4 SummaC-ZS vitc/albert-xlarge-vitaminc-mnli; sentence granularity, first 100 source sentences, max input 500, FP32, default max over source then mean over summary of entailment minus contradiction. Empty/too-short chunks score zero as upstream.',
          nll='Frozen GPT2-Large evaluator; descriptive average per-summary token NLL, native-boundary ten arms share eligibility; autonomous arm separate. Empty/unscorable outputs retained in other metrics and counted.',
          limitations='Automatic semantic/source-consistency proxies, not human evaluation or verified factual accuracy; native source truncation differs by tokenizer, no matched-compute architecture comparison.'),
      primary_contrasts=['pooled repair_rare minus mean(random11,random23,random37) BERTScore F1',
                         'pooled repair_rare minus mean(random11,random23,random37) SummaC-ZS'],
      inference=dict(unit='Paired document: average two generation seeds within each family, then equal family weights for pooled estimates. Resample same documents across seeds/families.',
          bootstrap_draws=4000,bootstrap_seed=61022901,primary_ci=.975,descriptive_ci=.95,
          multiplicity='Two prespecified primary contrasts use Bonferroni 97.5% intervals. All other intervals descriptive.',
          secondary='Per-family estimates; low-margin/numeric policies; all three transferred readouts; native/affine/repair task scores; autonomous boundary; predeclared editable subset e>=2 per generation. No repeated-denoiser-training inference.'),
      reporting='Retain positive, null and adverse results, all 4096 generations; no policy/metric selection using test outcomes.',
      resources='Two workers maximum; two CPU threads each; 5.5 GiB allocator cap and 2 GiB free headroom, low-priority CUDA stream, nice 10; downloads at most two streams at 6 MiB/s each.')
    path=OUT/'protocol.json'
    if not path.exists():path.write_text(json.dumps(protocol,indent=2)+'\n')
    if not a.freeze:return
    assert not (OUT/'generated').exists(),'Freeze before any formal generation'
    assert not (OUT/'frozen.json').exists(),'Never replace a freeze'
    for family in ['elf','aurora']:
        for i in range(2):
            m=json.loads((OUT/f'pilot/{family}/61022701/{i:04}.json').read_text())
            assert m['public_decode_equal'] and m['reference_exclusion_verified']
    paths=list(ROOT.glob('scripts/task_readout*20261022.py'))+[path,OUT/'cohort.jsonl',OUT/'pilot_cohort.jsonl',OUT/'source/cohort_manifest.json']
    for name in ['ELF-B-xsum-torch','aurora-gpt2-xsum']:
        manifest=OUT/'source'/f'{name}_download_verified.json';paths.append(manifest)
        paths.extend(ROOT/r['path'] for r in json.loads(manifest.read_text()))
    for f in [17,29,43]:
        paths.extend([ROOT/f'outputs/confirmatory_readout_20261013/fits/elf/n16384_seed{f}/selected.pt',ROOT/f'outputs/aurora_replication_20261022/fits/seed{f}/selected.pt'])
    paths.extend(ROOT/p for p in ['outputs/prospective_tail_20261013/token_counts.npy','outputs/prospective_tail_20261013/numeric.npy',
      'outputs/aurora_replication_20261022/counts.npy','outputs/aurora_replication_20261022/numeric.npy','models/aurora-lm/autoencoder/100000.safetensors'])
    paths.extend((OUT/'source').glob('*_metadata.json'))
    paths.append(OUT/'source/nltk_punkt.json')
    paths.append(ROOT/'outputs/phase0_phase1_20260520/scripts/phase2_trajectory_audit.py')
    for folder in ['models/t5-small','models/gpt2','models/gpt2-large']:
        paths.extend(p for p in (ROOT/folder).glob('*') if p.is_file() and p.suffix in ['.json','.txt','.model','.safetensors','.bin'])
    for folder in ['official-ELF-pytorch_elf/src/configs','third_party/AURORA-LM/conf']:
        paths.extend((ROOT/folder).rglob('*.yaml'));paths.extend((ROOT/folder).rglob('*.yml'))
    paths.extend((OUT/'nltk_data').rglob('*.txt'))
    paths.extend((OUT/'nltk_data').rglob('*.tab'))
    for folder in ['official-ELF-pytorch_elf/src','third_party/AURORA-LM/src','outputs/task_readout_20261022/deps/bert_score','outputs/task_readout_20261022/deps/summac']:
        paths.extend((ROOT/folder).rglob('*.py'))
    frozen=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),formal_generations_existing=0,
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths)) if '__pycache__' not in str(p)})
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n');print('FROZEN',len(frozen['files']))

if __name__=='__main__':main()
