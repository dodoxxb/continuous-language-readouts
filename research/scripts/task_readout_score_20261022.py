"""Fixed automatic task metrics, with cache keyed by document and output text."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/task_readout_20261022'
sys.path.insert(0,str(OUT/'deps'))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ['NLTK_DATA']=str(OUT/'nltk_data')
import numpy as np
import torch
from task_readout_runtime_20261022 import resources,dump,sha

def key(index,text):return f'{index:04}_'+hashlib.sha256(text.encode()).hexdigest()

def load_metric(name,device):
    if name=='rouge':
        from rouge_score.rouge_scorer import RougeScorer
        return RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
    if name=='bertscore':
        from bert_score import BERTScorer
        return BERTScorer(model_type=str(ROOT/'models/roberta-large'),num_layers=17,device=device,
                         batch_size=8,idf=False,rescale_with_baseline=False,use_fast_tokenizer=False)
    if name=='summac':
        from summac.model_summac import SummaCZS,model_map
        model_map['vitc']['model_card']=str(ROOT/'models/albert-xlarge-vitaminc-mnli')
        metric=SummaCZS(model_name='vitc',granularity='sentence',op1='max',op2='mean',
             use_ent=True,use_con=True,imager_load_cache=False,max_doc_sents=100,device=device)
        metric.imager.load_nli()
        original=metric.imager.tokenizer.batch_encode_plus
        def compatible_batch_encode(*args,**kwargs):
            legacy=kwargs.pop('truncation_strategy',None)
            if legacy is not None:kwargs['truncation']=legacy
            return original(*args,**kwargs)
        metric.imager.tokenizer.batch_encode_plus=compatible_batch_encode
        return metric
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(ROOT/'models/gpt2-large',local_files_only=True)
    lm=AutoModelForCausalLM.from_pretrained(ROOT/'models/gpt2-large',local_files_only=True,
       torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    return tok,lm

@torch.inference_mode()
def calculate(name,metric,source,reference,text,device):
    if name=='rouge':return {k:dict(precision=v.precision,recall=v.recall,f1=v.fmeasure) for k,v in metric.score(reference,text).items()},None
    if name=='bertscore':
        p,r,f=metric.score([text],[reference],batch_size=1)
        lengths=[len(metric._tokenizer.encode(t,add_special_tokens=True,truncation=False)) for t in [text,reference]]
        return dict(precision=float(p[0]),recall=float(r[0]),f1=float(f[0]),
                    output_tokens=lengths[0],reference_tokens=lengths[1],truncated=max(lengths)>512),None
    if name=='summac':
        result=metric.score([source],[text],batch_size=16);im=result['images'][0]
        assert np.isfinite(im).all()
        rawsource=metric.imager.split_sentences(source);rawtarget=metric.imager.split_sentences(text)
        return dict(score=float(result['scores'][0]),source_sentences=len(rawsource),summary_sentences=len(rawtarget),
                    source_truncated=len(rawsource)>100,empty_chunks=len(rawsource)==0 or len(rawtarget)==0),im
    tok,lm=metric
    ids=[tok.eos_token_id]+tok.encode(text,add_special_tokens=False)
    if not 2<=len(ids)<=1024:return dict(eligible=False,tokens=0,nll_sum=0.,encoded_length=len(ids)),None
    x=torch.tensor([ids],device=device);logits=lm(input_ids=x,use_cache=False).logits[:,:-1].float()
    losses=torch.nn.functional.cross_entropy(logits.transpose(1,2),x[:,1:],reduction='none')[0].cpu().numpy()
    return dict(eligible=True,tokens=len(ids)-1,nll_sum=float(losses.sum(dtype=np.float64)),encoded_length=len(ids)),losses

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--metric',choices=['rouge','bertscore','summac','nll'],required=True)
    p.add_argument('--family',choices=['elf','aurora'],required=True);p.add_argument('--device',default='cpu');p.add_argument('--pilot',action='store_true')
    p.add_argument('--seed',type=int);p.add_argument('--sanity',action='store_true');a=p.parse_args()
    if a.device.startswith('cuda'):resources(a.device)
    else:torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if a.metric in ['bertscore','summac']:
        asset='roberta-large' if a.metric=='bertscore' else 'albert-xlarge-vitaminc-mnli'
        for r in json.loads((OUT/'source'/f'{asset}_download_verified.json').read_text()):assert sha(ROOT/r['path'])==r['sha256']
    metric=load_metric(a.metric,a.device)
    if a.sanity:
        s='Mary won the race in London on Monday.';examples=[s,'Mary lost the race in London on Monday.','']
        scores=[calculate(a.metric,metric,s,s,t,a.device)[0] for t in examples]
        dump(OUT/f'source/sanity_{a.metric}.json',dict(source=s,examples=examples,scores=scores));print(scores,flush=True);return
    rows={r['index']:r for r in map(json.loads,(OUT/('pilot_cohort.jsonl' if a.pilot else 'cohort.jsonl')).read_text().splitlines())}
    base=OUT/('pilot_outputs' if a.pilot else 'texts')/a.family
    paths=sorted(base.glob(f'{a.seed if a.seed else "*"}/[0-9][0-9][0-9][0-9].json'))
    dest=OUT/('pilot_scores' if a.pilot else 'scores')/a.metric/a.family;dest.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();n=0
    stream=torch.cuda.Stream(device=a.device,priority=1) if a.device.startswith('cuda') else None
    from contextlib import nullcontext
    for path in paths:
        q=json.loads(path.read_text());row=rows[q['index']]
        for arm,text in q['texts'].items():
            k=key(q['index'],text);target=dest/(k+'.json')
            if target.exists():continue
            with torch.cuda.stream(stream) if stream else nullcontext():
                value,trace=calculate(a.metric,metric,row['document'],row['reference'],text,a.device)
            if trace is not None:
                np.save(target.with_suffix('.npy'),trace)
                value['trace_sha256']=sha(target.with_suffix('.npy'))
            dump(target,dict(metric=a.metric,index=q['index'],output_sha256=q['text_sha256'][arm],
                source_sha256=row['document_sha256'],reference_sha256=row['reference_sha256'],value=value));n+=1
            if a.metric=='summac':metric.imager.cache.clear()
        if q['index']%16==15:print(a.metric,a.family,q['generation_seed'],q['index'],'unique',n,'sec',round(time.monotonic()-started,1),flush=True)
    print('complete',a.metric,a.family,'unique',n,'sec',round(time.monotonic()-started,1),flush=True)

if __name__=='__main__':main()
