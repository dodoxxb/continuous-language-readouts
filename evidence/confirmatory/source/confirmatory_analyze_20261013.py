"""Prespecified native-agreement summaries, paired sequence intervals and controls."""
from __future__ import annotations
import argparse,csv,json,math,random,statistics,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from confirmatory_readout_20261013 import ROOT,OUT,P,dump,digest,evaluate

def csvwrite(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def interval(correct,denominator,other=None):
    # Each sequence is a cluster; average over the prespecified fit seeds.
    # This interval is conditional on those fits, not population training-seed uncertainty.
    c=np.mean(correct,axis=0);d=np.asarray(denominator,dtype=float)
    if other is not None:c=c-np.mean(other,axis=0)
    rng=np.random.default_rng(61013);samples=rng.integers(len(d),size=(4000,len(d)))
    b=c[samples].sum(1)/d[samples].sum(1)
    return float(c.sum()/d.sum()),*[float(z) for z in np.quantile(b,[.025,.975])]

@torch.inference_mode()
def lookup(device):
    from transformers import T5EncoderModel
    enc=T5EncoderModel.from_pretrained(ROOT/'models/t5-small',local_files_only=True)
    E=F.normalize(enc.get_input_embeddings().weight.detach().float(),dim=-1);del enc
    model=torch.nn.Linear(E.shape[1],E.shape[0],bias=False).to(device);model.weight.copy_(E.to(device));model.eval()
    out=OUT/'lookup/elf';out.mkdir(parents=True,exist_ok=True)
    for name in ['test','fresh_test']:
        if (out/(name+'.json')).exists():continue
        result=evaluate(model,OUT/'data/elf'/name,device,save=out/(name+'.npz'),eos_id=1,pad_id=0)
        dump(out/(name+'.json'),result)

def summarize():
    allrows=[];groups={};runrows=[];tailrows=[]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/t5-small',local_files_only=True)
    counts=np.bincount(np.load(OUT/'data/elf/train/y.npy',mmap_mode='r'),minlength=32128)
    numeric=np.array([any(ch.isdigit() for ch in tok.convert_ids_to_tokens(i)) for i in range(32100)]+[False]*28)
    for family,budgets,sets in [('elf',P['elf_train_budgets'],['test','fresh_test']),('langflow',[P['langflow_train_sequences']],['test'])]:
        for budget in budgets:
            for seed in P['fit_seeds']:
                dest=OUT/'fits'/family/f'n{budget}_seed{seed}';result=json.loads((dest/'selected.json').read_text())
                for name in sets:
                    pack=np.load(dest/(name+'.npz'));y=np.load(OUT/'data'/family/name/'y.npy');v=np.load(OUT/'data'/family/name/'valid.npy')
                    c=((pack['pred']==y)&v).sum(1);d=v.sum(1)
                    assert np.array_equal(c,pack['correct']) and np.array_equal(d,pack['denominator'])
                    assert abs(c.sum()/d.sum()-result[name]['agreement'])<1e-12
                    groups.setdefault((family,budget,name),[]).append((c,d,seed))
                    runrows.append({'family':family,'train_sequences':budget,'fit_seed':seed,'test_set':name,'selected_update':result['selected_update'],'validation_agreement':result['validation_agreement'],'agreement':result[name]['agreement'],'macro_agreement':result[name]['macro_agreement'],'autonomous_eos_agreement':result[name]['autonomous_eos_agreement'],'test_sequences':len(d),'valid_tokens':int(d.sum())})
                    for i,(cc,dd,aa) in enumerate(zip(c,d,pack['autonomous_correct'])):
                        allrows.append({'family':family,'train_sequences':budget,'fit_seed':seed,'test_set':name,'sequence_index':i,'correct':int(cc),'valid_tokens':int(dd),'autonomous_correct':int(aa)})
                    if family=='elf' and budget==max(P['elf_train_budgets']) and name=='fresh_test':
                        margin=np.load(OUT/'data/elf/fresh_test/native_margin.npy')
                        masks={'all':v,'numeric':v&numeric[y],'non_numeric':v&~numeric[y],'train_count_1_to_99':v&(counts[y]>0)&(counts[y]<100),'train_count_at_least_100':v&(counts[y]>=100),'unseen_training_token':v&(counts[y]==0),'margin_below_2':v&(margin<2),'margin_2_to_8':v&(margin>=2)&(margin<8),'margin_at_least_8':v&(margin>=8)}
                        for key,m in masks.items():
                            n=int(m.sum());correct=int(((pack['pred']==y)&m).sum())
                            tailrows.append({'fit_seed':seed,'group':key,'correct':correct,'tokens':n,'agreement':correct/n if n else None})
    summary=[]
    for (family,budget,name),items in groups.items():
        c=np.stack([i[0] for i in items]);d=items[0][1];m,lo,hi=interval(c,d)
        rates=c.sum(1)/d.sum()
        summary.append({'family':family,'train_sequences':budget,'test_set':name,'fit_seeds':len(items),'agreement_mean':m,'fit_seed_sd':float(np.std(rates,ddof=1)),'sequence_ci_low':lo,'sequence_ci_high':hi,'test_sequences':len(d),'valid_tokens':int(d.sum())})
    improvements=[]
    for name in ['test','fresh_test']:
        small=groups[('elf',min(P['elf_train_budgets']),name)];large=groups[('elf',max(P['elf_train_budgets']),name)]
        m,lo,hi=interval([x[0] for x in large],large[0][1],other=[x[0] for x in small])
        improvements.append({'comparison':'16384 minus 1024','test_set':name,'delta':m,'ci_low':lo,'ci_high':hi})
    lookuprows=[];controlseq=[]
    for name in ['test','fresh_test']:
        p=np.load(OUT/'lookup/elf'/(name+'.npz'));m,lo,hi=interval(p['correct'][None,:],p['denominator'])
        lookuprows.append({'family':'elf','test_set':name,'method':'frozen_cosine_lookup','agreement':m,'sequence_ci_low':lo,'sequence_ci_high':hi})
        for i,(c,d) in enumerate(zip(p['correct'],p['denominator'])):controlseq.append({'family':'elf','test_set':name,'step':-1,'method':'lookup','sequence_index':i,'correct':int(c),'valid_tokens':int(d)})
    lf=np.load(OUT/'data/langflow/test_phase_outputs.npz');phaserows=[]
    for step in [0,16,32,48,63]:
        y=lf[f'native_{step}'];den=np.full(len(y),y.shape[1])
        for name in ['lookup','residual']:
            correct=(lf[f'{name}_{step}']==y).sum(1);m,lo,hi=interval(correct[None,:],den)
            phaserows.append({'step':step,'method':name,'agreement':m,'sequence_ci_low':lo,'sequence_ci_high':hi,'test_sequences':len(y),'positions':int(den.sum())})
            for i,(c,d) in enumerate(zip(correct,den)):controlseq.append({'family':'langflow','test_set':'test','step':step,'method':name,'sequence_index':i,'correct':int(c),'valid_tokens':int(d)})
    for filename,rows in [('per_sequence.csv',allrows),('per_run.csv',runrows),('summary.csv',summary),('paired_improvements.csv',improvements),('tail_groups.csv',tailrows),('lookup.csv',lookuprows),('langflow_phase.csv',phaserows),('control_per_sequence.csv',controlseq)]:csvwrite(OUT/filename,rows)
    dump(OUT/'analysis.json',{'summary':summary,'paired_improvements':improvements,'lookup':lookuprows,'langflow_phase':phaserows,'ci_scope':'4000 paired sequence-cluster bootstrap draws, seed 61013, conditional on three fitted seeds; fit-seed SD separately reported','rare_definition':'largest-ELF-training-set token count from 1 to 99, fixed threshold; no test selection','numeric_definition':'tokenizer token string contains a Unicode digit','protocol_sha256':digest(OUT/'protocol.json')})
    print(json.dumps({'summary':summary,'paired_improvements':improvements,'langflow_phase':phaserows},indent=2))

@torch.inference_mode()
def ppl(device):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    folder=OUT/'ppl';folder.mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/t5-small',local_files_only=True)
    y=np.load(OUT/'data/elf/fresh_test/y.npy');v=np.load(OUT/'data/elf/fresh_test/valid.npy')
    preds={'native':y}
    for seed in P['fit_seeds']:preds[f'linear_seed{seed}']=np.load(OUT/'fits/elf'/f'n16384_seed{seed}'/'fresh_test.npz')['pred']
    preds['frozen_lookup']=np.load(OUT/'lookup/elf/fresh_test.npz')['pred']
    lm_tok=AutoTokenizer.from_pretrained(ROOT/'models/gpt2-large',local_files_only=True);lm_tok.pad_token=lm_tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models/gpt2-large',local_files_only=True,torch_dtype=torch.float16).to(device).eval()
    rows=[];perseq=[]
    for name,ids in preds.items():
        texts=tok.batch_decode(np.where(v,ids,0),skip_special_tokens=True)
        sums=[];counts=[];bad=[];started=time.time()
        for start in range(0,len(texts),2):
            enc=lm_tok(texts[start:start+2],padding=True,truncation=True,max_length=1024,return_tensors='pt').to(device)
            output=model(**enc)
            logits=output.logits[:,:-1].float();target=enc.input_ids[:,1:];mask=enc.attention_mask[:,1:]
            loss=F.cross_entropy(logits.transpose(1,2),target,reduction='none')
            ss=(loss*mask).sum(1).cpu().numpy();cc=mask.sum(1).cpu().numpy()
            sums.extend(ss.tolist());counts.extend(cc.tolist())
            del output,logits,loss
        for i,(ss,cc) in enumerate(zip(sums,counts)):
            if cc<1:bad.append(i)
            perseq.append({'method':name,'sequence_index':i,'nll_sum':ss,'scored_tokens':cc})
        rows.append({'method':name,'ppl':math.exp(sum(sums)/sum(counts)),'sequences':len(texts),'invalid_sequences':len(bad),'scored_tokens':sum(counts),'seconds':time.time()-started})
        print('PPL',rows[-1],flush=True)
        csvwrite(folder/'per_sequence.csv',perseq);csvwrite(folder/'summary.csv',rows)
    dump(folder/'protocol.json',{'evaluator':'local GPT-2-Large','max_length':1024,'batch_size':2,'dtype':'float16 weights; float32 NLL','formula':'exp(sum token NLL / sum scored tokens)','boundary':'identical native validity mask applied before T5 text decode for every readout; no independent predicted EOS termination','scope':'conditional readout fidelity diagnostic, not autonomous generation quality','fit_selection':'all three largest-budget models selected by validation; no test-best selection','invalid_sequences_reported':True})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['lookup','summarize','ppl']);ap.add_argument('--device',default='cuda:0');a=ap.parse_args();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    if a.mode=='summarize':summarize()
    else:
        torch.cuda.set_per_process_memory_fraction(.18,device=a.device)
        stream=torch.cuda.Stream(device=a.device,priority=1)
        with torch.cuda.stream(stream):(lookup if a.mode=='lookup' else ppl)(a.device)

if __name__=='__main__':main()
