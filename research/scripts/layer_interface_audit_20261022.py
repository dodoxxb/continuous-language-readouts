"""Post-hoc mechanistic controls for the Figure 21 layer discontinuities.

The protocol is frozen after inspecting the historical figure, not a new
confirmatory test. No weights are trained. Raw hidden states, terminal
normalization, and scalar/RMS controls are compared on identical texts.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, csv, hashlib, json, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/layer_interface_audit_20261022'
OLD = ROOT / 'outputs/phase0_phase1_20260520'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def writecsv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def freeze():
    if (OUT / 'protocol.json').exists(): raise RuntimeError('Protocol already frozen')
    import pandas as pd
    texts = []
    for f in sorted((ROOT / 'datasets/openwebtext/plain_text').glob('*.parquet')):
        for text in pd.read_parquet(f, columns=['text']).text:
            if isinstance(text, str) and len(text.split()) >= 20: texts.append(text)
            if len(texts) >= 640: break
        if len(texts) >= 640: break
    assert len(texts) == 640
    selected = [dict(split=split, index=i, text=texts[i], text_sha256=hashlib.sha256(texts[i].encode()).hexdigest())
                for split, indices in [('historical_overlap', range(128)), ('new_text_subset', range(512, 640))]
                for i in indices]
    dump(OUT / 'selected_texts.json', selected)
    files = [Path(__file__), ROOT / 'models/t5-small/model.safetensors', ROOT / 'models/t5-small/config.json',
             ROOT / 'models/gpt2/pytorch_model.bin', ROOT / 'models/gpt2/config.json',
             ROOT / 'models/ELF-B-owt-torch/checkpoint_95085', ROOT / 'models/ELF-B-owt-torch/config.yml',
             OLD / 'scripts/phase26_layer_decoder_margin_baseline.py',
             OLD / 'scripts/phase26_make_layer_margin_figures.py', OUT / 'selected_texts.json']
    protocol = dict(date='2026-10-22', purpose='Post-hoc exploration of historical Figure 21 anomalies; no generator training or quality claim.',
        samples_per_split=128, splits=['historical_overlap', 'new_text_subset'],
        text_selection='Same >=20 whitespace-word filter and sorted parquet order as the historical audit. Indices 0:128 overlap historical data; indices 512:640 are disjoint from the historical native-head 512-text subset.',
        t5='T5-small float32 encoder, full 1024-token ELF canvas, native ELF-B EMA1 decoder, CFG=3. Hidden indices 0:5 are pre-terminal normalization; pre6 is captured before final RMSNorm; 6 is the native normalized output. Check hidden_states[6] == last_hidden_state.',
        t5_controls='Raw at 0:6 plus pre6; final RMSNorm, global scalar RMS matching, and per-token RMS-only matching at layer 5 and pre6. Matching scale is RMS of the final norm weights, no labels or final-state oracle. Each state then receives ELF latent mean/std normalization.',
        t5_noise='Extra t=0.7 paired Gaussian controls at layer5 and pre6, all four variants, plus native layer6. Noise is scaled by each transformed full-canvas state standard deviation, matching the historical corruption definition. Same epsilon for all variants of a sequence. Clean t=1 for every variant.',
        gpt2='Float32 GPT-2, max_length=128, shifted next-token targets at all layers 0:12; capture pre12 before ln_f. Apply final ln_f and scalar/RMS controls to every pre-terminal state. Native layer12 is already normalized. Include previously unsampled layer11.',
        gpt2_scalar='Global positive rescaling to RMS(final ln_f weights) is an exact argmax-preserving control for the bias-free linear head; verify observed argmax equality and margin rescaling.',
        metrics='Per-sequence recovery, signed target-margin p10/mean, target NLL, input RMS, and margin/RMS. Sequence-average summaries and paired sequence-bootstrap descriptive intervals, not the historical batch-pooled p10.',
        uncertainty='4000 sequence bootstrap resamples within each split, seed 6102201; conditional descriptive intervals. No population claim over independently trained models.',
        precision_check='On the first eight historical texts only, repeat T5 raw layer5 and native layer6 with a float16 encoder to check whether the cliff is only a precision artifact.',
        implementation='Batch size one; frozen networks, inference mode. Full token outputs are reduced into per-sequence records, with invariants recorded separately.',
        files={str(p.relative_to(ROOT)): sha(p) for p in files})
    dump(OUT / 'protocol.json', protocol); print('Protocol frozen', sha(OUT / 'protocol.json'), flush=True)

def setup(device):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(6102201)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.25, device)
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert protocol['files'][str(Path(__file__).relative_to(ROOT))] == sha(__file__)
    selected = json.loads((OUT / 'selected_texts.json').read_text())
    assert sha(OUT / 'selected_texts.json') == protocol['files'][str((OUT / 'selected_texts.json').relative_to(ROOT))]
    return torch, selected

def stats(logits, targets, valid, hidden):
    import torch
    logits = logits.float()[valid]; targets = targets[valid]
    h = hidden.float()[valid]
    assert len(targets) > 0 and torch.isfinite(logits).all()
    top = logits.topk(2, -1)
    target = logits.gather(1, targets[:, None]).squeeze(1)
    rival = torch.where(top.indices[:, 0] == targets, top.values[:, 1], top.values[:, 0])
    margin = target - rival
    rms = h.square().mean(-1).sqrt().clamp_min(1e-12)
    return dict(tokens=len(targets), recovery=float((top.indices[:, 0] == targets).float().mean()),
                margin_p10=float(torch.quantile(margin, .1)), margin_mean=float(margin.mean()),
                nll=float((torch.logsumexp(logits, -1) - target).mean()),
                hidden_rms=float(h.square().mean().sqrt()),
                margin_over_rms_p10=float(torch.quantile(margin/rms, .1))), top.indices[:, 0], margin

def controls(h, norm):
    target_scale = norm.weight.float().square().mean().sqrt()
    global_rms = h.float().square().mean().sqrt().clamp_min(1e-12)
    token_rms = h.float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    return [('raw', h), ('final_norm', norm(h)),
            ('scalar_rms', h*(target_scale/global_rms)),
            ('token_rms', h*(target_scale/token_rms))]

def run_t5(device):
    torch, selected = setup(device)
    from transformers import T5EncoderModel, AutoTokenizer
    sys.path.insert(0, str(OLD / 'scripts'))
    from phase2_trajectory_audit import load_elf, decode_logits
    for p in ['models/t5-small/model.safetensors', 'models/ELF-B-owt-torch/checkpoint_95085']:
        assert sha(ROOT/p) == json.loads((OUT/'protocol.json').read_text())['files'][p]
    enc = T5EncoderModel.from_pretrained(ROOT/'models/t5-small', torch_dtype=torch.float32, local_files_only=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(ROOT/'models/t5-small', local_files_only=True)
    args = SimpleNamespace(root=ROOT, out=OUT/'t5', encoder_model=ROOT/'models/t5-small', tokenizer=ROOT/'models/t5-small',
        config=ROOT/'models/ELF-B-owt-torch/config.yml', checkpoint=ROOT/'models/ELF-B-owt-torch/checkpoint_95085',
        weights='ema_params1', num_samples=256, batch_size=1, ppl_model=ROOT/'models/gpt2-large')
    config, _, elf = load_elf(args, torch.device(device))
    norm = enc.encoder.final_layer_norm
    captured = {}
    hook = norm.register_forward_pre_hook(lambda module, inp: captured.update(pre=inp[0].detach().clone()))
    started = time.time()
    with torch.inference_mode():
        for j, sample in enumerate(selected):
            dest = OUT/'t5'/f"{sample['index']:04d}.json"
            if dest.exists(): continue
            inp = tok(sample['text'], padding='max_length', truncation=True, max_length=1024, return_tensors='pt').to(device)
            output = enc(**inp, output_hidden_states=True, return_dict=True)
            pre = captured['pre']; last = output.last_hidden_state
            alias_delta = float((output.hidden_states[-1]-last).abs().max())
            norm_delta = float((norm(pre)-last).abs().max())
            assert alias_delta == 0 and norm_delta == 0
            states = [(str(i), 'raw', h) for i, h in enumerate(output.hidden_states)] + [('pre6', 'raw', pre)]
            for layer, h in [('5', output.hidden_states[5]), ('pre6', pre)]:
                states.extend((layer, variant, value) for variant, value in controls(h,norm)[1:])
            rng = torch.Generator(device=device).manual_seed(6102201+sample['index'])
            epsilon = torch.randn(last.shape, generator=rng, device=device, dtype=last.dtype)
            rows=[]
            for layer, variant, h in states:
                z=(h-float(config.latent_mean))/float(config.latent_std)
                levels=[1., .7] if layer in ['5','pre6','6'] else [1.]
                for t in levels:
                    zz=z if t==1 else t*z+(1-t)*epsilon*z.float().std()
                    logits=decode_logits(elf,zz,1.,3.,config)
                    record,_,_=stats(logits,inp.input_ids,inp.attention_mask.bool(),zz)
                    rows.append(dict(model='t5', split=sample['split'], index=sample['index'], layer=layer, variant=variant, t=t, **record))
                    del logits
            dump(dest,dict(rows=rows, hidden_final_alias_max_abs=alias_delta, terminal_norm_reconstruction_max_abs=norm_delta, protocol_sha256=sha(OUT/'protocol.json')))
            if j%16==0: print('t5',j,'seconds',round(time.time()-started,1),flush=True)
        hook.remove()
        del elf
        torch.cuda.empty_cache()
        enc.half(); precision=[]
        # Compare encoder hidden states across precision, decode with the same ELF model.
        config, _, elf = load_elf(args,torch.device(device))
        for sample in selected[:8]:
            inp=tok(sample['text'],padding='max_length',truncation=True,max_length=1024,return_tensors='pt').to(device)
            o=enc(**inp,output_hidden_states=True,return_dict=True)
            for layer in [5,6]:
                z=(o.hidden_states[layer].float()-float(config.latent_mean))/float(config.latent_std)
                logits=decode_logits(elf,z,1.,3.,config)
                record,_,_=stats(logits,inp.input_ids,inp.attention_mask.bool(),z)
                precision.append(dict(model='t5_fp16_encoder',split=sample['split'],index=sample['index'],layer=str(layer),variant='raw',t=1.,**record))
        dump(OUT/'t5_precision.json',precision)
    print('t5 completed',round(time.time()-started,1),flush=True)

def run_gpt2(device):
    torch, selected=setup(device)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    p=ROOT/'models/gpt2/pytorch_model.bin'
    assert sha(p)==json.loads((OUT/'protocol.json').read_text())['files'][str(p.relative_to(ROOT))]
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models/gpt2',torch_dtype=torch.float32,use_safetensors=False,local_files_only=True).to(device).eval()
    tok=AutoTokenizer.from_pretrained(ROOT/'models/gpt2',local_files_only=True);tok.pad_token=tok.eos_token
    norm=model.transformer.ln_f;captured={}
    hook=norm.register_forward_pre_hook(lambda module,inp:captured.update(pre=inp[0].detach().clone()))
    assert model.lm_head.bias is None
    started=time.time()
    with torch.inference_mode():
        for j,sample in enumerate(selected):
            dest=OUT/'gpt2'/f"{sample['index']:04d}.json"
            if dest.exists():continue
            inp=tok(sample['text'],truncation=True,max_length=128,return_tensors='pt').to(device)
            out=model.transformer(**inp,output_hidden_states=True,return_dict=True)
            pre=captured['pre'];last=out.last_hidden_state
            assert torch.equal(out.hidden_states[-1],last)
            assert torch.equal(norm(pre),last)
            targets=inp.input_ids[:,1:];valid=inp.attention_mask[:,1:].bool()
            states=[(str(i),h[:,:-1]) for i,h in enumerate(out.hidden_states)]+[('pre12',pre[:,:-1])]
            rows=[];invariants=[]
            for layer,h in states:
                variants=[('raw',h)] if layer=='12' else controls(h,norm)
                base_pred=base_margin=None
                for variant,hh in variants:
                    logits=model.lm_head(hh)
                    record,pred,margin=stats(logits,targets,valid,hh)
                    if variant=='raw':base_pred=pred;base_margin=margin
                    if variant=='scalar_rms':
                        scale=float(norm.weight.square().mean().sqrt()/h.square().mean().sqrt())
                        differences=int((pred!=base_pred).sum())
                        err=float((margin-base_margin*scale).abs().max())
                        assert differences==0 and err<1e-3,(differences,err)
                        invariants.append(dict(layer=layer,argmax_changes=differences,margin_scaling_max_abs_error=err,scale=scale))
                    rows.append(dict(model='gpt2',split=sample['split'],index=sample['index'],layer=layer,variant=variant,t=1.,**record))
                    del logits
            dump(dest,dict(rows=rows,scalar_invariants=invariants,hidden_final_alias_max_abs=0.,terminal_norm_reconstruction_max_abs=0.,protocol_sha256=sha(OUT/'protocol.json')))
            if j%32==0:print('gpt2',j,'seconds',round(time.time()-started,1),flush=True)
    hook.remove();print('gpt2 completed',round(time.time()-started,1),flush=True)

def analyze():
    import pandas as pd
    rows=[]; invariants=[]
    for model in ['t5','gpt2']:
        files=sorted((OUT/model).glob('*.json'));assert len(files)==256,(model,len(files))
        for file in files:
            d=json.loads(file.read_text());assert d['protocol_sha256']==sha(OUT/'protocol.json')
            assert d['hidden_final_alias_max_abs']==d['terminal_norm_reconstruction_max_abs']==0
            rows+=d['rows'];invariants+=d.get('scalar_invariants',[])
    assert all(r['argmax_changes']==0 for r in invariants)
    writecsv(OUT/'records.csv',rows)
    df=pd.DataFrame(rows)
    keys=['model','split','layer','variant','t']
    summary=df.groupby(keys,as_index=False).agg(sequences=('index','size'),recovery=('recovery','mean'),margin_p10=('margin_p10','mean'),margin_mean=('margin_mean','mean'),nll=('nll','mean'),hidden_rms=('hidden_rms','mean'),margin_over_rms_p10=('margin_over_rms_p10','mean'))
    summary.to_csv(OUT/'summary.csv',index=False)
    contrasts=[]
    for model,layers in [('t5',['5','pre6']),('gpt2',['10','11','pre12'])]:
        for split in ['historical_overlap','new_text_subset']:
            for layer in layers:
                frame=df[(df.model==model)&(df.split==split)&(df.layer==layer)&(df.t==1)].set_index(['index','variant'])
                for variant in ['final_norm','scalar_rms','token_rms']:
                    for metric in ['recovery','margin_p10','nll']:
                        raw=frame.xs('raw',level='variant')[metric].sort_index()
                        alt=frame.xs(variant,level='variant')[metric].sort_index()
                        assert raw.index.equals(alt.index)
                        delta=(alt-raw).to_numpy();rng=np.random.default_rng(6102201)
                        boot=delta[rng.integers(len(delta),size=(4000,len(delta)))].mean(1)
                        lo,hi=np.quantile(boot,[.025,.975])
                        contrasts.append(dict(model=model,split=split,layer=layer,variant=variant,metric=metric,n=len(delta),delta=float(delta.mean()),ci_low=float(lo),ci_high=float(hi)))
    writecsv(OUT/'paired_contrasts.csv',contrasts)
    precision=json.loads((OUT/'t5_precision.json').read_text());writecsv(OUT/'precision.csv',precision)
    dump(OUT/'analysis.json',dict(status='passed',protocol_sha256=sha(OUT/'protocol.json'),sequences_per_model=512//2,
        model_text_pairs=512,records=len(rows),gpt2_scalar_checks=len(invariants),gpt2_scalar_argmax_changes=0,
        max_margin_scaling_error=max(x['margin_scaling_max_abs_error'] for x in invariants),
        terminal_norm_reconstruction_max_abs=0.,alias_equality_max_abs=0.,scope='Post-hoc interface/normalization interventions, not new generation or new training.',
        files={n:sha(OUT/n) for n in ['records.csv','summary.csv','paired_contrasts.csv','precision.csv']}))
    print(summary[(summary.t==1)&summary.layer.isin(['5','6','pre6','10','11','12','pre12'])].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','t5','gpt2','analyze']);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='t5':run_t5(a.device)
    elif a.action=='gpt2':run_gpt2(a.device)
    else:analyze()
