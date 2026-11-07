"""Post-result follow-up: locate scale heterogeneity in T5 residual states."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from layer_interface_audit_20261022 import OUT,dump,sha,writecsv

def main():
    import torch
    from transformers import T5EncoderModel,AutoTokenizer
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.18,0)
    torch.backends.cuda.matmul.allow_tf32=False
    selected=[r for r in json.loads((OUT/'selected_texts.json').read_text()) if r['split']=='new_text_subset']
    protocol=dict(date='2026-10-22',scope='Post-result descriptive follow-up prompted by different global- versus token-RMS outcomes; no new decoder outcomes.',
        sequences=128,indices=[r['index'] for r in selected],source_protocol_sha256=sha(OUT/'protocol.json'),
        script_sha256=sha(__file__),precision='float32',sequence_length=1024,
        measurements='Per-position residual RMS distributions, ordinary/EOS/padding energy, across T5 layers 0:5, pre6, and normalized final6. Same previously frozen new-text subset.')
    dest=OUT/'norm_distribution_protocol.json';assert not dest.exists()
    dump(dest,protocol)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/t5-small',local_files_only=True)
    model=T5EncoderModel.from_pretrained(ROOT/'models/t5-small',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    captured={};hook=model.encoder.final_layer_norm.register_forward_pre_hook(lambda m,inp:captured.update(pre=inp[0].detach().clone()))
    rows=[];started=time.time()
    with torch.inference_mode():
        for j,sample in enumerate(selected):
            inp=tok(sample['text'],padding='max_length',truncation=True,max_length=1024,return_tensors='pt').to('cuda')
            o=model(**inp,output_hidden_states=True,return_dict=True)
            mask=inp.attention_mask[0].bool();eos=(inp.input_ids[0]==tok.eos_token_id)&mask
            ordinary=mask&~eos;pad=~mask
            for layer,h in [(str(i),h[0]) for i,h in enumerate(o.hidden_states)]+[('pre6',captured['pre'][0])]:
                norm=h.square().mean(-1).sqrt();energy=h.square().sum(-1)
                r=dict(index=sample['index'],layer=layer,valid_tokens=int(mask.sum()),eos_tokens=int(eos.sum()),padding_tokens=int(pad.sum()),
                    canvas_rms=float(h.square().mean().sqrt()),valid_rms=float(h[mask].square().mean().sqrt()),
                    eos_valid_energy_share=float(energy[eos].sum()/energy[mask].sum()),padding_canvas_energy_share=float(energy[pad].sum()/energy.sum()))
                for name,keep in [('ordinary',ordinary),('eos',eos),('padding',pad)]:
                    x=norm[keep].cpu().numpy()
                    for q in [.1,.5,.9,.99]:r[f'{name}_rms_q{int(100*q)}']=float(np.quantile(x,q)) if len(x) else None
                rows.append(r)
            if j%32==0:print('norm distribution',j,round(time.time()-started,1),flush=True)
    hook.remove();writecsv(OUT/'norm_distribution.csv',rows)
    import pandas as pd
    d=pd.DataFrame(rows);s=d.groupby('layer',as_index=False).mean(numeric_only=True);s.to_csv(OUT/'norm_distribution_summary.csv',index=False)
    dump(OUT/'norm_distribution_analysis.json',dict(status='passed',sequences=len(selected),records=len(rows),
        protocol_sha256=sha(dest),csv_sha256=sha(OUT/'norm_distribution.csv'),scope=protocol['scope']))
    print(s[['layer','canvas_rms','ordinary_rms_q50','ordinary_rms_q99','eos_rms_q50','eos_valid_energy_share','padding_canvas_energy_share']].to_string(index=False))

if __name__=='__main__':main()
