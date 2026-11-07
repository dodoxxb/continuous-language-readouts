"""Post-initial-result token-budget control; preserves the original common test."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from confirmatory_readout_20261013 import ROOT,OUT,dump,digest,arrays_to_cache,train_cache_from_sequences,fit
Q=json.loads((OUT/'langflow_followup_protocol.json').read_text())

@torch.inference_mode()
def generate(device):
    dest=OUT/'data/langflow_extended';dest.mkdir(exist_ok=True)
    if (dest/'ready.json').exists():return
    sys.path.insert(0,str(ROOT/'LangFlow-main'))
    from langflow import LangFlow,LangFlowConfig
    from safetensors.torch import load_file
    md=ROOT/'LangFlow-main/langflow-lm1b';config=LangFlowConfig.from_pretrained(md);model=LangFlow(config)
    model.load_state_dict(load_file(str(md/'model.safetensors')));model.to(device).eval()
    torch.manual_seed(Q['generation_seed']);torch.cuda.manual_seed_all(Q['generation_seed'])
    gammas=model.proposal(torch.linspace(1.-1e-5,1e-5,Q['steps'],device=device));features=[];labels=[];started=time.time()
    for start in range(0,Q['additional_training_sequences'],Q['generation_batch_size']):
        b=min(Q['generation_batch_size'],Q['additional_training_sequences']-start);z=torch.randn(b,config.model_length,config.hidden_size,device=device);sc=None
        for k,g in enumerate(gammas):
            logits=model(noisy_embeds=z,timesteps=g.expand(b),x_self_cond=sc,return_dict=False)
            if k==len(gammas)-1:features.append(z.cpu());labels.append(logits.argmax(-1).cpu());break
            xp=model._embed_tokens(F.softmax(logits.float(),-1));sc=xp if config.self_conditioning else None
            z=model._euler_edm_step(z,xp,g,gammas[k+1])
        if start%256==0:print('additional LangFlow',start+b,'elapsed',time.time()-started,flush=True)
    original=OUT/'data/langflow'
    x=np.concatenate([np.load(original/'train/x.npy'),torch.cat(features).numpy()]);y=np.concatenate([np.load(original/'train/y.npy'),torch.cat(labels).numpy()])
    arrays_to_cache(dest/'train',x,y,np.ones_like(y,dtype=bool),np.arange(len(y)))
    for name in ['validation','test']:
        link=dest/name
        if not link.exists():link.symlink_to(original/name,target_is_directory=True)
    for file,h in Q['fixed_data_hashes'].items():assert digest(OUT/file)==h
    train_cache_from_sequences(dest)
    # Duplicate audit is recorded, never used to alter these frozen partitions.
    import hashlib
    hashrows=lambda arr:[hashlib.sha256(row.tobytes()).hexdigest() for row in arr]
    htrain=hashrows(y);hval=hashrows(np.load(original/'validation/y.npy'));htest=hashrows(np.load(original/'test/y.npy'))
    split={'first_1024':'original LangFlow training set','remaining_7168':'new independent generation seed 61013003','validation':'original 256, unchanged','test':'original 512, unchanged','exact_native_train_duplicates':len(htrain)-len(set(htrain)),'exact_native_train_validation_overlap':len(set(htrain)&set(hval)),'exact_native_train_test_overlap':len(set(htrain)&set(htest))}
    dump(dest/'split.json',split)
    mf=json.loads((original/'ready.json').read_text());mf.update({'family':'langflow_extended','train_sequences':len(y),'train_tokens':int(y.size),'additional_generation_seed':Q['generation_seed'],'generation_batch_size':Q['generation_batch_size'],'elapsed_seconds':time.time()-started,'followup_protocol_sha256':digest(OUT/'langflow_followup_protocol.json'),'train_feature_sha256':digest(dest/'train/x.npy'),'split_sha256':digest(dest/'split.json')})
    dump(dest/'ready.json',mf);print('additional data ready',split,flush=True)

def main():
    device='cuda:1';torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.16,device=device)
    stream=torch.cuda.Stream(device=device,priority=1)
    with torch.cuda.stream(stream):
        generate(device)
        for seed in Q['fit_seeds']:
            dest=OUT/'fits/langflow_extended'/f'n8192_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
            dump(dest/'followup_configuration.json',{'followup_protocol_sha256':digest(OUT/'langflow_followup_protocol.json'),'source_sha256':digest(Path(__file__)),'timing':'post-initial-result token-budget check','test_reused':True})
            fit('langflow_extended',8192,seed,device)
    for file,h in Q['fixed_data_hashes'].items():assert digest(OUT/file)==h

if __name__=='__main__':main()
