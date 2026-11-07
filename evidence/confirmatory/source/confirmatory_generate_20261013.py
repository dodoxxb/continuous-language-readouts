"""Generate an untouched ELF test and a native LangFlow interface control."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from confirmatory_readout_20261013 import ROOT,OUT,P,dump,digest,arrays_to_cache,train_cache_from_sequences

@torch.inference_mode()
def elf(device):
    target=OUT/'data/elf/fresh_test'
    if (target/'generation.json').exists():return
    sys.path.insert(0,str(ROOT/'outputs/phase0_phase1_20260520/scripts'))
    from phase2_trajectory_audit import load_elf,forward_x,decode_logits,get_sampling_steps,mask_after_eos
    a=SimpleNamespace(root=ROOT,out=target,encoder_model=ROOT/'models/t5-small',tokenizer=ROOT/'models/t5-small',ppl_model=ROOT/'models/gpt2-large',config=ROOT/'models/ELF-B-owt-torch/config.yml',checkpoint=ROOT/'models/ELF-B-owt-torch',weights=P['elf_weights'],num_samples=P['elf_fresh_test_sequences'],batch_size=4)
    old=Path.cwd();os.chdir(ROOT/'official-ELF-pytorch_elf')
    try:c,tok,model=load_elf(a,torch.device(device))
    finally:os.chdir(old)
    torch.manual_seed(P['elf_fresh_generation_seed']);torch.cuda.manual_seed_all(P['elf_fresh_generation_seed'])
    gen=torch.Generator().manual_seed(P['elf_fresh_generation_seed'])
    features=[];labels=[];margins=[];started=time.time()
    for start in range(0,a.num_samples,a.batch_size):
        b=min(a.batch_size,a.num_samples-start)
        steps=get_sampling_steps(P['elf_steps'],'logit_normal',c.denoiser_p_mean,c.denoiser_p_std,torch.device(device),torch.float32,gen)
        z=torch.randn(b,c.max_length,model.text_encoder_dim,device=device)*c.denoiser_noise_scale;sc=torch.zeros_like(z)
        for k in range(P['elf_steps']):
            t=float(steps[k]);tn=float(steps[k+1]);tm=t;zm=z
            if k<P['elf_steps']-1:
                alpha=max(0.,min(1.,1.-1.5*(tn-t)));tm=alpha*t
                zm=alpha*z+(1.-alpha)*torch.randn_like(z)*c.denoiser_noise_scale
            xp,_=forward_x(model,zm,tm,sc,3.,c,False)
            z=zm+(tn-tm)*(xp-zm)/max(1.-tm,float(c.t_eps));sc=xp
        ys=[];ms=[]
        for j in range(b):
            logits=decode_logits(model,z[j:j+1],1.,3.,c).float()
            top=logits.topk(2,-1);ys.append(mask_after_eos(top.indices[...,0],1,0).cpu());ms.append((top.values[...,0]-top.values[...,1]).cpu())
        features.append(z.cpu());labels.append(torch.cat(ys));margins.append(torch.cat(ms))
        if start%64==0:print('ELF generated',start+b,'elapsed',time.time()-started,flush=True)
    x=torch.cat(features).numpy();y=torch.cat(labels).numpy();v=y!=0
    arrays_to_cache(target,x,y,v,np.arange(a.num_samples));np.save(target/'native_margin.npy',torch.cat(margins).numpy())
    dump(target/'generation.json',{'seed':P['elf_fresh_generation_seed'],'steps':P['elf_steps'],'samples':a.num_samples,'batch_size':a.batch_size,'protocol_sha256':digest(OUT/'protocol.json'),'checkpoint_sha256':digest(ROOT/'models/ELF-B-owt-torch/checkpoint_95085'),'sampler':'same update equations as historical phase28; final SDE step excluded','x_sha256':digest(target/'x.npy'),'y_sha256':digest(target/'y.npy'),'elapsed_seconds':time.time()-started})

@torch.inference_mode()
def langflow(device):
    target=OUT/'data/langflow'
    if (target/'ready.json').exists():return
    sys.path.insert(0,str(ROOT/'LangFlow-main'))
    from langflow import LangFlow,LangFlowConfig
    from safetensors.torch import load_file
    md=ROOT/P['langflow_model'];c=LangFlowConfig.from_pretrained(md);model=LangFlow(c)
    model.load_state_dict(load_file(str(md/'model.safetensors')));model.to(device).eval()
    captured={}
    handle=model.backbone.register_forward_hook(lambda module,args,output:captured.__setitem__('residual',output[0]))
    torch.manual_seed(P['langflow_seed']);torch.cuda.manual_seed_all(P['langflow_seed'])
    nt=P['langflow_train_sequences'];nv=P['langflow_validation_sequences'];ne=P['langflow_test_sequences'];total=nt+nv+ne
    batch=8;features=[];labels=[];lookup=[];residual=[];decomp=[];saved_phase=[];started=time.time()
    gammas=model.proposal(torch.linspace(1.-1e-5,1e-5,P['langflow_steps'],device=device))
    E=model._get_embedding_matrix().float()
    np.save((target.mkdir(parents=True,exist_ok=True) or target)/'embedding.npy',E.cpu().numpy())
    for start in range(0,total,batch):
        b=min(batch,total-start);z=torch.randn(b,c.model_length,c.hidden_size,device=device);sc=None
        # Input state z is saved before the contextual forward, never its hidden output.
        for k,g in enumerate(gammas):
            logits=model(noisy_embeds=z,timesteps=g.expand(b),x_self_cond=sc,return_dict=False)
            if k==len(gammas)-1 or (start>=nt+nv and k in [0,16,32,48]):
                dot=z.float()@E.T
                res=captured['residual'].float()
                sigma=g.expand(b)
                skip=((F.softplus(-sigma)-sigma)/2).exp()[:,None,None]*dot.to(captured['residual'].dtype)
                # Capture the actual backbone output, avoiding cancellation or a
                # different rounding convention when the final skip is large.
                assert torch.equal(logits,res+skip),'Native residual/skip reconstruction failed'
                row={'start':start,'step':k,'gamma':float(g),'skip_coefficient':float(((F.softplus(-g)-g)/2).exp()),'lookup_agreement':float((dot.argmax(-1)==logits.argmax(-1)).float().mean()),'residual_agreement':float((res.argmax(-1)==logits.argmax(-1)).float().mean()),'residual_rms':float(res.float().square().mean().sqrt()),'skip_rms':float(skip.float().square().mean().sqrt())}
                decomp.append(row)
                if start>=nt+nv:saved_phase.append({'step':k,'start':start-(nt+nv),'native':logits.argmax(-1).cpu().numpy(),'lookup':dot.argmax(-1).cpu().numpy(),'residual':res.argmax(-1).cpu().numpy()})
                if k==len(gammas)-1:
                    features.append(z.cpu());labels.append(logits.argmax(-1).cpu());lookup.append(dot.argmax(-1).cpu());residual.append(res.argmax(-1).cpu())
                del dot,skip,res
            if k<len(gammas)-1:
                xp=model._embed_tokens(F.softmax(logits.float(),-1));sc=xp if c.self_conditioning else None
                z=model._euler_edm_step(z,xp,g,gammas[k+1])
        if start%128==0:print('LangFlow generated',start+b,'elapsed',time.time()-started,flush=True)
    x=torch.cat(features).numpy();y=torch.cat(labels).numpy();lo=torch.cat(lookup).numpy();rr=torch.cat(residual).numpy()
    for name,a,b in [('train',0,nt),('validation',nt,nt+nv),('test',nt+nv,total)]:
        arrays_to_cache(target/name,x[a:b],y[a:b],np.ones_like(y[a:b],dtype=bool),np.arange(a,b))
        np.save(target/name/'lookup.npy',lo[a:b]);np.save(target/name/'residual.npy',rr[a:b])
    # Test-only phase outputs are diagnostic; none are used to fit or select readouts.
    phase_arrays={}
    for k in [0,16,32,48,63]:
        rows=sorted([r for r in saved_phase if r['step']==k],key=lambda r:r['start'])
        for name in ['native','lookup','residual']:phase_arrays[f'{name}_{k}']=np.concatenate([r[name] for r in rows])
    np.savez_compressed(target/'test_phase_outputs.npz',**phase_arrays)
    dump(target/'decomposition.json',decomp)
    dump(target/'split.json',{'train':list(range(nt)),'validation':list(range(nt,nt+nv)),'test':list(range(nt+nv,total)),'generation_seed':P['langflow_seed'],'all_generated_after_protocol':True})
    train_cache_from_sequences(target)
    dump(target/'ready.json',{'family':'langflow','dim':c.hidden_size,'vocab':c.vocab_size,'train_sequences':nt,'validation_sequences':nv,'test_sequences':ne,'steps':len(gammas),'sequence_length':c.model_length,'mask':'all native positions; exact released generate_samples convention','checkpoint_sha256':digest(md/'model.safetensors'),'split_sha256':digest(target/'split.json'),'train_feature_sha256':digest(target/'train/x.npy'),'generation_seed':P['langflow_seed'],'batch_size':batch,'elapsed_seconds':time.time()-started,'protocol_sha256':digest(OUT/'protocol.json')})
    handle.remove()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('family',choices=['elf','langflow']);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.13,device=a.device)
    stream=torch.cuda.Stream(device=a.device,priority=1)
    with torch.cuda.stream(stream):(elf if a.family=='elf' else langflow)(a.device)

if __name__=='__main__':main()
