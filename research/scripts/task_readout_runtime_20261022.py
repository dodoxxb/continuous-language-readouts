"""Observe released conditional samplers and frozen OWT readouts on XSum."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,os,sys,time

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/task_readout_20261022'
sys.path[:0]=[str(OUT/'deps'),str(ROOT/'outputs/aurora_replication_20261022/deps')]
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import torch

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def resources(device,cap=5.5):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info(device)
    if free<(cap+2)*2**30:raise RuntimeError('Insufficient safe GPU headroom')
    torch.cuda.set_per_process_memory_fraction(cap*2**30/total,device)

def readouts(family,device):
    models={}
    for seed in [17,29,43]:
        path=(ROOT/f'outputs/confirmatory_readout_20261013/fits/elf/n16384_seed{seed}/selected.pt' if family=='elf'
              else ROOT/f'outputs/aurora_replication_20261022/fits/seed{seed}/selected.pt')
        state=torch.load(path,map_location=device,weights_only=True)
        model=torch.nn.Linear(state['weight'].shape[1],state['weight'].shape[0],device=device)
        model.load_state_dict(state);models[seed]=model.eval()
    return models

def load_elf(device):
    sys.path.insert(0,str(ROOT/'outputs/phase0_phase1_20260520/scripts'))
    from phase2_trajectory_audit import load_elf as original_load
    args=SimpleNamespace(root=ROOT,out=OUT/'runtime/elf',encoder_model=ROOT/'models/t5-small',
        tokenizer=ROOT/'models/t5-small',ppl_model=ROOT/'models/gpt2-large',
        config=ROOT/'models/ELF-B-xsum-torch/config.yml',checkpoint=ROOT/'models/ELF-B-xsum-torch/checkpoint_39800',
        weights='ema_params1',num_samples=1,batch_size=1)
    cfg,tok,model=original_load(args,torch.device(device))
    from modules.t5_encoder import get_encoder
    _,encoder=get_encoder(str(ROOT/'models/t5-small'),torch.float32)
    encoder=encoder.to(device).eval()
    return SimpleNamespace(cfg=cfg,tok=tok,model=model,encoder=encoder,device=device)

@torch.inference_mode()
def sample_elf(g,row,seed,parity=False):
    from utils.encoder_utils import build_self_attn_cond_masks,encode_text
    from utils.generation_utils import _generate_samples_single_batch,_dlm_decode_batch
    from utils.sampling_utils import get_sampling_steps
    from phase2_trajectory_audit import decode_logits
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    c=g.cfg;device=g.device;prefix=row['elf_prefix_ids'];n=len(prefix)
    assert n<=c.max_input_length and c.max_length==1088
    # The human reference never enters the generation input. Source positions
    # attend only to other source positions under the released encoder mask.
    ids=np.full((1,c.max_length),g.tok.eos_token_id,np.int64);ids[0,:n]=prefix
    iscond=np.arange(c.max_length)[None,:]<n
    isvalid=np.arange(c.max_length)[None,:]<n+64
    _,_,condmask=build_self_attn_cond_masks(iscond,isvalid)
    # Transformers 4.57 T5Stack expands a 3-D mask into 5-D, which breaks the
    # upstream conditional encoder call. Only source latents are used here:
    # a 2-D source-key mask gives every source query exactly the same allowed
    # keys as the upstream 3-D mask. Target-query outputs are discarded by the
    # sampler's conditioning mask. No reference tokens are needed or supplied.
    encmask=iscond.astype(np.float32)
    ids_t=torch.tensor(ids,device=device);mask_t=torch.tensor(encmask,device=device)
    cond_t=torch.tensor(condmask,device=device)
    cond=encode_text(ids_t,mask_t,g.encoder,c.latent_mean,c.latent_std)
    leakage=None
    if parity:
        changed=ids_t.clone();changed[:,n:]=17
        other=encode_text(changed,mask_t,g.encoder,c.latent_mean,c.latent_std)
        leakage=bool(torch.equal(cond[:,:n],other[:,:n]));assert leakage
    steps=get_sampling_steps(64,'logit_normal',c.denoiser_p_mean,c.denoiser_p_std,device=torch.device(device),dtype=torch.float32)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    z=torch.randn((1,c.max_length,g.model.text_encoder_dim),generator=generator)*c.denoiser_noise_scale
    z=z.to(device)
    sampling=SimpleNamespace(sampling_method='ode',sde_gamma=0.)
    z=_generate_samples_single_batch(g.model,generator,z,steps,cond,cond_t,c,sampling,2.,1.)
    logits=decode_logits(g.model,z,1.,1.,c).float()
    native=logits.argmax(-1)
    public_equal=None
    if parity:
        public=_dlm_decode_batch(z,g.model,1.,c,1.);public_equal=bool(torch.equal(public,native));assert public_equal
    target_logits=logits[:,n:n+64];top=target_logits.topk(2,-1).values
    return dict(full_x=z[0].float().cpu().numpy(),x=z[0,n:n+64].float().cpu().numpy(),
        native=native[0,n:n+64].cpu().numpy().astype(np.int32),
        margin=(top[0,:,0]-top[0,:,1]).cpu().numpy(),prefix_length=n,
        public_decode_equal=public_equal,reference_exclusion_verified=leakage)

def load_aurora(device):
    os.environ['DEVICE_TYPE']='cuda';os.environ['DEVICE_INDEX']=device.split(':')[-1]
    upstream=ROOT/'third_party/AURORA-LM';sys.path.insert(0,str(upstream/'src'))
    from aurora.eval.core.config import load_config_from_checkpoint
    from aurora.eval.generators.diffusion import DiffusionGenerator
    ckpt=ROOT/'models/aurora-lm/xsum/300000.safetensors'
    overrides=[f'causal_autoencoder.model.load_checkpoint={ROOT}/models/aurora-lm/autoencoder/100000.safetensors',
        f'causal_autoencoder.model.text_encoder={ROOT}/models/gpt2',f'project.path={upstream}',
        f'project.checkpoint_dir={OUT}/runtime/aurora','operator_optimization.profile="off"','precision.mode=bf16',
        'diffusion.model.num_workers=0','causal_autoencoder.model.num_workers=0']
    (OUT/'runtime/aurora').mkdir(parents=True,exist_ok=True)
    cfg,_=load_config_from_checkpoint(str(ckpt),overrides=overrides,project_root=str(upstream))
    assert cfg.conditional.enabled
    gen=DiffusionGenerator(cfg,str(ckpt),device=device)
    return gen

@torch.inference_mode()
def sample_aurora(g,row,seed,parity=False):
    ae=g.trainer.autoencoder;original=ae.decode_latent;captured=[];n=len(row['aurora_prefix_ids'])
    def hook(x,*args,**kwargs):
        logits=original(x,*args,**kwargs)
        target=logits[:,n:];top=target.float().topk(2,-1).values
        captured.append(dict(full_x=x[0].float().cpu().numpy(),x=x[0,n:].float().cpu().numpy(),
            native=target.argmax(-1)[0].cpu().numpy().astype(np.int32),
            margin=(top[0,:,0]-top[0,:,1]).cpu().numpy(),prefix_length=n))
        return logits
    kwargs=dict(prefix_texts=[''],prefix_token_ids=[row['aurora_prefix_ids']],num_steps=32,
        batch_size=1,seed=seed,max_gen_tokens=64,verbose=False)
    ae.decode_latent=hook
    try:texts,extras,stats=g.generate_multi_prefix(**kwargs)
    finally:ae.decode_latent=original
    assert len(captured)==1
    p=captured[0];eos=g.trainer.tokenizer.eos_token_id
    from aurora.utils.tokenizer_utils import truncate_token_sequences
    assert g.trainer.last_block_ar_stop_token_idx is None
    assert set(g.trainer.stop_token_ids)=={eos}
    assert extras['tokens'][0]==p['native'].tolist()
    trimmed=truncate_token_sequences([p['native'].tolist()],g.trainer.stop_token_ids)[0]
    decoded=g.trainer.tokenizer.decode(trimmed,skip_special_tokens=True)
    assert decoded.strip()==texts[0].strip(),(decoded,texts[0])
    public_equal=None
    if parity:
        direct,extra2,_=g.generate_multi_prefix(**kwargs)
        public_equal=(extras['tokens']==extra2['tokens'] and texts==direct);assert public_equal
    p.update(public_decode_equal=public_equal,reference_exclusion_verified=True,public_text=texts[0])
    return p

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['elf','aurora'],required=True)
    p.add_argument('--device',required=True);p.add_argument('--pilot',action='store_true')
    p.add_argument('--seed',type=int,default=61022701);p.add_argument('--start',type=int,default=0)
    p.add_argument('--end',type=int);a=p.parse_args()
    resources(a.device)
    if not a.pilot:
        frozen=json.loads((OUT/'frozen.json').read_text())
        for path,digest in frozen['files'].items():assert sha(ROOT/path)==digest,path
    rows=[json.loads(x) for x in (OUT/('pilot_cohort.jsonl' if a.pilot else 'cohort.jsonl')).read_text().splitlines()]
    end=a.end if a.end is not None else (2 if a.pilot else len(rows))
    g=load_elf(a.device) if a.family=='elf' else load_aurora(a.device)
    tok=g.tok if a.family=='elf' else g.trainer.tokenizer
    maps=readouts(a.family,a.device)
    sampler=sample_elf if a.family=='elf' else sample_aurora
    dest=OUT/('pilot' if a.pilot else 'generated')/a.family/str(a.seed);dest.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();complete=0
    stream=torch.cuda.Stream(device=a.device,priority=1)
    for row in rows[a.start:end]:
        path=dest/f"{row['index']:04}.npz";meta=path.with_suffix('.json')
        if meta.exists():
            assert sha(path)==json.loads(meta.read_text())['states_sha256'];continue
        while torch.cuda.mem_get_info(a.device)[0]<2*2**30:
            torch.cuda.empty_cache();print('Waiting for shared GPU headroom',a.device,flush=True);time.sleep(10)
        tic=time.monotonic();effective_seed=a.seed+row['index']
        with torch.cuda.stream(stream):
            pack=sampler(g,row,effective_seed,parity=a.pilot)
            for seed,m in maps.items():
                logits=m(torch.tensor(pack['x'],device=a.device,dtype=torch.float32))
                assert torch.isfinite(logits).all()
                pack[f'linear{seed}']=logits.argmax(-1).cpu().numpy().astype(np.int32)
        for name in ['full_x','x','margin']:assert np.isfinite(pack[name]).all(),name
        arrays={k:v for k,v in pack.items() if isinstance(v,np.ndarray)}
        with path.with_suffix('.npz.tmp').open('wb') as f:np.savez_compressed(f,**arrays)
        path.with_suffix('.npz.tmp').replace(path)
        record=dict(family=a.family,index=row['index'],document_id=row['id'],generation_seed=a.seed,effective_seed=effective_seed,
            prefix_length=pack['prefix_length'],generated_canvas_length=len(pack['native']),elapsed=time.monotonic()-tic,
            states_sha256=sha(path),max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            max_reserved_gib=torch.cuda.max_memory_reserved()/2**30,public_decode_equal=pack['public_decode_equal'],
            reference_exclusion_verified=pack['reference_exclusion_verified'],completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        dump(meta,record);complete+=1
        if complete%16==0 or a.pilot:print(a.family,a.seed,row['index'],'elapsed',round(time.monotonic()-started,1),'last',round(record['elapsed'],2),'GiB',round(record['max_allocated_gib'],3),flush=True)
    dump(dest/f'complete_{a.start}_{end}.json',dict(family=a.family,seed=a.seed,start=a.start,end=end,elapsed=time.monotonic()-started,newly_generated=complete))

if __name__=='__main__':main()
