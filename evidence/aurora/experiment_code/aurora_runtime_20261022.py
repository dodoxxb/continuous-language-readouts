"""Pinned AURORA inference adapter. Capture the input of the FULL native decoder.

No sampler, checkpoint, latent normalization or decoder math is replaced.
All resource controls are local to this process, never other users' jobs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/aurora_replication_20261022'
UPSTREAM = ROOT / 'third_party/AURORA-LM'
sys.path[:0] = [str(OUT/'deps'), str(UPSTREAM/'src')]
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
import numpy as np
import torch

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()

def setup(device='cuda:2', threads=2, memory_gib=3.5):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.startswith('cuda'):
        os.environ['DEVICE_TYPE']='cuda'
        os.environ['DEVICE_INDEX']=device.split(':')[-1]
        torch.cuda.set_device(device)
        free,total=torch.cuda.mem_get_info(device)
        if free < (memory_gib+2.0)*2**30:
            raise RuntimeError(f'Insufficient safe headroom before loading: {free/2**30:.2f} GiB free')
        torch.cuda.set_per_process_memory_fraction(memory_gib*2**30/total, device)
    elif device=='cpu':
        os.environ['DEVICE_TYPE']='cpu'
    else:
        raise ValueError(device)
    from aurora.eval.core.config import load_config_from_checkpoint
    from aurora.eval.generators.diffusion import DiffusionGenerator
    ckpt=ROOT/'models/aurora-lm/owt/250000.safetensors'
    overrides=[
        f'causal_autoencoder.model.load_checkpoint={ROOT}/models/aurora-lm/autoencoder/100000.safetensors',
        f'causal_autoencoder.model.text_encoder={ROOT}/models/gpt2',
        f'project.path={UPSTREAM}', f'project.checkpoint_dir={OUT}/runtime',
        'operator_optimization.profile="off"', 'precision.mode=bf16',
        'diffusion.model.num_workers=0', 'causal_autoencoder.model.num_workers=0',
    ]
    (OUT/'runtime').mkdir(exist_ok=True)
    cfg,_=load_config_from_checkpoint(str(ckpt), overrides=overrides, project_root=str(UPSTREAM))
    generator=DiffusionGenerator(cfg, str(ckpt), device=device)
    from omegaconf import OmegaConf
    dump(OUT/'runtime'/f'config_{device.replace(":","_")}.json', OmegaConf.to_container(generator.trainer.cfg,resolve=True))
    return generator

@torch.inference_mode()
def capture(generator, count, steps, length, seed, batch_size, sc=4.0):
    """Observe the original public generate path, including its EMA and solver."""
    captured=[]
    original=generator._decode_latent_to_logits
    def hook(latents, target_token_len=None):
        logits=original(latents, target_token_len=target_token_len)
        top=logits.float().topk(2,-1)
        x=latents.float().cpu().numpy()
        if not np.isfinite(x).all() or not torch.isfinite(top.values).all():
            raise RuntimeError('Nonfinite native state/logits')
        # Match the released decoder's argmax tie-breaking. topk's first index
        # can differ from argmax on BF16 logit ties even at identical states.
        captured.append(dict(x=x,native=logits.argmax(-1).cpu().numpy().astype(np.int32),
                             margin=(top.values[...,0]-top.values[...,1]).cpu().numpy()))
        return logits
    generator._decode_latent_to_logits=hook
    started=time.monotonic()
    try:
        texts,extras,stats=generator.generate(num_samples=count,num_steps=steps,batch_size=batch_size,
            seed=seed,verbose=False,eos_mode='eos_wrap',max_gen_tokens=length,
            solver='sde_dpmpp',self_cond_cfg_scale=sc)
    finally:
        generator._decode_latent_to_logits=original
    pack={k:np.concatenate([d[k] for d in captured],axis=0) for k in ['x','native','margin']}
    # Official eos_wrap returns the full generated token sequence, with an EOS prefix.
    emitted=np.asarray(extras['tokens'])
    expected=pack['native']
    if emitted.shape==expected.shape:
        assert np.array_equal(emitted,expected)
    elif emitted.shape==(len(expected),expected.shape[1]+1):
        assert np.all(emitted[:,0]==50256) and np.array_equal(emitted[:,1:],expected)
    else:
        raise AssertionError((emitted.shape,expected.shape))
    stats.update(elapsed=time.monotonic()-started,torch=torch.__version__,
        max_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if str(generator.device).startswith('cuda') else 0.,
        max_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if str(generator.device).startswith('cuda') else 0.,
        native_eos_fraction=float((expected==50256).mean()),
        source_commit=json.loads((OUT/'source/upstream_commit.json').read_text())['sha'])
    return pack,texts,stats

@torch.inference_mode()
def capture_streamed(generator, count, steps, length, seed, sc=4.0):
    """Same native sampler, sequential native decoding to bound B*L*V memory.

This changes only decoder batching. The public entry is independently checked
on pilot states before any confirmatory data are generated.
"""
    from aurora.utils.device_utils import manual_seed
    torch.manual_seed(seed);manual_seed(seed)
    trainer=generator.trainer
    params=generator._resolve_sampling_kwargs(dict(solver='sde_dpmpp',self_cond_cfg_scale=sc,eos_mode='eos_wrap'))
    block_size=trainer._resolve_effective_block_size(None)
    assert length % block_size == 0
    old_steps=trainer.cfg.diffusion.dynamic.N
    trainer.cfg.diffusion.dynamic.N=steps
    started=time.monotonic()
    try:
        z=generator._pred_latents_with_params(params,batch_size=count,block_size=None,num_generate_blocks=length//block_size)
        with trainer.autocast_context():
            x=trainer.autoencoder.denormalize_latent(z)
        del z
        native=[];margin=[];texts=[]
        for i in range(count):
            with trainer.autocast_context():
                logits=generator._decode_latent_to_logits(x[i:i+1],target_token_len=length)
            top=logits.float().topk(2,-1)
            if not torch.isfinite(top.values).all():raise RuntimeError('Nonfinite logits')
            native.append(logits.argmax(-1).cpu().numpy().astype(np.int32))
            margin.append((top.values[...,0]-top.values[...,1]).cpu().numpy())
            text,_=generator._decode_tokens_with_ar_stop(logits,eos_mode='eos_wrap');texts+=text
            del top,logits
        pack=dict(x=x.float().cpu().numpy(),native=np.concatenate(native),margin=np.concatenate(margin))
        if not np.isfinite(pack['x']).all():raise RuntimeError('Nonfinite states')
    finally:
        trainer.cfg.diffusion.dynamic.N=old_steps
    stats=dict(elapsed=time.monotonic()-started,steps=steps,sc=sc,batch=count,length=length,
        torch=torch.__version__,max_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if str(generator.device).startswith('cuda') else 0.,
        max_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if str(generator.device).startswith('cuda') else 0.,
        native_eos_fraction=float((pack['native']==50256).mean()))
    return pack,texts,stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--memory-gib',type=float,default=3.5);p.add_argument('--count',type=int,default=1)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--length',type=int,default=32)
    p.add_argument('--batch',type=int,default=1);p.add_argument('--seed',type=int,default=61022001)
    p.add_argument('--tag',default='smoke');p.add_argument('--streamed',action='store_true');a=p.parse_args()
    g=setup(a.device,a.threads,a.memory_gib)
    from contextlib import nullcontext
    stream_context=torch.cuda.stream(torch.cuda.Stream(device=a.device,priority=1)) if a.device.startswith('cuda') else nullcontext()
    with stream_context:
        if a.streamed:pack,texts,stats=capture_streamed(g,a.count,a.steps,a.length,a.seed)
        else:pack,texts,stats=capture(g,a.count,a.steps,a.length,a.seed,a.batch)
    dest=OUT/'pilot'/a.tag;dest.mkdir(parents=True,exist_ok=True)
    np.savez(dest/'states.npz',**pack)
    dump(dest/'result.json',dict(arguments=vars(a),stats=stats,texts=texts))
    print(json.dumps(stats),flush=True)

if __name__=='__main__': main()
