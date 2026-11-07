"""Fresh independent native-free endpoint-routing audit; not a deployed speed test.

The released samplers run unchanged. Both terminal outputs are evaluated for
paired counterfactual comparisons. Each arm stops at its own predicted EOS.
"""
from pathlib import Path
import argparse, importlib, json, os, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import torch
import task_readout_runtime_20261022 as parent
from task_readout_outputs_20261022 import boundary
import routing_features_20261023 as routing

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/native_free_routing_20261023'
PARENT = ROOT / 'outputs/task_readout_20261022'
parent.OUT = OUT


def verify_freeze():
    path = OUT / 'frozen.json'
    if not path.exists():
        raise RuntimeError('Fresh generation requires the frozen protocol, cohort and weights')
    frozen = json.loads(path.read_text())
    for relative, digest in frozen['files'].items():
        p = Path(relative)
        if not p.is_absolute():
            p = ROOT / p
        assert parent.sha(p) == digest, str(p)
    return parent.sha(path)


def export_texts(family, base, row, pack, tok, state_sha):
    texts, stops = {}, {}
    for arm, name in [('native', 'native'), ('autonomous17', 'linear17')]:
        ids = pack[name]
        start, stop = boundary(ids, tok.eos_token_id, family)
        texts[arm] = tok.decode(ids[:stop], skip_special_tokens=True,
                               clean_up_tokenization_spaces=False).strip()
        stops[arm] = dict(start=int(start), end=int(stop), eos_present=bool(stop < len(ids)))
    import hashlib
    return dict(index=row['index'], family=family, generation_seed=base,
                document_id=row['id'], texts=texts, independent_boundaries=stops,
                source_sha256=row['document_sha256'], reference_sha256=row['reference_sha256'],
                text_sha256={a: hashlib.sha256(t.encode()).hexdigest() for a,t in texts.items()},
                states_sha256=state_sha,
                boundary_note='Each arm uses its own predicted EOS; no native clamp or native lexical mask.')


def primary_features_one_head(logits, counts, numeric, eos_id, family):
    """Identical eight primary features without executing auxiliary heads."""
    z=logits.float()
    a=z.argmax(-1).detach().cpu().numpy()
    top=z.topk(2,-1).values
    lg=torch.log1p((top[:,0]-top[:,1]).clamp_min(0))
    logp=z-torch.logsumexp(z,dim=-1,keepdim=True)
    entropy=-(logp.exp()*logp).sum(-1)
    _,stop=routing.own_boundary(a,eos_id,family)
    return np.array([float(lg.mean()),float(torch.quantile(lg,0.1)),float(entropy.mean()),
        float(np.log1p(counts[a]).mean()),float(numeric[a].mean()),float(np.any(a==eos_id)),
        float(stop/len(a)),float(len(a))],dtype=np.float64)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True, choices=['elf', 'aurora'])
    p.add_argument('--device', required=True)
    p.add_argument('--seed', required=True, type=int, choices=[61032301,61034301])
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=256)
    args = p.parse_args()
    freeze_sha = verify_freeze()
    parent.resources(args.device, cap=5.5)
    rows = [json.loads(s) for s in (OUT/'cohort/cohort.jsonl').read_text().splitlines()]
    assert len(rows) == 256
    g = parent.load_elf(args.device) if args.family=='elf' else parent.load_aurora(args.device)
    tok = g.tok if args.family=='elf' else g.trainer.tokenizer
    maps = parent.readouts(args.family,args.device)
    _,countpath,numericpath,eos_id = routing.assets(args.family)
    counts,numeric = np.load(countpath),np.load(numericpath)
    risk_model = json.loads((OUT/'calibration/weights.json').read_text())['models'][args.family]
    sampler = parent.sample_elf if args.family=='elf' else parent.sample_aurora
    stream = torch.cuda.Stream(device=args.device, priority=1)
    stream.wait_stream(torch.cuda.current_stream(args.device))
    decode_events = []
    if args.family=='elf':
        module = importlib.import_module('phase2_trajectory_audit')
        original_decode = module.decode_logits
    else:
        module = g.trainer.autoencoder
        original_decode = module.decode_latent
    def timed_decode(*a, **kw):
        begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        value = original_decode(*a,**kw)
        end.record()
        decode_events.append((begin,end))
        return value
    if args.family=='elf': module.decode_logits = timed_decode
    else: module.decode_latent = timed_decode
    dest = OUT/'generated'/args.family/str(args.seed)
    textdest = OUT/'texts'/args.family/str(args.seed)
    dest.mkdir(parents=True,exist_ok=True)
    textdest.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    newly_generated = 0
    for row in rows[args.start:args.end]:
        path = dest/f"{row['index']:04}.npz"
        meta = path.with_suffix('.json')
        if meta.exists():
            saved = json.loads(meta.read_text())
            assert saved['states_sha256'] == parent.sha(path)
            assert saved['freeze_sha256'] == freeze_sha
            assert (textdest/meta.name).exists()
            continue
        free,_ = torch.cuda.mem_get_info(args.device)
        if free < 2*2**30:
            raise RuntimeError('Shared GPU free headroom fell below 2 GiB; resumable')
        decode_events.clear()
        t0 = time.monotonic()
        with torch.cuda.stream(stream):
            pack = sampler(g,row,args.seed+row['index'],parity=False)
            stream.synchronize()
            sample_wall = time.monotonic()-t0
            assert len(decode_events)==1, 'The terminal decoder must not determine earlier sampling'
            decode_ms = decode_events[0][0].elapsed_time(decode_events[0][1])
            h0 = time.monotonic()
            begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            x = torch.tensor(pack['x'],device=args.device,dtype=torch.float32)
            begin.record()
            logits_by_seed = {}
            for seed,m in maps.items():
                logits = m(x)
                assert bool(torch.isfinite(logits).all())
                logits_by_seed[seed] = logits
            features,secondary,preds = routing.features_from_logits(
                logits_by_seed,counts,numeric,eos_id,args.family)
            predicted_gain = float(routing.predict(features[None,:],risk_model)[0])
            end.record()
            for seed,value in preds.items():
                pack[f'linear{seed}'] = value.astype(np.int32)
            pack['routing_features'] = features
            stream.synchronize()
            affine_cuda_ms = begin.elapsed_time(end)
            affine_wall = time.monotonic()-h0
            del logits_by_seed,logits
            # Separate instrumented replay on the same already generated latent:
            # the primary ridge needs only fit17. Auxiliary heads are descriptive.
            single_begin,single_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            single_wall_start=time.monotonic()
            single_begin.record()
            single_features=primary_features_one_head(maps[17](x),counts,numeric,eos_id,args.family)
            single_gain=float(routing.predict(single_features[None,:],risk_model)[0])
            single_end.record()
            stream.synchronize()
            single_cuda_ms=single_begin.elapsed_time(single_end)
            single_wall=time.monotonic()-single_wall_start
            np.testing.assert_array_equal(single_features,features)
            assert single_gain==predicted_gain
            del x
        arrays = {k:v for k,v in pack.items() if isinstance(v,np.ndarray)}
        for name,value in arrays.items():
            assert np.isfinite(value).all(),name
        with path.with_suffix('.npz.tmp').open('wb') as f:
            np.savez_compressed(f,**arrays)
        path.with_suffix('.npz.tmp').replace(path)
        state_sha = parent.sha(path)
        textrecord = export_texts(args.family,args.seed,row,pack,tok,state_sha)
        parent.dump(textdest/meta.name,textrecord)
        record = dict(family=args.family,index=row['index'],document_id=row['id'],
            generation_seed=args.seed,effective_seed=args.seed+row['index'],
            prefix_length=pack['prefix_length'],generated_canvas_length=len(pack['native']),
            sampling_with_native_and_capture_wall_seconds=sample_wall,
            native_terminal_forward_cuda_ms=decode_ms,
            three_affine_heads_features_and_ridge_cuda_ms=affine_cuda_ms,
            three_affine_heads_features_and_ridge_wall_seconds=affine_wall,
            isolated_affine17_features_and_ridge_cuda_ms=single_cuda_ms,
            isolated_affine17_features_and_ridge_wall_seconds=single_wall,
            native_free_features={k:float(v) for k,v in zip(routing.FEATURE_NAMES,features)},
            native_free_secondary=secondary,native_free_predicted_bertscore_gain=predicted_gain,
            elapsed=time.monotonic()-t0,states_sha256=state_sha,freeze_sha256=freeze_sha,
            max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            max_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            terminal_native_decode_calls=len(decode_events),
            timing_limitations='Shared GPU descriptive measurements; audit computes both outputs. Native forward excludes argmax. Affine intervals include features, CPU summaries and ridge prediction; CUDA events span host gaps. Isolated affine17 is a separate replay on the same latent with exact feature parity. No deployed end-to-end speed measurement.',
            completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        parent.dump(meta,record)
        newly_generated += 1
        if newly_generated%16==0 or newly_generated==1:
            print(args.family,args.seed,row['index'],'wall_s',round(time.monotonic()-started,1),
                  'sample_s',round(sample_wall,3),'native_ms',round(decode_ms,3),
                  'affine_ms',round(affine_cuda_ms,3),'GiB',round(record['max_allocated_gib'],3),flush=True)
    parent.dump(dest/f'complete_{args.start}_{args.end}.json',dict(
        family=args.family,seed=args.seed,start=args.start,end=args.end,newly_generated=newly_generated,
        elapsed=time.monotonic()-started,freeze_sha256=freeze_sha))


if __name__=='__main__':main()
