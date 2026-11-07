"""Official-path parity and post hoc provenance diagnostics, no policy selection."""
from new_interfaces_calm_20261102 import *

def main():
    setup('cuda:2');device='cuda:2'
    from models.modeling_energy import EnergyTransformer
    from models.configuration_calm import CALMConfig
    import transformers
    cfg=CALMConfig.from_pretrained(ROOT/'models/CALM-M',local_files_only=True);cfg.ae_path=str(ROOT/'models/CALM-Autoencoder')
    model,info=EnergyTransformer.from_pretrained(ROOT/'models/CALM-M',config=cfg,local_files_only=True,output_loading_info=True)
    assert not info.get('missing_keys') and not info.get('unexpected_keys') and not info.get('mismatched_keys'),info
    original=torch.load(ROOT/'models/CALM-M/pytorch_model.bin',weights_only=True,map_location='cpu',mmap=True)
    loaded=model.state_dict()
    assert all(torch.equal(v,loaded[k]) for k,v in original.items())
    checked=len(original);del original,loaded
    model=model.float().to(device).eval()
    with torch.inference_mode():
        ids=torch.as_tensor(np.load(OUT/'cohort/test.npy')[:2,:12],device=device)
        h=model.transformer(inputs_embeds=model.embed_proj(model.transformer.embed_tokens(ids).reshape(2,3,-1)),use_cache=False)[0]
        torch.manual_seed(61102931);z=model.generative_head.sample(h)
        direct=model.ae_model.decoder(z).argmax(-1).reshape(2,3,4)
        torch.manual_seed(61102931);official=model.temperature_sampling(h,temperature=1.)
        assert torch.equal(direct,official)
        ae=model.ae_model
        d=np.load(OUT/'generated_predictions.npz');post=np.load(OUT/'data/test/x.npy',mmap_mode='r')
        rng=np.random.default_rng(61102932);which=np.stack([rng.choice(63,4,replace=False) for _ in range(512)])
        gen_ix=(np.arange(512)[:,None]*63+which).ravel();post_ix=(np.arange(512)[:,None]*64+which+1).ravel()
        rows=[]
        for name,x in [('posterior',np.array(post[post_ix])),('generated',d['x'][gen_ix])]:
            for i in range(0,len(x),64):
                zz=torch.as_tensor(x[i:i+64],device=device);yy,_,margin=native(ae,zz)
                re=ae.encoder(yy)[:,0,:128]
                norms=zz.norm(dim=-1);cycle=(zz-re).norm(dim=-1)/norms
                for j in range(len(zz)):
                    rows.append(dict(family=name,document=(i+j)//4,chunk=int(which.ravel()[i+j]),norm=float(norms[j]),mean_native_margin=float(margin[j].mean()),relative_cycle_distance=float(cycle[j])))
        summary={name:{key:{'mean':float(np.mean([r[key] for r in rows if r['family']==name])),
                            'median':float(np.median([r[key] for r in rows if r['family']==name]))} for key in ['norm','mean_native_margin','relative_cycle_distance']} for name in ['posterior','generated']}
    with (OUT/'generation_diagnostics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    dump(OUT/'generation_checks.json',dict(status='passed',official_from_pretrained_all_checkpoint_tensors_equal=checked,
        official_temperature1_tokens_equal=True,parity_tokens=int(official.numel()),transformers=transformers.__version__,
        diagnostic_status='Post hoc observational comparison: four paired chunk positions per document,512documents; no causal identification or policy selection',diagnostics=summary,
        boundary='Base sampling temperature1 only. Does not evaluate the authors recommended temperature<=0.5 or its candidate-collision selection, free-running generation, or published BrierLM protocol.'))
    print(summary,flush=True)

if __name__=='__main__':main()
