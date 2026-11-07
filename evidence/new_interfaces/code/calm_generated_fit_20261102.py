"""Matched posterior-vs-prior fitting control; common validation and fresh test."""
from new_interfaces_calm_20261102 import *
from calm_fit_control_20261102 import Tied
G=OUT/'generated_fit'

def prior_pool(model,ids,name,seed,device):
    dest=G/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():return
    n=len(ids)*63
    x=np.lib.format.open_memmap(dest/'x.npy',mode='w+',shape=(n,128),dtype='float32')
    y=np.lib.format.open_memmap(dest/'y.npy',mode='w+',shape=(n,4),dtype='int32')
    torch.manual_seed(seed)
    with torch.inference_mode():
        for start in range(0,len(ids),8):
            t=torch.as_tensor(ids[start:start+8,:252],device=device)
            h=model.transformer(inputs_embeds=model.embed_proj(model.transformer.embed_tokens(t).reshape(len(t),63,-1)),use_cache=False)[0]
            z=model.generative_head.sample(h).reshape(-1,128);lo=start*63;hi=lo+len(z);x[lo:hi]=z.cpu().numpy()
            for j in range(0,len(z),64):y[lo+j:lo+j+len(z[j:j+64])]=model.ae_model.decoder(z[j:j+64,None,:]).argmax(-1).cpu().numpy()
            if start%512==0:print('collect',name,start,len(ids),flush=True)
    x.flush();y.flush();dump(dest/'manifest.json',dict(documents=len(ids),chunks=n,seed=seed,x_sha256=sha(dest/'x.npy'),y_sha256=sha(dest/'y.npy')))

def load_model(device):
    from models.modeling_energy import EnergyTransformer
    from models.configuration_calm import CALMConfig
    cfg=CALMConfig.from_pretrained(ROOT/'models/CALM-M',local_files_only=True);cfg.ae_path=str(ROOT/'models/CALM-Autoencoder')
    m,info=EnergyTransformer.from_pretrained(ROOT/'models/CALM-M',config=cfg,local_files_only=True,output_loading_info=True)
    assert not info.get('missing_keys') and not info.get('unexpected_keys') and not info.get('mismatched_keys'),info
    return m.float().to(device).eval()

def run(device):
    G.mkdir(parents=True,exist_ok=True);model=load_model(device)
    for split,seed in [('train',61102301),('validation',61102302)]:prior_pool(model,np.load(OUT/'cohort'/(split+'.npy')),split,seed,device)
    E=model.ae_model.encoder.embed_tokens.weight.detach().clone();del model
    ridge=torch.as_tensor(np.load(OUT/'ridge_weights.npy'),device=device);Ef=E.double();gram=Ef.T@Ef
    coef=torch.stack([torch.linalg.solve(gram,Ef.T@w.double()) for w in ridge]).float();del Ef,gram,ridge
    xv=np.load(G/'validation/x.npy',mmap_mode='r');yv=np.load(G/'validation/y.npy')
    xpost=np.load(OUT/'data/train/x.npy').reshape(4096,64,128)[:,1:].reshape(-1,128)
    ypost=np.load(OUT/'data/train/native.npy').reshape(4096,64,4)[:,1:].reshape(-1,4)
    xprior=np.load(G/'train/x.npy',mmap_mode='r');yprior=np.load(G/'train/y.npy',mmap_mode='r')
    assert len(xpost)==len(xprior)==4096*63
    reports=[]
    for family,x,y in [('posterior',xpost,ypost),('prior',xprior,yprior)]:
        for seed in [17,29,43]:
            folder=G/f'{family}_seed{seed}';folder.mkdir(exist_ok=True)
            if (folder/'manifest.json').exists():reports.append(json.loads((folder/'manifest.json').read_text()));continue
            torch.manual_seed(seed);rng=np.random.default_rng(seed);m=Tied(coef,E).to(device)
            opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01);best=-1;history=[]
            for step in range(2049):
                if step in [0,512,1024,2048]:
                    m.eval();pred,ce=predict(m,xv,device,yv);acc=float((pred==yv).mean());history.append(dict(step=step,agreement=acc,ce=ce));print('fit',family,seed,step,acc,flush=True)
                    if acc>best:
                        best=acc;selected=step;torch.save({'weight':m.effective().cpu()},folder/'selected.pt')
                        torch.save({'coef':m.coef.detach().cpu()},folder/'selected_coef.pt')
                    m.train()
                if step==2048:break
                ix=rng.integers(len(x),size=64);xx=torch.as_tensor(np.array(x[ix]),device=device);yy=torch.as_tensor(np.array(y[ix]).reshape(-1),device=device,dtype=torch.long)
                opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(xx).flatten(0,1),yy);loss.backward();opt.step()
            row=dict(family=family,seed=seed,selected_step=selected,validation_agreement=best,selected_sha256=sha(folder/'selected.pt'),coef_sha256=sha(folder/'selected_coef.pt'))
            dump(folder/'manifest.json',row);dump(folder/'history.json',history);reports.append(row);del m,opt
    dump(G/'frozen.json',dict(fits=reports,frozen_before_test_generation=True,time=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    del E,coef,xpost,ypost;torch.cuda.empty_cache()
    model=load_model(device);prior_pool(model,np.load(OUT/'fit_control/fresh_ids.npy'),'test',61102303,device);del model
    x=np.load(G/'test/x.npy',mmap_mode='r');y=np.load(G/'test/y.npy');preds={}
    for family in ['posterior','prior']:
        for seed in [17,29,43]:
            m=Readout(np.load(OUT/'ridge_weights.npy')).to(device)
            m.load_state_dict(torch.load(G/f'{family}_seed{seed}/selected.pt',map_location=device,weights_only=True));m.eval()
            preds[f'{family}_{seed}']=predict(m,x,device)[0];del m
    np.savez_compressed(G/'test_predictions.npz',native=y,**preds)
    boot=np.random.default_rng(61102309).integers(512,size=(4000,512));means={}
    rows=[]
    for family in ['posterior','prior']:
        a=np.stack([(preds[f'{family}_{s}']==y).reshape(512,-1).mean(1) for s in [17,29,43]]).mean(0);means[family]=a
        rows.append(dict(family=family,agreement=float(a.mean()),ci=np.quantile(a[boot].mean(1),[.025,.975]).tolist(),individual_agreements={str(s):float((preds[f'{family}_{s}']==y).mean()) for s in [17,29,43]}))
    delta=means['prior']-means['posterior']
    dump(G/'summary.json',dict(documents=512,chunks=512*63,results=rows,gain=float(delta.mean()),gain_ci=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),max_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    print('final',rows,'gain',float(delta.mean()),flush=True)

if __name__=='__main__':setup('cuda:2');run('cuda:2')
