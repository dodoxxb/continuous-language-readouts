"""Fit-adequacy control with untouched follow-up documents; see frozen protocol."""
from new_interfaces_calm_20261102 import *

class Tied(torch.nn.Module):
    def __init__(self,coef,embedding):
        super().__init__();self.coef=torch.nn.Parameter(coef.clone());self.register_buffer('embedding',embedding)
    def forward(self,x):
        z=torch.cat((x,torch.ones_like(x[:,:1])),-1)
        h=torch.einsum('nd,phd->nph',z,self.coef)
        return F.linear(h,self.embedding)
    @torch.no_grad()
    def effective(self):return torch.einsum('vh,phd->pvd',self.embedding,self.coef)

def fit_controls(device):
    dest=OUT/'fit_control';dest.mkdir(exist_ok=True)
    if (dest/'frozen.json').exists():return
    ae=load_ae(device);E=ae.encoder.embed_tokens.weight.detach().clone();del ae
    ridge=torch.as_tensor(np.load(OUT/'ridge_weights.npy'),device=device)
    # Float64 least squares makes the recovered initialization explicit.
    Ef=E.double();gram=Ef.T@Ef
    coef=torch.stack([torch.linalg.solve(gram,Ef.T@w.double()) for w in ridge]).float()
    error=float((torch.einsum('vh,phd->pvd',E,coef)-ridge).abs().max());assert error<1e-4
    del Ef,gram
    x=np.load(OUT/'data/train/x.npy',mmap_mode='r');y=np.load(OUT/'data/train/native.npy',mmap_mode='r')
    xv=np.load(OUT/'data/validation/x.npy',mmap_mode='r');yv=np.load(OUT/'data/validation/native.npy')
    rows=[]
    for family,lr in [('free',1e-5),('tied',1e-5),('tied',3e-5),('tied',1e-4)]:
        for seed in [17,29,43]:
            name=f'{family}_lr{lr}_seed{seed}';folder=dest/name;folder.mkdir(exist_ok=True)
            if (folder/'manifest.json').exists():rows.append(json.loads((folder/'manifest.json').read_text()));continue
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            m=Readout(ridge.cpu().numpy()).to(device) if family=='free' else Tied(coef,E).to(device)
            opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.01);best=-1;hist=[]
            for step in range(2049):
                if step in [0,512,1024,2048]:
                    m.eval();pred,ce=predict(m,xv,device,yv);acc=float((pred==yv).mean())
                    hist.append(dict(step=step,agreement=acc,ce=ce));print(name,step,acc,flush=True)
                    if acc>best:
                        best=acc;selected=step
                        effective=m.weight.detach() if family=='free' else m.effective()
                        torch.save({'weight':effective.cpu()},folder/'selected.pt')
                    m.train()
                if step==2048:break
                ix=rng.integers(len(x),size=64);xx=torch.as_tensor(np.array(x[ix]),device=device)
                yy=torch.as_tensor(np.array(y[ix]).reshape(-1),device=device,dtype=torch.long)
                opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(xx).flatten(0,1),yy);loss.backward();opt.step()
            row=dict(name=name,family=family,lr=lr,seed=seed,best_step=selected,validation_agreement=best,weight_sha256=sha(folder/'selected.pt'))
            dump(folder/'manifest.json',row);dump(folder/'history.json',hist);rows.append(row);del m,opt
    rates=[1e-5,3e-5,1e-4]
    means={str(lr):float(np.mean([r['validation_agreement'] for r in rows if r['family']=='tied' and r['lr']==lr])) for lr in rates}
    chosen=max(rates,key=lambda lr:means[str(lr)])
    dump(dest/'frozen.json',dict(selected_tied_lr=chosen,validation_means=means,initialization_max_abs_error=error,
        selection='Original validation only, before preparing fresh test',runs=rows,freeze_time=time.strftime('%Y-%m-%dT%H:%M:%S%z')))

def fresh_cohort():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    dest=OUT/'fit_control';assert (dest/'frozen.json').exists()
    if (dest/'cohort_manifest.json').exists():return
    tok=AutoTokenizer.from_pretrained(ROOT/'models/CALM-Autoencoder',local_files_only=True)
    source=ROOT/'datasets/openwebtext/plain_text/train-00079-of-00080.parquet'
    seen=set();rows=[];ids=[];index=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=512,columns=['text']):
        texts=batch.column(0).to_pylist();enc=tok(texts,add_special_tokens=False,truncation=True,max_length=256)['input_ids']
        for text,tokens in zip(texts,enc):
            h=hashlib.sha256(text.encode()).hexdigest()
            if len(tokens)==256 and h not in seen:seen.add(h);rows.append(dict(row_index=index,document_sha256=h));ids.append(tokens)
            index+=1
    choice=np.random.default_rng(61102001).permutation(len(rows))[:5376]
    old=[json.loads(l) for l in (OUT/'cohort/documents.jsonl').read_text().splitlines()]
    assert [rows[int(i)]['document_sha256'] for i in choice[:4864]]==[r['document_sha256'] for r in old]
    old_tokens=np.concatenate([np.load(OUT/'cohort'/(s+'.npy')) for s in ['train','validation','test']])
    assert np.array_equal(np.asarray([ids[int(i)] for i in choice[:4864]]),old_tokens), 'Tokenizer version changed source tokens'
    selected=[rows[int(i)] for i in choice[4864:]];assert len(selected)==512
    assert not {r['document_sha256'] for r in old}&{r['document_sha256'] for r in selected}
    np.save(dest/'fresh_ids.npy',np.asarray([ids[int(i)] for i in choice[4864:]],dtype=np.int64))
    (dest/'fresh_documents.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    dump(dest/'cohort_manifest.json',dict(documents=512,disjoint_old_hashes=True,original_prefix_reproduced=True,frozen_sha256=sha(dest/'frozen.json'),cohort_sha256=sha(dest/'fresh_documents.jsonl')))

def fresh_test(device):
    dest=OUT/'fit_control';fresh_cohort();frozen=json.loads((dest/'frozen.json').read_text());lr=frozen['selected_tied_lr']
    ids=np.load(dest/'fresh_ids.npy').reshape(-1,4);ae=load_ae(device);rng=np.random.default_rng(61102202);xx=[];yy=[]
    with torch.inference_mode():
        for start in range(0,len(ids),64):
            e=ae.encoder(torch.as_tensor(ids[start:start+64],device=device))[:,0];mu,ls=e.chunk(2,-1)
            z=mu+ls.exp()*torch.as_tensor(rng.standard_normal(mu.shape).astype('float32'),device=device)
            y,_,_=native(ae,z);xx.append(z.cpu().numpy());yy.append(y.cpu().numpy())
    x=np.concatenate(xx);y=np.concatenate(yy);del ae
    preds={}
    for family in ['ridge','free','tied']:
        for seed in ([17] if family=='ridge' else [17,29,43]):
            m=Readout(np.load(OUT/'ridge_weights.npy')).to(device)
            if family!='ridge':
                rate=1e-5 if family=='free' else lr
                m.load_state_dict(torch.load(dest/f'{family}_lr{rate}_seed{seed}/selected.pt',map_location=device,weights_only=True))
            m.eval();preds[f'{family}_{seed}']=predict(m,x,device)[0];del m
    np.savez_compressed(dest/'fresh_predictions.npz',native=y,reference=ids,**preds)
    boot=np.random.default_rng(61102209).integers(512,size=(4000,512));base=(preds['ridge_17']==y).reshape(512,-1).mean(1);rows=[]
    for family in ['ridge','free','tied']:
        ss=[17] if family=='ridge' else [17,29,43]
        a=np.stack([(preds[f'{family}_{s}']==y).reshape(512,-1).mean(1) for s in ss]).mean(0)
        gain=a-base
        count=np.load(OUT/'token_counts.npy')[y];groups={}
        for tag,mask in [('unseen',count==0),('seen_below100',(count>0)&(count<100)),('at_least100',count>=100)]:
            groups[tag]=dict(tokens=int(mask.sum()),agreement=float(np.mean([(preds[f'{family}_{s}'][mask]==y[mask]).mean() for s in ss])) if mask.any() else None)
        rows.append(dict(family=family,agreement=float(a.mean()),ci=np.quantile(a[boot].mean(1),[.025,.975]).tolist(),gain_over_ridge=float(gain.mean()),gain_ci=np.quantile(gain[boot].mean(1),[.025,.975]).tolist(),
            individual_agreements={str(s):float((preds[f'{family}_{s}']==y).mean()) for s in ss},frequency_groups=groups))
    dump(dest/'summary.json',dict(documents=512,native_reference=float((y==ids).mean()),selected_tied_lr=lr,results=rows))
    print('fresh summary',rows,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','fresh']);p.add_argument('--device',default='cuda:2');args=p.parse_args();setup(args.device)
    if args.mode=='fit':fit_controls(args.device)
    else:fresh_test(args.device)
