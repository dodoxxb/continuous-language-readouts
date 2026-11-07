"""Fixed-document CALM compressed-interface audit. One bounded GPU worker."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/new_interfaces_20261102/calm'
sys.path.insert(0,str(ROOT/'third_party/CALM'))
if os.environ.get('CALM_SYSTEM_TRANSFORMERS')!='1':
    sys.path.insert(0,str(ROOT/'outputs/new_interfaces_20261102/deps'))

def dump(p,j):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def setup(device):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if device.startswith('cuda'):
        torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
        if free<14*2**30:raise RuntimeError('Need 14 GiB free before starting bounded worker')
        torch.cuda.set_per_process_memory_fraction(12*2**30/total,device)
    p=ROOT/'outputs/new_interfaces_20261102/protocol/protocol.json'
    assert sha(p)==json.loads(p.with_name('frozen.json').read_text())['sha256']

def cohort():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    dest=OUT/'cohort'
    if (dest/'manifest.json').exists():return
    tok=AutoTokenizer.from_pretrained(ROOT/'models/CALM-Autoencoder',local_files_only=True)
    source=ROOT/'datasets/openwebtext/plain_text/train-00079-of-00080.parquet'
    seen=set();rows=[];ids=[];index=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=512,columns=['text']):
        texts=batch.column(0).to_pylist()
        encoded=tok(texts,add_special_tokens=False,truncation=True,max_length=256)['input_ids']
        for text,tokens in zip(texts,encoded):
            h=hashlib.sha256(text.encode()).hexdigest()
            if len(tokens)==256 and h not in seen:
                seen.add(h);rows.append(dict(row_index=index,document_sha256=h));ids.append(tokens)
            index+=1
    choice=np.random.default_rng(61102001).permutation(len(rows))[:4864]
    assert len(choice)==4864
    dest.mkdir(parents=True,exist_ok=True)
    selected=[]
    for name,lo,hi in [('train',0,4096),('validation',4096,4352),('test',4352,4864)]:
        ix=choice[lo:hi];np.save(dest/(name+'.npy'),np.asarray([ids[i] for i in ix],dtype=np.int64))
        selected.extend(dict(split=name,split_index=j,**rows[int(i)]) for j,i in enumerate(ix))
    (dest/'documents.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in selected))
    dump(dest/'manifest.json',dict(source=str(source.relative_to(ROOT)),source_sha256=sha(source),eligible=len(rows),
        source_rows=index,seed=61102001,counts=dict(train=4096,validation=256,test=512),
        unique_hashes=len({r['document_sha256'] for r in selected}),documents_sha256=sha(dest/'documents.jsonl')))
    print('cohort',len(selected),flush=True)

def load_ae(device):
    from models.modeling_autoencoder import Autoencoder
    from models.configuration_autoencoder import AutoencoderConfig
    from safetensors.torch import load_file
    from safetensors import safe_open
    cfg=AutoencoderConfig.from_pretrained(ROOT/'models/CALM-Autoencoder',local_files_only=True)
    path=ROOT/'models/CALM-Autoencoder/model.safetensors'
    state=load_file(path)
    with safe_open(path,framework='pt',device='cpu') as f:meta=f.metadata()
    # The official safetensors conversion stores only one tied weight. Restore
    # its exact alias, then require a strict load: never accept random weights.
    if 'encoder.embed_tokens.weight' not in state:
        assert meta.get('encoder.embed_tokens.weight')=='decoder.lm_head_weight',meta
        state['encoder.embed_tokens.weight']=state['decoder.lm_head_weight']
    model=Autoencoder(cfg)
    model.load_state_dict(state,strict=True)
    assert model.decoder.lm_head_weight.data_ptr()==model.encoder.embed_tokens.weight.data_ptr()
    assert torch.equal(model.encoder.embed_tokens.weight,state['encoder.embed_tokens.weight'])
    return model.float().to(device).eval()

@torch.inference_mode()
def native(ae,x):
    features=[]
    hook=ae.decoder.norm.register_forward_hook(lambda m,a,o:features.append(o))
    try:logits=ae.decoder(x[:,None,:])
    finally:hook.remove()
    values=logits.topk(2,-1).values
    return logits.argmax(-1),features[0],(values[...,0]-values[...,1])

def prepare(device):
    assert (OUT/'cohort/manifest.json').exists(), 'Prepare the immutable cohort first'
    ae=load_ae(device);all_stats={}
    for si,split in enumerate(['train','validation','test']):
        dest=OUT/'data'/split
        if (dest/'manifest.json').exists():continue
        dest.mkdir(parents=True,exist_ok=True)
        ids=np.load(OUT/'cohort'/(split+'.npy')).reshape(-1,4);n=len(ids)
        rng=np.random.default_rng(61102002+si)
        arrays={name:np.lib.format.open_memmap(dest/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
            for name,dtype,shape in [('mean','float32',(n,128)),('std','float32',(n,128)),('x','float32',(n,128)),
                                     ('native','int32',(n,4)),('margin','float32',(n,4))]}
        gram=np.zeros((129,129),np.float64);rhs=np.zeros((129,2048),np.float64)
        count=np.zeros(ae.config.vocab_size,np.int64);correct=0
        for start in range(0,n,256):
            end=min(start+256,n)
            with torch.inference_mode():
                enc=ae.encoder(torch.as_tensor(ids[start:end],device=device))[:,0]
                mean,logstd=enc.chunk(2,-1);sd=logstd.exp()
                noise=torch.as_tensor(rng.standard_normal(mean.shape).astype('float32'),device=device)
                x=mean+sd*noise
                # Stream vocabulary logits in 64-chunk slices.
                ys=[];hs=[];ms=[]
                for k in range(0,len(x),64):
                    y,h,m=native(ae,x[k:k+64]);ys.append(y.cpu().numpy());hs.append(h.cpu().numpy());ms.append(m.cpu().numpy())
                y=np.concatenate(ys);h=np.concatenate(hs);m=np.concatenate(ms)
                z=x.cpu().numpy()
                for key,val in [('mean',mean.cpu().numpy()),('std',sd.cpu().numpy()),('x',z),('native',y),('margin',m)]:arrays[key][start:end]=val
            correct+=int((y==ids[start:end]).sum());count+=np.bincount(y.ravel(),minlength=len(count))
            if split=='train':
                xx=np.c_[z,np.ones(len(z))].astype('float64');hh=h.reshape(len(h),-1).astype('float64')
                gram+=xx.T@xx;rhs+=xx.T@hh
            if start%32768==0:print('prepared',split,start,n,flush=True)
        for a in arrays.values():a.flush()
        if split=='train':
            reg=1e-3*np.diag(gram)[:-1].mean();penalty=np.eye(129)*reg;penalty[-1,-1]=0
            coef=np.linalg.solve(gram+penalty,rhs).reshape(129,4,512).transpose(1,2,0)
            E=ae.encoder.embed_tokens.weight.detach().cpu().numpy().astype('float64')
            weights=np.stack([E@coef[p] for p in range(4)]).astype('float32')
            np.save(OUT/'ridge_weights.npy',weights)
            np.save(OUT/'token_counts.npy',count)
            dump(OUT/'ridge.json',dict(alpha=reg,condition_number=float(np.linalg.cond(gram+penalty))))
        dump(dest/'manifest.json',dict(chunks=n,documents=n//64,reference_reconstruction=correct/(n*4),
            x_sha256=sha(dest/'x.npy'),native_sha256=sha(dest/'native.npy'),source_ids_sha256=sha(OUT/'cohort'/(split+'.npy'))))
    print('prepared all',flush=True)

class Readout(torch.nn.Module):
    def __init__(self,weights):
        super().__init__();self.weight=torch.nn.Parameter(torch.as_tensor(weights.copy()))
    def forward(self,x):
        z=torch.cat((x,torch.ones_like(x[:,:1])),-1)
        return torch.einsum('nd,pvd->npv',z,self.weight)

@torch.inference_mode()
def predict(model,x,device,reference=None):
    out=[];loss=0
    for i in range(0,len(x),64):
        logits=model(torch.as_tensor(np.array(x[i:i+64]),device=device))
        out.append(logits.argmax(-1).cpu().numpy().astype('int32'))
        if reference is not None:
            loss+=float(F.cross_entropy(logits.flatten(0,1),torch.as_tensor(reference[i:i+64].reshape(-1),device=device,dtype=torch.long),reduction='sum'))
    return np.concatenate(out),loss/(4*len(x))

def fit(device,seed):
    dest=OUT/'fits'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():return
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    x=np.load(OUT/'data/train/x.npy',mmap_mode='r');y=np.load(OUT/'data/train/native.npy',mmap_mode='r')
    xv=np.load(OUT/'data/validation/x.npy',mmap_mode='r');yv=np.load(OUT/'data/validation/native.npy')
    model=Readout(np.load(OUT/'ridge_weights.npy')).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    best=-1.;history=[];t=time.monotonic()
    for step in range(2049):
        if step in [0,512,1024,2048]:
            model.eval();pred,ce=predict(model,xv,device,yv);acc=float((pred==yv).mean())
            row=dict(step=step,validation_agreement=acc,validation_ce=ce,elapsed=time.monotonic()-t);history.append(row)
            print('fit',seed,row,flush=True)
            if acc>best:
                best=acc;best_step=step;torch.save(model.state_dict(),dest/'selected.pt')
            model.train()
        if step==2048:break
        ix=rng.integers(len(x),size=64);xx=torch.as_tensor(np.array(x[ix]),device=device)
        yy=torch.as_tensor(np.array(y[ix]).reshape(-1),device=device,dtype=torch.long)
        opt.zero_grad(set_to_none=True);logits=model(xx)
        loss=F.cross_entropy(logits.flatten(0,1),yy);loss.backward();opt.step()
        if step%256==0:print('update',seed,step,float(loss),flush=True)
    dump(dest/'history.json',history)
    dump(dest/'manifest.json',dict(seed=seed,best_step=best_step,best_validation_agreement=best,
        selected_sha256=sha(dest/'selected.pt'),elapsed=time.monotonic()-t,max_allocated_gib=torch.cuda.max_memory_allocated()/2**30))

def test(device):
    dest=OUT/'test';dest.mkdir(exist_ok=True)
    ae=load_ae(device)
    raw=np.load(OUT/'data/test/x.npy');mu=np.load(OUT/'data/test/mean.npy')
    refs=np.load(OUT/'cohort/test.npy').reshape(-1,4)
    noise=np.random.default_rng(61102012).standard_normal(mu.shape).astype('float32')
    conditions={'posterior':raw,'mean':mu,'noise025':mu+.25*noise,'radial05':.5*raw,'radial2':2*raw}
    records=[];counts=np.load(OUT/'token_counts.npy');native_base=np.load(OUT/'data/test/native.npy')
    for name,x in conditions.items():
        path=dest/(name+'.npz')
        if path.exists():continue
        yy=[];margin=[]
        with torch.inference_mode():
            for i in range(0,len(x),64):
                y,_,m=native(ae,torch.as_tensor(x[i:i+64],device=device));yy.append(y.cpu().numpy());margin.append(m.cpu().numpy())
        y=np.concatenate(yy);m=np.concatenate(margin)
        if name=='posterior':assert np.array_equal(y,native_base)
        preds={}
        for seed in ['ridge',17,29,43]:
            model=Readout(np.load(OUT/'ridge_weights.npy')).to(device)
            if seed!='ridge':model.load_state_dict(torch.load(OUT/f'fits/seed{seed}/selected.pt',weights_only=True,map_location=device))
            model.eval();pred,_=predict(model,x,device);preds[str(seed)]=pred
            del model
        np.savez_compressed(path,native=y,margin=m,reference=refs,native_base=native_base,**preds)
        print('tested',name,{k:float((v==y).mean()) for k,v in preds.items()},flush=True)
    analyze()

def analyze():
    records=[];summary=[];rng=np.random.default_rng(61102009);boot=rng.integers(512,size=(4000,512))
    counts=np.load(OUT/'token_counts.npy')
    for p in sorted((OUT/'test').glob('*.npz')):
        d=np.load(p);y=d['native'];ref=d['reference'];n=len(y)//64
        if n!=512:continue
        for seed in ['ridge','17','29','43']:
            pred=d[seed];correct=(pred==y).reshape(n,-1);chunk=(pred==y).all(-1).reshape(n,-1)
            for i in range(n):records.append(dict(condition=p.stem,fit=seed,document=i,native_agreement=float(correct[i].mean()),
                chunk_exact=float(chunk[i].mean()),native_reference=float((y.reshape(n,-1)[i]==ref.reshape(n,-1)[i]).mean()),
                native_radial_preservation=float((y.reshape(n,-1)[i]==d['native_base'].reshape(n,-1)[i]).mean())))
            a=correct.mean(1);b=chunk.mean(1)
            row=dict(condition=p.stem,fit=seed,token_agreement=float(a.mean()),token_ci=np.quantile(a[boot].mean(1),[.025,.975]).tolist(),
                chunk_agreement=float(b.mean()),chunk_ci=np.quantile(b[boot].mean(1),[.025,.975]).tolist(),
                native_reference=float((y==ref).mean()),native_radial_preservation=float((y==d['native_base']).mean()))
            for tag,mask in [('unseen',counts[y]==0),('rare',counts[y]<100),('common',counts[y]>=100)]:
                row[tag+'_tokens']=int(mask.sum());row[tag+'_agreement']=float((pred[mask]==y[mask]).mean()) if mask.any() else None
            summary.append(row)
        a=np.stack([(d[str(s)]==y).reshape(n,-1).mean(1) for s in [17,29,43]]).mean(0)
        b=np.stack([(d[str(s)]==y).all(-1).reshape(n,-1).mean(1) for s in [17,29,43]]).mean(0)
        summary.append(dict(condition=p.stem,fit='three_fit_mean',token_agreement=float(a.mean()),token_ci=np.quantile(a[boot].mean(1),[.025,.975]).tolist(),chunk_agreement=float(b.mean()),chunk_ci=np.quantile(b[boot].mean(1),[.025,.975]).tolist(),native_reference=float((y==ref).mean()),native_radial_preservation=float((y==d['native_base']).mean())))
    with (OUT/'per_document.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    dump(OUT/'summary.json',summary)
    print('summary',[(r['condition'],round(r['token_agreement'],4)) for r in summary if r['fit']=='three_fit_mean'],flush=True)

def generated(device):
    """Official base generator, teacher prefixes, exact same latent for all readouts."""
    from models.modeling_energy import EnergyTransformer
    from models.configuration_calm import CALMConfig
    from models.modeling_autoencoder import Autoencoder
    path=ROOT/'models/CALM-M';cfg=CALMConfig.from_pretrained(path,local_files_only=True)
    cfg.ae_path=str(ROOT/'models/CALM-Autoencoder')
    # Restore the tied AE alias explicitly during construction as above.
    assert 'from_pretrained' not in Autoencoder.__dict__
    Autoencoder.from_pretrained=classmethod(lambda cls,*args,**kwargs:load_ae('cpu'))
    try:model=EnergyTransformer(cfg)
    finally:del Autoencoder.from_pretrained
    state=torch.load(path/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    assert model.generative_head is model.mlp_generator
    aliases={k.replace('mlp_generator.','generative_head.',1):v for k,v in state.items() if k.startswith('mlp_generator.')}
    for k,v in aliases.items():
        if k in state:assert torch.equal(state[k],v)
        else:state[k]=v
    model.load_state_dict(state,strict=True);del state
    dump(ROOT/'outputs/new_interfaces_20261102/source/calm_m_load.json',dict(strict_load_passed=True,shared_generator_aliases=len(aliases),generator_is_same_module=True,source='Official upstream generative_head is an alias of mlp_generator; checkpoint stores only mlp_generator keys'))
    model=model.float().to(device).eval();torch.manual_seed(61102021)
    ids=np.load(OUT/'cohort/test.npy');x=[];native_labels=[]
    with torch.inference_mode():
        for start in range(0,len(ids),8):
            # All 64 next-chunk states in each held-out document, teacher prefixes.
            t=torch.as_tensor(ids[start:start+8,:252],device=device)
            emb=model.transformer.embed_tokens(t).reshape(len(t),63,-1)
            hidden=model.transformer(inputs_embeds=model.embed_proj(emb),use_cache=False)[0]
            z=model.generative_head.sample(hidden).float().reshape(-1,128)
            x.append(z.cpu().numpy())
            for j in range(0,len(z),64):native_labels.append(model.ae_model.decoder(z[j:j+64,None,:]).argmax(-1).cpu().numpy())
    x=np.concatenate(x);y=np.concatenate(native_labels);del model
    preds={}
    for seed in ['ridge',17,29,43]:
        read=Readout(np.load(OUT/'ridge_weights.npy')).to(device)
        if seed!='ridge':read.load_state_dict(torch.load(OUT/f'fits/seed{seed}/selected.pt',map_location=device,weights_only=True))
        read.eval();preds[str(seed)]=predict(read,x,device)[0];del read
    control=OUT/'fit_control/frozen.json'
    if control.exists():
        lr=json.loads(control.read_text())['selected_tied_lr']
        for seed in [17,29,43]:
            read=Readout(np.load(OUT/'ridge_weights.npy')).to(device)
            read.load_state_dict(torch.load(OUT/f'fit_control/tied_lr{lr}_seed{seed}/selected.pt',map_location=device,weights_only=True))
            read.eval();preds['tied'+str(seed)]=predict(read,x,device)[0];del read
    np.savez_compressed(OUT/'generated_predictions.npz',x=x,native=y,**preds)
    rng=np.random.default_rng(61102022);boot=rng.integers(512,size=(4000,512));summary=[]
    for seed,p in preds.items():
        a=(p==y).reshape(512,-1).mean(1);c=(p==y).all(-1).reshape(512,-1).mean(1)
        summary.append(dict(fit=seed,agreement=float(a.mean()),ci=np.quantile(a[boot].mean(1),[.025,.975]).tolist(),chunk_exact=float(c.mean())))
    pooled=np.stack([(preds[str(s)]==y).reshape(512,-1).mean(1) for s in [17,29,43]]).mean(0)
    summary.append(dict(fit='three_fit_mean',agreement=float(pooled.mean()),ci=np.quantile(pooled[boot].mean(1),[.025,.975]).tolist(),chunk_exact=float(np.stack([(preds[str(s)]==y).all(-1).reshape(512,-1).mean(1) for s in [17,29,43]]).mean())))
    if control.exists():
        tp=np.stack([(preds['tied'+str(s)]==y).reshape(512,-1).mean(1) for s in [17,29,43]]).mean(0)
        delta=tp-(preds['ridge']==y).reshape(512,-1).mean(1)
        summary.append(dict(fit='tied_three_fit_mean',agreement=float(tp.mean()),ci=np.quantile(tp[boot].mean(1),[.025,.975]).tolist(),gain_over_ridge=float(delta.mean()),gain_ci=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),chunk_exact=float(np.stack([(preds['tied'+str(s)]==y).all(-1).reshape(512,-1).mean(1) for s in [17,29,43]]).mean())))
    dump(OUT/'generated_summary.json',dict(scope='Teacher-prefix-conditioned one-step latent predictions; not free-running generation or reference correctness',documents=512,chunks_per_document=63,results=summary,torch=torch.__version__))

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['cohort','prepare','fit','test','analyze','generated']);a.add_argument('--seed',type=int,default=17);a.add_argument('--device',default='cuda:2');args=a.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode not in ['analyze','cohort']:setup(args.device)
    else:torch.set_num_threads(2)
    if args.mode=='cohort':cohort()
    elif args.mode=='prepare':prepare(args.device)
    elif args.mode=='fit':fit(args.device,args.seed)
    elif args.mode=='test':test(args.device)
    elif args.mode=='generated':generated(args.device)
    else:analyze()

if __name__=='__main__':main()
