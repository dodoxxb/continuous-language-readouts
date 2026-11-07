"""Exploratory, equal-parameter context-residual intervention on frozen readouts.

Original train/validation splits only fit/select adapters. Previously examined
OWT/XSum tests are reused and explicitly remain exploratory. No native decoder
labels or margins enter adapter features; evaluation uses native lexical masks.
"""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,os,time
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/deep_mechanism_20261022/context'
ARMS=['focal','context','shuffled'];SEEDS=[71,73]

def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,**a):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.with_name(p.name+'.tmp').open('wb') as f:np.savez(f,**a)
    p.with_name(p.name+'.tmp').replace(p)
def writecsv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def basepath(f):
    return ROOT/('outputs/confirmatory_readout_20261013/fits/elf/n16384_seed17/selected.pt' if f=='elf'
                 else 'outputs/aurora_replication_20261022/fits/seed17/selected.pt')
def datadir(f,split):
    return ROOT/(f'outputs/confirmatory_readout_20261013/data/elf/{split}' if f=='elf' else
                 f'outputs/aurora_replication_20261022/data/{"training" if split=="train" else split}')
def protocol():
    p=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      question='Can preceding endpoint context explain residual decisions beyond equal additional focal-only parameters?',
      status='Exploratory new intervention on previously inspected test cohorts; not a new prospective confirmation.',
      families=['elf','aurora'],base='Previously frozen seed17 affine OWT readout, all weights frozen',
      formula='logits_i = W x_i + b + U V standardized(q_i) + c; rank=8',
      arms=dict(focal='q_i=x_i',context='mean of up to four immediately preceding full-canvas endpoint states, excluding focal; zero at position0',
                shuffled='same context feature replaced by an independently chosen other-sequence context during training and a fixed other-sequence context during evaluation'),
      boundaries='Context uses full-canvas positions, including source positions on XSum and latent positions whose native token may be special. Native labels/margins never select features. All compared outputs use the same native lexical evaluation mask.',
      training=dict(tokens=131072,selection='Uniform with replacement over original valid training tokens; deterministic seed61022912; original raw ELF shards recover exact full-canvas preceding states',
        seeds=SEEDS,rank=8,updates=1024,batch_tokens=256,lr=.0005,weight_decay=.01,init='V normal std 1/sqrt(D), U and output bias zero',
        normalization='Feature dimension means/SD from sampled training features only; context and shuffled share statistics',
        validation_tokens=16384,validation_checkpoints=[0,256,512,1024],selection_rule='Maximum native agreement on fixed original validation tokens; earliest checkpoint on ties; baseline step0 eligible',
        precision='FP32 parameters and evaluation; BF16 training matrix multiplications; TF32 disabled'),
      evaluation='All original prospective OWT conditions/seeds and all4096 XSum generations; no XSum fitting, selection, or retuning. Store full selected predictions.',
      primary_diagnostics=['context minus focal native agreement, macro per sequence','context minus shuffled agreement','rare-token native agreement and corrected-versus-new errors'],
      inference='Exploratory descriptive paired bootstrap; average two adapter fits; preserve condition/seed and paired XSum document structure. No quality or acceleration claim from fidelity alone.',
      limits='Rank8 and four preceding states test a narrow cheap context model, not all contextual decoders or the full diffusion mechanism. Prior tests were already observed. Additional fit seeds are not independently trained generators.',
      resources='One low-priority GPU worker,3.5GiB allocator ceiling,2CPUthreads,nice10; no denoiser training/generation.',
      base_sha256={f:sha(basepath(f)) for f in ['elf','aurora']})
    path=OUT/'protocol.json'
    if not path.exists():dump(path,p)
    if not (OUT/'protocol_frozen.json').exists():
        dump(OUT/'protocol_frozen.json',dict(protocol_sha256=sha(path),code_sha256=sha(Path(__file__)),new_fits_existing=False))
    return json.loads(path.read_text())

def context_rows(x,pos):
    # x is one full sequence. Never compress positions by native labels.
    out=np.zeros((len(pos),x.shape[-1]),np.float32)
    for lag in range(1,5):
        ok=pos>=lag;out[ok]+=np.asarray(x[pos[ok]-lag],np.float32)
    return (out/np.maximum(np.minimum(pos,4),1)[:,None]).astype(np.float32)

def sample_data(f,split,n,seed):
    rng=np.random.default_rng(seed);d=datadir(f,split)
    if f=='elf' and split=='train':
        off=np.load(d/'offsets.npy');flat=rng.integers(int(off[-1]),size=n)
        s=np.searchsorted(off,flat,side='right')-1;k=flat-off[s]
        ids=np.load(d/'sequence_ids.npy');original=ids[s]
        info=json.loads((d.parent/'ready.json').read_text());D=info['dim']
        xx=np.empty((n,D),np.float32);cc=np.empty_like(xx);yy=np.empty(n,np.int64);pp=np.empty(n,np.int64)
        first=0
        for rec in info['source_shards']:
            pack=torch.load(ROOT/rec['path'],map_location='cpu',mmap=True,weights_only=False)
            size=len(pack['labels']);ix=np.flatnonzero((original>=first)&(original<first+size))
            for local in np.unique(original[ix]-first):
                js=ix[original[ix]-first==local];valid=pack['valid'][local].numpy()
                pos=np.flatnonzero(valid)[k[js]];a=pack['final_z'][local].numpy()
                xx[js]=a[pos];cc[js]=context_rows(a,pos);yy[js]=pack['labels'][local].numpy()[pos];pp[js]=pos
            first+=size;del pack
            print('prepare',f,split,'raw shard end',first,flush=True)
        return dict(x=xx,context=cc,y=yy,sequence=s,position=pp)
    x=np.load(d/'x.npy',mmap_mode='r');y=np.load(d/'y.npy')
    v=np.load(d/'valid.npy') if f=='elf' else y!=50256
    eligible=np.flatnonzero(v);flat=eligible[rng.integers(len(eligible),size=n)]
    seq,pos=np.unravel_index(flat,y.shape)
    xx=np.empty((n,x.shape[-1]),np.float32);cc=np.empty_like(xx)
    for i in np.unique(seq):
        js=np.flatnonzero(seq==i);xx[js]=x[i,pos[js]];cc[js]=context_rows(x[i],pos[js])
    return dict(x=xx,context=cc,y=y[seq,pos].astype(np.int64),sequence=seq,position=pos)

def prepare(f):
    p=protocol();dest=OUT/f;dest.mkdir(parents=True,exist_ok=True)
    for split,n,seed in [('train',131072,61022912),('validation',16384,61022913)]:
        path=dest/f'{split}.npz'
        if path.exists():continue
        q=sample_data(f,split,n,seed)
        rng=np.random.default_rng(seed+1);donor=rng.integers(n,size=n)
        same=q['sequence'][donor]==q['sequence']
        while same.any():
            donor[same]=rng.integers(n,size=int(same.sum()));same=q['sequence'][donor]==q['sequence']
        q['donor']=donor;save(path,**q)
    q=np.load(dest/'train.npz');stats={}
    if f=='elf':
        flat=datadir(f,'train');off=np.load(flat/'offsets.npy')
        # Independently reconstruct sampled flat rows from the frozen RNG.
        ix=np.random.default_rng(61022912).integers(int(off[-1]),size=len(q['y']))
        original_x=np.load(flat/'x.npy',mmap_mode='r');original_y=np.load(flat/'y.npy',mmap_mode='r')
        assert np.array_equal(q['y'],original_y[ix]),'Raw/flat training-label mismatch'
        check=np.linspace(0,len(ix)-1,1024,dtype=np.int64)
        cached_x=q['x']
        assert np.array_equal(cached_x[check],original_x[ix[check]]),'Raw/flat sampled state mismatch'
    assert np.all(q['sequence'][q['donor']]!=q['sequence'])
    assert all(np.isfinite(q[k]).all() for k in ['x','context'])
    for arm in ['focal','context']:
        a=q['x' if arm=='focal' else 'context'].astype(np.float64)
        stats[arm]=dict(mean=a.mean(0).tolist(),std=np.maximum(a.std(0),1e-4).tolist())
    dump(dest/'normalization.json',stats)
    dump(dest/'data_manifest.json',dict(train_sha256=sha(dest/'train.npz'),validation_sha256=sha(dest/'validation.npz'),
        train_tokens=len(q['y']),original_training_only=True,full_canvas_context=True,
        raw_flat_labels_verified=len(q['y']) if f=='elf' else None,
        raw_flat_feature_rows_verified=1024 if f=='elf' else None,protocol_sha256=sha(OUT/'protocol.json')))

class Adapter(torch.nn.Module):
    def __init__(self,d,v,mean,std,device):
        super().__init__();self.v=torch.nn.Linear(d,8,bias=False,device=device)
        self.u=torch.nn.Linear(8,v,bias=True,device=device)
        torch.nn.init.normal_(self.v.weight,std=d**-.5);torch.nn.init.zeros_(self.u.weight);torch.nn.init.zeros_(self.u.bias)
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32,device=device))
        self.register_buffer('std',torch.tensor(std,dtype=torch.float32,device=device))
    def forward(self,q):return self.u(self.v((q-self.mean)/self.std))

def setup(device):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
    assert free>5.5*2**30,'Not enough shared GPU headroom'
    torch.cuda.set_per_process_memory_fraction(3.5*2**30/total,device)
    return torch.cuda.Stream(device=device,priority=1)
def base(f,device):
    state=torch.load(basepath(f),weights_only=True,map_location=device)
    model=torch.nn.Linear(state['weight'].shape[1],state['weight'].shape[0],device=device)
    model.load_state_dict(state);model.requires_grad_(False);return model.eval()
def feature(z,arm,idx):
    return z['x'][idx] if arm=='focal' else z['context'][z['donor'][idx] if arm=='shuffled' else idx]

@torch.inference_mode()
def validation(b,a,z,arm,device):
    correct=0
    for i in range(0,len(z['y']),256):
        idx=np.arange(i,min(i+256,len(z['y'])))
        x=torch.as_tensor(z['x'][idx],device=device);q=torch.as_tensor(feature(z,arm,idx),device=device)
        pred=(b(x)+a(q)).argmax(-1).cpu().numpy();correct+=int((pred==z['y'][idx]).sum())
    return correct/len(z['y'])

def fit(f,device):
    protocol();stream=setup(device);b=base(f,device)
    dest=OUT/f;tr=dict(np.load(dest/'train.npz'));va=dict(np.load(dest/'validation.npz'))
    stats=json.loads((dest/'normalization.json').read_text())
    for seed in SEEDS:
        for arm in ARMS:
            folder=dest/'fits'/f'{arm}_{seed}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'selected.json').exists():continue
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            st=stats['focal' if arm=='focal' else 'context'];a=Adapter(b.in_features,b.out_features,st['mean'],st['std'],device)
            opt=torch.optim.AdamW(a.parameters(),lr=.0005,weight_decay=.01)
            stream.wait_stream(torch.cuda.current_stream(device))
            best=-1.;history=[];started=time.monotonic()
            for step in range(1025):
                with torch.cuda.stream(stream):
                    if step:
                        idx=rng.integers(len(tr['y']),size=256)
                        x=torch.as_tensor(tr['x'][idx],device=device);q=torch.as_tensor(feature(tr,arm,idx),device=device)
                        y=torch.as_tensor(tr['y'][idx],device=device)
                        with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(b(x)+a(q),y)
                        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                    if step in [0,256,512,1024]:
                        score=validation(b,a,va,arm,device);history.append(dict(step=step,agreement=score))
                        if score>best:
                            best=score;beststep=step;torch.save({k:v.detach().cpu() for k,v in a.state_dict().items()},folder/'selected.pt')
                        dump(folder/'history.json',history)
                        print(f,arm,seed,step,'val',score,'elapsed',round(time.monotonic()-started,1),flush=True)
                if step%32==0:time.sleep(.03)
            dump(folder/'selected.json',dict(family=f,arm=arm,seed=seed,step=beststep,validation=best,
                added_parameters=sum(p.numel() for p in a.parameters()),sha256=sha(folder/'selected.pt'),
                max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,elapsed=time.monotonic()-started))
            del a,opt;torch.cuda.empty_cache()

def load_adapters(f,b,device):
    result={};stats=json.loads((OUT/f/'normalization.json').read_text())
    for seed in SEEDS:
        for arm in ARMS:
            s=stats['focal' if arm=='focal' else 'context'];m=Adapter(b.in_features,b.out_features,s['mean'],s['std'],device)
            m.load_state_dict(torch.load(OUT/f/'fits'/f'{arm}_{seed}'/'selected.pt',map_location=device,weights_only=True))
            result[f'{arm}_{seed}']=m.eval()
    return result

@torch.inference_mode()
def predict(b,models,x,context,donor,device):
    out={k:np.empty(len(x),np.int32) for k in ['base']+list(models)}
    for start in range(0,len(x),256):
        s=slice(start,start+256);xx=torch.as_tensor(np.array(x[s],copy=True),device=device)
        logits=b(xx);out['base'][s]=logits.argmax(-1).cpu().numpy()
        qs={'focal':xx,'context':torch.as_tensor(context[s],device=device),'shuffled':torch.as_tensor(donor[s],device=device)}
        for key,m in models.items():out[key][s]=(logits+m(qs[key.split('_')[0]])).argmax(-1).cpu().numpy()
    return out

def evaluate(f,device,scope):
    protocol();stream=setup(device);b=base(f,device);models=load_adapters(f,b,device)
    stream.wait_stream(torch.cuda.current_stream(device))
    countpath=ROOT/('outputs/prospective_tail_20261013/token_counts.npy' if f=='elf' else 'outputs/aurora_replication_20261022/counts.npy')
    counts=np.load(countpath);rows=[];dest=OUT/f/'evaluation'/scope;dest.mkdir(parents=True,exist_ok=True)
    def record(pred,y,valid,meta):
        rare=valid&(counts[y]<100);err=(pred['base']!=y)&valid
        for key,z in pred.items():
            rows.append(dict(**meta,arm=key,tokens=int(valid.sum()),correct=int(((z==y)&valid).sum()),
              rare_tokens=int(rare.sum()),rare_correct=int(((z==y)&rare).sum()),
              fixed=int(((z==y)&err).sum()),introduced=int(((z!=y)&~err&valid).sum())))
    if scope=='owt':
        dirs=sorted((ROOT/'outputs/prospective_tail_20261013/test').glob('*/*')) if f=='elf' else sorted((ROOT/'outputs/aurora_replication_20261022/data/test').glob('*/*'))
        for d in dirs:
            if not (d/'x.npy').exists():continue
            x=np.load(d/'x.npy',mmap_mode='r')
            if f=='elf':
                zz=np.load(d/'tokens.npz');y=zz['native'];valid=zz['valid'];expected=zz['linear_seed17']
            else:
                y=np.load(d/'y.npy');valid=y!=50256;expected=np.load(d/'linear17.npy')
            target=dest/f'{d.parent.name}_{d.name}.npz'
            if target.exists():predall=dict(np.load(target))
            else:
                predall={k:np.empty(y.shape,np.int32) for k in ['base']+list(models)}
                for i in range(len(x)):
                    pos=np.arange(x.shape[1]);ctx=context_rows(x[i],pos);donor=context_rows(x[(i+len(x)//2)%len(x)],pos)
                    with torch.cuda.stream(stream):pred=predict(b,models,x[i],ctx,donor,device)
                    for key,value in pred.items():predall[key][i]=value
                    if i%32==31:print('evaluate',f,scope,d.parent.name,d.name,i+1,flush=True);time.sleep(.1)
                assert np.array_equal(predall['base'],expected),'FP32 frozen readout parity failed'
                save(target,**predall)
            for i in range(len(y)):
                record({k:v[i] for k,v in predall.items()},y[i],valid[i],dict(family=f,scope=scope,condition=d.parent.name,generation_seed=int(d.name),sequence=i))
    else:
        task=ROOT/'outputs/task_readout_20261022'
        for sd in sorted((task/'generated'/f).iterdir()):
            paths=sorted(sd.glob('[0-9][0-9][0-9][0-9].npz'));n=len(paths)
            for i,path in enumerate(paths):
                target=dest/sd.name/path.name;z=np.load(path);meta=json.loads(path.with_suffix('.json').read_text())
                full=z['full_x'];p=int(meta['prefix_length']);pos=np.arange(p,p+len(z['x']))
                if target.exists():pred=dict(np.load(target))
                else:
                    other=np.load(paths[(i+n//2)%n]);om=json.loads(paths[(i+n//2)%n].with_suffix('.json').read_text())
                    op=np.minimum(np.arange(len(pos))+int(om['prefix_length']),len(other['full_x'])-1)
                    ctx=context_rows(full,pos);donor=context_rows(other['full_x'],op)
                    with torch.cuda.stream(stream):pred=predict(b,models,z['x'],ctx,donor,device)
                    assert np.array_equal(pred['base'],z['linear17']),'Task frozen readout parity failed'
                    save(target,**pred)
                valid=np.load(task/'texts'/f/sd.name/path.name)['valid']
                record(pred,z['native'],valid,dict(family=f,scope=scope,condition='XSum',generation_seed=int(sd.name),sequence=int(path.stem)))
                if i%64==63:print('evaluate',f,scope,sd.name,i+1,flush=True);time.sleep(.1)
    writecsv(dest/'per_sequence.csv',rows)
    dump(dest/'complete.json',dict(family=f,scope=scope,rows=len(rows),max_allocated_gib=torch.cuda.max_memory_allocated()/2**30))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['protocol','prepare','fit','evaluate'])
    p.add_argument('--family',choices=['elf','aurora']);p.add_argument('--device',default='cuda:1');p.add_argument('--scope',choices=['owt','xsum'],default='owt');a=p.parse_args()
    if a.mode=='protocol':protocol()
    elif a.mode=='prepare':prepare(a.family)
    elif a.mode=='fit':fit(a.family,a.device)
    else:evaluate(a.family,a.device,a.scope)

if __name__=='__main__':main()
