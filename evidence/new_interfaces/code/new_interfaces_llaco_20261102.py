"""Released LLaCO geometry/adapter audit, not a full-model evaluation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,sys,time,types
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/new_interfaces_20261102/llaco'
SOURCE=ROOT/'third_party/LLaCO/Megatron-Bridge/src/megatron/bridge/diffusion/models/nemotron_labs_diffusion_spherical'

def dump(p,j):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def module(name):
    pkg='llaco_component_audit'
    if pkg not in sys.modules:
        m=types.ModuleType(pkg);m.__path__=[str(SOURCE)];sys.modules[pkg]=m
    spec=importlib.util.spec_from_file_location(pkg+'.'+name,SOURCE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

class DenseAdapter:
    """Equivalent FP32 expectation with explicit 4 GiB codebook cache cost."""
    def __init__(self,adapter,normalize):
        self.adapter=adapter;self.normalize=normalize
        self.values=adapter.geometry.E_geometry.float()
        self.directions=F.normalize(self.values,dim=-1)
        self.beta=adapter.beta.detach();self.bias=adapter.bias.detach().float()
        self.maximum_norm=self.values.norm(dim=-1).max()
    def probabilities(self,state):
        x=self.normalize(state.reshape(-1,state.shape[-1])).float()
        logits=(self.beta*x)@self.directions.T+self.bias
        return logits.softmax(-1)
    def __call__(self,state):
        return (self.probabilities(state)@self.values).reshape_as(state).to(state.dtype)
    def sparse(self,state,epsilon,return_stats=False):
        p=self.probabilities(state);v,ix=p.topk(64,-1,sorted=True);mass=v.cumsum(-1)
        eligible=mass[:,-1]>=1-epsilon
        k=(mass<1-epsilon).sum(-1)+1;k=k.clamp(max=64)
        out=torch.empty((len(p),self.values.shape[1]),device=p.device,dtype=p.dtype)
        delta=torch.zeros(len(p),device=p.device)
        if eligible.any():
            vv=v[eligible];ii=ix[eligible];kk=k[eligible]
            mask=torch.arange(64,device=p.device)[None,:]<kk[:,None]
            vv=vv*mask;retained=vv.sum(-1);delta[eligible]=1-retained
            vv=vv/retained[:,None]
            out[eligible]=(vv[:,:,None]*self.values[ii]).sum(1)
        if (~eligible).any():out[~eligible]=p[~eligible]@self.values
        out=out.reshape_as(state).to(state.dtype)
        if return_stats:return out,dict(eligible=eligible,k=k,delta=delta,bound=2*delta*self.maximum_norm)
        return out

def load(device):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
    if free<14*2**30:raise RuntimeError('Insufficient free headroom')
    torch.cuda.set_per_process_memory_fraction(12*2**30/total,device)
    geo=module('geometry');am=module('adapter')
    base=ROOT/'models/LLaCO-8B/components'
    E=torch.load(base/'spherical_geometry.E_geometry.pt',map_location='cpu',weights_only=True)
    assert E.shape==(131072,4096) and E.dtype==torch.bfloat16 and torch.isfinite(E).all()
    geometry=geo.EGeometry(E.to(device));del E
    adapter=am.SphericalAdapter(geometry,chunk_size=512,compute_dtype=torch.float32).to(device).eval()
    with torch.no_grad():
        adapter.log_beta.copy_(torch.load(base/'spherical_adapter.log_beta.pt',map_location=device,weights_only=True))
        adapter.bias.copy_(torch.load(base/'spherical_adapter.bias.pt',map_location=device,weights_only=True))
    return geo,geometry,adapter,DenseAdapter(adapter,geo.normalize)

@torch.inference_mode()
def study(device):
    OUT.mkdir(parents=True,exist_ok=True)
    geo,g,a,dense=load(device)
    meta=json.loads((ROOT/'models/LLaCO-8B/tokenizer/tokenizer.json').read_text())
    vocab=meta['model']['vocab'];vocab_ids=set(vocab.values()) if isinstance(vocab,dict) else set(range(len(vocab)))
    specials={r['id'] for r in meta['added_tokens'] if r.get('special')}
    norms=dense.values.norm(dim=-1)
    norms_cpu=norms.cpu().numpy()
    eligible=np.array(sorted(i for i in vocab_ids-specials if i<g.vocab_size and norms_cpu[i]>1e-6))
    rng=np.random.default_rng(61102101);ids=rng.choice(eligible,1024,replace=False)
    ids_t=torch.as_tensor(ids,device=device);x0=dense.directions[ids_t];radius=norms[ids_t,None]
    gen=torch.Generator(device=device).manual_seed(61102101)
    noise=geo.random_sphere(tuple(x0.shape),device=device,dtype=torch.float32,generator=gen)
    endpoints={'uniform':noise,'mask_biased':geo.terminal_noise(noise,g.codeword(100,device=device,dtype=torch.float32))}
    rows=[];sparse_rows=[];summary=[]
    for endpoint,z in endpoints.items():
        for gamma in [0.,.5,.8,.9,.95,.98,1.]:
            states=geo.slerp(x0,z,torch.tensor(gamma,device=device))
            for start in range(0,1024,32):
                state=states[start:start+32];source=ids_t[start:start+32]
                cosine=state@dense.directions.T
                cosine_source=cosine.gather(1,source[:,None])
                rank=1+(cosine>cosine_source).sum(-1)
                cos_id=cosine.argmax(-1)
                # Restore each source's original radius so gamma=0 equals the
                # original embedding, avoiding a unit/raw-codebook mismatch.
                raw=state*radius[start:start+32]
                euclid=(2*(raw@dense.values.T)-norms.square()).argmax(-1)
                p=dense.probabilities(state);mode=p.argmax(-1)
                full=p@dense.values
                for i in range(len(state)):
                    rows.append(dict(endpoint=endpoint,gamma=gamma,source_index=start+i,split='development' if start+i<256 else 'test',
                        source_token=int(source[i]),cosine_correct=int(cos_id[i]==source[i]),euclidean_correct=int(euclid[i]==source[i]),
                        adapter_correct=int(mode[i]==source[i]),adapter_cosine_disagree=int(mode[i]!=cos_id[i]),source_rank=int(rank[i]),
                        posterior_max=float(p[i].max()),source_norm=float(radius[start+i,0])))
                if endpoint=='mask_biased':
                    for eps in [.001,.01]:
                        sparse,info=dense.sparse(state,eps,True);err=(sparse-full).norm(dim=-1)
                        # FP32 roundoff is recorded separately from the exact-arithmetic bound.
                        if not torch.all(err<=info['bound']+1e-4):raise AssertionError('Omitted-mass bound violated beyond FP32 tolerance')
                        for i in range(len(state)):
                            sparse_rows.append(dict(gamma=gamma,epsilon=eps,source_index=start+i,split='development' if start+i<256 else 'test',
                                sparse=int(info['eligible'][i]),kept_k=int(info['k'][i]) if info['eligible'][i] else g.vocab_size,
                                omitted_mass=float(info['delta'][i]),l2_error=float(err[i]),bound=float(info['bound'][i])))
            print('geometry',endpoint,gamma,flush=True)
    csvwrite(OUT/'per_source.csv',rows);csvwrite(OUT/'sparse_per_source.csv',sparse_rows)
    for endpoint in endpoints:
        for gamma in [0.,.5,.8,.9,.95,.98,1.]:
            r=[v for v in rows if v['endpoint']==endpoint and v['gamma']==gamma and v['split']=='test']
            summary.append(dict(endpoint=endpoint,gamma=gamma,n=len(r),**{k:float(np.mean([v[k] for v in r])) for k in ['cosine_correct','euclidean_correct','adapter_correct','adapter_cosine_disagree','source_rank','posterior_max']}))
    # Positive radial scaling: raw source times alpha, no added noise.
    radial=[]
    for alpha in [.1,.5,1.,2.]:
        for start in range(256,1024,32):
            state=dense.values[ids_t[start:start+32]]*alpha
            cosine=F.normalize(state,dim=-1)@dense.directions.T
            ep=(2*(state@dense.values.T)-norms.square()).argmax(-1)
            ap=dense.probabilities(state).argmax(-1);cp=cosine.argmax(-1)
            base=dense.probabilities(dense.values[ids_t[start:start+32]]).argmax(-1)
            for i in range(len(state)):
                radial.append(dict(alpha=alpha,source_index=start+i,cosine_correct=int(cp[i]==ids_t[start+i]),euclidean_correct=int(ep[i]==ids_t[start+i]),adapter_correct=int(ap[i]==ids_t[start+i]),adapter_scale_preserved=int(ap[i]==base[i])))
    csvwrite(OUT/'radial_per_source.csv',radial)
    radial_summary=[dict(alpha=s,**{k:float(np.mean([r[k] for r in radial if r['alpha']==s])) for k in ['cosine_correct','euclidean_correct','adapter_correct','adapter_scale_preserved']}) for s in [.1,.5,1.,2.]]
    # Independent numerical parity checks against unmodified upstream implementation.
    parity=[]
    for gamma in [0.,.9,.98,1.]:
        states=geo.slerp(x0[256:260],endpoints['mask_biased'][256:260],torch.tensor(gamma,device=device))
        for dtype in [torch.float32,torch.bfloat16]:
            s=states.to(dtype);reference=a(s);replacement=dense(s)
            max_error=float((reference.float()-replacement.float()).abs().max())
            rel=float((reference.float()-replacement.float()).norm()/reference.float().norm().clamp_min(1e-12))
            assert max_error<(1e-4 if dtype==torch.float32 else .005)
            parity.append(dict(gamma=gamma,dtype=str(dtype),max_abs_error=max_error,relative_l2=rel))
    # Complete adapter timing with allocation and all selection work included.
    timings=[]
    for gamma in [0.,.9,.98,1.]:
        s=geo.slerp(x0[256:260],endpoints['mask_biased'][256:260],torch.tensor(gamma,device=device))
        funcs={'upstream_streamed':a,'dense_exact':dense,'sparse_eps001':lambda x:dense.sparse(x,.001),'sparse_eps01':lambda x:dense.sparse(x,.01)}
        for name,fn in funcs.items():
            for _ in range(5):fn(s)
            torch.cuda.synchronize();times=[]
            for _ in range(20):
                t=time.perf_counter();fn(s);torch.cuda.synchronize();times.append(1000*(time.perf_counter()-t))
            timings.append(dict(gamma=gamma,method=name,median_ms=float(np.median(times)),p10_ms=float(np.quantile(times,.1)),p90_ms=float(np.quantile(times,.9)),measurements=times))
            print('runtime',gamma,name,round(np.median(times),3),'ms',flush=True)
    ss=[]
    for eps in [.001,.01]:
        for gamma in [0.,.5,.8,.9,.95,.98,1.]:
            r=[v for v in sparse_rows if v['split']=='test' and v['epsilon']==eps and v['gamma']==gamma]
            ss.append(dict(epsilon=eps,gamma=gamma,sparse_fraction=float(np.mean([v['sparse'] for v in r])),mean_k=float(np.mean([v['kept_k'] for v in r])),max_l2_error=max(v['l2_error'] for v in r),max_bound=max(v['bound'] for v in r)))
    dump(OUT/'summary.json',dict(scope='Component audit on forward corruption; no full LLaCO inference or task-quality result',
        beta=float(a.beta),bias_range=[float(a.bias.min()),float(a.bias.max())],source_eligible=len(eligible),
        source_ids=ids.tolist(),geometry=summary,radial=radial_summary,sparse=ss,parity=parity,runtime=timings,
        cache_additional_bytes=dense.values.numel()*4+dense.directions.numel()*4,torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),max_allocated_gib=torch.cuda.max_memory_allocated()/2**30))

@torch.inference_mode()
def competition_audit(device):
    """Post hoc accounting of special-token competitors; no policy retuning."""
    from collections import Counter
    geo,g,a,d=load(device)
    old=json.loads((OUT/'summary.json').read_text());ids=torch.tensor(old['source_ids'],device=device)
    x0=d.directions[ids];gen=torch.Generator(device=device).manual_seed(61102101)
    noise=geo.random_sphere(tuple(x0.shape),device=device,dtype=torch.float32,generator=gen)
    endpoints={'uniform':noise,'mask_biased':geo.terminal_noise(noise,g.codeword(100,device=device,dtype=torch.float32))}
    meta=json.loads((ROOT/'models/LLaCO-8B/tokenizer/tokenizer.json').read_text());v=meta['model']['vocab']
    vocab=set(v.values());specials={r['id'] for r in meta['added_tokens'] if r.get('special')}
    allowed=torch.zeros(g.vocab_size,device=device,dtype=torch.bool);allowed[list(vocab-specials)]=True
    records=[]
    for endpoint in endpoints:
        for gamma in [.8,.9]:
            state=geo.slerp(x0,endpoints[endpoint],torch.tensor(gamma,device=device))
            cc=[];aa=[];restricted=[]
            for start in range(256,1024,32):
                s=state[start:start+32];c=s@d.directions.T;p=d.probabilities(s)
                cc.extend(c.argmax(-1).cpu().tolist());aa.extend(p.argmax(-1).cpu().tolist())
                c[:,~allowed]=-torch.inf;restricted.extend(c.argmax(-1).cpu().tolist())
            true=ids[256:].cpu().numpy()
            records.append(dict(endpoint=endpoint,gamma=gamma,cosine_mask_fraction=float(np.mean(np.array(cc)==100)),
                adapter_mask_fraction=float(np.mean(np.array(aa)==100)),lexical_only_cosine_reconstruction=float(np.mean(np.array(restricted)==true)),
                cosine_winners=Counter(cc).most_common(5),adapter_winners=Counter(aa).most_common(5)))
    eu=[];lex=[];norm=d.values.norm(dim=-1)
    for start in range(256,1024,32):
        s=.5*d.values[ids[start:start+32]];scores=2*s@d.values.T-norm.square()
        eu.extend(scores.argmax(-1).cpu().tolist());scores[:,~allowed]=-torch.inf;lex.extend(scores.argmax(-1).cpu().tolist())
    dump(OUT/'competition_audit.json',dict(status='Post hoc competitor accounting after primary results; original full-vocabulary outcomes unchanged',
        angular=records,radial_half=dict(unrestricted_winners=Counter(eu).most_common(8),unrestricted_special_fraction=float(np.mean([i in specials or i not in vocab for i in eu])),
        lexical_only_reconstruction=float(np.mean(np.array(lex)==ids[256:].cpu().numpy())))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');p.add_argument('--competition-audit',action='store_true');a=p.parse_args()
    if a.competition_audit:competition_audit(a.device)
    else:study(a.device)
