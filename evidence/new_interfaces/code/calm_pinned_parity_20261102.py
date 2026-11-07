"""Validate stored native data against upstream-pinned transformers/tokenizers."""
from new_interfaces_calm_20261102 import *
import transformers,tokenizers

def main():
    setup('cuda:2');assert transformers.__version__=='4.43.0';assert tokenizers.__version__=='0.19.1'
    ae=load_ae('cuda:2');rows=[]
    with torch.inference_mode():
        for split in ['train','validation','test']:
            ids=np.load(OUT/'cohort'/(split+'.npy'))[:8].reshape(-1,4)
            enc=torch.cat([ae.encoder(torch.as_tensor(ids[i:i+256],device='cuda:2'))[:,0] for i in range(0,len(ids),256)])
            mu,ls=enc.chunk(2,-1)
            oldmu=np.load(OUT/f'data/{split}/mean.npy',mmap_mode='r')[:len(ids)]
            oldstd=np.load(OUT/f'data/{split}/std.npy',mmap_mode='r')[:len(ids)]
            e1=float(np.max(abs(mu.cpu().numpy()-oldmu)));e2=float(np.max(abs(ls.exp().cpu().numpy()-oldstd)))
            x=np.load(OUT/f'data/{split}/x.npy',mmap_mode='r')[:len(ids)];y=np.load(OUT/f'data/{split}/native.npy')[:len(ids)]
            pred=[]
            for i in range(0,len(x),64):pred.append(native(ae,torch.as_tensor(np.array(x[i:i+64]),device='cuda:2'))[0].cpu().numpy())
            mismatches=int((np.concatenate(pred)!=y).sum());assert e1<1e-5 and e2<1e-5 and mismatches==0
            rows.append(dict(split=split,documents=8,mean_max_abs_error=e1,std_max_abs_error=e2,native_mismatches=mismatches))
        for split in ['validation','test']:
            x=np.load(OUT/f'data/{split}/x.npy',mmap_mode='r');y=np.load(OUT/f'data/{split}/native.npy');mismatches=0
            for i in range(0,len(x),64):
                p=native(ae,torch.as_tensor(np.array(x[i:i+64]),device='cuda:2'))[0].cpu().numpy()
                mismatches+=int((p!=y[i:i+64]).sum())
            assert mismatches==0
            rows.append(dict(split=split,scope='entire stored validation or test; identical decoder batch shape',tokens=int(y.size),native_mismatches=mismatches))
    dump(ROOT/'outputs/new_interfaces_20261102/source/calm_pinned_parity.json',dict(status='passed',transformers=transformers.__version__,tokenizers=tokenizers.__version__,torch=torch.__version__,checks=rows))
    print(rows,flush=True)
if __name__=='__main__':main()
