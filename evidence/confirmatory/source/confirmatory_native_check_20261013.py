"""Check historical labels against the released native decoder used for fresh tests."""
import os,sys,json
from types import SimpleNamespace
import numpy as np
import torch
from confirmatory_readout_20261013 import ROOT,OUT,P,dump
sys.path.insert(0,str(ROOT/'outputs/phase0_phase1_20260520/scripts'))
from phase2_trajectory_audit import load_elf,decode_logits,mask_after_eos
torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
device=torch.device('cuda:0')
a=SimpleNamespace(root=ROOT,out=OUT,encoder_model=ROOT/'models/t5-small',tokenizer=ROOT/'models/t5-small',ppl_model=ROOT/'models/gpt2-large',config=ROOT/'models/ELF-B-owt-torch/config.yml',checkpoint=ROOT/'models/ELF-B-owt-torch',weights=P['elf_weights'],num_samples=32,batch_size=1)
old=os.getcwd();os.chdir(ROOT/'official-ELF-pytorch_elf')
try:c,tok,model=load_elf(a,device)
finally:os.chdir(old)
split=json.loads((OUT/'data/elf/split.json').read_text());ids=split['sequence_ids']['train'][:32]
groups={}
for sid in ids:groups.setdefault(sid//1024,[]).append(sid)
rows=[]
with torch.inference_mode():
 for shard,seqs in groups.items():
  d=torch.load(ROOT/P['elf_source']/f'shard{shard}.pt',map_location='cpu',mmap=True,weights_only=False)
  for sid in seqs:
   i=sid%1024;z=d['final_z'][i:i+1].to(device)
   logits=decode_logits(model,z,1.,3.,c).float().cpu()
   top=logits.topk(2,-1)
   pred=mask_after_eos(top.indices[...,0],1,0)
   y=d['labels'][i:i+1];v=d['valid'][i:i+1]
   wrong=(pred!=y)&v
   gaps=top.values[...,0]-top.values[...,1]
   rows.append({'historical_sequence_id':sid,'correct':int(((pred==y)&v).sum()),'valid':int(v.sum()),'all_positions_exact':bool(torch.equal(pred,y)),'current_top_two_gaps_at_mismatch':gaps[wrong].tolist()})
dump(OUT/'native_checkpoint_check.json',{'sample_source':'first 32 NEW training-split sequence IDs; no validation or test selection','rows':rows,'native_agreement':sum(x['correct'] for x in rows)/sum(x['valid'] for x in rows),'all_sequences_bitwise_equal':all(x['all_positions_exact'] for x in rows),'weights':P['elf_weights'],'scope':'Decoder lineage/numerical consistency check; not generation-seed reproduction.'})
print(json.dumps({'agreement':sum(x['correct'] for x in rows)/sum(x['valid'] for x in rows)}))
