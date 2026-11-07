"""Recorded post hoc global-scale control for the context-residual intervention.

The first intervention's OWT-only coordinate standardization amplified a nearly
constant training coordinate on XSum. This second, explicitly exploratory round
changes only standardization to a single training RMS scale per feature family.
All first-round failures remain preserved. This is not test-independent design.
"""
from pathlib import Path
import argparse,datetime,json,os
import numpy as np
import deep_context_adapter_20261022 as experiment

ROOT=experiment.ROOT
PARENT=ROOT/'outputs/deep_mechanism_20261022/context'
OUT=ROOT/'outputs/deep_mechanism_20261022/context_global'

def protocol():
    p=OUT/'protocol.json'
    if not p.exists():
        obj=json.loads((PARENT/'protocol.json').read_text())
        obj.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          question='Does a training-only global scale avoid instability from whitening a near-constant coordinate, and does local context then help beyond focal refitting?',
          status='Second-round post hoc diagnostic after seeing first-round AURORA OWT/XSum fidelity. No test labels fit weights or select checkpoints; the choice of normalization was informed by those outcomes.',
          parent_protocol_sha256=experiment.sha(PARENT/'protocol.json'),
          change='Keep per-dimension training means but divide all dimensions by s=sqrt(mean_j Var_train(q_j)); no coordinate-specific variance inversion, clipping, learned gate, new rank, or learning-rate tuning.',
          observed_trigger='First-round AURORA adapters fail on XSum despite near-baseline OWT fidelity; OWT coordinate86 has SD below1e-4 and XSum variance is much larger. The replacement rule is global, identical for all coordinates/families/arms, and uses training statistics only.')
        obj['training']['normalization']='Per-dimension training means; one scalar root mean coordinate variance from training only, repeated over all coordinates. Context and shuffled share statistics.'
        experiment.dump(p,obj)
        experiment.dump(OUT/'protocol_frozen.json',dict(protocol_sha256=experiment.sha(p),
            wrapper_code_sha256=experiment.sha(Path(__file__)),core_code_sha256=experiment.sha(ROOT/'scripts/deep_context_adapter_20261022.py'),new_global_fits_existing=False))
    return json.loads(p.read_text())

def prepare(family):
    protocol();d=OUT/family;d.mkdir(parents=True,exist_ok=True)
    for name in ['train.npz','validation.npz']:
        target=d/name;source=PARENT/family/name
        assert source.exists()
        if not target.exists():os.link(source,target)
    z=np.load(d/'train.npz');stats={}
    for arm in ['focal','context']:
        x=z['x' if arm=='focal' else 'context'].astype(np.float64)
        mean=x.mean(0);scale=float(np.sqrt(x.var(0).mean()));assert scale>1e-4
        stats[arm]=dict(mean=mean.tolist(),std=np.full(x.shape[1],scale).tolist(),scalar_scale=scale)
    experiment.dump(d/'normalization.json',stats)
    experiment.dump(d/'data_manifest.json',dict(parent=str(PARENT/family),
        train_sha256=experiment.sha(d/'train.npz'),validation_sha256=experiment.sha(d/'validation.npz'),
        data_identical_to_first_round=True,protocol_sha256=experiment.sha(OUT/'protocol.json')))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['protocol','prepare','fit','evaluate'])
    p.add_argument('--family',choices=['elf','aurora']);p.add_argument('--device',default='cuda:1')
    p.add_argument('--scope',choices=['owt','xsum'],default='owt');a=p.parse_args()
    experiment.OUT=OUT;experiment.protocol=protocol
    if a.mode=='protocol':protocol()
    elif a.mode=='prepare':prepare(a.family)
    elif a.mode=='fit':experiment.fit(a.family,a.device)
    else:experiment.evaluate(a.family,a.device,a.scope)

if __name__=='__main__':main()
