"""Verify the portable scalar evidence with direct positive-negative comparisons.

Run after recompute.py; requires only Python's standard library, with no network.
"""
import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent.parent
data=root/"data/heldout_perturbations.csv"
manifest=json.loads((root/"data/provenance.json").read_text())
assert hashlib.sha256(data.read_bytes()).hexdigest()==manifest["sha256"]
records=list(csv.DictReader(data.open()))
assert len(records)==manifest["rows"]==4992
for row in records:
    for key in ["t","relative_rms","raw_margin","perturbation_norm","first_order_radius","normalized_failure_score","predicted_margin","observed_margin"]:
        row[key]=float(row[key])
    for key in ["sequence","position"]:row[key]=int(row[key])
    row["flipped"]=row["flipped"].lower()=="true"
    row["amplitude"]=row["perturbation_norm"]/max(row["raw_margin"],1e-12)
    assert row["split"]=="heldout"
report=json.loads((root/"results/calibration_falsifiability_20261011.json").read_text())
strata=list(csv.DictReader((root/"results/calibration_phase_scale_cells_20261011.csv").open()))
violations=list(csv.DictReader((root/"results/calibration_proxy_radius_violations_20261011.csv").open()))
checks=0
bad_keys=[]
for model in sorted(set(r["model"] for r in records)):
    for direction in ["gradient","random"]:
        rows=[r for r in records if r["model"]==model and r["direction"]==direction]
        nums={"raw":0.,"amplitude":0.,"radius":0.};denominator=0
        for t in [1.,.8,.6]:
            for scale in [.01,.05,.2,.5]:
                part=[r for r in rows if r["t"]==t and r["relative_rms"]==scale]
                yes=[r for r in part if r["flipped"]];no=[r for r in part if not r["flipped"]]
                n=len(yes)*len(no);denominator+=n
                expected=next(r for r in strata if r["model"]==model and r["direction"]==direction and float(r["t"])==t and float(r["relative_rms"])==scale)
                assert len(part)==int(expected["n"]) and len(yes)==int(expected["flips"])
                for name,key,sign in [("raw","raw_margin",-1),("amplitude","amplitude",1),("radius","normalized_failure_score",1)]:
                    numerator=sum((sign*p[key]>sign*q[key])+.5*(p[key]==q[key]) for p in yes for q in no)
                    nums[name]+=numerator
                    expected_score=expected["auc_"+name]
                    assert not expected_score if not n else abs(float(expected_score)-numerator/n)<1e-12
                    checks+=1
        cell=next(r for r in report["checkpoint_direction_cells"] if r["model"]==model and r["direction"]==direction)
        assert denominator==cell["discordant_label_pairs_within_cells"]
        for key in nums:
            assert cell["conditional_auc_"+key] is None if not denominator else abs(cell["conditional_auc_"+key]-nums[key]/denominator)<1e-12
        below=[r for r in rows if r["perturbation_norm"]<r["first_order_radius"]]
        bad=[r for r in below if r["observed_margin"]<0]
        assert len(below)==cell["below_proxy_radius_rows"] and len(bad)==cell["below_proxy_radius_flips"]
        assert all(r["flipped"] and r["predicted_margin"]>0 for r in bad)
        bad_keys.extend((model,direction,r["sequence"],r["position"],r["t"],r["relative_rms"]) for r in bad)
saved_bad={(r["model"],r["direction"],int(r["sequence"]),int(r["position"]),float(r["t"]),float(r["relative_rms"])) for r in violations}
assert len(bad_keys)==27 and set(bad_keys)==saved_bad and checks==792
result={"status":"passed","de_texted_rows":len(records),"independent_pairwise_auc_checks":checks,
        "all_strata_retained":len(strata),"signed_margin_confirmed_violations":len(bad_keys),
        "input_sha256_matches_manifest":True,"bootstrap_intervals_not_independently_regenerated_by_this_verifier":True}
(root/"results/portable_validation.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
