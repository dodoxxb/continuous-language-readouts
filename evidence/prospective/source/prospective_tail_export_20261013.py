"""Export compact evidence, preserving all inherited ancillary files."""
from pathlib import Path
import hashlib,json,shutil
from prospective_tail_20261013 import ROOT,OUT,CONDITIONS
DOC=ROOT/'docs/arxiv_v3_full_20261013_predictive'

def main():
    dest=DOC/'audit/evidence_supplement/prospective';dest.mkdir(parents=True,exist_ok=True)
    names=['protocol.json','frozen.json','predictors.json','calibration.csv','inference_amendment_before_generation.json','loader_dependency_fix.json','analysis.json','predictions.csv','per_sequence.csv','text_scores.csv','agreement.csv','prediction_summary.csv','repair_summary.csv','ppl_summary.csv','overlap.csv']
    names+=['secondary_protocol.json','secondary_analysis.json','secondary_text_scores.csv','secondary_per_sequence.csv','secondary_repair_summary.csv','secondary_ppl_summary.csv']
    names+=['integrity.json']
    for name in names:shutil.copy2(OUT/name,dest/name)
    weights=dest/'official_weights';weights.mkdir(exist_ok=True)
    for path in (OUT/'official_weights').glob('*.json'):shutil.copy2(path,weights/path.name)
    (weights/'README.md').write_text('''# Official checkpoint inventory, checked 2026-10-13

Sources: https://huggingface.co/api/models?author=embedded-language-flows&full=true&sort=lastModified&direction=-1&limit=100 and the per-model API (`?blobs=true`); official code https://github.com/lillian039/ELF, including its distillation branch.

The API returned 13 model repositories. The most recently modified repository is ELF-B-PD-owt-torch (2026-06-26), containing checkpoint_19017. No later repository update was returned in this official organization at retrieval time. Repository modification dates are not training dates. The separate ELF-B-owt-teacher repository is included in the inventory; it is not asserted to be identical to the ELF-B-owt-torch checkpoint. The distillation README currently points to ELF-B-owt-torch as its teacher.

The three checkpoints used in the frozen prospective experiment (B, M, PD) exactly match the official current LFS SHA-256 values. `used_weights_comparison.json` compares official values with the local checkpoint hashes frozen before generation. The newest released PD student is therefore already represented in the new 512-sequence, two-generation-seed PD8 condition, without changing the frozen primary protocol. This inventory is provenance, not a new trained-model result. Third-party forks and conditional checkpoints are not represented as new unconditional ELF releases.
''')
    shutil.copy2(ROOT/'scripts/prospective_tail_verify_20261013.py',dest/'verify.py')
    source=dest/'source';source.mkdir(exist_ok=True)
    for name in ['prospective_tail_20261013.py','prospective_tail_analyze_20261013.py','prospective_tail_manuscript_20261013.py','prospective_tail_export_20261013.py','prospective_tail_secondary_20261013.py','prospective_tail_secondary_analyze_20261013.py','prospective_tail_integrity_20261013.py']:shutil.copy2(ROOT/'scripts'/name,source/name)
    shutil.copy2(ROOT/'outputs/phase0_phase1_20260520/scripts/phase2_trajectory_audit.py',source/'phase2_trajectory_audit.py')
    shutil.copy2(ROOT/'scripts/p1_pd_official_fewstep_sampling.yml',source/'pd_sampling_dependency.yml')
    generated={}
    for folder in sorted((OUT/'test').glob('*/*')):
        key=folder.relative_to(OUT/'test').as_posix();generated[key]=json.loads((folder/'generation.json').read_text())
        generated[key]['repair_tokens_sha256']=hashlib.sha256((folder/'repairs.npz').read_bytes()).hexdigest()
    (dest/'generation_manifest.json').write_text(json.dumps(generated,indent=2)+'\n')
    configs={}
    for name,c in CONDITIONS.items():
        path=ROOT/'models'/c['model']/'config.yml';target=source/(name+'_config.yml');shutil.copy2(path,target);configs[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (dest/'config_manifest.json').write_text(json.dumps(configs,indent=2)+'\n')
    (dest/'README.md').write_text('''# Prospective readout prediction and equal-error repair

Run `python verify.py` here with Python 3.10+ and NumPy. The verifier refits the calibration regressions, reproduces frozen predictions, paired bootstrap intervals and corpus PPL, and checks identical per-sequence repair counts. It does not rerun GPU measurements or certify generated content.

The calibration set was previously inspected. Before new generation, we froze the protocol, three readouts, regression coefficients and checkpoint hashes. `inference_amendment_before_generation.json` records a pre-generation correction preserving common-random-number dependence in bootstrap. `loader_dependency_fix.json` records a missing PD configuration dependency repaired without changing experimental sampling settings. The primary experiment script retains its originally frozen hash.

Tests comprise three checkpoint/sampling conditions × two generation seeds × 256 sequences = 1,536 trajectories. Three readouts give 4,608 prediction records. Ten text-score arms give 15,360 per-sequence measurements. Six repair arms use the prespecified readout seed 17, all correcting exactly floor(errors/2) positions. These are native-label oracle interventions, not a usable decoder or acceleration method.

`text_scores.csv` contains measured evaluator NLL sums and token counts. `predictions.csv` contains every fixed predictor input/output. `calibration.csv` and `predictors.json` permit refitting and prediction checks. `generation_manifest.json` records local latent/token cache hashes; checkpoints and large caches remain outside this package. The source scripts require the repository paths and local model installations, whereas `verify.py` is portable. Original generation uses PyTorch, Transformers and the released ELF port; analysis additionally uses SciPy; figures use Matplotlib.

Native-mask scoring is conditional; token agreement is not factual correctness. Different detokenizations have different evaluator lengths and truncation coverage. MAE and repair primary intervals are 97.5% Bonferroni intervals for two tests. Per-condition and other policy intervals are descriptive 95% intervals. Failed hypotheses and all prespecified arms are retained.

The OPT-125M secondary evaluation was added after the initial protocol and generation, before inspecting the primary scoring results. It evaluates the exact same 15,360 text arms, without fitting or selecting a predictor or repair policy. It is reported separately as evaluator sensitivity, not a third confirmatory primary hypothesis.
''')
    root_readme=DOC/'audit/evidence_supplement/README.md'
    rs=root_readme.read_text()
    if 'prospective/verify.py' not in rs:
        root_readme.write_text(rs+'\n## Prospective prediction and matched repair extension\n\nSee `prospective/README.md` for 1,536 new trajectories, fixed readout transfer, quality prediction, equal-error repair, secondary OPT scoring, and official checkpoint provenance. Run `python prospective/verify.py` with NumPy to independently reproduce the new scalar results. Earlier sections describe their original study snapshots.\n')
    # Adapt inherited validators without weakening numerical preservation checks.
    path=DOC/'audit/validate_full.py';s=path.read_text()
    s=s.replace('assert len(used)==72','assert len(used)==73').replace('assert figs==72 and tabs==38','assert figs==73 and tabs==41')
    s=s.replace("'tab:confirmatory-langflow'}","'tab:confirmatory-langflow','tab:prospective-prediction','tab:prospective-repair','tab:prospective-ppl'}")
    before="(out/'audit/full_validation.json').write_text(json.dumps(report,indent=2))"
    extra="""confirm=out.parent/'arxiv_v3_full_20261013_confirmatory'
confirm_manifest=json.loads((out/'audit/confirmatory_preservation_baseline.json').read_text())
for name,h in confirm_manifest['members'].items():assert sha(confirm/name)==h,name
confirm_source,_=expand(confirm)
for name,nums in tables(confirm_source).items():assert nums==nt[name],name
assert set(re.findall(r'\\\\label\\{([^}]+)\\}',confirm_source))<=set(labels)
confirm_figures=re.findall(r'\\\\includegraphics(?:\\[[^]]*\\])?\\{([^}]+)\\}',confirm_source)
assert len(confirm_figures)==72
for name in confirm_figures:assert sha(out/name)==sha(confirm/name),name
report.update(confirmatory_parent_unchanged=True,confirmatory_tables_exactly_preserved=38,confirmatory_figures_unchanged=72)
"""
    if 'confirmatory_parent_unchanged' not in s:s=s.replace(before,extra+before)
    if "selected={1,2,a['fig:prospective-tail']" not in s:s=s.replace("selected={1,2,","selected={1,2,a['fig:prospective-tail']['page'],a['app:prospective-readout']['page'],")
    path.write_text(s)
    path=DOC/'audit/package_evidence_20261013.py';s=path.read_text();s=s.replace("'../confirmatory/verify_followup.py']","'../confirmatory/verify_followup.py','../prospective/verify.py']")
    if "'prospective_trajectories'" not in s:s=s.replace("'source_measurement_rows':4992","'prospective_trajectories':1536,'prospective_prediction_records':4608,'prospective_text_scores':15360,'secondary_text_scores':15360,'source_measurement_rows':4992")
    path.write_text(s)
    print('Exported',len(list(dest.rglob('*'))),'prospective entries')

if __name__=='__main__':main()
