"""Generate manuscript numbers from the retained experiment summaries."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'outputs/new_interfaces_20261102';P=ROOT/'docs/arxiv_v3_full_20261102'
def pct(x):return f'{100*x:.2f}'
def interval(x):return '['+', '.join(pct(v) for v in x)+']'
def table(caption,label,columns,header,rows):
    return '\\begin{table}[!htbp]\n\\centering\n\\footnotesize\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'

def main():
    c=json.loads((O/'calm/summary.json').read_text());c={r['condition']:r for r in c if r['fit']=='three_fit_mean'}
    rows=[]
    for key,label in [('posterior','Posterior sample'),('mean','Posterior mean'),('noise025',r'Mean $+\,0.25\epsilon$'),('radial05',r'Posterior $\times0.5$'),('radial2',r'Posterior $\times2$')]:
        r=c[key];rows.append(f"{label} & {pct(r['native_reference'])} & {pct(r['token_agreement'])} {interval(r['token_ci'])} & {pct(r['chunk_agreement'])} \\\\")
    s=table('CALM boundary audit on the same 512 test documents. Percentages; brackets are descriptive 95\\% document-bootstrap intervals. Native/reference measures reconstruction, whereas affine/native measures decision fidelity on identical states. Exact chunk requires all four decisions to match. All three primary optimization runs select their shared ridge initialization, so the selected map is identical across runs; the interval does not represent independent fitting variation.',
        'tab:calm-compressed-readout','lrrr','State & Native/reference & Affine/native [95\\% CI] & Exact chunk',rows)
    s+='\nThe primary validation-selected map recovers '+pct(c['posterior']['token_agreement'])+'\\% of native decisions on sampled posterior states, while native reconstruction is '+pct(c['posterior']['native_reference'])+'\\%. All three unrestricted CE runs select update zero. This outcome motivates a fit-adequacy control rather than an intrinsic nonlinear-decoding claim.\n\n'
    f=json.loads((O/'calm/fit_control/summary.json').read_text());frows=[]
    for r in f['results']:
        name={'ridge':'Original ridge','free':'Unrestricted, lower LR','tied':'Fixed-unembedding affine'}[r['family']]
        frows.append(f"{name} & {pct(r['agreement'])} {interval(r['ci'])} & {pct(r['gain_over_ridge'])} {interval(r['gain_ci'])} \\\\")
    s+=r'''The follow-up is exploratory because the first test was already inspected. It compares the original ridge map, a lower-learning-rate unrestricted affine fit, and four affine latent-to-feature maps with the released unembedding fixed. The latter fit 264,192 parameters instead of 66,180,612; frozen vocabulary weights still contribute storage and inference computation. Learning rate and checkpoint selection use only the original validation documents. The selected fits are then frozen before evaluating the next 512 documents in the original seeded permutation; exact hashes and token sequences verify that these documents are disjoint from all original splits.\n\n'''.replace(r'\n','\n')
    s+=table('Fit-adequacy follow-up on 512 additional documents, encoded after model selection. Agreement and paired gain are percentages/percentage points with descriptive 95\\% document-bootstrap intervals, averaged across three fits for each fitted family. Native/reference reconstruction is '+pct(f['native_reference'])+'\\%. This tests readout fitting, not generation quality or a nonlinear expressivity lower bound.',
        'tab:calm-fit-adequacy','lrr','Readout & Native agreement [95\\% CI] & Gain over ridge [95\\% CI]',frows)
    g=json.loads((O/'calm/generated_summary.json').read_text());gg={r['fit']:r for r in g['results']}
    r=gg['ridge'];t=gg['tied_three_fit_mean']
    s+='\nWe also use the official CALM-M generative head to sample one next-chunk latent at temperature 1 for each of 63 teacher prefixes in each of the original 512 test documents (32,256 latent predictions). The original ridge map agrees with the native decoder on '+pct(r['agreement'])+'\\% '+interval(r['ci'])+' of decisions; the validation-selected fixed-unembedding maps reach '+pct(t['agreement'])+'\\% '+interval(t['ci'])+', a paired change of '+pct(t['gain_over_ridge'])+' points '+interval(t['gain_ci'])+'. This additional transfer test uses the same state and native label for every readout. It is a conditional one-step interface test, not a free-running CALM benchmark or a comparison against a unique correct continuation.\n'
    h=json.loads((O/'calm/generated_fit/summary.json').read_text());hrows=[]
    for r in h['results']:
        name={'posterior':'Supplied posterior states','prior':'Sampled prior states'}[r['family']]
        hrows.append(f"{name} & {pct(r['agreement'])} {interval(r['ci'])} \\\\")
    s+='\nAn additional exploratory control changes the fitting population while holding the affine family, initialization, AdamW settings, seeds, minibatch indices, and 2,048-update budget fixed. Both arms use the same 4,096 fitting documents and 63 aligned chunk positions per document; one uses supplied posterior states and the other samples prior states from the corresponding teacher prefixes. Both select their checkpoint on the same 256-document prior-state validation pool. After all six fits are frozen, we generate 32,256 new prior states from the 512 held-out documents used in the posterior follow-up above. The documents are reused for this new state condition; they are not a third independent document cohort.\n\n'
    s+=table('Matched fitting-population control on newly sampled CALM-M prior states. Both arms use fixed-unembedding affine readouts and the same prior-state validation pool. Three-fit mean native agreement is reported with descriptive 95\\% document-bootstrap intervals. Temperature 1 and teacher prefixes are unchanged.','tab:calm-generated-fit','lr','Fitting population & Native agreement [95\\% CI]',hrows)
    s+='\nMatching the fitting population increases prior-state agreement by '+pct(h['gain'])+' percentage points '+interval(h['gain_ci'])+'. This modest gain does not close the observed gap under the declared budget. All six fits select the last evaluated checkpoint and are still improving there, so the residual is not an affine expressivity lower bound or a demonstrated property of compression itself. The control establishes the effect of this fitting-data substitution, not a full causal account of the generated states.\n'
    (P/'v3_calm_results.tex').write_text(s)
    l=json.loads((O/'llaco/summary.json').read_text());rows=[]
    for endpoint,gamma in [('uniform',.8),('uniform',.9),('uniform',.95),('mask_biased',.5),('mask_biased',.8),('mask_biased',.9),('mask_biased',1.)]:
        r=next(r for r in l['geometry'] if r['endpoint']==endpoint and r['gamma']==gamma)
        label='Uniform' if endpoint=='uniform' else 'Mask-biased'
        rows.append(f"{label} & {gamma:g} & {pct(r['cosine_correct'])} & {pct(r['adapter_correct'])} & {pct(1-r['adapter_cosine_disagree'])} & {pct(r['posterior_max'])} \\\\")
    (P/'v3_llaco_results.tex').write_text(table('LLaCO component measurements on 768 paired held-out source codewords. All entries after $\\gamma$ are percentages. Cosine/source and adapter/source measure recovery of the supplied token. Mode agreement compares two component readouts, not the contextual model. Max mass is the mean largest adapter posterior probability. Under mask-biased endpoints at $\\gamma=0.8/0.9$, both modes are always mask; their agreement is not lexical recovery.','tab:llaco-interface-boundary','lrrrrr',r'Endpoint & $\gamma$ & Cosine/source & Adapter/source & Mode agreement & Max mass',rows))
    rows=[]
    for gamma in [0.,.9,.98,1.]:
        r={v['method']:v for v in l['runtime'] if v['gamma']==gamma}
        rows.append(f"{gamma:g} & {r['upstream_streamed']['median_ms']:.2f} & {r['dense_exact']['median_ms']:.2f} & {r['sparse_eps001']['median_ms']:.2f} & {r['sparse_eps01']['median_ms']:.2f} \\\\")
    (P/'v3_llaco_runtime.tex').write_text(table('Adapter-only latency in milliseconds on one RTX 5090, batch four, FP32 accumulation: median of 20 synchronized repetitions after five warmups on the same states. Selection and fallback work are included. Dense and sparse implementations cache both FP32 codebooks (4\\,GiB additional memory). These shared-machine measurements exclude the 8B backbone, KV cache, and full generation; neither end-to-end speedup nor unchanged generated text is established.','tab:llaco-adapter-runtime','rrrrr',r'$\gamma$ & Upstream streamed & Dense exact & Sparse $10^{-3}$ & Sparse $10^{-2}$',rows))
    files=['calm/summary.json','calm/fit_control/summary.json','calm/generated_summary.json','calm/generated_fit/summary.json','llaco/summary.json','llaco/competition_audit.json']
    report={x:hashlib.sha256((O/x).read_bytes()).hexdigest() for x in files}
    (P/'audit/new_interface_table_sources.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
