"""Prepare evidence-linked manuscript inserts only after both scorers pass audit.

This creates reviewable additions, not a silently promoted submission release.
The existing long-manuscript release remains the baseline for final integration.
"""
import csv
import json
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/aurora_replication_20261022'

def rows(path):return list(csv.DictReader(path.open()))
def interval(r):return f"{r['mean']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
def direction(r,positive,negative,uncertain):
    return positive if r['ci_low']>0 else negative if r['ci_high']<0 else uncertain

def main():
    audit=json.loads((OUT/'integrity.json').read_text());assert audit['status']=='passed'
    p=json.loads((OUT/'protocol.json').read_text())
    primary=json.loads((OUT/'analysis/gpt2-large/summary.json').read_text())
    secondary=json.loads((OUT/'analysis/opt-125m/summary.json').read_text())
    controls={}
    if (OUT/'source/prediction_control_plan.json').exists():
        started=time.monotonic()
        for family in ['elf','aurora']:
            path=OUT/'regularization_controls'/family/'summary.json'
            while not path.exists():
                if time.monotonic()-started>600:raise RuntimeError('Declared regression controls did not complete')
                time.sleep(1)
            controls[family]=json.loads(path.read_text())
    pred={r['condition']:r for r in primary['prediction']}
    repairs={(r['condition'],r['policy']):r for r in primary['repair']}
    second={(r['condition'],r['policy']):r for r in secondary['repair']}
    agreements=rows(OUT/'analysis/gpt2-large/agreement.csv')
    linear_agreements=[q for q in agreements if q['method'].startswith('linear')]
    pooled_agreement=sum(int(q['lexical_correct']) for q in linear_agreements)/sum(int(q['lexical_tokens']) for q in linear_agreements)
    primary_eligible=sum(q['common_scorable'] for q in primary['validity'])
    secondary_eligible=sum(q['common_scorable'] for q in secondary['validity'])
    primary_ppl={(q['condition'],q['method']):float(q['ppl']) for q in rows(OUT/'analysis/gpt2-large/ppl.csv')}
    secondary_ppl={(q['condition'],q['method']):float(q['ppl']) for q in rows(OUT/'analysis/opt-125m/ppl.csv')}
    dest=OUT/'manuscript_ready';dest.mkdir(parents=True,exist_ok=True)
    a=pred['all'];r=repairs['all','rare'];l=repairs['all','lowmargin']
    change=('reduces' if a['relative_mae_reduction']>=0 else 'increases')
    prediction_claim=direction(a,'supports the prespecified prediction gain','finds worse prediction under the augmented predictor','does not establish a prediction gain')
    repair_claim=direction(r,'supports a rare-first advantage','finds a rare-first disadvantage','does not establish a rare-first advantage')
    main_text=rf'''\subsection{{An Independent-Interface Replication}}
\label{{sec:aurora-replication}}

We repeat the frozen prediction and equal-count repair protocol on the released AURORA-LM-S OpenWebText checkpoint~\citep{{aurora2026}}. Its separately trained text autoencoder provides a different decoder-facing representation. The native decoder starts from shared learned queries, reads the latent sequence through six causal contextual layers, and applies a tied GPT-2 token projection. It has no separate affine latent-to-logit bypass. The pretrained backbone supplies the frozen token table rather than a contextual Transformer in this autoencoder. We fit new token-wise affine readouts in its native 768-dimensional coordinates; no ELF readout weights are transferred. The inputs are generated states before the complete native decoder. We use 2,048 training, 256 validation, and 512 calibration sequences, and freeze all three selected readouts and their ridge predictors before generating 1,536 test sequences at 16, 32, and 64 sampling steps per 16-token block. These are new seeds and sampling conditions within one independently released architecture, not independent denoiser training replicas.

Pooled lexical agreement to the native decoder is ${100*pooled_agreement:.2f}\%$ across the three fitted readouts on all new test states. In the pooled AURORA quality test, the composition-aware predictor {change} MAE by ${abs(a['relative_mae_reduction'])*100:.1f}\%$ (baseline ${a['baseline_mae']:.4f}$; augmented ${a['tail_mae']:.4f}$). The paired baseline-minus-augmented difference is ${interval(a)}$ nats, using the prespecified 97.5\% interval. Thus this test {prediction_claim}. At exactly the same number of corrections within each sequence, mean random repair minus rare-first repair is ${interval(r)}$ nats (97.5\% interval), which {repair_claim}. Low-native-margin-first has contrast ${interval(l)}$ (descriptive 95\% interval). All step-specific outcomes, both evaluators, every readout fit, and the training-frequency sensitivity are retained in \secref{{app:aurora-replication}}; neither feature nor repair policy is selected on the new test.

The comparison concerns conditional evaluator scores under a shared native output boundary. Native EOS separators are fixed across arms; repairs do not feed back into the generated latent trajectory. Training-pool sizes, tokenizers, and sequence horizons differ from ELF, so the replication tests the diagnostic protocol rather than attributing numerical differences solely to architecture. The ELF prospective study uses 1,024 source positions; this audit uses 512. A frequency-normalized sensitivity matches the earlier ELF rare-token frequency cutoff without altering the primary count-based test. These oracle interventions neither establish factual correctness nor supply a deployable repair policy.
'''
    if controls:
        main_text+='\n'+r'An additional regression audit separates composition information from changes in effective regularization: repeated baseline columns add no information but weaken the ridge penalty, while unpenalized fits remove it. These post-protocol controls retain the original comparisons and all condition-specific results (\secref{app:aurora-replication}).'+'\n'
    (dest/'v3_aurora_main.tex').write_text(main_text)
    appendix=[r'\section{Independent AURORA Readout Replication}',r'\label{app:aurora-replication}','',
      r'The local protocol was frozen before formal data collection; it is not a public preregistration. Mechanical pilot sequences were excluded from every fitting, validation, calibration, and test pool. The released EMA denoiser and shared text autoencoder were downloaded at pinned revisions and checked against the upstream SHA-256 digests. The memory-bounded adapter preserves the released model and sampler operations. Native labels use the released decoder\textquotesingle s \texttt{argmax}, rather than the potentially different first index of \texttt{topk} on tied BF16 logits. The public inference entry and the adapter agree exactly in the recorded single-sequence pilot.',
      '',r'\parhead{Fitting and calibration}',
      r'All development sequences use the released SDE-DPM++ sampler with 64 diffusion updates per 16-token block and self-conditioning guidance 4.0. We generate 512-token prefixes (32 blocks); the checkpoint\textquotesingle s training and default maximum is 1,024. S16, S32, and S64 denote updates per block, not whole-sequence function evaluations; compute is not matched to ELF. Decoder evaluation is performed one sequence at a time to bound memory. Each affine fit uses seeds 17, 29, or 43, 4,096 AdamW updates of 1,024 sampled token positions, learning rate $5\times10^{-4}$, weight decay 0.01, and input noise 0.02 times the batch feature standard deviation. Native EOS positions are excluded from fitting. Checkpoints at updates 1,024, 2,048, and 4,096 are selected by validation lexical agreement, with earlier updates breaking ties. Calibration fits one baseline and one augmented ridge predictor per selected readout, using $\alpha=10$, calibration-only standardization, and an unpenalized intercept.',
      '',r'\parhead{Scoring and matched edits}',
      r'OWT samples contain document separators, so first-EOS truncation would change the task. Every arm retains the same native EOS separators and is decoded and re-tokenized by its evaluator. A sequence belongs to that evaluator\textquotesingle s common scorable set only when all ten arms contain 2--1,024 evaluator tokens; no arm is truncated. The native decoder is an oracle comparator, not a semantic truth label. Repairs use readout seed 17 and correct exactly $\lfloor e/2\rfloor$ of its lexical disagreements, using ascending native training count, numeric-first, ascending native margin, or three uniform random selections. All six arms therefore retain identical source-token error counts within each sequence. The underlying latent trajectory and its contextual decoder inputs are held fixed.',
      '',rf'The common scorable sets retain {primary_eligible:,} of 1,536 test sequences for GPT-2-Large and {secondary_eligible:,} for OPT-125M. Excluded indices, every per-sequence score, and exact native-output overlap counts are retained in the evidence. Native lexical agreement uses all 1,536 generated sequences.',
      '',r'\parhead{Inference and scope}',
      r'Test seeds are 61022401 and 61022402, with 256 sequences per seed at each of 16, 32, and 64 steps per block. Paired sequence bootstrap intervals use 4,000 draws, preserve common-seed dependence across step conditions, and average the three fixed readout errors within each sequence. The two primary pooled contrasts use 97.5\% intervals; all other intervals are descriptive 95\% intervals. The primary scorer is GPT-2-Large; OPT-125M evaluates exactly the same output edits. Both use BF16 inference and float32 cross entropy. These intervals condition on the released model, the matching pool, and the fitted readouts. No result establishes a universal ceiling for affine fitting or an architecture-isolated quality ranking.',
      '',r'\begin{figure}[!htbp]',r'\centering',r'\includegraphics[width=\linewidth]{figures/figure_aurora_replication_20261022.pdf}',
      r'\caption{Independent AURORA replication on new generated states. Step counts denote updates per 16-token block. Positive values favor (a) the composition-aware predictor or (b) targeted repair over the mean of three random repair arms. Each repair arm corrects exactly the same number of native disagreements within its sequence. Pooled primary contrasts use 97.5\% paired intervals; the other intervals are descriptive 95\%. These are conditional evaluator-score effects, not factuality or deployed speedups.}',r'\label{fig:aurora-replication}',r'\end{figure}','',
      r'\begin{table}[!htbp]',r'\centering\small',r'\caption{Frozen prediction on AURORA. The frequency-matched sensitivity substitutes the ELF relative-frequency cutoff for the primary training-count cutoff. All conditions are retained.}',r'\label{tab:aurora-prediction}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Condition & Sequences & Baseline MAE & Tail MAE & Freq.-matched MAE \\',r'\midrule']
    for cond in ['S16','S32','S64','all']:
        q=pred[cond];appendix.append(f"{cond} & {q['sequences']} & {q['baseline_mae']:.4f} & {q['tail_mae']:.4f} & {q['frequency_matched_mae']:.4f} "+chr(92)*2)
    appendix += [r'\bottomrule',r'\end{tabular}',r'\end{table}','',r'\begin{table}[!htbp]',r'\centering\small',
      r'\caption{Random-minus-targeted repair contrasts in nats per evaluator token. Brackets are paired intervals; pooled GPT-2-Large rare-first uses 97.5\%, and other entries use descriptive 95\%. Positive values favor the targeted policy.}',r'\label{tab:aurora-repair}',r'\begin{tabular}{llll}',r'\toprule',r'Condition & Policy & GPT-2-Large & OPT-125M \\',r'\midrule']
    for cond in ['S16','S32','S64','all']:
        for policy in ['rare','numeric','lowmargin']:
            appendix.append(f"{cond} & {policy} & {interval(repairs[cond,policy])} & {interval(second[cond,policy])} "+chr(92)*2)
    appendix += [r'\bottomrule',r'\end{tabular}',r'\end{table}','',r'\begin{table}[!htbp]',r'\centering\small',
      r'\caption{Every independently fitted AURORA readout and its pooled text score. S16--S64 columns give lexical agreement (\%), excluding native EOS positions. PPL is token-weighted on each evaluator\textquotesingle s common scorable set, with native EOS fixed across arms. These agreements do not identify an intrinsic decoder-complexity ceiling; full fitting-coverage and condition-specific scores accompany the evidence.}',r'\label{tab:aurora-fits}',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Readout & Update & S16 & S32 & S64 & GPT-2-L PPL & OPT PPL \\',r'\midrule']
    def ppl_text(value):
        if value<10000:return f'{value:.2f}'
        exponent=int(np.floor(np.log10(value)))
        return rf'${value/10**exponent:.2f}\!\times\!10^{{{exponent}}}$'
    appendix.append('Native & -- & 100 & 100 & 100 & '+ppl_text(primary_ppl['all','native'])+' & '+ppl_text(secondary_ppl['all','native'])+' '+chr(92)*2)
    for seed in [17,29,43]:
        fit=json.loads((OUT/f'fits/seed{seed}/selected.json').read_text());vals=[]
        for cond in ['S16','S32','S64']:
            rr=[q for q in agreements if q['condition']==cond and q['method']==f'linear{seed}']
            vals.append(100*sum(int(q['lexical_correct']) for q in rr)/sum(int(q['lexical_tokens']) for q in rr))
        appendix.append(f"Affine {seed} & {fit['selected_step']} & "+' & '.join(f'{v:.2f}' for v in vals)+' & '+ppl_text(primary_ppl['all',f'linear{seed}'])+' & '+ppl_text(secondary_ppl['all',f'linear{seed}'])+r' \\')
    appendix += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    if controls:
        appendix += [r'\parhead{Regularization controls}',
          r'This sensitivity audit was added after the local protocol freeze, before inspection of the new AURORA test-quality outcomes; it is not part of the original confirmatory contrasts. ELF outcomes were already known, so its control comparisons are retrospective. All predictors are fitted only on the saved calibration set. Duplicating each of the three baseline features once halves its effective ridge penalty without adding information. Adding three copies of the baseline error rate quarters that feature\textquotesingle s penalty while leaving the other penalties unchanged. We compare the original augmented predictor against both controls and separately compare unpenalized baseline and augmented fits (standardized OLS with fixed \texttt{rcond=1e-10}). Every condition is retained; intervals here are descriptive paired 95\% intervals.', '']
        for family,name in [('elf','ELF'),('aurora','AURORA')]:
            pooled=controls[family]['summary'][0]['contrasts']
            def ci5(q):return f"{q['mean']:+.5f} [{q['ci_low']:+.5f}, {q['ci_high']:+.5f}]"
            appendix.append(rf'On pooled {name}, duplicate-all-baseline minus augmented MAE is ${ci5(pooled["tail_over_duplicate_all"])}$ nats; duplicate-error-baseline minus augmented MAE is ${ci5(pooled["tail_over_duplicate_error"])}$. Under OLS, baseline minus augmented MAE is ${ci5(pooled["ols_tail_gain"])}$. Coefficients, per-sequence errors, algebraic penalty-equivalence checks, and all step/checkpoint-specific contrasts accompany the evidence.')
            appendix.append('')
    (dest/'v3_aurora_additions.tex').write_text('\n'.join(appendix))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.7),layout='constrained');conds=['S16','S32','S64','all'];x=np.arange(4)
    def points(ax,positions,rr,color,label=None):
        ax.vlines(positions,[z['ci_low'] for z in rr],[z['ci_high'] for z in rr],color=color,lw=1)
        for pos,z in zip(positions,rr):ax.hlines([z['ci_low'],z['ci_high']],pos-.035,pos+.035,color=color,lw=1)
        ax.scatter(positions,[z['mean'] for z in rr],color=color,s=17,label=label,zorder=3)
    points(axes[0],x,[pred[c] for c in conds],'#176b87');axes[0].set_title('(a) Prediction gain');axes[0].set_ylabel('Baseline − augmented MAE')
    for offset,policy,color,label in [(-.18,'rare','#176b87','Rare first'),(0,'numeric','#bd7b26','Numeric first'),(.18,'lowmargin','#b34d55','Low margin first')]:
        points(axes[1],x+offset,[repairs[c,policy] for c in conds],color,label)
    axes[1].set_title('(b) Equal-count repair');axes[1].set_ylabel('Random − targeted NLL');axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,frameon=False)
    for ax in axes:
        ax.axhline(0,color='.45',lw=.8);ax.set_xticks(x,['16 steps','32 steps','64 steps','Pooled']);ax.spines[['top','right']].set_visible(False)
    fig.savefig(dest/'figure_aurora_replication_20261022.pdf');fig.savefig(dest/'figure_aurora_replication_20261022.png',dpi=180);plt.close(fig)
    report=dict(status='evidence_complete_manuscript_inserts_ready_for_integration_review',prediction=primary['prediction'],repair=primary['repair'],secondary_repair=secondary['repair'],
        primary_prediction_supported=a['ci_low']>0,primary_rare_repair_supported=r['ci_low']>0,pooled_lexical_agreement=pooled_agreement,
        review_required=['Read all condition-specific outcomes and frequency coverage','Update abstract/introduction/conclusion/limitations consistently with these results','Preserve all prior long-manuscript evidence','Compile and visually inspect the integrated PDF, then rebuild the arXiv archive'],
        native_reference='Exact native argmax; EOS separators fixed across score arms',published=False,
        primary_ppl=rows(OUT/'analysis/gpt2-large/ppl.csv'),secondary_ppl=rows(OUT/'analysis/opt-125m/ppl.csv'),
        frequency_coverage=rows(OUT/'analysis/gpt2-large/coverage.csv'),regularization_controls=controls)
    temporary=dest/'RESULTS_AND_INTEGRATION.json.tmp'
    temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(dest/'RESULTS_AND_INTEGRATION.json')
    print(json.dumps({'status':report['status'],'primary_prediction_supported':report['primary_prediction_supported'],'primary_rare_repair_supported':report['primary_rare_repair_supported']},indent=2))

if __name__=='__main__':main()
