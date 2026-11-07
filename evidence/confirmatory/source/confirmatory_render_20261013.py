"""Render new results and integrate them without changing inherited evidence."""
from pathlib import Path
import csv,json,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/confirmatory_readout_20261013'
DOC=ROOT/'docs/arxiv_v3_full_20261013_confirmatory'
def read(name):return list(csv.DictReader((OUT/name).open()))
summary=read('summary.csv');runs=read('per_run.csv');phase=read('langflow_phase.csv');tail=read('tail_groups.csv');ppl=read('ppl/summary.csv')
follow=read('followup/summary.csv');follow_runs=read('followup/per_run.csv');follow_delta=read('followup/paired_improvement.csv')[0]
large=next(r for r in follow if int(r['train_sequences'])==8192);small=next(r for r in follow if int(r['train_sequences'])==1024)
follow_split=json.loads((OUT/'data/langflow_extended/split.json').read_text())
assert all(follow_split[k]==0 for k in ['exact_native_train_duplicates','exact_native_train_validation_overlap','exact_native_train_test_overlap'])
assert len(ppl)==5 and len(runs)==21
def pct(x,n=2):return f'{100*float(x):.{n}f}'
def avg(rows,key):return statistics.mean(float(r[key]) for r in rows)
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(6.65,2.6),layout='constrained')
for name,color,mark,label in [('test','#2678a6','o','Fixed historical test'),('fresh_test','#c36b36','s','Fresh test')]:
    rows=[r for r in summary if r['family']=='elf' and r['test_set']==name]
    x=np.array([int(r['train_sequences']) for r in rows]);y=np.array([float(r['agreement_mean'])*100 for r in rows]);lo=np.array([float(r['sequence_ci_low'])*100 for r in rows]);hi=np.array([float(r['sequence_ci_high'])*100 for r in rows])
    axes[0].errorbar(x,y,yerr=[y-lo,hi-y],color=color,marker=mark,lw=1.35,capsize=2,ms=4,label=label)
axes[0].set(xscale='log',xticks=[1024,4096,16384],xticklabels=['1,024','4,096','16,384'],xlabel='Training sequences',ylabel='Native-token agreement (%)',ylim=(95.95,97.6),title='(a) ELF: fixed fitting compute')
axes[0].legend(loc='lower right',frameon=False);axes[0].grid(axis='y',alpha=.16)
for method,color,mark,label in [('lookup','#2678a6','o','Native embedding lookup'),('residual','#c36b36','s','Backbone term only')]:
    rows=[r for r in phase if r['method']==method];x=np.array([int(r['step']) for r in rows])/63;y=np.array([float(r['agreement'])*100 for r in rows]);lo=np.array([float(r['sequence_ci_low'])*100 for r in rows]);hi=np.array([float(r['sequence_ci_high'])*100 for r in rows])
    axes[1].plot(x,y,color=color,marker=mark,lw=1.35,ms=4,label=label);axes[1].fill_between(x,lo,hi,color=color,alpha=.15,lw=0)
axes[1].set(xlabel='Normalized sampler phase',ylabel='Same-phase native agreement (%)',ylim=(-3,103),xticks=[0,.25,.5,.75,1],title='(b) LangFlow: native logit terms')
axes[1].legend(loc='upper center',bbox_to_anchor=(.53,.99),frameon=False);axes[1].grid(axis='y',alpha=.16)
fig.savefig(DOC/'figures/figure_confirmatory_readout_20261013.pdf');fig.savefig(DOC/'audit/confirmatory_readout_preview.png',dpi=170);plt.close(fig)

app=r'''\section{Readout repetitions on fixed tests and a native-interface control}
\label{app:confirmatory-readout}

\parhead{Protocol fixed before fitting}
This study adds new readout fits and new generations to the retrospective audit in \secref{app:readout-selection}. A local protocol file, saved before the initial fits and generations, fixes the splits, fitting seeds, update budget, validation rule, and reporting scope; it is not a public preregistration. No denoiser or native decoder is retrained. All twelve initial fits are reported, including the weaker small-budget LangFlow result. Three additional LangFlow fits belong to an explicitly post-result token-budget follow-up below. Training inputs are the sampler state before the complete contextual decoder, never a decoder hidden state.

For ELF-B, we partition the historical 32,768-sequence cache with split seed 20261013 into nested training sets of 1,024, 4,096, and 16,384 sequences, one validation set of 512, and one common test set of 1,024. No sequence belongs to more than one split. There are no empty or exactly duplicated saved native sequences in the source pool. The nested training sets contain 1,047,155, 4,188,683, and 16,754,550 valid token positions. These historical sequences were used in earlier work, so the common test is disjoint for the new fits but not globally untouched. We additionally generate 512 test sequences with independent seed 61013001 after fixing the protocol, using the released ELF-B OWT checkpoint's \texttt{ema\_params1} weights, 32 SDE steps, noise coefficient 1.5, self-conditioning guidance 3, the logit-normal grid, and a final ODE step. None duplicates a historical native token sequence exactly. The fixed and fresh tests contain 1,047,171 and 523,560 valid positions, respectively.

Each affine readout uses fitting seeds 17, 29, and 43, with 4,096 AdamW updates of 1,024 token positions sampled with replacement, learning rate $5\times10^{-4}$, weight decay $0.01$, and Gaussian input noise $0.02$ times the batch standard deviation. Parameters are float32; fitting uses bfloat16 matrix products and evaluation uses float32 with TF32 disabled. Validation native-token agreement is checked at updates 1,024, 2,048, and 4,096, with ties selecting the earlier update. All twelve initial fits select update 4,096. Tests are evaluated only after selection. Thus the comparison fixes 4,194,304 sampled token exposures, not epochs or convergence; changing the available pool changes the diversity and revisit rate of those exposures. It is not an exact replication of the older optimization schedule or an asymptotic sample-complexity estimate. In particular, the higher agreement of the new smallest-pool fit is not an improvement claim over historical MDP: fitting exposures, training counts, and test composition differ.

ELF agreement counts native nonpadding positions before the first native EOS. We separately apply each readout's predicted EOS truncation: it changes none of the nine learned-readout agreement scores on either test. Frozen cosine lookup uses the complete released T5 embedding table; its fresh-test agreement changes only from $93.2143\%$ to $93.2128\%$ under predicted EOS truncation. Neither measure evaluates token behavior beyond the native boundary. Means below average the three fitted readouts. Fit-seed standard deviations quantify optimizer/initialization variation on this one split. The 95\% intervals use 4,000 sequence-cluster bootstrap draws with seed 61013, reusing each sampled sequence across fits and compared budgets. They are conditional on these fitted models and this data split, not confidence intervals over new denoiser training seeds or new train-set partitions.

\begin{table}[!htbp]
\centering\small
\caption{Common-test ELF readout repetition. All entries are percentages; $\pm$ is the standard deviation across three fitting seeds. Brackets give conditional sequence-bootstrap 95\% intervals. Pool sizes here count training sequences only, unlike the historical total-pool labels.}
\label{tab:confirmatory-elf}
\begin{tabular}{rcc}
\toprule
Train sequences & Fixed historical test & Fresh test \\
\midrule
'''
for n in [1024,4096,16384]:
    cells=[]
    for name in ['test','fresh_test']:
        r=next(r for r in summary if r['family']=='elf' and int(r['train_sequences'])==n and r['test_set']==name)
        cells.append(f"${pct(r['agreement_mean'])}\\pm{pct(r['fit_seed_sd'],3)}$ [{pct(r['sequence_ci_low'])}, {pct(r['sequence_ci_high'])}]")
    app+=f'{n:,} & '+' & '.join(cells)+r' \\'+'\n'
app+=r'''\bottomrule
\end{tabular}
\end{table}

The 16,384-minus-1,024 paired gain is $0.851$ percentage points on the fixed historical test (95\% interval $[0.811,0.891]$) and $0.849$ on the fresh test ($[0.800,0.901]$). Both tests support a gain from the larger available fitting pool at this fixed update budget. They do not establish a universal scaling curve. Frozen cosine lookup attains $93.21\%$ on the fresh test, below all three learned readouts at the largest budget.

\begin{table}[!htbp]
\centering\small
\caption{Every scheduled fit, selected on validation only. Agreements are percentages. ELF columns use the same two tests at all budgets; LangFlow has its own independent 512-sequence test. No best-seed result is selected.}
\label{tab:confirmatory-fits}
\begin{tabular}{lrrrrr}
\toprule
Interface & Train sequences & Fit seed & Validation & Fixed test & Fresh test \\
\midrule
'''
for r in runs:
    if r['test_set']!='test':continue
    fr=next((f for f in runs if f['family']==r['family'] and f['train_sequences']==r['train_sequences'] and f['fit_seed']==r['fit_seed'] and f['test_set']=='fresh_test'),None)
    family='ELF' if r['family']=='elf' else 'LangFlow'
    app+=f"{family} & {int(r['train_sequences']):,} & {r['fit_seed']} & {pct(r['validation_agreement'])} & {pct(r['agreement'])} & {pct(fr['agreement']) if fr else '--'}"+r' \\'+'\n'
app+=r'\midrule'+'\n'
for r in follow_runs:
    if int(r['train_sequences'])!=8192:continue
    app+=f"LangFlow follow-up & 8,192 & {r['fit_seed']} & {pct(r['validation_agreement'])} & {pct(r['agreement'])} & --"+r' \\'+'\n'
app+=r'''\bottomrule
\end{tabular}
\end{table}

\parhead{Residual errors and conditional text scoring}
On the fresh ELF test, we retain the numeric-token definition (a digit in the tokenizer token string), report training-frequency groups, and inspect native-margin strata. The frequency threshold is a count in the largest training pool, not a test-frequency rank; it differs from the historical rare-token quintile. Native-margin groups are diagnostic oracle information. Their concentrations do not imply a deployable confidence gate. \tabref{tab:confirmatory-tail} shows that more fitting data has not removed the structured tail. All 127 valid positions whose target type was unseen in training remain wrong for all three fits.

\begin{table}[!htbp]
\centering\small
\caption{Fresh ELF residual groups for all three 16,384-sequence readouts. Agreement is mean $\pm$ fitting-seed SD in percent. Groups from different definitions overlap and must not be added. Native margins are unnormalized confidence gaps.}
\label{tab:confirmatory-tail}
\begin{tabular}{lrr}
\toprule
Group & Native positions & Agreement (\%) \\
\midrule
'''
names={'all':'All','numeric':'Numeric','non_numeric':'Nonnumeric','train_count_1_to_99':'Training count 1--99','train_count_at_least_100':'Training count $\geq100$','unseen_training_token':'Unseen target type','margin_below_2':'Native margin $<2$','margin_2_to_8':'Native margin $[2,8)$','margin_at_least_8':'Native margin $\geq8$'}
for key,label in names.items():
    rows=[r for r in tail if r['group']==key];rates=[100*float(r['agreement']) for r in rows]
    app+=f"{label} & {int(rows[0]['tokens']):,} & ${statistics.mean(rates):.2f}\\pm{statistics.stdev(rates):.2f}$"+r' \\'+'\n'
app+=r'''\bottomrule
\end{tabular}
\end{table}

For a paired text diagnostic, all readouts use the same native validity mask before T5 detokenization. GPT-2-Large scores each resulting text with length capped at 1,024 evaluator tokens, and corpus PPL is $\exp(\sum\mathrm{NLL}/\sum\mathrm{tokens})$. Detokenization can change evaluator token counts even with a common native boundary. No text has zero scored tokens. This oracle-boundary comparison tests conditional readout fidelity, not autonomous generation quality, and is not pooled with older PPL protocols.

\begin{table}[!htbp]
\centering\small
\caption{Conditional text scoring on the same 512 fresh ELF states. Every largest-budget fit is included. Native boundaries and evaluator truncation are fixed; lower PPL alone does not establish a better generator.}
\label{tab:confirmatory-ppl}
\begin{tabular}{lrr}
\toprule
Readout & GPT-2-Large PPL & Scored evaluator tokens \\
\midrule
'''
for r in ppl:
    name={'native':'Native decoder','frozen_lookup':'Frozen cosine lookup'}.get(r['method'],r['method'].replace('linear_seed','Linear, seed '))
    app+=f"{name} & {float(r['ppl']):.2f} & {int(r['scored_tokens']):,}"+r' \\'+'\n'
app+=r'''\bottomrule
\end{tabular}
\end{table}

\parhead{LangFlow tests an architectural alternative}
The released LangFlow LM1B checkpoint has a learned 768-dimensional embedding table and a 30,522-token BERT vocabulary. We generate 1,024 training, 256 validation, and 512 test sequences of 128 positions at 64 sampler evaluations, with seed 61013002. All positions follow the released sampling convention; unlike ELF, no EOS mask is applied. The three partitions contain no exactly duplicated native token sequences within or across partitions. Readout fitting and validation follow the same update protocol, but the number of available training positions is only 131,072. Thus this is an interface-specific feasibility check, not a matched-token-budget comparison to ELF.

The native logits have the implementation-level decomposition
\[
\ell_\gamma(z,s)=R_\gamma(z,s)+c(\gamma)zE^\top,
\qquad
c(\gamma)=\exp\!\left(\frac{\operatorname{softplus}(-\gamma)-\gamma}{2}\right),
\]
where $E$ is the checkpoint's layer-normalized learned embedding table and $s$ is the sampler's self-conditioning input. We capture the actual backbone output $R_\gamma$ and verify exact reconstruction of native logits with the implementation's floating-point casts. The frozen lookup uses float32 dot products before the native skip's bfloat16 cast; it is not a bitwise ablation of the quantized skip. At five phases on the same 512 test trajectories, we compare the frozen dot-product lookup $\arg\max zE^\top$ and backbone-only $\arg\max R_\gamma$ to the full native decision at that phase. This is a fixed-state logit ablation: the trajectories and self-conditioning remain those of the complete model. It does not measure generation after removing a term throughout sampling, nor agreement to the eventual final token at earlier phases.

\begin{table}[!htbp]
\centering\small
\caption{LangFlow native-interface controls. Each row uses 65,536 test positions. Intervals are conditional 95\% sequence-bootstrap intervals in percent. Learned rows average three fits per budget; the 8,192-sequence row is a post-result follow-up. Phase rows require no new fitting.}
\label{tab:confirmatory-langflow}
\begin{tabular}{rcc}
\toprule
Phase index & Native embedding lookup & Backbone term only \\
\midrule
'''
for step in [0,16,32,48,63]:
    vals=[]
    for method in ['lookup','residual']:
        r=next(r for r in phase if int(r['step'])==step and r['method']==method)
        vals.append(f"{pct(r['agreement'])} [{pct(r['sequence_ci_low'])}, {pct(r['sequence_ci_high'])}]")
    app+=f'{step} & '+' & '.join(vals)+r' \\'+'\n'
app+=r'''\midrule
\multicolumn{2}{l}{Learned map, 1,024 train sequences} & 62.62 [62.26, 62.99] \\
\multicolumn{2}{l}{Learned map, 8,192 train sequences} & FOLLOW_MEAN [FOLLOW_LO, FOLLOW_HI] \\
\bottomrule
\end{tabular}
\end{table}

At the initial fitting budget, the affine map reaches $62.62\%\pm0.037$ percentage points on this LangFlow test. In an explicitly post-fit coverage diagnostic, target types seen at least 100 times during training occupy $51.02\%$ of test positions and have $99.04\%$ mean agreement. Types seen 1--99 times occupy $44.79\%$ with $26.99\%$ agreement; unseen types occupy $4.19\%$ with zero agreement. This concentration prevents interpreting the low overall score as proof of intrinsic nonlinear decoding difficulty or as a ceiling for larger fitting budgets. Frozen lookup has access to the checkpoint's complete learned vocabulary geometry, an advantage the independently initialized affine map must learn from its finite matching set.

The phase audit gives a narrower, useful boundary: at the final phase, native embedding lookup agrees on $89.69\%$ of positions and the backbone term alone on $55.86\%$, whereas at the initial phase the ordering is reversed ($0.014\%$ versus $94.06\%$). The full native decision still differs from float32 lookup on roughly one in ten final positions. These same-phase comparisons expose an explicit linear contribution to late readability; they do not independently reproduce ELF's frozen T5 geometry, prove a term unnecessary for transport, or rank the two systems.

\parhead{Post-result token-budget follow-up}
The initial LangFlow fitting pool contains eight times fewer token positions than the smallest ELF pool. After inspecting the initial results, a separate protocol freezes an extension to 8,192 LangFlow training sequences: the original 1,024 plus 7,168 generated with seed 61013003, batch size 16, and the same native sampler. The resulting 1,048,576 positions approximately match ELF's 1,047,155-position smallest pool. Validation and test files are unchanged byte for byte; optimizer settings, update budget, validation rule, and three fitting seeds are unchanged. Each additional fit restarts from scratch using the original seed list; it does not continue the smaller-pool weights. This is a result-motivated diagnostic extension on a previously inspected test, not an independent confirmation or a preregistered cross-architecture comparison. Its conditional intervals do not adjust for the choice to run this follow-up.

At this larger available pool, mean LangFlow agreement is $FOLLOW_MEAN\%\pm FOLLOW_SD$ percentage points, with conditional interval $[FOLLOW_LO,FOLLOW_HI]\%$. The paired change from the original three fits is $FOLLOW_DELTA$ percentage points ($[FOLLOW_DLO,FOLLOW_DHI]$). The six LangFlow fits and both fitting budgets are all retained. Even this larger fit remains below the $89.69\%$ frozen-lookup baseline. Increasing the available pool also changes exposure frequencies under fixed compute, so the gain does not separately identify vocabulary coverage, optimization, and representation effects. The unchanged test has 3,172 native special-token positions ($4.84\%$, all \texttt{[CLS]}). As a post-fit mask sensitivity check, excluding these positions gives SMALL_NORMAL\% and LARGE_NORMAL\% mean agreement at the two budgets; the all-position metric remains primary. The enlarged training set has no exact native-sequence duplicates or overlap with validation or test. This within-interface pool intervention measures how sensitive the learned probe is to finite matching data; it does not isolate tokenizer, architecture, or dataset effects between ELF and LangFlow.

\parhead{Label lineage and reproducibility}
As a training-split lineage check, we decode the first 32 newly assigned historical training states with the currently released ELF EMA decoder. It agrees with the stored labels on 32,707 of 32,720 valid positions ($99.9603\%$), but is not bitwise identical. The 13 mismatches have current top-two logit gaps between approximately $7.6\times10^{-6}$ and $0.1224$; the check alone does not identify their cause. Historical results retain their saved native targets, while fresh-test targets are computed directly in the new generation run. The ancillary package includes the frozen protocol, split identifiers, checkpoint and selected-readout hashes, validation histories, every per-sequence agreement count, phase controls, tail counts, and per-sequence evaluator NLL/token sums. A CPU verifier reproduces the aggregate agreements, seed variation, paired intervals, and corpus PPL from these scalar records. Original generation and fitting scripts are included as provenance; rerunning them also requires the local checkpoints and latent caches, which are not bundled.
'''
# Independently verify the explicitly printed historical-label numerator/denominator.
check=json.loads((OUT/'native_checkpoint_check.json').read_text());valid=sum(r['valid'] for r in check['rows']);correct=sum(r['correct'] for r in check['rows'])
app=app.replace('32,707 of 32,720',f'{correct:,} of {valid:,}')
for key,val in {'FOLLOW_MEAN':pct(large['agreement_mean']),'FOLLOW_LO':pct(large['sequence_ci_low']),'FOLLOW_HI':pct(large['sequence_ci_high']),'FOLLOW_SD':pct(large['fit_seed_sd'],3),'FOLLOW_DELTA':pct(follow_delta['delta']),'FOLLOW_DLO':pct(follow_delta['ci_low']),'FOLLOW_DHI':pct(follow_delta['ci_high']),'SMALL_NORMAL':pct(small['nonspecial_agreement_mean']),'LARGE_NORMAL':pct(large['nonspecial_agreement_mean'])}.items():app=app.replace(key,val)
(DOC/'v3_confirmatory_additions.tex').write_text(app)

# Edit from the preserved release, making repeated rendering idempotent.
main=(ROOT/'docs/arxiv_v3_full_20261013/main.tex').read_text()
old='After successful transport, frozen token lookup recovers $93$--$96\\%$ of native decisions and a fixed-variant linear readout reaches $97.9\\%$, leaving a structured rare/numeric tail.'
new='After successful transport, frozen token lookup recovers $93$--$96\\%$ of native decisions. New fixed-test, three-seed readout fits reach $97.17\\%$ on independently generated ELF states, leaving a structured rare/numeric tail; a LangFlow control exposes the role of its explicit linear skip.'
assert old in main;main=main.replace(old,new)
old=r'Frozen lookup recovers $93$--$96\%$ of native decisions, and a fixed-variant linear readout reaches $97.9\%$ on its held-out generated states. Residual errors concentrate in rare, numeric, and low-margin tokens.'
new=r'Frozen lookup recovers $93$--$96\%$ of native decisions. A new comparison with nested fitting pools, common tests, and three fitting seeds reaches $97.17\%$ on fresh ELF states; a native LangFlow decomposition tests the alternative explanation of an explicit linear skip. Residual ELF errors concentrate in rare, numeric, and low-margin tokens.'
assert old in main;main=main.replace(old,new)
main=main.replace('Matching-pool size includes training and evaluation sequences. The splits change with pool size,','Historical matching-pool size includes training and evaluation sequences. Those splits change with pool size,')
anchor=r'The cross-scale confirmation in \figref{fig:mdp-crossscale}'
linear=[float(r['ppl']) for r in ppl if r['method'].startswith('linear')];native=float(next(r['ppl'] for r in ppl if r['method']=='native'))
insert=r'''\leadin{New fits on common tests.} To remove the changing-test-set ambiguity, we freeze nested training pools of 1,024, 4,096, and 16,384 sequences, one 512-sequence validation set, and one 1,024-sequence historical test before fitting. Three fitting seeds use the same 4,096-update budget and validation-only checkpoint selection. All nine readouts are also evaluated on 512 newly generated test sequences. Mean agreement rises from $96.43\%$ to $97.28\%$ on the historical test and from $96.32\%$ to $97.17\%$ on the fresh test; the latter paired gain is $0.85$ percentage points (95\% sequence-bootstrap interval $[0.80,0.90]$). Fit-seed variation is reported separately from these conditional test-sequence intervals. This supports a fitting-pool effect at fixed optimization work, without treating the previously used historical pool as globally untouched or equating the new schedule with the original sweep. The fresh-test linear-readout PPL remains LINEAR_RANGE versus NATIVE_PPL for native decoding under a common native boundary. The structured error tail persists (\secref{app:confirmatory-readout}).

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{figures/figure_confirmatory_readout_20261013.pdf}
\caption{New readout controls with the protocol fixed before fitting. (a) ELF mean agreement across three fits per nested training budget, evaluated on the same historical and fresh test sequences. Error bars are conditional 95\% sequence-bootstrap intervals; fitting-seed SD is separate in \tabref{tab:confirmatory-elf}. (b) LangFlow's native embedding lookup and isolated backbone output on the complete model's stored trajectories. Agreement is with the full decoder at the same phase, not with eventual final tokens. Shading gives sequence-bootstrap intervals. Different tokenizers, sequence lengths, and training-token budgets preclude cross-panel score ranking.}
\label{fig:confirmatory-readout}
\end{figure}

'''.replace('LINEAR_RANGE',f'${min(linear):.2f}$--${max(linear):.2f}$').replace('NATIVE_PPL',f'${native:.2f}$')
assert anchor in main;main=main.replace(anchor,insert+anchor)
main=main.replace('Because the held-out sequences change with pool size, this is not a controlled learning curve on a common test set.','The historical comparison changes held-out sequences; the new fixed-test repetition above supplies the corresponding controlled comparison at one fixed optimization budget.')
anchor=r'\begin{figure}[!htbp]'+'\n  \\centering\n  \\includegraphics[width=\\linewidth]{figures/figure35_external_interface_boundary.png}'
# Locate the end of the main external-boundary paragraph without depending on image filenames.
end='ELF carries the main mechanism evidence, while LangFlow, CoBit, and Cola-DLM are boundary checks that ask whether denoisability, decoder compatibility, and trajectory reliability remain meaningful when the state object and decoder change.'
assert end in main
main=main.replace(end,end+r'''

A new native-interface control makes the LangFlow boundary concrete (\secref{app:confirmatory-readout}). On 512 fresh LM1B trajectories, frozen lookup in LangFlow's own learned embedding table agrees with $89.69\%$ of final native decisions; keeping only its contextual backbone term agrees with $55.86\%$. The native logits explicitly add a linear embedding skip to the backbone output, so high late lookup agreement has an architectural explanation and is not independent evidence for ELF's T5 mechanism. A separately fitted affine map reaches $62.62\%$ across three fits at the initial 131,072-token budget; a post-fit coverage audit localizes most errors to sparsely observed token types. A subsequent, explicitly result-motivated extension to 1,048,576 training positions reaches $FOLLOW_MAIN\%$ on the same test with the same update budget. The enlarged readout still falls below frozen lookup. These controls expose the probe's budget dependence without isolating vocabulary coverage from optimization effects or supporting a cross-model ranking. The native-term audit holds trajectories and self-conditioning fixed; it does not remove either term during generation.
''')
main=main.replace('After successful transport, the original frozen and linear readouts still recover most native decisions, leaving the full rare/numeric-tail evidence intact.','After successful transport, new common-test readout repetitions retain the bulk-recoverability result on fresh ELF states and preserve the rare/numeric-tail evidence. The LangFlow decomposition makes the architectural scope explicit: a native linear skip can contribute substantially to late readability, while finite fitting coverage limits an independently learned probe.')
anchor='The external checks are too few and too heterogeneous to support a rank correlation between basin width and PPL across architectures.'
main=main.replace(anchor,'The new fixed-test study repeats readout fitting, not denoiser training, on one historical split and one fresh generation seed. Its fixed update budget does not establish convergence or asymptotic sample efficiency. LangFlow changes tokenizer, sequence length, and the native linear-skip architecture; its approximately token-matched extension was chosen after inspecting the initial test results. These comparisons do not isolate architecture effects or supply an independent confirmation of the post-fit coverage explanation. '+anchor)
main=main.replace(r'\input{v3_october_additions.tex}',r'\input{v3_october_additions.tex}'+'\n\n'+r'\input{v3_confirmatory_additions.tex}')
main=main.replace('FOLLOW_MAIN',pct(large['agreement_mean']))
(DOC/'main.tex').write_text(main)
october=(ROOT/'docs/arxiv_v3_full_20261013/v3_october_additions.tex').read_text()
october+='\n'+r'The new fits in \secref{app:confirmatory-readout} address the changing-test-set limitation separately; they do not replace these historical numerical results or use their optimization schedule.'+'\n'
(DOC/'v3_october_additions.tex').write_text(october)
print(json.dumps({'new_figure':'figure_confirmatory_readout_20261013.pdf','new_tables':5,'initial_fits':12,'post_result_followup_fits':3,'linear_ppl_range':[min(linear),max(linear)],'native_ppl':native,'langflow_followup_agreement':float(large['agreement_mean'])},indent=2))
