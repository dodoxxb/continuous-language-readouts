"""Integrate the prospective evidence into a preserved full-length revision."""
from pathlib import Path
import json,re,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prospective_tail_20261013 import ROOT,OUT,readcsv

DOC=ROOT/'docs/arxiv_v3_full_20261013_predictive'
OLD=ROOT/'docs/arxiv_v3_full_20261013_confirmatory'

def main():
    analysis=json.loads((OUT/'analysis.json').read_text());pred=readcsv(OUT/'prediction_summary.csv');repair=readcsv(OUT/'repair_summary.csv');ppl=readcsv(OUT/'ppl_summary.csv');allrow=pred[0]
    secondary=readcsv(OUT/'secondary_repair_summary.csv');secondary_ppl=readcsv(OUT/'secondary_ppl_summary.csv')
    supported=analysis['primary_prediction_supported'];rare_supported=analysis['primary_rare_repair_supported']
    sign='reduces' if float(allrow['mae_reduction'])>0 else 'increases'
    pc=abs(float(allrow['mae_reduction']))/float(allrow['baseline_mae'])*100
    pdrow=next(r for r in pred if r['condition']=='PD8')
    pd_worse=-float(pdrow['mae_reduction'])/float(pdrow['baseline_mae'])*100
    prediction_sentence=(f'The frozen tail-aware predictor {sign} mean absolute prediction error by ${pc:.1f}\\%$ in the pooled test. ' + ('The multiplicity-adjusted interval supports this aggregate improvement. The predictor improves on ELF-B and ELF-M but worsens MAE by '+f'${pd_worse:.1f}\\%$ on the distilled student, delimiting its transfer.' if supported else 'The prespecified positive-transfer hypothesis is not supported by its multiplicity-adjusted interval.'))
    # Compact data-only plot. Interpretation, sample sizes and limitations stay in caption.
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(1,3,figsize=(7.0,2.2),constrained_layout=True);names=['B32','M32','PD8'];xx=np.arange(3)
    for j,method in enumerate(['baseline','tail']):
        ax[0].bar(xx+(j-.5)*.32,[float(next(r for r in pred if r['condition']==c)[method+'_mae']) for c in names],width=.3,label='Aggregate' if j==0 else '+ Tail features',color=['#75869b','#087f8c'][j])
    ax[0].set(xticks=xx,xticklabels=names,ylim=(0,.095),yticks=[0,.02,.04,.06,.08],ylabel='Prediction MAE (nats)',title='(a) Frozen quality prediction');ax[0].legend(frameon=False,loc='upper left',fontsize=6.5)
    for j,policy in enumerate(['rare','numeric','lowmargin']):
        rr=[next(r for r in repair if r['condition']==c and r['policy']==policy) for c in names];means=np.array([float(r['mean_random_minus_target_nll']) for r in rr]);lo=np.array([float(r['ci_low']) for r in rr]);hi=np.array([float(r['ci_high']) for r in rr]);pos=xx+(j-1)*.19
        ax[1].errorbar(pos,means,yerr=[means-lo,hi-means],fmt='o',markersize=3,capsize=2,label={'rare':'Rare first','numeric':'Numeric first','lowmargin':'Native low margin'}[policy],color=['#bd6a23','#7757a0','#087f8c'][j])
    ax[1].axhline(0,color='.55',lw=.6);ax[1].set(xticks=xx,xticklabels=names,ylabel='Random − targeted NLL',title='(b) Equal-error repair');ax[1].legend(frameon=False,fontsize=6.5)
    for method,label,color in [('native','Native','#242c36'),('linear_seed17','Linear (seed 17)','#75869b'),('repair_random11','Random repair A','#7757a0'),('repair_rare','Rare-first repair','#bd6a23')]:
        ax[2].plot(xx,[float(next(r for r in ppl if r['condition']==c and r['method']==method)['ppl']) for c in names],'-o',markersize=3,label=label,color=color,lw=1)
    ax[2].set(xticks=xx,xticklabels=names,ylabel='GPT-2-Large PPL',title='(c) Conditional text score');ax[2].legend(frameon=False,fontsize=6)
    for a in ax:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
    fig.savefig(DOC/'figures/figure_prospective_tail_20261013.pdf');fig.savefig(DOC/'audit/prospective_tail_preview.png',dpi=180);plt.close(fig)
    # Preserve the inherited source and all numerical tables; edit framing only.
    s=(OLD/'main.tex').read_text()
    title='The Limits of Decoder Readability in Continuous Language Diffusion'
    s=s.replace('pdftitle={Decoder Basins Are Targets, Not Transport in Continuous Language Diffusion}',f'pdftitle={{{title}}}')
    s=s.replace(r'\title{\vspace{-1.5em}Decoder Basins Are Targets, Not Transport\\in Continuous Language Diffusion}',r'\title{\vspace{-1.5em}The Limits of Decoder Readability\\in Continuous Language Diffusion}')
    abstract=r'''How much does an easily decoded endpoint explain about continuous language generation? In Embedded Language Flows (ELF), a token-wise affine readout recovers $97.17\%$ of native decisions on fresh generated states, yet retains a text-score gap and a structured error tail. We test this boundary prospectively: readouts and quality predictors calibrated on ELF-B are transferred without refitting to 1,536 new generations from ELF-B, ELF-M, and a released distillation student. Adding tail-specific error rates reduces pooled prediction MAE by $18.2\%$, improving on B/M but worsening it by $7.0\%$ on the student. A paired intervention then repairs exactly the same number of disagreements within each sequence. Rare-first repair outperforms random repair in the pooled test, whereas low-native-margin-first repair underperforms it; both directions persist under a second evaluator. Thus error enrichment, decoder confidence, and external-score impact are distinct. Supporting controls expose two further limits: explicit counterexamples prevent a sensitivity-normalized margin proxy from serving as a stability certificate, and decoder-dose and fixed-state decoder-swap experiments show that stronger corruption recovery need not improve generation from noise. An external LangFlow decomposition delimits transfer through its architectural linear skip. Together, these results characterize what simple readouts preserve and where local decoder measurements fail. The contribution is controlled empirical evidence, not a new generator or end-to-end acceleration method.'''
    s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',s,count=1,flags=re.S)
    a=s.index('Continuous language diffusion succeeds');b=s.index(r'\begin{figure}',a)
    intro=r'''Continuous language diffusion performs most of its computation before converting a continuous state into tokens. Embedded Language Flows (ELF)~\citep{elf2026} makes this design concrete: a learned flow generates states in a frozen T5~\citep{t52020} representation space, and a shared contextual network performs the final decoding. This raises a specific empirical question: after the full generation process, how much contextual decoding is still needed, and what does a simpler readout fail to preserve?

The first observation is a large simplification. A fixed token-wise affine map recovers $97.17\%$ of the native decoder's decisions on newly generated ELF states, averaged over three readout fits selected on validation data. Frozen lookup in the T5 token table already recovers more than $93\%$. Nevertheless, the learned readouts retain a conditional text-score gap: GPT-2-Large PPL is $26.27$--$26.33$, compared with native $24.71$ under the same native output boundary. Disagreements are concentrated in rare, numeric, and low-margin positions. Thus the bulk of native token realization becomes simple on these generated states, while the remaining errors are structured.

That observation alone does not explain the quality gap. Rare-token enrichment is descriptive; it need not predict the effect of replacing a readout on another checkpoint, and a token that matters for exact fidelity need not have the largest effect on an external language-model score. We therefore freeze a quality predictor on previously inspected ELF-B calibration data and test it, without refitting, on new ELF-B, ELF-M, and distilled-student generations. A second intervention holds the number of repaired disagreements fixed within every sequence and changes only their selection. These tests ask whether tail composition adds information beyond total disagreement, and whether equal agreement entails equal conditional text scores (\secref{sec:prospective-readout}).

The broader controls locate this readout result within generation. Representation and decoder define which states are readable; the denoiser and sampler determine which states are reached. ELF already combines flow and noisy decoder training, so this operational decomposition is not our contribution. Our experiments instead test the limits of the measurements: a higher handed-state margin need not imply better generated text, a first-order stability proxy need not certify robustness, and an easily read endpoint does not remove the work needed to generate it. Because denoising and decoding share weights, these roles are not independently identified causal modules. LangFlow's explicit linear skip provides an architectural boundary check rather than a matched replication of ELF geometry.

The full manuscript preserves the representation, trajectory, objective, and sampler evidence behind these distinctions. Its central claim is narrower than a universal theory of language generation: local decoder measurements quantify different properties, and their relationship to output quality must be tested under frozen comparisons and interventions. \figref{fig:overview} illustrates the operational roles; the prospective readout test leads the results, followed by the controls that establish what local readability can and cannot support.

'''
    s=s[:a]+intro+s[b:]
    a=s.index('The measurement distinction is central:');b=s.index(r'\section{Related Work}',a)
    contrib=r'''\parhead{Contributions}
\begin{enumerate}[leftmargin=1.4em,itemsep=0.15em,topsep=0.15em]
\item \textbf{A measured and prospectively tested readout boundary.} Common-test repetitions establish high bulk recoverability with a structured error tail. Frozen transfer to new checkpoint conditions and equal-error repair test whether that tail predicts or changes a conditional text score. All outcomes, including failed predictions, are retained.
\item \textbf{A falsifiable assessment of local confidence.} Matched representation controls separate denoisability, recoverability, and order sensitivity. A standard local stability bound motivates a sensitivity-normalized proxy; phase-and-scale-controlled perturbations measure its predictive value, and explicit flips below its radius rule out interpreting it as a certificate.
\item \textbf{Controlled evidence separating recovery from generation.} A two-seed, four-dose training grid, contextual-decoder swaps, and compute-aware sampler controls test whether stronger recovery on supplied states implies better outputs from noise. These results establish operational boundaries in a shared-weight system, with heterogeneous external interfaces delimiting their scope.
\end{enumerate}

'''
    s=s[:a]+contrib+s[b:]
    a=s.index('The experiments follow four questions');b=s.index(r'\parhead{Models and data}',a)
    s=s[:a]+r'''We begin with the frozen readout-quality test (\secref{sec:prospective-readout}), which directly tests the inferential step from a structured error tail to output quality. The supporting experiments then address three questions. \textbf{Which properties make states readable?} Covariance, shuffle, PCA, semantic, and decoder controls separate candidate measurements (\secref{sec:denoise-recover-decouple}--\secref{sec:order-sensitivity}). \textbf{What does local reliability predict about generation?} Trajectory, perturbation, decoder-dose, and sampler controls test the limits of confidence and handed-state recovery (\secref{sec:trajectory-reliability-results}--\secref{sec:interface-phase-diagram}). \textbf{How far can decoding be simplified?} The complete historical and common-test readout studies, long-form checks, and external systems delimit transfer and residual errors (\secref{sec:three-minimal-probes}--\secref{sec:ruled-out-explanations}). Detailed evidence and negative controls remain in the appendix. \tabref{tab:scale-summary} retains the scale of the supporting studies; the new prospective test is specified separately in \secref{app:prospective-readout}.

'''+s[b:]
    s=s.replace(r'\section{Diagnostic Protocol}',r'\input{v3_predictive_main.tex}'+'\n\n'+r'\section{Diagnostic Protocol}',1)
    s=s.replace('The next section decomposes this compatibility into five measurable axes that together define a diagnostic protocol for evaluating candidate state spaces.','We first test the relationship between endpoint readability and a conditional text score. The subsequent diagnostic protocol decomposes interface compatibility into five measurable axes for the supporting experiments.')
    s=s.replace('This phase-diagram view is the main conceptual object of the paper.','This phase-diagram view supplies a temporal description supporting the readout and local-stability tests.')
    s=s.replace('The fixed-variant comparison retains the bulk-readout conclusion and its residual calibration gap.','The fixed-variant comparison retains the bulk-readout conclusion and the remaining native-agreement and PPL gaps.')
    s=s.replace('Even at 32k, however, the native decoder retains a measurable calibration advantage.','Even at 32k, however, the native decoder retains lower measured conditional PPL.')
    s=s.replace('while the remaining PPL gap shows that the native decoder still calibrates the tail.','while native agreement and conditional PPL remain distinct measurements. Tail composition and repair effects are tested in \\secref{sec:prospective-readout}.')
    s=s.replace('The remaining quality gap is concentrated in a structured residual tail that the native decoder calibrates better than a frozen lookup or linear readout, especially on rare and out-of-domain tokens.','Native-decision disagreements are concentrated in a structured residual tail, including rare and out-of-domain tokens. Their concentration alone does not establish which corrections improve an external text score.')
    s=s.replace('The operational implication is sharp. Most token decisions are already made easy by basin entry; the remaining quality gap is dominated by a small, structured tail of low-margin, token-embedding-hard, and lexical-boundary cases that the native decoder calibrates better than a single linear readout.','Most native token decisions are recoverable by a simple map on these final states; the residual disagreements concentrate in low-margin, token-embedding-hard, and lexical-boundary cases. This diagnoses fidelity, not the effect of each category on PPL. The equal-error intervention in \\secref{sec:prospective-readout} finds that native-margin priority can perform worse than random repair despite this enrichment.')
    s=s.replace('It mixes scale-sensitive lexical stress cases with an interface-floor component that still requires native decoder calibration.','It mixes scale-sensitive lexical stress cases with a persistent difference between the tested lookup rule and the native decoder.')
    s=s.replace('late states are nearly readable by a lightweight head, but the residual tail still requires confidence routing or native calibration.','late states are nearly readable by a lightweight head, while the tested heads leave residual native-decision disagreements.')
    s=s.replace('Transport should then be trained against neighborhoods the actual trajectory visits: combine the flow loss with late-phase decoder consistency, reweight the persistent low-margin token tail, and expose the decoder to trajectory-aligned perturbations rather than maximizing a scalar clean-state margin.','A candidate training direction is to combine the flow loss with late-phase decoder consistency and exposure to trajectory-aligned perturbations. The present evidence does not establish that prioritizing low native margins is beneficial: the equal-error output repair gives the opposite score ordering. Any training-time tail weighting therefore needs its own matched evaluation.')
    s=s.replace('Our contribution is to test which properties of its readable target predict local stability, which fail to predict generation, and what its native decoder still contributes after transport.','Our contribution is to test what simple readouts preserve on generated states, whether their error composition predicts a conditional text-score gap on new checkpoints, and which local decoder measurements fail to establish stable or useful generation.')
    s=s.replace(r'\subsection{Mechanism: Denoising as Basin Navigation}',r'\subsection{Trajectory Evidence for Entry into Readable Regions}')
    s=s.replace('A wider decoder basin is not necessarily a better generator: decoder-only training improves native and OOD recovery but supplies no path from the prior.','Stronger corruption recovery does not imply a better generator: decoder-only training improves native and OOD recovery but supplies no learned flow objective from the prior.')
    s=s.replace('The five-axis checklist in \\tabref{tab:checklist} is therefore part of the main claim:','The five-axis checklist in \\tabref{tab:checklist} is a supporting reporting tool:')
    s=s.replace('while the residual tail still needs native decoder calibration.','while a residual native-decision and conditional-score gap remains.')
    s=s.replace('The experiments explain how successful ELF trajectories become locally readable,','The experiments measure how successful ELF trajectories become locally readable,')
    discussion='The readout experiment makes the central distinction concrete. '+prediction_sentence+' At exactly matched agreement, rare-first repair improves the pooled external score over random repair, low-native-margin-first repair worsens it, and numeric-first repair has no clear advantage. Both evaluators preserve these directions. Error concentration therefore does not determine the value of correcting an error. These findings concern conditional evaluator scores; a useful quality criterion must also preserve diversity, task success, and the intended output boundary.\n\n'
    pos=s.index('Continuous language diffusion is a property of a coupled',s.index(r'\section{Discussion}'));s=s[:pos]+discussion+s[pos:]
    s=s.replace('The residual-tail analysis below shows why:','The residual-tail analysis below locates the disagreements:')
    s=s.replace('Most token decisions are therefore linearly recoverable after basin entry; the native decoder calibrates a small structured tail not fully recovered by the tested linear readouts and fitting budgets.','Most token decisions are therefore linearly recoverable on these final states; the tested linear readouts and fitting budgets leave a structured residual tail. Whether its composition explains text-score differences is tested separately in \\secref{sec:prospective-readout}.')
    s=s.replace('supporting the view that the residual PPL gap is a tail-calibration issue rather than a failure of bulk readability.','locating a residual confidence difference in the tail. This correlation alone does not identify the cause of the PPL gap.')
    s=s.replace('supporting the view that the native decoder primarily calibrates a predictable residual tail on generated final ELF states.','showing concentrated native-decision disagreements in this historical fit. These evaluation-set gates do not establish transferable thresholds or end-to-end savings.')
    # Distinguish destructive post-hoc PCA truncation from trainable input bottlenecks.
    needle=r'\subsection{Order Sensitivity Is Graded}'
    s=s.replace(needle,r'''These PCA interventions delete coordinates from an already trained decoder-facing representation. They do not test a learned input bottleneck trained jointly with a denoiser, and should not be generalized to that different architectural intervention.

'''+needle,1)
    # Preserve detailed conclusions while replacing their opening with the central question.
    needle='The evidence supports a separation between a decoder-readable target and transport from the prior.'
    s=s.replace(needle,'The strongest readout result is conditional: successful ELF trajectories expose states whose native token decisions are largely recoverable by a fixed affine map. Tail-specific errors improve a frozen quality predictor on B/M but not on the distilled student. At identical agreement, rare-first and low-native-margin-first repair move external scores in opposite directions relative to random repair. These outcomes distinguish fidelity, confidence, and evaluator impact without inferring quality from agreement alone. The supporting evidence also separates a decoder-readable target from transport from the prior.',1)
    s=s.replace('Basin entry explains local token reliability and exposes a structured rare/numeric residual tail, but it does not certify long-form discourse planning;','Measured basin entry tracks local confidence and exposes a structured rare/numeric residual tail, but it does not certify token robustness or long-form discourse planning;')
    limitation=r'''The prospective extension transfers three fixed ELF-B readouts to 1,536 new trajectories from three released checkpoint conditions and two generation seeds. The calibration set was already inspected; only the new test generations are prospective. ELF-M and the distilled student change multiple training and architecture variables jointly and do not constitute independent denoiser training-seed repetitions. The quality predictors use native labels, native margins, and native evaluator scores; they are oracle diagnostics, not cheap deployable gates. Equal-error repair uses native decisions and holds the native output boundary fixed. Its paired effects concern an external language-model score, not factual accuracy, human preference, or a learned decoder's end-to-end utility. Different T5 outputs can induce different evaluator tokenizations and truncation coverage. Confidence intervals retain paired sequence dependence across shared generation seeds and remain conditional on the selected checkpoints and fits.

'''
    limitation=limitation.replace('Equal-error repair uses native decisions','The frozen margin threshold is not scale invariant across decoders. Failure of this specified ridge predictor would not rule out other features or nonlinear predictors, and success would not identify a causal origin for the tail. Equal-error repair uses native decisions')
    pos=s.index('The study analyzes released checkpoints',s.index(r'\section{Limitations}'));s=s[:pos]+limitation+s[pos:]
    s=s.replace(r'\input{v3_confirmatory_additions.tex}',r'\input{v3_confirmatory_additions.tex}'+'\n\n'+r'\input{v3_predictive_additions.tex}')
    (DOC/'main.tex').write_text(s)
    # Numerical result text and full protocols are generated from audited scalars.
    maintext=r'''\section{A Frozen Test of Readout Quality}
\label{sec:prospective-readout}

The common-test study reaches $97.17\%$ native agreement with a token-wise affine readout, while numeric and low-frequency disagreements remain concentrated (\secref{app:confirmatory-readout}). This motivates a prediction that is stronger than tail enrichment: do those error categories improve prediction of the text-score gap on new conditions? Before generating the test states, we freeze both the readouts and a ridge regression on 512 previously inspected ELF-B sequences. The baseline predicts excess per-sequence GPT-2-Large NLL from total disagreement, native sequence length, and native NLL. The augmented predictor adds the rates of rare, numeric, and low-native-margin disagreements, using the same ridge penalty and no test tuning.

We evaluate 256 new sequences per generation seed, with two seeds for each of ELF-B at 32 steps, ELF-M at 32 steps, and the released distilled student at 8 steps: 1,536 sequences in total. Three fixed ELF-B readouts transfer without refitting. PREDICTION The measurements and all condition-specific results appear in \figref{fig:prospective-tail} and \tabref{tab:prospective-prediction}. These predictors deliberately use native oracle information to test an explanation; they are not proposed as inference-time quality gates.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{figures/figure_prospective_tail_20261013.pdf}
\caption{Prospective readout-quality tests on new states, with predictors and readouts frozen before generation. B32 and M32 use 32-step sampling; PD8 uses the released 8-step student. (a) Mean absolute error predicting the per-sequence excess evaluator NLL of a linear readout over native decoding, averaged over all three fixed readouts. (b) Paired NLL differences between the mean of three random repair arms and targeted repair; positive values favor targeted repair. Every repair arm corrects exactly half the errors, rounded down, within each sequence. Bars are conditional 95\% paired sequence-bootstrap intervals. (c) Corpus PPL under the common native boundary; the plotted random arm A is fixed in advance, with all three reported in \tabref{tab:prospective-ppl}. Oracle repairs diagnose a text-score effect, not a deployable routing policy or factual correctness.}
\label{fig:prospective-tail}
\end{figure}

The repair intervention separates error count from error selection. For the prespecified readout seed 17, each sequence receives exactly $\lfloor e/2\rfloor$ native corrections, where $e$ is its number of disagreements. We compare rare-first, numeric-first, low-native-margin-first, and three uniform random selections. Consequently every repaired output has exactly the same native agreement within its sequence. REPAIR The protocol and outcomes are retained regardless of direction; no policy is selected on this test. This intervention identifies the conditional score effect of these output edits, not the effect of retraining a decoder or changing transport.

These tests precede the supporting controls because they address the central inferential gap: bulk readability is an observable, and a proposed explanation of its quality boundary must survive an independent prediction or a matched intervention. The following sections characterize representation, local stability, and transport without assuming that any one of their scalar diagnostics determines generation quality.

'''.replace('PREDICTION',prediction_sentence)
    rr=next(r for r in repair if r['condition']=='all' and r['policy']=='rare')
    repair_sentence=f"The prespecified rare-first contrast is ${float(rr['mean_random_minus_target_nll']):+.4f}$ nats per evaluator token (97.5\\% interval $[{float(rr['ci_low']):+.4f},{float(rr['ci_high']):+.4f}]$; random minus rare-first). " + ('It supports a score advantage for this targeted repair in the pooled test, with the other policies and condition-specific results reported separately.' if rare_supported else 'It does not support the hypothesized pooled score advantage for rare-first repair; the other policies and condition-specific results remain separate controls.')
    maintext=maintext.replace('REPAIR',repair_sentence)
    low=next(r for r in repair if r['condition']=='all' and r['policy']=='lowmargin')
    boundary_sentence=f"In contrast, low-native-margin-first repair is worse than random by ${-float(low['mean_random_minus_target_nll']):.4f}$ nats in the pooled GPT-2-Large test; numeric-first repair shows no clear advantage. The B32 rare-first interval includes zero, so the pooled benefit does not establish a benefit in every condition. An error category can be enriched among disagreements without being the best category to correct for this evaluator. "
    maintext=maintext.replace('The protocol and outcomes are retained',boundary_sentence+'The protocol and outcomes are retained')
    agreements={c:np.mean([float(r['agreement']) for r in ppl if r['condition']==c and r['method'].startswith('linear_seed')])*100 for c in ['B32','M32','PD8']}
    sentence=f"The frozen maps retain mean native agreement of ${agreements['B32']:.2f}\\%$, ${agreements['M32']:.2f}\\%$, and ${agreements['PD8']:.2f}\\%$ on B32, M32, and PD8 respectively, separating bulk transfer from exact interface invariance. "
    maintext=maintext.replace('Three fixed ELF-B readouts transfer without refitting. ','Three fixed ELF-B readouts transfer without refitting. '+sentence)
    opt=next(r for r in secondary if r['condition']=='all' and r['policy']=='rare')
    opt_sentence=f"A secondary OPT-125M evaluation, specified before inspecting the primary scoring outcomes, gives a pooled rare-first contrast of ${float(opt['mean_random_minus_target_nll']):+.4f}$ nats (descriptive 95\\% interval $[{float(opt['ci_low']):+.4f},{float(opt['ci_high']):+.4f}]$) on the exact same outputs. Low-native-margin-first repair remains worse than random, while the numeric-first interval again includes zero. This checks evaluator sensitivity without adding a confirmatory primary test."
    maintext=maintext.replace('These tests precede the supporting controls',opt_sentence+'\n\nThese tests precede the supporting controls')
    (DOC/'v3_predictive_main.tex').write_text(maintext)
    appendix=r'''\section{Prospective Readout Prediction and Equal-Error Repair}
\label{app:prospective-readout}

\parhead{Protocol, calibration, and transfer}
The local protocol, calibration coefficients, readout weights, and source checkpoints were hashed before any new test generation. This is a prospectively frozen local test, not a publicly registered study. Calibration reuses the already inspected 512-sequence ELF-B set from \secref{app:confirmatory-readout}; its generation seed is 61013001. All three validation-selected affine readouts trained on 16,384 ELF-B sequences are frozen. Their fitting seeds are 17, 29, and 43. Test generation seeds are 61013201 and 61013202, with 256 sequences per seed and condition, batch size four, and current native labels computed during generation. ELF-B and ELF-M use 32 SDE steps, $\gamma=1.5$, and self-conditioning CFG $3.0$; the released PD student uses eight steps, $\gamma=1.5$, and CFG $2.5$. All use their native logit-normal grids and the final ODE update. These are within-interface checkpoint transfers; checkpoint size, training, and sampling differences are not separately identified.

Let $v_{i\ell}$ denote the native valid-position mask, $y_{i\ell}$ the native token, and $\widehat y_{i\ell}$ a frozen readout decision. The target is the difference of mean evaluator NLLs for the two detokenized texts,
\[
q_i=\frac{\operatorname{NLL}(\widehat y_i)}{n(\widehat y_i)}-
\frac{\operatorname{NLL}(y_i)}{n(y_i)}.
\]
Here $n$ counts scored GPT-2 tokens after truncation, not T5 positions. The baseline covariates are the native-position disagreement rate, log native valid length, and native mean evaluator NLL. Three added covariates are the number of disagreements in each specified group divided by the total native valid length: target-token training count below 100 (including unseen types), a digit in the native token string, and native top-two margin below 2. Groups overlap. Training frequencies come only from the frozen 16,384-sequence ELF-B fitting pool. For each readout fit, ridge regression uses calibration-only means and standard deviations, an unpenalized intercept, and fixed penalty $\alpha=10$. There is no hyperparameter search, test-based selection, or target-checkpoint refitting.

\begin{table}[!htbp]
\centering\small
\caption{Frozen prediction of excess per-sequence NLL. MAE is averaged across the three fixed readouts. Positive reduction favors adding tail-specific features. Per-condition intervals are conditional 95\%; the pooled primary interval is 97.5\% to account for the two prespecified primary tests.}
\label{tab:prospective-prediction}
\begin{tabular}{lrrr}
\toprule
Condition & Aggregate MAE & Tail MAE & Reduction [interval] \\
\midrule
'''
    for r in pred:
        name='Pooled' if r['condition']=='all' else r['condition']
        appendix+=f"{name} & {float(r['baseline_mae']):.4f} & {float(r['tail_mae']):.4f} & ${float(r['mae_reduction']):+.4f}$ [$ {float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f} $] \\\\\n"
    appendix+=r'''\bottomrule
\end{tabular}
\end{table}

\parhead{Matched repair and uncertainty}
For readout seed 17 only, each repair corrects $\lfloor e_i/2\rfloor$ of the $e_i$ disagreements in sequence $i$ to their native targets. The rare-first policy orders erroneous positions by ascending target training count; numeric-first prioritizes a digit in the native target token; low-margin-first orders native margins. Random tie breaking and three uniform random selections are fixed by the protocol. These are oracle edits: all policies know the native errors, and all achieve exactly identical per-sequence agreement. They cannot be interpreted as inference-time routers or as edits toward ground-truth factual answers.

Primary prediction inference averages the absolute prediction errors across the three fixed readouts within each sequence. Repair contrasts compare each targeted arm with the mean of the three random arms on the same sequence. We use 4,000 paired sequence-cluster bootstrap draws with seed 61013203, equal condition/seed weighting, and reuse sampled indices across checkpoint conditions sharing a generation seed. This common-random-number correction was documented before test generation as an amendment to the initial stratified-bootstrap wording. The two primary tests---pooled prediction MAE reduction and pooled rare-first repair advantage---use Bonferroni 97.5\% two-sided intervals. Other intervals are descriptive 95\% intervals without simultaneous-coverage claims. None represents uncertainty over new denoiser training seeds or new calibration partitions.

\begin{table}[!htbp]
\centering\small
\caption{Equal-error repair: mean per-sequence NLL of random repair minus targeted repair. Positive values favor targeted repair. All repair arms have identical native agreement within each sequence. Pooled rare-first is the prespecified primary contrast; remaining contrasts are secondary.}
\label{tab:prospective-repair}
\begin{tabular}{llrr}
\toprule
Condition & Priority & NLL difference & Interval \\
\midrule
'''
    for r in repair:
        name='Pooled' if r['condition']=='all' else r['condition'];policy={'rare':'Rare','numeric':'Numeric','lowmargin':'Low margin'}[r['policy']]
        appendix+=f"{name} & {policy} & ${float(r['mean_random_minus_target_nll']):+.5f}$ & [$ {float(r['ci_low']):+.5f}, {float(r['ci_high']):+.5f} $] \\\\\n"
    appendix+=r'''\midrule
\multicolumn{4}{l}{Secondary OPT-125M (all intervals descriptive 95\%)} \\
'''
    for r in secondary:
        name='Pooled' if r['condition']=='all' else r['condition'];policy={'rare':'Rare','numeric':'Numeric','lowmargin':'Low margin'}[r['policy']]
        appendix+=f"{name} & {policy} & ${float(r['mean_random_minus_target_nll']):+.5f}$ & [$ {float(r['ci_low']):+.5f}, {float(r['ci_high']):+.5f} $] \\\\\n"
    appendix+=r'''\bottomrule
\end{tabular}
\end{table}

\parhead{Scores and measurement scope}
Every output uses the same native validity mask before T5 detokenization. GPT-2-Large uses float16 weights with float32 losses and at most 1,024 evaluator tokens. Corpus PPL exponentiates summed NLL divided by summed scored tokens, whereas prediction and repair inference use per-sequence mean NLL. Different outputs can have different evaluator tokenizations and truncation coverage; the measurements concern these scored texts, not equal-position teacher-forced probabilities. Common-scorable sequence counts, token counts, exact-overlap checks, frozen coefficients, all predictions, and scalar score records are included in the ancillary evidence. Exact overlap checks do not assess semantic near-duplicates. Native agreement and this external text score are distinct outcomes; neither is factual correctness or document planning.

\begin{table}[!htbp]
\centering\small
\caption{Conditional corpus PPL on all 512 sequences per checkpoint condition. The three learned maps are frozen ELF-B fits. Repair arms use fit seed 17 and identical per-sequence correction counts. All prespecified random arms are retained.}
\label{tab:prospective-ppl}
\begin{tabular}{lrrr}
\toprule
Readout & B32 & M32 & PD8 \\
\midrule
'''
    labels={'native':'Native','linear_seed17':'Linear, seed 17','linear_seed29':'Linear, seed 29','linear_seed43':'Linear, seed 43','repair_random11':'Random repair A','repair_random23':'Random repair B','repair_random37':'Random repair C','repair_rare':'Rare-first repair','repair_numeric':'Numeric-first repair','repair_lowmargin':'Low-margin-first repair'}
    for method,label in labels.items():
        vals=[float(next(r for r in ppl if r['condition']==c and r['method']==method)['ppl']) for c in ['B32','M32','PD8']]
        appendix+=label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\'+'\n'
    appendix+=r'''\midrule
\multicolumn{4}{l}{Secondary OPT-125M} \\
'''
    for method,label in labels.items():
        vals=[float(next(r for r in secondary_ppl if r['condition']==c and r['method']==method)['ppl']) for c in ['B32','M32','PD8']]
        appendix+=label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\'+'\n'
    appendix+=r'''\bottomrule
\end{tabular}
\end{table}

The portable NumPy verifier refits the calibration regressions, reproduces the frozen predictions and paired intervals, checks equal repair counts from saved per-sequence records, and recomputes corpus PPL. This scalar audit does not rerun the GPU models. Local token arrays additionally verify the actual edited positions and the unchanged nonselected outputs; their hashes are recorded, while large latent arrays and source checkpoints are not bundled.
'''
    appendix=appendix.replace('Random tie breaking and three uniform random selections are fixed by the protocol.','Random tie breaking and three uniform random selections are fixed by the protocol. The NumPy seed sequence combines the generation seed, condition index, sequence index, policy index, and constant 61013. Random arms A/B/C correspond to policy indices 0/1/2 (internal file labels 11/23/37); these file labels are identifiers, not standalone random seeds.')
    appendix=appendix.replace('Primary prediction inference averages',r'The OPT-125M secondary scoring protocol was added after generation but before inspecting primary scoring outcomes. It scores the exact same ten arms without new prediction fitting or output selection. All OPT intervals are descriptive 95\% intervals. This is evaluator sensitivity, not an independent confirmation of the primary predictor. '+'\n\nPrimary prediction inference averages')
    appendix=appendix.replace('None represents uncertainty over new denoiser training seeds or new calibration partitions.','All intervals are conditional on the fixed random-repair realizations as well as the selected models. None represents uncertainty over new denoiser training seeds or new calibration partitions.')
    appendix=appendix.replace('They cannot be interpreted as inference-time routers or as edits toward ground-truth factual answers.','They cannot be interpreted as inference-time routers or as edits toward ground-truth factual answers. The low-margin priority uses the native decoder, whereas the historical routing experiment uses confidence of the learned linear readout and a different correction budget; these signals and interventions must not be conflated.')
    appendix+=r'''

\parhead{Released-checkpoint provenance}
A live inventory of the official \href{https://huggingface.co/embedded-language-flows}{ELF model organization} on October 13, 2026 returned 13 repositories. The newest repository update was the June 26 release of ELF-B-PD-owt-torch, whose student is included in PD8. All three checkpoints used here exactly match their current official LFS SHA-256 values; the API snapshots and comparisons are included in the ancillary evidence. Modification dates identify repository updates, not new independent training runs. This provenance check did not change the frozen test protocol.
'''
    # This is a labeled post-result breakdown of already specified secondary metrics.
    pd_predictions=[r for r in readcsv(OUT/'predictions.csv') if r['condition']=='PD8']
    base_bias=np.mean([float(r['baseline_prediction'])-float(r['target']) for r in pd_predictions])
    tail_bias=np.mean([float(r['tail_prediction'])-float(r['target']) for r in pd_predictions])
    appendix+=f"\n\\parhead{{Interpreting the student failure}}\nA post-result breakdown distinguishes absolute prediction from ranking. On PD8, adding tail features increases mean prediction bias from ${base_bias:+.4f}$ to ${tail_bias:+.4f}$ nats (prediction minus measured excess NLL), while descriptive Spearman correlation rises from ${float(pdrow['baseline_spearman']):.3f}$ to ${float(pdrow['tail_spearman']):.3f}$. These summaries pool the three fixed readout records per sequence and carry no independent-sample significance claim. Better ordering therefore coexists with worse absolute MAE. This does not identify the cause of the shift: the checkpoint and sampler change together, and no post-test recalibration is performed.\n"
    (DOC/'v3_predictive_additions.tex').write_text(appendix)
    print(json.dumps(dict(title=title,pooled_prediction_supported=supported,pooled_rare_repair_supported=rare_supported,summary=prediction_sentence,repair=repair_sentence),indent=2))

if __name__=='__main__':main()
