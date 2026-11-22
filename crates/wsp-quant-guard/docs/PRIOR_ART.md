# Certified quantized weight edits: literature audit

Date: 22 November 2026. Scope: a bounded research audit, not an exhaustive novelty or patent search. All proposed contributions below remain hypotheses. No third party implementation was executed.

## Verified premise

The official Lean community statement dated 21 November 2026 endorses Alpoge's RH proof and reports statement matching and independent kernel checks. Andrew Sutherland's MIT page records concrete verification files and a mix of completed and still running checks. Thus the earlier conversation's central RH announcement is supported by the currently retrieved primary sources. This does not establish that its specific mathematics transfers to neural editing. The reusable research discipline is explicit statements, adversarial checking, independent verification, and reproducible artifacts.

Sources:

1. https://leanprover-community.github.io/blog/posts/riemann-hypothesis/
2. https://math.mit.edu/~drew/grh/

## Evidence matrix

| Work | What is established in the source | Consequence for our novelty claim |
|:--|:--|:--|
| AlphaEdit, ICLR 2025 | Projects weight changes into the null space of protected representations. Gives a preservation argument and LLM experiments. | The condition `delta_W H = 0`, its elementary proof, and null space editing are prior art. |
| BetaEdit, IJCAI 2026 | Identifies leakage from approximate null spaces and analyzes history aware updates. Its text notes that empirical covariance can be full rank, forcing a trivial exact null space. | Sequential edit preservation, history protection, and approximate null space leakage are prior art. A smaller integer patch certificate must offer a distinct operational advantage. |
| QNNRepair, SEFM 2023 | Locates damaged neurons and formulates quantized model repair using constrained parameter optimization with passing and failing tests. | Repairing quantized weights while preserving specified examples is prior art. |
| Provable Repair of Deep Neural Networks, PLDI 2021 | Defines provable repair, including minimal repair for finite point specifications. | A certificate over finite anchors is valuable but not a new general research category. |
| PREPARED, NeurIPS 2024 | Uses parametric linear relaxation for parameter edits satisfying output constraints throughout an input polytope. | Region wide provable editing and small parameter changes are established. |
| Scalable Verification of Quantized Neural Networks, AAAI 2021 | Studies exact quantized semantics and warns that ignoring rounding can invalidate conclusions. | Exact integer verification is established; use it as a foundation rather than claim it as a discovery. |
| CellFill, arXiv 2608.20873, August 2026 | Constrains residual updates inside weight quantization cells while freezing integer codes and scales. Requantization returns the original artifact; residual removal revokes the update. | Weight cell invariance, frozen scale constraints, and revocable residuals are prior art. Our possible distinction is editing integer weights while preserving specified activation/output cells. |
| WeightCode, arXiv 2611.25476, November 2026 | Frozen LLM writes programs that modify weights to install capabilities, with reuse and sequential experiments. | WSP capability installation, transfer of edit programs, and adding capabilities sequentially are already claimed experimentally. |
| Composable interventions, ICLR 2024 workshop | Studies MEMIT combined with pruning and quantization; quantization can erase edits while broad metrics degrade less. | Compare the actual deployed integer artifact, not a float surrogate or general accuracy alone. |
| Weight Space Programming for Vision Language Model Repair, arXiv 2611.23615, November 2026 | Finds planted float32 corruption can expose low bit fingerprints absent from original 16 bit weights; rounded controls impair repair. | Avoid synthetic faults whose serialization identifies the damaged weights. If using synthetic faults, include a format matched rounded control. |

Primary sources for the matrix:

1. AlphaEdit: https://proceedings.iclr.cc/paper_files/paper/2025/hash/29c8c615b3187ee995029284702d3f43-Abstract-Conference.html
2. BetaEdit: https://ijcai.org/proceedings/2026/508
3. QNNRepair: https://research.manchester.ac.uk/en/publications/qnnrepair-quantized-neural-network-repair/ and https://arxiv.org/abs/2306.13793
4. Provable Repair: https://thakur.cs.ucdavis.edu/bibliography/PLDI2021.html
5. PREPARED: https://proceedings.neurips.cc/paper_files/paper/2024/hash/ce6326ac4794bb04d5eb16f597446baf-Abstract-Conference.html
6. Exact QNN verification: https://arxiv.org/abs/2012.08185 and https://ojs.aaai.org/index.php/AAAI/article/view/16496
7. CellFill: https://arxiv.org/abs/2608.20873. Primary abstract retrieved by search; full page open was unavailable. Do not infer unreviewed details from secondary summaries.
8. WeightCode: https://arxiv.org/abs/2611.25476 and https://github.com/Gen-Verse/WeightCode
9. Composable interventions: https://openreview.net/pdf?id=quK3DN5kUO
10. WSP repair controls: https://arxiv.org/abs/2611.23615

## Two strongest falsifiable research hypotheses

### H1. Integer synthesis can preserve protected behavior after quantization where float projected edits fail

Setting: fixed upstream features `H`, a single editable integer affine map, fixed quantizer scales and zero points, and an unchanged downstream network. Synthesize an integer `delta_W` directly, with bounded stored weights and proven accumulator headroom, instead of rounding a real valued null space update.

Hypothesis: on a declared distribution of small quantized models and edit tasks, integer constrained synthesis accepts useful target edits with zero protected integer output changes, while rounding the corresponding float projected updates causes nonzero protected changes or loses the target edit.

Baselines: unconstrained target solve then quantize; float exact null space solve then quantize; exhaustive integer search on tiny instances as a correctness oracle. Include an infeasible target. Do not compare only against a deliberately thresholded pseudo null space.

Evidence to collect: useful feasible edit rate, target error before/after, protected output changes, fraction of changed parameters, synthesis time, certificate checking time, and exact replay of the saved artifact. For integer null space synthesis, denominator cleared rational bases may span only a sublattice; the verifier can remain sound while synthesis is incomplete. An integer kernel basis or a bounded exhaustive oracle can expose this limitation.

Novelty status: integer linear algebra and constrained repair are known. The promising contribution is an efficient deployable implementation plus a measured regime where it gives a useful advantage. A new mathematical theorem is not presently established.

### H2. Preserving output quantization cells gives useful edit capacity when exact activation preservation is impossible

For a deterministic integer quantizer, each protected output code corresponds to an exact interval of accumulator values. Instead of imposing zero accumulator change, constrain the edited accumulator to stay within its original output cell. This permits nonzero integer changes even if protected features have full rank.

A minimal existence example with `q(z) = floor(z / 8)`: scalar weight `W = 3`, protected feature `h = 1`, and edit `delta = 1` preserve the protected output code 0, but change the output code for target feature 2 from 0 to 1. The scalar protected feature has full rank, so exact accumulator preservation permits only the zero edit. This is an elementary counterexample to an overly broad impossibility claim, not a new general theorem.

Hypothesis: using exact output cell constraints plus cumulative edit certificates yields a materially higher useful edit acceptance rate than exact accumulator null space preservation on full rank protected sets, without changing any specified protected integer output.

Baselines: exact accumulator preservation, output cell constrained search, and unconstrained integer search. Track cell margins and cumulative signed accumulator changes after every accepted edit. The target success condition must reflect a meaningful held out behavior, not merely a changed internal number. Compare both target capacity and verification cost.

Novelty status: quantization cells and constraint solving are established. The combined operational result is a candidate for evaluation. No claim of priority is justified from this search.

## Hard soundness rules

1. State whether protection covers a finite list of features, an input region, predicted labels, or exact outputs. These are different contracts.
2. Freeze and bind upstream features, the remaining network, quantizer scales, zero points, clipping, rounding mode, and weight layout to the certificate.
3. Prove or explicitly check accumulator ranges under the deployed arithmetic. A theorem over mathematical integers does not cover wraparound or different backend saturation.
4. For sequential edits, protect the original anchors and explicitly decide whether previous successful target edits become protected. Checking only the original anchors does not prevent forgetting edits.
5. Count effective distinct edits. One hundred zero patches, reversals, or repeated identical targets do not demonstrate capacity for one hundred retained capabilities.
6. Require immutable artifact hashes, replayable verification, rejection of stale bases and altered certificates, and exact rollback to a known base. Hashes bind artifacts; they do not establish mathematical validity by themselves.
7. Do not equate no anchor changes with no distributional forgetting. Use a disjoint evaluation set or prove coverage of an entire declared input domain.
8. A failed search is not an infeasibility proof unless the search is complete over the stated bounded space.
9. Keep a mathematical proof, an executable arithmetic checker, exhaustive finite tests, and a Lean theorem distinct in reporting.

## Recommendation

Deliver a reproducible research artifact with three outcomes: a positive construction under explicit conditions, counterexamples showing why relaxed claims fail, and a measured comparison against meaningful baselines. Label the result a candidate research contribution pending external novelty review. Business value is lower risk deployment of small model updates; cost savings versus fine tuning remain an empirical claim requiring matched end to end accounting.

Acceptance: an independent checker must reject altered or stale patches, accept the original artifact, reproduce every protected integer output after each effective accepted edit, and restore the base byte for byte on rollback. Report target performance and unseen retention separately.
