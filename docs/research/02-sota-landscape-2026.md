# SOTA Landscape 2026: Research Report for the ruv-FANN Ecosystem

**Date:** 2026-08-08
**Author:** SOTA research agent (swarm session)
**Scope:** Agentic coding harnesses, self-improving agents, model routing, data flywheels, neural forecasting, and the Rust/WASM ML ecosystem — with implications for ruv-FANN, ruv-swarm, and neuro-divergent.

**Epistemic conventions used throughout:**
- **[Verified]** — confirmed against a primary or independent source fetched during this research pass (URL cited).
- **[Secondary]** — reported by a blog/aggregator found in search; plausible but not confirmed against the primary source.
- **[Prior knowledge]** — from the author's training data (cutoff ~Jan 2026), not re-verified today.

---

## 1. Agentic Coding Harnesses & SWE-bench State of the Art

### 1.1 Where the leaderboards stand (mid-2026)

SWE-bench Verified is effectively saturating at the frontier:

- **[Verified]** The Steel.dev tracker lists **Claude Mythos 5 at 95.5%** (Jun 2026), **Claude Fable 5 at 95.0%** (Jun 2026), **Claude Mythos Preview at 93.9%** (Apr 2026), **Claude Opus 4.8 at 88.6%** (May 2026), and **Claude Opus 4.7 at 87.6%** (Apr 2026) on SWE-bench Verified. The tracker explicitly warns the benchmark is "now mature and heavily exposed in public training data" and that frontier scores "should be interpreted with contamination and test-design caveats."
  https://leaderboard.steel.dev/leaderboards/swe-bench-verified/
- **[Secondary]** CodeAnt's 2026 roundup reports **Claude Opus 5 at 97.0%**, **GPT-5.6 Sol at 96.2%**, Claude Fable 5 at 95.0%, **Kimi K3 at 93.4%**, GPT-5.6 Luna at 93.0%. https://codeant.ai/blogs/swe-bench-scores
- **[Secondary]** The field has migrated to harder benchmarks: **SWE-bench Pro** (Scale AI) shows Claude Opus 4.8 leading the active set at **69.2%**, and models that score >93% on Verified score **~46% on Pro** (Claude Mythos Preview: 93.9% Verified vs 45.9% Pro). https://www.morphllm.com/swe-bench-pro
- **[Verified/Secondary]** **Terminal-Bench v2.0** is the other de-facto successor benchmark; frontier agents score in the 45–60% range there (see §1.3), leaving real headroom.
- **[Prior knowledge]** SWE-bench Lite is essentially retired as a frontier signal; scores above ~55% were already common in 2025. ruv-swarm's advertised "84.8% SWE-Bench solve rate" (README) dates from the 2025 Lite/Verified era and should be re-baselined against 2026 conditions before being used in any comparison.

### 1.2 Harness vs model: the harness now dominates marginal gains

This is the single most important finding for ruv-swarm's positioning:

- **[Secondary]** On SWE-bench Pro, three different agent systems running the *same* Claude Opus 4.5 produced a **50.2%–55.4% spread (5.2 points) from scaffold differences alone**; holding the model fixed and varying only the harness widens to **9.5 points** (SEAL vs Claude Code). One analysis attributes a **22+ point swing to scaffold** vs **~1 point to model swaps at the frontier**. https://www.digitalapplied.com/blog/swe-bench-verified-june-2026-benchmark-vs-scaffolding-analysis , https://www.mindstudio.ai/blog/harness-vs-model-distinction-agent-wrapper
- **[Secondary]** Extreme documented cases show harness changes alone swinging the same model **42% → 78%** on coding benchmarks. https://particula.tech/blog/agent-scaffolding-beats-model-upgrades-swe-bench
- **[Verified]** An arXiv position paper, "Stop Comparing LLM Agents Without Disclosing the Harness" (2605.23950), formalizes the reproducibility complaint: scaffolds, prompts, budgets, and termination policies vary per system, making cross-system deltas unattributable. https://arxiv.org/pdf/2605.23950

### 1.3 Test-time compute scaling for long-horizon agents

- **[Verified]** *Scaling Test-Time Compute for Agentic Coding* (Kim et al., arXiv 2604.16529, Apr 2026) argues classic best-of-n breaks down for long trajectories and that the problem is **representation, selection, and reuse** of prior rollouts. Two mechanisms: **Recursive Tournament Voting (RTV)** — parallel scaling that narrows a population of rollout *summaries* via small-group comparisons — and **Parallel-Distill-Refine (PDR)** — sequential scaling conditioning new rollouts on distilled summaries of prior attempts. Results: Claude-4.5-Opus **70.9% → 77.6%** on SWE-bench Verified (mini-swe-agent harness) and **46.9% → 59.1%** on Terminal-Bench v2.0 (Terminus 1 harness). https://arxiv.org/abs/2604.16529
- **[Verified]** Related: **SWE-TRACE** uses rubric process-reward models plus heuristic test-time scaling for long-horizon SWE agents (arXiv 2604.14820). https://arxiv.org/html/2604.14820v1

### 1.4 Notable open-source harnesses

- **[Verified]** **mini-swe-agent** (SWE-bench/SWE-agent team, Princeton lineage): a radically minimal bash-only loop, now the standard *reference harness* used for apples-to-apples model comparisons on the official leaderboard. https://vibecodinghub.org/tools/mini-swe-agent , https://www.swebench.com/
- **[Verified]** **OpenHands** (ex-OpenDevin): the dominant research substrate — "most agentic-coding research papers in 2025–2026 use OpenHands." https://tensorfeed.ai/harnesses/openhands
- **[Secondary]** New benchmark families evaluate *harnesses* rather than models: **Claw-SWE-Bench** for OpenClaw-style harnesses (arXiv 2606.12344), and the **Holistic Agent Leaderboard** (arXiv 2510.11977) for standardized agent evaluation infrastructure.

### Implications for ruv-FANN / ruv-swarm

1. **The harness is the moat.** With frontier models within ~1 point of each other, ruv-swarm's differentiation lives in topology, memory, and orchestration — exactly the 5–22 point territory the harness studies describe. Invest engineering there, not in model chasing.
2. **Re-baseline the 84.8% claim.** Publish which benchmark/variant/harness/model produced it, or replace it with a 2026 measurement (SWE-bench Pro or Terminal-Bench v2) run through mini-swe-agent-style controlled conditions. Saturated-benchmark claims now invite contamination skepticism.
3. **RTV/PDR-style summary-based scaling is directly implementable in ruv-swarm**: parallel agent rollouts already exist; adding trajectory summarization + tournament selection is an orchestration-layer feature, cheap relative to its measured 6–12 point gains.
4. Adopt harness-disclosure norms (arXiv 2605.23950) in all published numbers.

---

## 2. Self-Improving / Evolutionary Agent Systems

### 2.1 AlphaEvolve — now a product, not just a paper

- **[Verified]** **AlphaEvolve reached general availability in July 2026** on the Gemini Enterprise Agent Platform: users define a baseline algorithm + client-side scoring function; the service generates mutated candidates via Gemini models and iterates. Evaluators run **client-side** (code never leaves customer infra). Customer numbers: **Klarna** doubled ML training throughput (~6,000 candidate programs over 3 weeks); **JetBrains** cut IDE completion latency 15–20%; **FM Logistic** shortened warehouse picking routes 10.4%; **Kinaxis** +22% forecast accuracy with 90% runtime reduction; Google internally cut Spanner LSM-tree write amplification 20%. https://www.infoq.com/news/2026/07/alphaevolve-generally-available/ ; paper: https://arxiv.org/abs/2506.13131
- **[Verified]** Open-source replications are competitive: **CodeEvolve** (arXiv 2510.14150) matches or beats reported AlphaEvolve results on 5/9 benchmark problems and outperforms **OpenEvolve** and **ShinkaEvolve** on 6/9 under matched conditions; with an open-weight **Qwen3-Coder-30B** backbone it beat AlphaEvolve on both CirclePackingSquare instances at **~10x lower cost** than a frontier closed ensemble. https://arxiv.org/abs/2510.14150

### 2.2 Darwin Gödel Machine lineage

- **[Verified]** **DGM** (Sakana AI + UBC, arXiv 2505.22954): a self-modifying coding agent maintaining an expanding archive of agent variants; improved itself **20.0% → 50.0% on SWE-bench** and **14.2% → 30.7% on Polyglot**. https://arxiv.org/abs/2505.22954
- **[Secondary]** Sakana formalized a **Recursive Self-Improvement Lab** in 2026. A companion ICLR 2026 paper (MIT + Sakana) introduces **SIFT (Self-Improvement via Fast Tree search)**, identifying *evaluation cost* as the RSI bottleneck and reporting an **11-point SWE-bench gain in three steps for $25** of API cost. https://stackfutures.com/blog/sakana-rsi-lab-launch/
- **[Verified]** **Agentic Harness Engineering** (arXiv 2604.25850, Apr 2026) is the most directly ruv-swarm-relevant paper found: an observability-driven loop that *evolves the harness itself* (tools, middleware, memory components as file-level editable units; every edit ships with a falsifiable prediction verified against outcomes). Results: Terminal-Bench 2 pass@1 **69.7% → 77.0%** over ten iterations (beating human-designed Codex-CLI at 71.9%); top aggregate SWE-bench-Verified success with **12% fewer tokens**; **+5.1 to +10.1 pp cross-model transfer**. Key ablation: *"factual harness structure transfers while prose-level strategy does not."* https://arxiv.org/pdf/2604.25850

### 2.3 Gradient-free config/prompt evolution

- **[Verified]** **GEPA** (Genetic-Pareto reflective prompt evolution; arXiv 2507.19457, ICLR 2026 Oral) uses natural-language reflection + Pareto evolutionary search instead of policy gradients: beats GRPO-style RL by **up to 20% with 35x fewer rollouts**, beats MIPROv2 by >10%; shipped as `dspy.GEPA` and generalized in the `optimize_anything` API (arXiv 2605.19633). https://arxiv.org/pdf/2507.19457 , https://github.com/gepa-ai/gepa
- **[Verified]** Field taxonomy: *A Survey of Self-Evolving Agents* (arXiv 2507.21046) organizes the space by **what evolves** (prompts / code / weights / architecture) and **how** (gradient-based / LLM-guided / evolutionary / experience-driven). https://arxiv.org/abs/2507.21046
- **[Secondary]** The dominant "what works" pattern per 2026 reviews: agents generating their own training data via self-play (e.g., SWE-RL-style bug-injector/solver alternation), plus memory-based evolution (§4) as the cheapest reliable channel.

### Implications for ruv-FANN

1. **AHE (2604.25850) is a blueprint ruv-swarm should copy**: ruv-swarm's hooks, memory namespaces, and agent configs are already file-level artifacts — the exact representation AHE requires. An "evolve-the-harness" mode (mutate hook configs/tool definitions, score on a task suite, keep winners in an archive) is feasible with existing plumbing.
2. **Evaluation cost is the bottleneck** (SIFT's finding). ruv-FANN's cheap CPU-native networks could serve as *learned fitness predictors* to pre-filter candidate mutations before expensive real evaluations — a genuinely differentiated angle for the ephemeral-intelligence thesis.
3. **GEPA-style reflective evolution beats RL at ruv-scale budgets** (35x fewer rollouts). For evolving swarm configs/prompts, prefer reflective-evolutionary over gradient methods; no GPU training loop needed.
4. Archive-based open-ended search (DGM) maps naturally onto ruv-swarm's memory system: store agent-variant configs + scores in AgentDB, sample parents by HNSW similarity to the task.

---

## 3. Model Routing & Cost Cascades

### 3.1 Routing research since RouteLLM

- **[Prior knowledge, confirmed by survey]** **RouteLLM** (LMSYS/Ong et al.) remains the reference point: preference-data-trained routers achieving ~85% cost reduction at 95% of GPT-4 quality on MT-Bench.
- **[Verified]** The space now has a proper survey — *Dynamic Model Routing and Cascading for Efficient LLM Inference* (arXiv 2603.04445) — framing designs by **when** the decision is made (pre-request / mid-inference / post-response), **what informs it** (query features, model metadata, history), and **how** (rules, classifiers, RL, cascades). https://arxiv.org/html/2603.04445v2
- **[Verified]** Notable 2025–2026 systems: **IRT-Router** (item-response theory over query difficulty x model ability), **UniRoute** (arXiv/ICLR 2026; handles *unseen* models at test time by embedding each model as a vector of its predictions on representative prompts), **Cluster-Route-Escalate** (arXiv 2606.27457; pre-routes predicted-hard queries straight to the strong model under a latency budget, avoiding wasted cheap-tier calls), and **LLMRouterBench** (arXiv 2601.07206), a unified routing benchmark. Decision-theoretic treatment of when escalation pays: arXiv 2605.06350.

### 3.2 Cheap-tier open models (distillation targets)

- **[Secondary]** Current open-weight coder landscape (mid-2026): **DeepSeek V4 Pro-Max at 80.6% SWE-bench Verified** (vendor-reported, MIT license) with a near-free Flash variant at **$0.14/$0.28 per Mtok**; **MiniMax M3 80.5%**; **GLM-5.2** (744B MoE / 40B active, MIT, 1M context) top open model on Artificial Analysis; **Kimi K3 93.4% Verified** on Vals AI's independent harness; **Qwen3.6-27B at 77.2%** runnable on a single consumer GPU; **Qwen3-Coder-Next 70.6% at 3B active params**. https://www.morphllm.com/best-open-source-coding-model-2026 , https://agyn.io/blog/top-open-weight-llms-2026
- **[Verified]** Distilling frontier trajectories into cheap tiers works: **SWE-Zero→SWE-Hero** (arXiv 2604.01496) — 300k execution-free + 13k execution-backed trajectories distilled from Qwen3-Coder-480B — yields **swe-hero-32B at 62.2% SWE-bench Verified**. Rejection fine-tuning (SFT on filtered successful attempts) is repeatedly the most effective simple recipe. https://arxiv.org/html/2604.01496 ; see also SWE-smith (arXiv 2504.21798) and Nebius's public OpenHands+Qwen3-Coder-480B trajectory dataset. https://nebius.com/blog/posts/openhands-trajectories-with-qwen3-coder-480b

### 3.3 OpenRouter ecosystem

- **[Verified]** OpenRouter now ships first-party routers: **Auto Router (Beta)** — task-classifies each request and routes to the most-used model for that task under a user-selected cost/quality tradeoff; a **Pareto Router** for coding — tiered shortlist ranked by Artificial Analysis coding percentiles with a `min_coding_score` knob; and a **Free Router** over zero-cost models (rate-limited 20 req/min / 200 req/day). API stays OpenAI-compatible; provider pricing passes through at cost. https://openrouter.ai/openrouter/auto , https://openrouter.ai/docs/faq

### Implications for ruv-FANN

1. **The 3-tier routing in CLAUDE.md (ADR-026) is directionally validated** by the literature, but SOTA routing is now *learned*, not thresholded. A tiny ruv-FANN classifier trained on the swarm's own routing outcomes (task features → tier success probability) is exactly the IRT-Router/UniRoute pattern — and is a showcase for CPU-native micro-networks (<1ms, $0, fits "Tier 1").
2. **Cluster-Route-Escalate's key lesson**: send *predicted-hard* tasks straight to the top tier; naive cascades waste money attempting the cheap tier first on everything.
3. **Cheap tier candidates for a claude-flow cascade**: DeepSeek V4 Flash / Qwen3.6-27B / GLM-5.2 via OpenRouter, with the Pareto Router's `min_coding_score` as an off-the-shelf fallback before building custom routing.
4. **Flywheel-distilled house models are viable at modest scale**: a 32B model at 62% Verified from pure SFT on frontier trajectories means an org collecting its own gold trajectories (§4) can plausibly own its Tier-2.

---

## 4. Data Flywheels for Agents

### 4.1 Memory-as-learning (test-time evolution)

- **[Verified]** **ReasoningBank** (Ouyang et al., arXiv 2509.25140, Google Cloud AI Research + UIUC): distills *both successful and failed* trajectories into structured strategy memories (title / one-line description / reasoning content), retrieved at test time. Up to **20% relative effectiveness gain and 16% fewer interaction steps** vs trajectory- or workflow-reuse baselines; introduces **memory-aware test-time scaling (MaTTS)** — the paper's core claim is that memory curation compounds with parallel rollouts. https://arxiv.org/pdf/2509.25140
- **[Verified]** Successor threads (2026): **SwiftMem** (query-aware indexing for fast agentic memory, arXiv 2601.08160), **Experience Memory Graph** (one-shot error correction, arXiv 2607.13884), selective memory retention for long horizons (arXiv 2606.29178), and meta-learned memory designs (arXiv 2602.07755). Structured memory for code agents specifically: arXiv 2603.13258.
- **[Prior knowledge]** This validates the claude-flow "reasoningbank-*" skills' design lineage; the 2026 literature adds *forgetting/curation* as the differentiator, not just accumulation.

### 4.2 Trajectory collection → distillation loops

- **[Verified]** **NVIDIA's Data Flywheel Blueprint** (NeMo microservices) automated the loop: log production traffic → curate → fine-tune smaller models → auto-evaluate → promote when latency/cost/accuracy criteria pass. Reported up to **98.6% inference-cost reduction** (Llama 3.1 70B → fine-tuned 8B at ~96% task accuracy, 70% latency cut). **Caveat: the blueprint repo was deprecated April 2026** (retained for reference) — treat as pattern, not dependency. https://github.com/NVIDIA-AI-Blueprints/data-flywheel , https://developer.nvidia.com/blog/build-efficient-ai-agents-through-model-distillation-with-nvidias-data-flywheel-blueprint
- **[Verified]** **Adaptive Data Flywheel** (arXiv 2510.27051) applies MAPE control loops (Monitor-Analyze-Plan-Execute) to agent improvement — a useful formal framing for claude-flow's hooks system.
- **[Verified]** Gold-archive distillation at benchmark scale: SWE-Zero/SWE-Hero (§3.2), **Open-SWE-Traces** (dual-mode multilingual distillation for SWE agents), and *What Makes Interaction Trajectories Effective for Training Terminal Agents?* (arXiv 2606.03461) — the latter is the closest thing to a recipe paper for which trajectories are worth keeping.

### Implications for ruv-FANN

1. **claude-flow already has the substrate** (AgentDB + HNSW + reasoningbank skills). The 2026 edge is (a) storing *distilled strategies* including failure lessons, not raw trajectories, and (b) MaTTS — coupling the memory bank to parallel-rollout selection (§1.3). Both are orchestration features, not model features.
2. **A gold-trajectory archive should be a first-class product surface**: every swarm run's successful trajectory, verdict-judged and stored, becomes distillation fuel for a house Tier-2 model (§3.4). NVIDIA's deprecation shows the *platform* market is unsettled — an open, Rust-native flywheel has room.
3. Trajectory *selection* quality (2606.03461) matters more than volume — verdict/rubric judging at store-time (claude-flow's "verification-quality" 0.95 truth-score machinery) is the right gate.

---

## 5. Neural Forecasting SOTA

### 5.1 Foundation models have displaced per-dataset training at the frontier

The N-BEATS/NHITS/PatchTST generation (neuro-divergent's core library) is no longer the accuracy frontier for zero-shot/general forecasting:

- **[Verified]** **Chronos-2** (Amazon, arXiv 2510.15821, Oct 2025): universal forecaster handling univariate, multivariate, and covariate-informed tasks via a group-attention in-context-learning mechanism, trained largely on *synthetic* multivariate structure. State-of-the-art on **fev-bench, GIFT-Eval, and Chronos Benchmark II**; secondary sources report it beating TiRex and TimesFM-2.5 on GIFT-Eval win-rate under both WQL and MASE, at 300+ forecasts/sec on one GPU. https://arxiv.org/abs/2510.15821
- **[Verified]** **TiRex** (NX-AI, arXiv 2505.23719): an **xLSTM-based, 35M-parameter** zero-shot model that topped GIFT-Eval and Chronos-ZS leaderboards in 2025 — notable because it is *recurrent, tiny, and CPU-friendly*, beating far larger transformers. **TiRex-2** (arXiv 2607.01204, 2026) extends to multivariate + streaming. https://github.com/NX-AI/tirex , https://arxiv.org/pdf/2607.01204
- **[Verified]** **Moirai 2.0** (Salesforce, arXiv 2511.11698, "When Less Is More"): simplified decoder-only redesign; ranks highly on GIFT-Eval among non-data-leaking models. https://arxiv.org/pdf/2511.11698
- **[Verified]** **TimesFM 2.x** (Google) remains the production-hardened baseline. **Tiny-TSM** (arXiv 2511.19272) shows lightweight SOTA TSFMs can be trained cheaply — the "small foundation model" thread is active.
- **[Verified]** Benchmark context: **GIFT-Eval** = 97 tasks / 55 datasets (Salesforce). Caveat papers worth noting: calibration of TSFMs is questioned (arXiv 2510.16060), and *How Foundational are Foundation Models for Time Series?* (arXiv 2510.00742) finds classical/tuned baselines still win in many in-domain settings.
- **[Prior knowledge]** For per-dataset supervised training on ample in-domain data, NHITS/PatchTST/TiDE-class models remain competitive and vastly cheaper — the foundation-model advantage is concentrated in zero-shot, cold-start, and covariate-rich settings.

### Implications for neuro-divergent

1. **The library's "27+ models, NeuralForecast-compatible" positioning is now a legacy-compatibility story, not a SOTA story.** The gap: no zero-shot foundation model in the lineup.
2. **TiRex is the strategic opening**: 35M params, recurrent (xLSTM), CPU-efficient — philosophically identical to ruv-FANN's "tiny purpose-built brains, GPU-poor" thesis. A Rust/WASM TiRex-class inference path (or an original small xLSTM/recurrent TSFM trained à la Tiny-TSM) would put neuro-divergent on GIFT-Eval with a CPU-native differentiator no Python incumbent has.
3. Add **GIFT-Eval and fev-bench harnesses** to neuro-divergent's benchmark suite; "2–4x faster than Python" claims should be accompanied by accuracy standings on the benchmarks the field now watches.
4. Chronos-2's synthetic-data training recipe (imposing multivariate structure on univariate series) is reproducible without web-scale data — relevant to any in-house pretraining attempt.

---

## 6. Rust ML / WASM Ecosystem

### 6.1 Framework state

- **[Verified]** **Burn** (tracel-ai): now the flagship Rust train+inference framework, with the **CubeCL** compute language targeting CUDA/ROCm/WGPU/CPU from one kernel source; production case studies include a 2026 healthcare deployment on 16xA100 reporting **2.3x speedup over PyTorch 2.3** for a training workload [Secondary]. Backends include `burn-wgpu` (WebGPU, browser-capable), `burn-ndarray` (pure CPU), and a `burn-candle` bridge. https://github.com/tracel-ai/burn
- **[Verified]** **Candle** (Hugging Face): the minimalist inference-first framework; the standard choice for serverless/WASM/edge deployment of transformer models. https://github.com/huggingface/candle
- **[Prior knowledge]** Neither offers FANN-style classic-NN ergonomics; ruv-FANN occupies a distinct niche (tiny classic networks, cascade correlation, zero-unsafe) that burn/candle don't target — but any *new* model family (e.g., an xLSTM TSFM, §5) would be faster to build on burn/candle than on ruv-FANN's own tensor layer.

### 6.2 WASM compute is no longer the bottleneck

- **[Secondary]** **WebAssembly 3.0 is stable across Chrome 119+/Firefox 120+/Safari 18.2+**: WasmGC, Memory64, **Relaxed SIMD** (new dot-product/FMA instructions, 1.5–3x over strict SIMD for ML kernels), tail calls, typed references. 128-bit SIMD + threads (SharedArrayBuffer) put browser compute at **85–95% of native** for hot loops. https://www.alldevtoolshub.com/blog/webassembly-browser-tools-2026-simd-threads-wasm-3/
- **[Secondary]** **WebGPU ships in all major browsers including iOS Safari (since late 2025), ~70% global support.** Practitioner rule of thumb: **WebGPU for models >~100M params; WASM-SIMD for smaller models and preprocessing** — i.e., ruv-FANN's entire target range sits on the WASM-SIMD side, where it is already native. https://www.sitepoint.com/webgpu-vs-webasm-transformers-js/
- **[Verified]** WebLLM (arXiv 2412.15803) and similar engines demonstrate full LLM inference in-browser via WebGPU — the ceiling for "ephemeral intelligence in the browser" keeps rising.

### Implications for ruv-FANN

1. **Adopt Relaxed SIMD** in ruv-FANN's WASM builds (dot-product/FMA paths) — a near-free 1.5–3x on exactly the kernels FANN-style networks spend time in. Audit whether current builds still target baseline WASM SIMD from 2024-era toolchains.
2. **The <100M-param regime is officially the WASM-CPU sweet spot** per 2026 practitioner guidance — this externally validates the "CPU-native, GPU-optional" positioning; cite it.
3. For anything beyond classic networks, **interop with burn/candle rather than compete**: e.g., neuro-divergent's future TSFM on burn with a wgpu fallback, while ruv-FANN remains the micro-network layer for routers, fitness predictors, and swarm controllers.
4. cuda-wasm work in this repo should track CubeCL, which has effectively become the community's answer to portable GPU kernels from Rust.

---

## 7. Cross-Cutting Synthesis: A 2026 Playbook for ruv-FANN

The six areas converge on one loop, and every piece has published, quantified precedent:

| Loop stage | 2026 SOTA precedent | ruv-FANN asset |
|---|---|---|
| Route task to cheapest capable tier | UniRoute / Cluster-Route-Escalate / OpenRouter Pareto | ADR-026 3-tier routing; ruv-FANN micro-classifier as learned router |
| Run parallel rollouts, select by summary tournament | RTV/PDR (+6.7 pp Verified) | ruv-swarm parallel agents |
| Store verdict-judged strategy memories incl. failures | ReasoningBank (+20% eff., −16% steps) | AgentDB + HNSW + reasoningbank skills |
| Evolve the harness itself from trajectory evidence | AHE (69.7→77.0 TB2), DGM (20→50 SWE-bench), GEPA (35x cheaper than RL) | hooks, file-level configs, memory namespaces |
| Distill gold archive into a house cheap-tier model | SWE-Hero 32B @ 62.2% Verified; NVIDIA flywheel (−98.6% cost) | trajectory logs + verification-quality gating |
| Cheap fitness/eval prediction to cut evolution cost | SIFT ($25 for +11 pp) | ruv-FANN tiny nets, <1ms CPU inference |

**Biggest risks flagged:** (1) the repo's headline SWE-bench claim is stale against saturated-benchmark skepticism; (2) neuro-divergent's model library missed the foundation-model turn; (3) the NVIDIA flywheel deprecation shows platform churn — build the loop on open primitives (OpenRouter API, mini-swe-agent-style eval, GIFT-Eval) rather than vendor blueprints.

---

## Appendix: Primary sources consulted

- SWE-bench Verified tracker: https://leaderboard.steel.dev/leaderboards/swe-bench-verified/ ; official site: https://www.swebench.com/
- Scaling Test-Time Compute for Agentic Coding: https://arxiv.org/abs/2604.16529
- Agentic Harness Engineering: https://arxiv.org/pdf/2604.25850
- Harness disclosure position paper: https://arxiv.org/pdf/2605.23950
- Darwin Gödel Machine: https://arxiv.org/abs/2505.22954
- AlphaEvolve paper / GA news: https://arxiv.org/abs/2506.13131 , https://www.infoq.com/news/2026/07/alphaevolve-generally-available/
- CodeEvolve: https://arxiv.org/abs/2510.14150
- GEPA: https://arxiv.org/pdf/2507.19457 , https://github.com/gepa-ai/gepa
- Self-evolving agents survey: https://arxiv.org/abs/2507.21046
- Routing/cascading survey: https://arxiv.org/html/2603.04445v2 ; Cluster-Route-Escalate: https://arxiv.org/pdf/2606.27457 ; LLMRouterBench: https://arxiv.org/html/2601.07206v1
- OpenRouter Auto/Pareto routers: https://openrouter.ai/openrouter/auto , https://openrouter.ai/docs/faq
- SWE-Zero→SWE-Hero distillation: https://arxiv.org/html/2604.01496 ; SWE-smith: https://arxiv.org/pdf/2504.21798 ; Nebius trajectories: https://nebius.com/blog/posts/openhands-trajectories-with-qwen3-coder-480b
- ReasoningBank: https://arxiv.org/pdf/2509.25140 ; trajectory-effectiveness study: https://arxiv.org/pdf/2606.03461
- NVIDIA Data Flywheel Blueprint (deprecated Apr 2026): https://github.com/NVIDIA-AI-Blueprints/data-flywheel ; MAPE flywheel: https://arxiv.org/pdf/2510.27051
- Chronos-2: https://arxiv.org/abs/2510.15821 ; TiRex: https://arxiv.org/abs/2505.23719 ; TiRex-2: https://arxiv.org/pdf/2607.01204 ; Moirai 2.0: https://arxiv.org/pdf/2511.11698 ; Tiny-TSM: https://arxiv.org/pdf/2511.19272 ; TSFM calibration critique: https://arxiv.org/pdf/2510.16060
- Burn: https://github.com/tracel-ai/burn ; Candle: https://github.com/huggingface/candle
- WASM 3.0 / WebGPU status (secondary): https://www.alldevtoolshub.com/blog/webassembly-browser-tools-2026-simd-threads-wasm-3/ , https://www.sitepoint.com/webgpu-vs-webasm-transformers-js/
- Open-weight coder landscape (secondary): https://www.morphllm.com/best-open-source-coding-model-2026 , https://agyn.io/blog/top-open-weight-llms-2026
- Harness-vs-model analyses (secondary): https://www.digitalapplied.com/blog/swe-bench-verified-june-2026-benchmark-vs-scaffolding-analysis , https://particula.tech/blog/agent-scaffolding-beats-model-upgrades-swe-bench
