# Workbench benchmark: real-data comparison and controlled repair

This offline benchmark compares tiny ruv-FANN models on two classical UCI classification datasets. It tests real WASM execution, inspection, tracing, numerical sensitivity, neuron ablation, hash-bound repair, promotion and rollback. These datasets establish neither agent-task performance nor state of the art.

The benchmark code and Python dependencies are repository tooling; they are excluded from the production npm package.

## Run

Use Node 22.13+ and CPython 3.12 on Linux x86_64. The requirements lock pins the hashes of compatible manylinux wheels. Create an isolated environment:

```sh
python3.12 -m venv /tmp/fann-benchmark-python
/tmp/fann-benchmark-python/bin/python -m pip install --require-hashes -r packages/wsp/benchmarks/requirements.lock
npm install --prefix packages/wsp --ignore-scripts --no-save --package-lock=false ruv-fann@0.1.2
node --test scripts/wsp-benchmark.test.mjs
node scripts/wsp-benchmark.mjs --python /tmp/fann-benchmark-python/bin/python --output /tmp/wsp-benchmark.json
```

The output must be an absolute path that does not already exist. Invalid CLI options fail. Missing or incompatible scikit-learn fails; the tree comparison is never silently skipped. `WSP_BENCHMARK_PYTHON` provides the same Python selection if `--python` is omitted.

For release qualification, run against the single sealed archive installed into a fresh consumer:

```sh
node scripts/wsp-benchmark.mjs --python /tmp/fann-benchmark-python/bin/python --wsp-module /tmp/consumer/node_modules/ruv-fann-wsp/src/index.js --output /tmp/packed-wsp-benchmark.json
```

The runtime is resolved from that installed package, with exact `ruv-fann@0.1.2` required. The report records source SHA, clean checkout status, package versions, module/script/lock hashes, raw data hashes, seed-specific row IDs, preprocessing parameters, model hashes, every candidate's validation result, held-out predictions, hardware, dependency versions and timings. The release harness separately binds the report to the exact sealed archive hash. Development runs with a dirty checkout report that state and do not qualify a release.

## Fixed comparison plan

- Five seeds: 17, 43, 101, 211, 509. Each class is independently shuffled using the declared LCG/Fisher–Yates implementation. Each class allocates `floor(0.6*n)` training and `floor(0.2*n)` validation rows; the remainder is test. Iris has 90/30/30 rows; Wine has 105/34/39. All split row IDs and hashes are reported.
- Standardization fits feature means and population standard deviations using training rows only. A zero standard deviation becomes one. The same transformation applies to every compared model.
- Baselines: training-label majority; nearest training centroid; real scalar WASM `D→3 Linear`; real scalar WASM `D→8 Tanh→3 Linear`; scikit-learn ExtraTrees with 128 trees and depths `[3, None]`, one worker.
- Neural candidates use full-batch AdamW, learning rate 0.01, and **100 or 200 epochs**, each in one runtime call with continuous optimizer state. Every candidate starts from the same deterministic initial model for that family and seed. The original proposed 300-epoch budget exceeds the published runtime's 100M work cap for the Wine MLP; 200 was selected from the work formula and applied globally. It was not selected by test performance.
- Each model family selects by validation accuracy, then validation MSE, then fewer epochs/earlier declared configuration. No validation refit. The held-out split only scores the selected result. Compute costs and training objectives differ; two configurations each do not imply equal compute.
- Primary metric: accuracy with first-class argmax ties. Secondary: macro F1 and one-hot target MSE. MSE encodings differ: hard one-hot labels for simple baselines, probabilities for trees, raw regression outputs for neural models. Python's independent sklearn metrics verify the JavaScript tree metrics.
- Summaries retain all outcomes, including baseline wins. The reported 95% Student t intervals across five overlapping splits are descriptive; they do not establish independent population significance. Intervals are not truncated to the metric range.

## Controlled ablation and exact repair

For each selected MLP, evaluate all eight possible first-layer neuron ablations on **validation only**. Select the largest validation accuracy decrease, then largest MSE increase, then lowest neuron index. Zeroing that neuron's outgoing weights uses the public hash-bound ablation API.

Restore the known original outgoing weights using a separate patch. Evaluate and promote the repair using the validation MSE policy, verify the exact original model hash, and then verify identical outputs on the held-out split. Rollback must recover the ablated model. A three-row trace must equal full actual WASM inference; a selected outgoing weight also receives central finite-difference sensitivity analysis.

This is a controlled intervention and restoration proof. Access to known original weights makes restoration possible. It is not evidence that an agent can discover repairs to arbitrary corruption or improve an unknown model autonomously.

## Blind repair with independent checkpoints

This separate experiment uses the same Iris and Wine splits and fits a new
checkpoint for every case. It injects large outliers, a sign flip or an all-zero
row into a seeded weight row. Both the hidden and readout layers are exercised.
There are 30 damaged cases and 10 clean controls.

```sh
npm install --prefix packages/wsp --ignore-scripts --no-save --package-lock=false ruv-fann@0.1.2
node --test scripts/wsp-benchmark.test.mjs scripts/wsp-blind-repair-controls.test.mjs
blind_dir=$(mktemp -d)
node scripts/wsp-blind-repair.mjs prepare "$blind_dir/input.json"
node scripts/wsp-blind-repair.mjs score "$blind_dir/report.json" "$PWD/packages/wsp/benchmarks/blind-repair-plan.json"
```

The frozen plan records one platform agent's batch of typed proposals and
binds the exact diagnostics by SHA-256. To run with another proposer, give it
only the exported JSON and use its plan in the score command. The diagnostics
contain damaged weights, activation summaries and development metrics. The
scorer retains intact weights, fault locations, gate rows and test rows. The
historical agent was instructed to use only the diagnostics. This was a
procedural separation on public fixtures.

Controls are no edit, fixed median clipping, scalar coordinate search, a ridge
refit of the readout, and grouped row search. Search and agent strategies may
submit up to 20 candidates per case. Development accuracy ranks candidates,
with MSE as the tiebreaker. Only the top qualifying candidate sees the
independent gate. Gate accuracy and MSE must preserve or improve the damaged
baseline. Rejected candidates score as the baseline, including abstentions.
Accepted patches use the WSP workspace to check stale revision rejection and
exact rollback to the damaged checkpoint. Reports group outcomes by dataset and
fault and include a timing independent result digest.

The domain level advancement criterion is a two point mean test accuracy gain
over the strongest control on both datasets, zero accepted agent proposals on
clean checkpoints, and exact rollback. The overlapping data splits provide
descriptive comparisons. Selection timing excludes external reasoning and
inference costs.

| Mean test accuracy on damaged cases | Iris | Wine |
| --- | ---: | ---: |
| No edit | 77.11% | 74.02% |
| Median clipping | 77.11% | 74.53% |
| Scalar coordinate search | 81.11% | 82.91% |
| Grouped row search | 87.78% | 91.79% |
| Ridge readout refit | 96.00% | 95.90% |
| Agent proposals | 94.67% | 94.70% |

The agent missed the accuracy gate, abstained on all ten clean controls and
passed exact rollback for all 27 accepted edits. The refit passed the local
gate on two clean controls and reduced held-out accuracy on one of them. A
separate test of fault detection is needed before automatic activation.
The full cohort result digest is
`cb10f844c43b333ed98583a2d7163885fda1ca5f408c34e836ef4dd7cbdecc74`.

## Dataset attribution and integrity

The original raw files under `data/` were downloaded unchanged from the official UCI server on 2026-11-10. No runtime network access is needed. Every run verifies the pinned bytes and SHA-256 values against `datasets.lock.json` and fixed identities in the benchmark helper. Iris uses the original `iris.data` version; known discrepancies documented on UCI's page have not been corrected.

Both official dataset pages explicitly declare [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Credit:

- Fisher, R. (1936). **Iris** [Dataset]. UCI Machine Learning Repository. [DOI:10.24432/C56C76](https://doi.org/10.24432/C56C76). [Official dataset and license](https://archive.ics.uci.edu/dataset/53/iris). Original data: `https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data`. SHA-256 `6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0`.
- Aeberhard, S. & Forina, M. (1992). **Wine** [Dataset]. UCI Machine Learning Repository. [DOI:10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J). [Official dataset and license](https://archive.ics.uci.edu/dataset/109/wine). Original data: `https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data`. SHA-256 `6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659`.

Adaptations made by this benchmark: deterministic partitioning, training-only standardization, conversion of labels into three-class one-hot targets, model fitting and derived measurements. Raw files remain unchanged. No endorsement by the creators or UCI is implied.

## Report schema

`ruv-fann.workbench-benchmark/1` has `status`, `sourceSha`, `sourceWorktreeClean`, package/module hashes, `datasetLockSha256`, `dataProvenance`, `tooling`, `hardware`, `plan`, `checks`, `aggregate`, `runs` and `limitations`. A passing report requires ten complete runs, real sklearn results and all workbench repair proofs. Each run includes every validation candidate and final held-out scores so metrics can be recomputed independently. Local timings describe the reported host and call boundary; they are not portable latency guarantees or speedup evidence.
