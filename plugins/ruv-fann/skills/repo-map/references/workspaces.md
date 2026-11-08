# Repository evidence map

Read these paths inside the selected checkout, not relative to the plugin cache.

| Question | Source |
| --- | --- |
| Package roles and runnable root example | `README.md` |
| Live package identity and feature graph | Each family's `Cargo.toml`/`package.json` |
| Candidate versus dated registry observations | `docs/release-inventory.json` |
| Completed modernization scope and limits | `docs/modernization.md` |
| Machine-readable prior validation | `docs/validation-modernization.json` |
| Candidate archive identities | `docs/release-artifacts.json` |
| Research sources and adoption decisions | `docs/research-modernization.md` |
| Boundary and inference design | `docs/adr/ADR-006-inference-and-boundary-safety.md` |
| Cross-workspace driver | `scripts/verify-packages.py` |
| Isolated Cargo archive verification | `scripts/verify-release.py`, `docs/release-process.md` |

Prior receipts describe a particular source snapshot, toolchain and date. New source changes require relevant new checks; do not repeat an old test count as proof for a new artifact.

## Selecting direct checks

All commands below run from the checkout root:

```sh
python3 scripts/verify-packages.py --check --workspace .
python3 scripts/verify-packages.py --test --workspace ruv-swarm
python3 scripts/verify-packages.py --test --workspace neuro-divergent
python3 scripts/verify-packages.py --test --workspace cuda-wasm
python3 scripts/verify-packages.py --test --workspace opencv-rust
```

The last command requires native OpenCV. The `opencv` skill gives a narrower portable alternative. `--check` includes registered examples/benches without proving their runtime behavior. `--test` may execute benchmark smoke cases but does not measure a speedup.

## Coordination vocabulary

The repository's `ruv-swarm` library, the Ruflo CLI, the private Ruflo Ai Team app, and the ruOS executor are separate surfaces. A local assignment record or private task ledger does not execute a model. Platform workers or an explicitly configured executor perform the work; mark tasks complete only after validating their outputs. Keep private repository content off public federation channels.
