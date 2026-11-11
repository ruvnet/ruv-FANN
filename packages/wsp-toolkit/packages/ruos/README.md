# ruv-fann-wsp-ruos

`runWasmCanary()` exercises the real scalar WASM backend: two-weight XOR repair, exhaustive activation oracle, rejected neuron ablation, numerical sensitivity and revision-safe rollback. It throws on any failed assertion. This Node/browser function has no desktop connection or credentials.

`createValidationCommand(absoluteDirectory)` returns a shell-quoted `npm ci --ignore-scripts && npm run validate` plan for a trusted toolkit checkout. It does not execute anything. Select an exact observed desktop with ruOS's live `desktop_status`, then invoke its supported `desktop_exec` route. `ready` describes provisioning, not liveness; check heartbeat and execution separately. Do not wake or select a different computer implicitly.

`assertRuOSCompletion(actualToolResult, expectedExecutorMachineId)` requires executor-owned structured completion version 1, verified true, status ok, integer exit 0, matching executor identity and no reported truncation, timeout or transport error. Use the executor machine identity reported by a harmless connection probe; it may differ from the desktop selection ID. The helper rejects success strings/JSON in stdout. Its result explicitly does not independently attest assertions or cryptographically bind code to a run. Missing output completeness metadata still requires caller review. Never pass a fabricated tool envelope or use success to authorize deployment. No credentials or host tokens are exported.

<!-- constellation-links:start -->

Explore the [constellation onramp](../constellation/README.md) and [shared contracts](../contracts/README.md). Optional ecosystem primitives: [ruv-fann-wsp](https://www.npmjs.com/package/ruv-fann-wsp), [AgentDB](https://www.npmjs.com/package/agentdb), [RuVector](https://www.npmjs.com/package/ruvector). Optional orchestration: [MetaHarness](https://www.npmjs.com/package/metaharness) and [Ruflo](https://www.npmjs.com/package/ruflo).

These links describe integrations; they do not add install-time dependencies. Workspace package candidates and namespace proposals are listed explicitly in the catalog.

<!-- constellation-links:end -->
