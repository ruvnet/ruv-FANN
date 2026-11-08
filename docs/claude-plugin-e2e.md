# Claude plugin end-to-end validation

The `ruv-fann` Claude Code plugin version `0.1.0` passed its scoped local and
ruOS checks on 2026-11-08. It provides eight detailed skills, eight supporting
references, three commands, a read-only reviewer and two Python tools. The
machine-readable record is [claude-plugin-e2e.json](claude-plugin-e2e.json).

## What ran

| Check | Evidence |
| --- | --- |
| Local tests | 39 passed, including real checkout inventory and README/SVG checks |
| Independent review | Found and fixed malformed TOML handling and background-child cleanup; regression tests passed |
| Plugin structure | 27 contained files; eight skills, three commands, one reviewer; no automatic hooks or MCP server |
| Official host validation | Claude Code 2.1.283 accepted the plugin and marketplace with no final errors or warnings |
| Project installation | Installed into a temporary project/configuration; all 27 installed files matched the reviewed source bytes |
| Host discovery | Live Claude registered all eight skills, three namespaced commands and the reviewer |
| Doctor invocation | Claude actually ran the Python helper and observed 26 Cargo packages and three npm manifests |
| Verification invocation | Claude ran doctor, plan and docs verification; the child check exited zero and reported valid README links/SVG sources |
| Detailed skill invocation | `/ruv-fann:forecasting` loaded and described the supported models; it disclosed reliance on the skill because the partial fixture lacked the review source |
| Remote plugin tests | 36 passed; the three full-checkout methods were omitted through one skipped class in the standalone layout |
| Real packaged runtime | Clean-installed `ruv-swarm@2.0.0` on Node 22.13.0; nine JSON-RPC frames, 17 tools, inference `[2]`, forecast `[7]`, benchmark checksum `20` |
| Negative runtime inputs | Rejected `constructor` with `-32601` and training with `UNSUPPORTED_FEATURE` |

The Work Desktop used Python 3.12.3. The shell's default Node was older; the
runtime test selected its existing Node 22.13.0 explicitly. No Rust toolchain
was installed there. Verification used exact copies of 57 current manifests,
README targets, SVGs and the verification script, plus an explicit fixture
scope marker. This establishes metadata/documentation behavior rather than a
new remote Rust, native OpenCV, browser or GPU build.

## Host permissions and retries

The live Claude sessions disabled hooks, persistent sessions and automatic MCP
loading, and used the host's existing authenticated model. Initial restricted
attempts stopped after denied shell exploration or an appended `echo` command.
These were not counted as successful workflows. Subsequent calls executed
independently permitted Python commands. The successful docs run used an
additional test instruction to use `Read` for instruction discovery and avoid
shell chaining. This did not bypass or permanently expand host permissions.

Existing MCP configuration, permission rules and global settings were preserved.
The installation itself added only the intended plugin/catalog entries in the
temporary project's configuration. Five completed model calls reported a total
list-price estimate of USD 0.5110856; this is not an account billing receipt.

An optional direct-ZIP session probe lost its ruOS transport and produced no
completed model receipt. It remains unverified, including its model cost. The
unpacked directory and local marketplace flows passed. The distributed ZIP is
the exact payload that was unpacked, installed and invoked successfully.

## Reproduce the verified flow

From a full checkout:

```sh
python3 -B -m unittest discover -s plugins/ruv-fann/tests -p 'test_*.py' -v
python3 -B plugins/ruv-fann/scripts/validate_plugin.py
claude plugin validate ./plugins/ruv-fann --json
claude plugin validate ./plugins --json
claude --plugin-dir ./plugins/ruv-fann
```

Inside that Claude session, invoke `/ruv-fann:doctor`, `/ruv-fann:verify docs`
and `/ruv-fann:forecasting`. A command still needs the host's ordinary tool
permission. The standalone helper can also run without a model:

```sh
python3 -B plugins/ruv-fann/scripts/repoctl.py doctor --repo .
python3 -B plugins/ruv-fann/scripts/repoctl.py verify --repo . --profile docs
```

See the [installation guide](claude-plugin.md) for the opt-in marketplace flow.
When using the ZIP, extract into a dedicated directory first, then pass that
directory to `--plugin-dir`. Its `.claude-plugin/plugin.json` sits at the
archive root; the separate marketplace catalog is supplied by the repository.

## Artifacts and coordination

The downloadable plugin archive is `ruv-fann-claude-plugin-0.1.0.zip`, 54,051
bytes, SHA-256:

```text
eccdd7dd19117256e6f0a6ae87b181689a49fc1606a216be5b8b558b2f84935b
```

The separate evidence archive is `ruv-fann-claude-plugin-e2e-2026-11-08.zip`,
73,557 bytes, containing 39 receipts/transcripts. Its SHA-256 is:

```text
41c407b30df6353a12d2db69e444a266549e4e109064c164413d0f86d7e4073d
```

Ruflo Ai Team recorded private task ownership and evidence. Platform workers
implemented and reviewed the files; ruOS executed the host and package checks.
A team task's status was not used as execution proof. Coordination identifiers
and the evidence export remain in the private task receipts.

All eight skill names were discovered; forecasting was the directly invoked
detailed skill. Reviewer registration was tested; delegated reviewer execution
was not exercised. No public publication, push or commit was performed.
