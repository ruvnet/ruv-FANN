# Patched SQLite and bounded MCP

The release audit found two affected native dependencies while validating dense
training: [RUSTSEC-2026-0459](https://rustsec.org/advisories/RUSTSEC-2026-0459)
in bundled SQLite and
[RUSTSEC-2026-0462](https://rustsec.org/advisories/RUSTSEC-2026-0462)
in rmcp. This change replaces those suppliers and tests the actual database and
protocol implementations. It adds no advisory exceptions.

| Component | Updated implementation |
| --- | --- |
| Rust persistence | Exact `rusqlite 0.41.1`, bundled `libsqlite3-sys 0.39.1`, actual SQLite 3.54.1 |
| Rust connection pool | Small owned `r2d2` manager; unfinished transactions are discarded |
| Benchmark store | Same SQLite supplier, bound SQL, one owned connection, bounded asynchronous admission and atomic metric batches |
| Rust MCP | Exact `rmcp 3.9.0`, actual SDK transport with a 1 MiB input-line limit |
| Node databases | Exact `better-sqlite3 12.11.1` wrapper, built against verified SQLite 3.54.1 sources |

Native swarm packages now require Rust **1.89**. The core library keeps **1.81**;
the minimal swarm WASM crate and napi-rs binding keep **1.88**. The optional
Candle adapter has its own newer toolchain and lockfile.

## Installing Node packages

At review time, published Node wrappers still bundled an affected SQLite. The
compatible wrapper is therefore built from source during installation. It
requires Python 3, a C++20 toolchain and access to the pinned public source and
Node headers. The source request honors host proxy and CA settings, requires TLS
verification, and does not forward registry credentials.

The [source contract](../ruv-swarm/npm/scripts/sqlite-source.json) pins the
archive, its size, all three amalgamation files, the SQLite version and source
ID. The builder bounds downloads, decompression, compiler output and execution
time. It selects files from the archive without using archive paths as output
paths. A private subprocess checks the compiled engine before publishing the
binary and receipt. Each platform, architecture and Node ABI has its own output.

Before opening an application database, the constructor checks the build inputs,
binary hash, wrapper version, ABI and actual `sqlite_version()` and
`sqlite_source_id()` in a private memory database. A missing, stale or invalid
receipt causes an actionable error. A failed build cannot select an older
prebuilt addon. Installations using `--ignore-scripts` can still load independent
WASM exports and help commands; database commands require the verified build.

## Validation

Local Rust verification passed 144 library tests, two real SQLite runtime and
durability tests, and the compiled server's bounded stdio protocol test. The
protocol test initializes a real SDK session, checks ten tools, persists memory
and agents, rejects invalid tool arguments, recovers after an oversized line,
and closes on EOF. It uses private temporary state and no external provider.

Node 22.13 and Node 24 each pass 11 focused SQLite tests and 108 maintained
security/runtime tests without a loader override. The archive test performs a
normal installation in a new external consumer, runs the source-build lifecycle,
verifies the installed engine and reopens persisted state across four public CLI
processes. Quoted Unicode task text survives a parameterized SQL round trip.

The mandatory [security workflow](../.github/workflows/sqlite-security.yml)
repeats the runtime, protocol, archive and seven Cargo lockfile audit checks.
The [structured training validation](candle-validation-2026-11-09.json) records
the local evidence hashes. Advisory audits of those seven graphs and npm audits
of the root and swarm graphs found zero vulnerabilities in the reviewed state;
maintenance warnings and legacy scan findings are recorded separately.

Reproduce the Node installation and archive checks with new external paths:

```sh
npm ci
node ruv-swarm/npm/scripts/sqlite-build.mjs --verify
npm --prefix ruv-swarm/npm run test:security
node ruv-swarm/npm/test/check-sqlite-package.mjs --output /tmp/new-sqlite-consumer
```
