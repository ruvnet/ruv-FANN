# ruv-swarm-mcp

Expose swarm operations to MCP clients over stdio, or use the separate local JSON-RPC/WebSocket server. These transports have different resource-management paths.

**Rust 1.89 or later is required** by this package's patched native SQLite
persistence dependency. The MCP SDK itself supports Rust 1.88.

The stdio SDK is pinned to `rmcp 3.9.0`, which fixes RUSTSEC-2026-0462 by
bounding nonstreaming HTTP responses. This crate enables only the SDK's server,
macros and stdio features; its HTTP client is not enabled. Incoming stdio lines
are separately capped at 1 MiB before buffering. Oversized lines are discarded,
and subsequent valid requests can continue on the same connection.

## Start a local server

From `ruv-swarm/`:

```sh
# MCP stdio transport; protocol output uses stdout, logging uses stderr.
cargo run --locked -p ruv-swarm-mcp --bin ruv-swarm-mcp-stdio

# Separate WebSocket example, bound to 127.0.0.1:3000.
cargo run --locked -p ruv-swarm-mcp --bin ruv-swarm-mcp
```

Set `RUV_SWARM_MCP_BIND_ADDR` to an IP socket address to run an isolated WebSocket instance, for example `127.0.0.1:4321`. The default remains `127.0.0.1:3000`; port `0` lets the operating system choose an available port. The server logs `MCP server listening on ...` only after binding successfully, with the actual assigned address. Invalid address values fail before creating the orchestrator/database.

The WebSocket endpoint is `/mcp`, and `/tools` lists its tool schemas. The stdio service uses `rmcp`; the WebSocket implementation has its own JSON-RPC handlers and does not imply compliance with every MCP transport specification.

## WebSocket budgets

The defaults allow 100 agents, 1,000 tasks, 64 MiB of accounted session memory, 10,000 memory entries, 1 MiB per value/message and a 24-hour session lifetime. Reservations enforce quotas while holding the same lock that updates usage, so simultaneous requests cannot all pass an advisory preflight check. Failed agent creation releases its reservation. Memory byte accounting rejects integer overflow.

The server applies connection permits, frame/message limits and per-request timeouts, and validates JSON-RPC version and live sessions. Parameter bodies are not logged by the handler entry point.

These protections do not cover all stdio service paths. The service has no complete authenticated multi-tenant security model; memory replacement/expiry accounting and workflow behavior remain incomplete. Use local clients within a trusted environment while those boundaries are developed. Task orchestration includes simulated processing and does not run a complete autonomous coding service.

## Check it

```sh
cargo test --locked -p ruv-swarm-mcp --lib
cargo check --locked -p ruv-swarm-mcp --bins
cargo test --locked -p ruv-swarm-mcp --bin ruv-swarm-mcp
# After building the actual stdio binary, run a bounded protocol round trip.
python3 crates/ruv-swarm-mcp/test_stdio.py --binary /absolute/target/debug/ruv-swarm-mcp-stdio
```

New regressions test concurrent quota reservations, memory arithmetic and bounded defaults. Several older security tests only validate request construction; their names do not establish end-to-end protection.

Published on [crates.io](https://crates.io/crates/ruv-swarm-mcp). Builds use rustls for TLS; no OpenSSL dependency. See the [workspace guide](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm) for all 14 packages and the [review record](https://github.com/ruvnet/ruv-FANN/blob/main/docs/review-rust-swarm.md) for checks, dependency findings and remaining limits.

![Animated map of the ruv-FANN packages](https://raw.githubusercontent.com/ruvnet/ruv-FANN/main/docs/assets/package-map.svg)
