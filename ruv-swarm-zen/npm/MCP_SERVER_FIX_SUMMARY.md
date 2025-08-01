# MCP Server Startup Fix - Issue #155

## Problem Summary
The `npm run mcp:server` command was failing with a binary ambiguity error:
```
error: cargo run could not determine which binary to run. Use the --bin option to specify a binary
available binaries: ruv-swarm-mcp, ruv-swarm-mcp-stdio
```

## Root Cause
The `ruv-swarm-mcp` crate in `/home/mhugo/code/ruv-FANN/ruv-swarm/crates/ruv-swarm-mcp/Cargo.toml` defines two binary targets:
1. `ruv-swarm-mcp` (main binary at `src/main.rs`)
2. `ruv-swarm-mcp-stdio` (stdio binary at `src/bin/stdio.rs`)

When running `cargo run` without specifying which binary to use, cargo cannot determine which one to execute.

## Solution Applied
Updated the npm scripts in `package.json` to explicitly specify the binary:

### Before (Line 63):
```json
"mcp:server": "cd ../crates/ruv-swarm-mcp && cargo run"
```

### After (Line 63):
```json
"mcp:server": "cd ../crates/ruv-swarm-mcp && cargo run --bin ruv-swarm-mcp-stdio"
```

### Development Script Also Updated (Line 64):
```json
"mcp:server:dev": "cd ../crates/ruv-swarm-mcp && cargo watch -x 'run --bin ruv-swarm-mcp-stdio'"
```

## Verification

### 1. Error Demonstration (Before Fix):
```bash
$ cd ../crates/ruv-swarm-mcp && cargo run
error: `cargo run` could not determine which binary to run. Use the `--bin` option to specify a binary, or the `default-run` manifest key.
available binaries: ruv-swarm-mcp, ruv-swarm-mcp-stdio
```

### 2. Success Demonstration (After Fix):
```bash
$ npm run mcp:server
> ruv-swarm@1.0.18 mcp:server
> cd ../crates/ruv-swarm-mcp && cargo run --bin ruv-swarm-mcp-stdio

    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.39s
     Running `target/debug/ruv-swarm-mcp-stdio`
2025-07-24T14:01:05.056510Z  INFO ruv_swarm_mcp_stdio: Starting ruv-swarm-mcp stdio server
2025-07-24T14:01:05.065606Z  INFO ruv_swarm_persistence::migrations: Current schema version: 4
2025-07-24T14:01:05.066020Z  INFO ruv_swarm_persistence::migrations: No pending migrations
2025-07-24T14:01:05.066350Z  INFO ruv_swarm_persistence::sqlite: SQLite storage initialized at: ruv-swarm-mcp.db
2025-07-24T14:01:05.066378Z  INFO ruv_swarm_mcp::orchestrator: Using SQLite database at: ruv-swarm-mcp.db
```

Note: The final error `ConnectionClosed("initialized request")` is expected behavior for an MCP stdio server when run without a client connection.

## Files Modified
1. `/home/mhugo/code/ruv-FANN/ruv-swarm/npm/package.json`
   - Line 63: Updated `mcp:server` script
   - Line 64: Updated `mcp:server:dev` script
   - Line 47: Added `test:mcp-startup` script

2. `/home/mhugo/code/ruv-FANN/ruv-swarm/npm/test/mcp-server-startup.test.js` (Created)
   - Validation test for the fix

## Test Script Added
A new test script `test:mcp-startup` was added to validate the fix:
```bash
npm run test:mcp-startup
```

This test verifies:
1. Binary selection fix works (no ambiguity error)
2. Package.json scripts are correctly configured
3. MCP server starts successfully

## Status
✅ **RESOLVED** - Issue #155 is fixed and verified
- MCP server starts without binary ambiguity errors
- Both production and development scripts updated
- Test coverage added for regression prevention

## Impact
- **Before**: `npm run mcp:server` failed completely
- **After**: `npm run mcp:server` starts the MCP stdio server successfully
- **Risk**: Minimal - only changed binary specification, no functional changes