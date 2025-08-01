# Performance Benchmarking

## MCP Tool Usage in Claude Code

**Tool:** `mcp__ruv-swarm-zen__benchmark_run`

## Parameters
```json
{
  "type": "all",        // all, wasm, swarm, agent, task
  "iterations": 10      // Number of benchmark iterations (1-100)
}
```

## Examples

**Full system benchmark:**
- Tool: `mcp__ruv-swarm-zen__benchmark_run`
- Parameters: `{"type": "all", "iterations": 10}`

**WASM performance test:**
- Tool: `mcp__ruv-swarm-zen__benchmark_run`
- Parameters: `{"type": "wasm", "iterations": 20}`

**Swarm coordination benchmark:**
- Tool: `mcp__ruv-swarm-zen__benchmark_run`
- Parameters: `{"type": "swarm", "iterations": 15}`

**Agent spawn performance:**
- Tool: `mcp__ruv-swarm-zen__benchmark_run`
- Parameters: `{"type": "agent", "iterations": 25}`

**Task execution benchmark:**
- Tool: `mcp__ruv-swarm-zen__benchmark_run`
- Parameters: `{"type": "task", "iterations": 30}`

## Benchmark Types
- **all** - Complete system performance evaluation
- **wasm** - WebAssembly execution speed and memory
- **swarm** - Multi-agent coordination efficiency  
- **agent** - Individual agent performance metrics
- **task** - Task orchestration and completion times

## Performance Targets
- **SWE-Bench solve rate:** 84.8%
- **Token reduction:** 32.3%
- **Speed improvement:** 2.8-4.4x faster
- **Neural models:** 27+ cognitive diversity models