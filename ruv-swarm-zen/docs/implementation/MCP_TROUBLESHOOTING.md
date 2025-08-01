# ruv-swarm MCP Troubleshooting Guide

## Quick Check
```bash
# Check if MCP server is configured
claude mcp list

# Check server details
claude mcp get ruv-swarm

# Test MCP server directly
echo '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}' | /workspaces/ruv-FANN/ruv-swarm/npm/mcp-server.sh
```

## Common Issues

### 1. Multiple MCP Processes
If you see multiple ruv-swarm processes:
```bash
# Check for stale processes
ps aux | grep "ruv-swarm mcp"

# Kill all MCP processes
pkill -f "ruv-swarm mcp"
```

### 2. MCP Tools Not Available
After configuring, you must:
1. Restart Claude Code (Cmd/Ctrl+Shift+P → "Developer: Reload Window")
2. Wait a few seconds for MCP to initialize
3. Try `/mcp` command to see tools

### 3. Server Fails to Start
Check the wrapper script:
```bash
# Test directly
/workspaces/ruv-FANN/ruv-swarm/npm/mcp-server.sh
```

Should output JSON-RPC messages only, no debug text.

### 4. Reconfigure from Scratch
```bash
# Remove existing
claude mcp remove ruv-swarm -s local

# Add fresh
claude mcp add ruv-swarm /workspaces/ruv-FANN/ruv-swarm/npm/mcp-server.sh

# Restart Claude Code
```

## Expected MCP Tools
Once working, these 12 tools should be available:
- swarm_init
- swarm_status  
- swarm_monitor
- agent_spawn
- agent_list
- agent_metrics
- task_orchestrate
- task_status
- task_results
- benchmark_run
- features_detect
- memory_usage

## Testing MCP Tools
In Claude Code, after restart:
1. Type `/mcp` to see available tools
2. Click on any tool to use it
3. Or ask Claude to use a specific tool

## Debug Mode
To see what's happening:
```bash
# Watch MCP processes
watch -n 1 'ps aux | grep "ruv-swarm mcp" | grep -v grep'

# Check Claude Code logs
# View → Output → Select "MCP" from dropdown
```