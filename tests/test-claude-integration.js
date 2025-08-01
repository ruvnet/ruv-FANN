#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');

async function setupClaudeIntegration() {
  console.log('Setting up Claude Code integration...');

  try {
    // Create claude.md
    const claudeMdContent = `# Claude Code Configuration for ruv-swarm

## Quick Setup (Stdio MCP - Recommended)

### 1. Add MCP Server (Stdio - No Port Needed)
\`\`\`bash
# Add ruv-swarm MCP server to Claude Code using stdio
claude mcp add ruv-swarm npx ruv-swarm mcp start
\`\`\`

### 2. Use MCP Tools Directly in Claude Code
Once configured, ruv-swarm MCP tools are available directly in Claude Code:

**Initialize a swarm:**
- Use the mcp__ruv-swarm-zen__swarm_init tool with topology: mesh, hierarchical, ring, or star
- Set maxAgents (default: 5)
- Enable cognitive diversity and neural agents

**Spawn agents:**
- Use mcp__ruv-swarm-zen__agent_spawn tool
- Agent types: researcher, coder, analyst, architect, reviewer, optimizer, coordinator
- Automatically includes neural network capabilities

**Orchestrate tasks:**
- Use mcp__ruv-swarm-zen__task_orchestrate tool
- Provide task description
- Choose strategy: parallel, sequential, adaptive
- Set priority: low, medium, high, critical

## Available MCP Tools

- mcp__ruv-swarm-zen__swarm_init - Initialize swarm topology (mesh/hierarchical/ring/star)
- mcp__ruv-swarm-zen__agent_spawn - Create specialized agents with neural capabilities
- mcp__ruv-swarm-zen__task_orchestrate - Coordinate distributed tasks across agents
- mcp__ruv-swarm-zen__swarm_status - Get real-time swarm status and metrics
- mcp__ruv-swarm-zen__agent_list - List active agents with filtering
- mcp__ruv-swarm-zen__agent_metrics - Get agent performance metrics
- mcp__ruv-swarm-zen__task_status - Check task progress and status
- mcp__ruv-swarm-zen__task_results - Retrieve completed task results
- mcp__ruv-swarm-zen__memory_usage - Monitor resource usage and optimization
- mcp__ruv-swarm-zen__neural_status - Check neural network performance
- mcp__ruv-swarm-zen__neural_train - Train neural agents
- mcp__ruv-swarm-zen__neural_patterns - Analyze cognitive patterns
- mcp__ruv-swarm-zen__benchmark_run - Execute performance benchmarks
- mcp__ruv-swarm-zen__features_detect - Detect runtime capabilities
- mcp__ruv-swarm-zen__swarm_monitor - Real-time monitoring

## Performance

- **84.8% SWE-Bench solve rate** - Industry-leading performance
- **32.3% token reduction** - Significant cost savings
- **2.8-4.4x speed improvement** - Faster than alternatives
- **27+ neural models** - Maximum cognitive diversity

## Support

- Documentation: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm
- Issues: https://github.com/ruvnet/ruv-FANN/issues
- Examples: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm/examples
`;

    await fs.writeFile('claude.md', claudeMdContent);
    console.log('✅ Created claude.md');

    // Create .claude directory structure
    await fs.mkdir('.claude', { recursive: true });
    await fs.mkdir('.claude/commands', { recursive: true });

    // Create basic command files
    const initContent = `# Initialize ruv-swarm

## MCP Tool Usage in Claude Code

**Tool:** mcp__ruv-swarm-zen__swarm_init

## Parameters
\`\`\`json
{"topology": "mesh", "maxAgents": 5, "strategy": "balanced"}
\`\`\`

## Examples

**Basic mesh topology:**
- Tool: mcp__ruv-swarm-zen__swarm_init
- Parameters: {"topology": "mesh", "maxAgents": 5}

## Topology Types
- **mesh**: Full connectivity, best for collaboration
- **hierarchical**: Tree structure, best for large projects
- **ring**: Circular coordination, best for sequential tasks
- **star**: Central coordination, best for controlled workflows
`;

    const spawnContent = `# Spawn Agents

## MCP Tool Usage in Claude Code

**Tool:** mcp__ruv-swarm-zen__agent_spawn

## Parameters
\`\`\`json
{"type": "researcher", "name": "AI Research Specialist", "capabilities": ["analysis"]}
\`\`\`

## Agent Types
- **researcher** - Research and analysis tasks
- **coder** - Code generation and development
- **analyst** - Data analysis and insights
- **architect** - System design and planning

## Examples

**Spawn research agent:**
- Tool: mcp__ruv-swarm-zen__agent_spawn
- Parameters: {"type": "researcher", "name": "AI Research Specialist"}
`;

    const orchestrateContent = `# Orchestrate Tasks

## MCP Tool Usage in Claude Code

**Tool:** mcp__ruv-swarm-zen__task_orchestrate

## Parameters
\`\`\`json
{"task": "Build REST API with authentication", "strategy": "parallel", "priority": "high", "maxAgents": 5}
\`\`\`

## Examples

**Research task:**
- Tool: mcp__ruv-swarm-zen__task_orchestrate
- Parameters: {"task": "Research modern web frameworks", "strategy": "adaptive"}

**Development with parallel strategy:**
- Tool: mcp__ruv-swarm-zen__task_orchestrate
- Parameters: {"task": "Build REST API", "strategy": "parallel", "priority": "high"}
`;

    await fs.writeFile('.claude/commands/init.md', initContent);
    await fs.writeFile('.claude/commands/spawn.md', spawnContent);
    await fs.writeFile('.claude/commands/orchestrate.md', orchestrateContent);

    console.log('✅ Created .claude/commands/ directory with command files');

    // Create ruv-swarm wrapper script
    const wrapperScript = `#!/usr/bin/env bash
# ruv-swarm local wrapper
# This script ensures ruv-swarm runs from your project directory

# Save the current directory
PROJECT_DIR="\${PWD}"

# Set environment to ensure correct working directory
export PWD="\${PROJECT_DIR}"
export RUVSW_WORKING_DIR="\${PROJECT_DIR}"

# Try to find ruv-swarm
# 1. Local npm/npx ruv-swarm
if command -v npx &> /dev/null; then
  cd "\${PROJECT_DIR}"
  exec npx ruv-swarm "\$@"

# 2. Local node_modules
elif [ -f "\${PROJECT_DIR}/node_modules/.bin/ruv-swarm" ]; then
  cd "\${PROJECT_DIR}"
  exec "\${PROJECT_DIR}/node_modules/.bin/ruv-swarm" "\$@"

# 3. Global installation (if available)
elif command -v ruv-swarm &> /dev/null; then
  cd "\${PROJECT_DIR}"
  exec ruv-swarm "\$@"

# 4. Fallback to direct npx with latest
else
  cd "\${PROJECT_DIR}"
  exec npx ruv-swarm@latest "\$@"
fi
`;

    await fs.writeFile('ruv-swarm', wrapperScript, { mode: 0o755 });
    console.log('✅ Created ruv-swarm wrapper script');

    console.log('\n🎉 Claude Code integration setup complete!');
    console.log('\n📋 Next steps:');
    console.log('1. In Claude Code: claude mcp add ruv-swarm npx ruv-swarm mcp start');
    console.log('2. Test with MCP tools: mcp__ruv-swarm-zen__agent_spawn');
    console.log('3. Check .claude/commands/ for detailed usage guides');

  } catch (error) {
    console.error('❌ Failed to setup Claude integration:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  setupClaudeIntegration();
}

module.exports = { setupClaudeIntegration };
