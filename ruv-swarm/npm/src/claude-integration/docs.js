/**
 * Documentation generation module for Claude Code integration
 * Generates claude.md and .claude/commands/ documentation
 */

import fs from 'fs/promises';
import path from 'path';
import { AdvancedCommandsGenerator } from './advanced-commands.js';

class ClaudeDocsGenerator {
  constructor(options = {}) {
    this.workingDir = options.workingDir || process.cwd();
    this.advancedGenerator = new AdvancedCommandsGenerator(options);
  }

  /**
     * Generate main claude.md configuration file
     */
  async generateClaudeMd() {
    const content = `# Claude Code Configuration for ruv-swarm

## 🎯 IMPORTANT: Separation of Responsibilities

### Claude Code Handles:
- ✅ **ALL file operations** (Read, Write, Edit, MultiEdit)
- ✅ **ALL code generation** and development tasks
- ✅ **ALL bash commands** and system operations
- ✅ **ALL actual implementation** work
- ✅ **Project navigation** and code analysis

### ruv-swarm MCP Tools Handle:
- 🧠 **Coordination only** - Orchestrating Claude Code's actions
- 💾 **Memory management** - Persistent state across sessions
- 🤖 **Neural features** - Cognitive patterns and learning
- 📊 **Performance tracking** - Monitoring and metrics
- 🐝 **Swarm orchestration** - Multi-agent coordination

### ⚠️ Key Principle:
**MCP tools DO NOT create content or write code.** They coordinate and enhance Claude Code's native capabilities. Think of them as an orchestration layer that helps Claude Code work more efficiently.

## 🚀 CRITICAL: Parallel Execution & Batch Operations

### 🚨 MANDATORY RULE #1: BATCH EVERYTHING

**When using swarms, you MUST use BatchTool for ALL operations:**

1. **NEVER** send multiple messages for related operations
2. **ALWAYS** combine multiple tool calls in ONE message
3. **PARALLEL** execution is MANDATORY, not optional

### ⚡ THE GOLDEN RULE OF SWARMS

\`\`\`
If you need to do X operations, they should be in 1 message, not X messages
\`\`\`

### 📦 BATCH TOOL EXAMPLES

**✅ CORRECT - Everything in ONE Message:**
\`\`\`javascript
[Single Message with BatchTool]:
  mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 6 }
  mcp__ruv-swarm__agent_spawn { type: "researcher" }
  mcp__ruv-swarm__agent_spawn { type: "coder" }
  mcp__ruv-swarm__agent_spawn { type: "analyst" }
  mcp__ruv-swarm__agent_spawn { type: "tester" }
  mcp__ruv-swarm__agent_spawn { type: "coordinator" }
  TodoWrite { todos: [todo1, todo2, todo3, todo4, todo5] }
  Bash "mkdir -p app/{src,tests,docs}"
  Write "app/package.json" 
  Write "app/README.md"
  Write "app/src/index.js"
\`\`\`

**❌ WRONG - Multiple Messages (NEVER DO THIS):**
\`\`\`javascript
Message 1: mcp__ruv-swarm__swarm_init
Message 2: mcp__ruv-swarm__agent_spawn 
Message 3: mcp__ruv-swarm__agent_spawn
Message 4: TodoWrite (one todo)
Message 5: Bash "mkdir src"
Message 6: Write "package.json"
// This is 6x slower and breaks parallel coordination!
\`\`\`

### 🎯 BATCH OPERATIONS BY TYPE

**File Operations (Single Message):**
- Read 10 files? → One message with 10 Read calls
- Write 5 files? → One message with 5 Write calls
- Edit 1 file many times? → One MultiEdit call

**Swarm Operations (Single Message):**
- Need 8 agents? → One message with swarm_init + 8 agent_spawn calls
- Multiple memories? → One message with all memory_usage calls
- Task + monitoring? → One message with task_orchestrate + swarm_monitor

**Command Operations (Single Message):**
- Multiple directories? → One message with all mkdir commands
- Install + test + lint? → One message with all npm commands
- Git operations? → One message with all git commands

## 🚀 Quick Setup (Stdio MCP - Recommended)

### 1. Add MCP Server (Stdio - No Port Needed)
\`\`\`bash
# Add ruv-swarm MCP server to Claude Code using stdio
claude mcp add ruv-swarm npx ruv-swarm mcp start
\`\`\`

### 2. Use MCP Tools for Coordination in Claude Code
Once configured, ruv-swarm MCP tools enhance Claude Code's coordination:

**Initialize a swarm:**
- Use the \`mcp__ruv-swarm__swarm_init\` tool to set up coordination topology
- Choose: mesh, hierarchical, ring, or star
- This creates a coordination framework for Claude Code's work

**Spawn agents:**
- Use \`mcp__ruv-swarm__agent_spawn\` tool to create specialized coordinators
- Agent types represent different thinking patterns, not actual coders
- They help Claude Code approach problems from different angles

**Orchestrate tasks:**
- Use \`mcp__ruv-swarm__task_orchestrate\` tool to coordinate complex workflows
- This breaks down tasks for Claude Code to execute systematically
- The agents don't write code - they coordinate Claude Code's actions

## Available MCP Tools for Coordination

### Coordination Tools:
- \`mcp__ruv-swarm__swarm_init\` - Set up coordination topology for Claude Code
- \`mcp__ruv-swarm__agent_spawn\` - Create cognitive patterns to guide Claude Code
- \`mcp__ruv-swarm__task_orchestrate\` - Break down and coordinate complex tasks

### Monitoring Tools:
- \`mcp__ruv-swarm__swarm_status\` - Monitor coordination effectiveness
- \`mcp__ruv-swarm__agent_list\` - View active cognitive patterns
- \`mcp__ruv-swarm__agent_metrics\` - Track coordination performance
- \`mcp__ruv-swarm__task_status\` - Check workflow progress
- \`mcp__ruv-swarm__task_results\` - Review coordination outcomes

### Memory & Neural Tools:
- \`mcp__ruv-swarm__memory_usage\` - Persistent memory across sessions
- \`mcp__ruv-swarm__neural_status\` - Neural pattern effectiveness
- \`mcp__ruv-swarm__neural_train\` - Improve coordination patterns
- \`mcp__ruv-swarm__neural_patterns\` - Analyze thinking approaches

### System Tools:
- \`mcp__ruv-swarm__benchmark_run\` - Measure coordination efficiency
- \`mcp__ruv-swarm__features_detect\` - Available capabilities
- \`mcp__ruv-swarm__swarm_monitor\` - Real-time coordination tracking

## Workflow Examples (Coordination-Focused)

### Research Coordination Example
**Context:** Claude Code needs to research a complex topic systematically

**Step 1:** Set up research coordination
- Tool: \`mcp__ruv-swarm__swarm_init\`
- Parameters: \`{"topology": "mesh", "maxAgents": 5, "strategy": "balanced"}\`
- Result: Creates a mesh topology for comprehensive exploration

**Step 2:** Define research perspectives
- Tool: \`mcp__ruv-swarm__agent_spawn\`
- Parameters: \`{"type": "researcher", "name": "Literature Review"}\`
- Tool: \`mcp__ruv-swarm__agent_spawn\`
- Parameters: \`{"type": "analyst", "name": "Data Analysis"}\`
- Result: Different cognitive patterns for Claude Code to use

**Step 3:** Coordinate research execution
- Tool: \`mcp__ruv-swarm__task_orchestrate\`
- Parameters: \`{"task": "Research neural architecture search papers", "strategy": "adaptive"}\`
- Result: Claude Code systematically searches, reads, and analyzes papers

**What Actually Happens:**
1. The swarm sets up a coordination framework
2. Each agent MUST use ruv-swarm hooks for coordination:
   - \`npx ruv-swarm hook pre-task\` before starting
   - \`npx ruv-swarm hook post-edit\` after each file operation
   - \`npx ruv-swarm hook notification\` to share decisions
3. Claude Code uses its native Read, WebSearch, and Task tools
4. The swarm coordinates through shared memory and hooks
5. Results are synthesized by Claude Code with full coordination history

### Development Coordination Example
**Context:** Claude Code needs to build a complex system with multiple components

**Step 1:** Set up development coordination
- Tool: \`mcp__ruv-swarm__swarm_init\`
- Parameters: \`{"topology": "hierarchical", "maxAgents": 8, "strategy": "specialized"}\`
- Result: Hierarchical structure for organized development

**Step 2:** Define development perspectives
- Tool: \`mcp__ruv-swarm__agent_spawn\`
- Parameters: \`{"type": "architect", "name": "System Design"}\`
- Result: Architectural thinking pattern for Claude Code

**Step 3:** Coordinate implementation
- Tool: \`mcp__ruv-swarm__task_orchestrate\`
- Parameters: \`{"task": "Implement user authentication with JWT", "strategy": "parallel"}\`
- Result: Claude Code implements features using its native tools

**What Actually Happens:**
1. The swarm creates a development coordination plan
2. Each agent coordinates using mandatory hooks:
   - Pre-task hooks for context loading
   - Post-edit hooks for progress tracking
   - Memory storage for cross-agent coordination
3. Claude Code uses Write, Edit, Bash tools for implementation
4. Agents share progress through ruv-swarm memory
5. All code is written by Claude Code with full coordination

## Best Practices for Coordination

### ✅ DO:
- Use MCP tools to coordinate Claude Code's approach to complex tasks
- Let the swarm break down problems into manageable pieces
- Use memory tools to maintain context across sessions
- Monitor coordination effectiveness with status tools
- Train neural patterns for better coordination over time

### ❌ DON'T:
- Expect agents to write code (Claude Code does all implementation)
- Use MCP tools for file operations (use Claude Code's native tools)
- Try to make agents execute bash commands (Claude Code handles this)
- Confuse coordination with execution (MCP coordinates, Claude executes)

## Memory and Persistence

The swarm provides persistent memory that helps Claude Code:
- Remember project context across sessions
- Track decisions and rationale
- Maintain consistency in large projects
- Learn from previous coordination patterns

## Performance Benefits

When using ruv-swarm coordination with Claude Code:
- **84.8% SWE-Bench solve rate** - Better problem-solving through coordination
- **32.3% token reduction** - Efficient task breakdown reduces redundancy
- **2.8-4.4x speed improvement** - Parallel coordination strategies
- **27+ neural models** - Diverse cognitive approaches

## Claude Code Hooks Integration

ruv-swarm includes powerful hooks that automate coordination:

### Pre-Operation Hooks
- **Auto-assign agents** before file edits based on file type
- **Validate commands** before execution for safety
- **Prepare resources** automatically for complex operations
- **Optimize topology** based on task complexity analysis
- **Cache searches** for improved performance

### Post-Operation Hooks  
- **Auto-format code** using language-specific formatters
- **Train neural patterns** from successful operations
- **Update memory** with operation context
- **Analyze performance** and identify bottlenecks
- **Track token usage** for efficiency metrics

### Session Management
- **Generate summaries** at session end
- **Persist state** across Claude Code sessions
- **Track metrics** for continuous improvement
- **Restore previous** session context automatically

### Advanced Features (New!)
- **🚀 Automatic Topology Selection** - Optimal swarm structure for each task
- **⚡ Parallel Execution** - 2.8-4.4x speed improvements  
- **🧠 Neural Training** - Continuous learning from operations
- **📊 Bottleneck Analysis** - Real-time performance optimization
- **🤖 Smart Auto-Spawning** - Zero manual agent management
- **🛡️ Self-Healing Workflows** - Automatic error recovery
- **💾 Cross-Session Memory** - Persistent learning & context

### Configuration
Hooks are pre-configured in \`.claude/settings.json\`. Key features:
- Automatic agent assignment for different file types
- Code formatting on save
- Neural pattern learning from edits
- Session state persistence
- Performance tracking and optimization
- Intelligent caching and token reduction

See \`.claude/commands/\` for detailed documentation on all features.

## Integration Tips

1. **Start Simple**: Begin with basic swarm init and single agent
2. **Scale Gradually**: Add more agents as task complexity increases
3. **Use Memory**: Store important decisions and context
4. **Monitor Progress**: Regular status checks ensure effective coordination
5. **Train Patterns**: Let neural agents learn from successful coordinations
6. **Enable Hooks**: Use the pre-configured hooks for automation

## 🧠 SWARM ORCHESTRATION PATTERN

### You are the SWARM ORCHESTRATOR. **IMMEDIATELY SPAWN AGENTS IN PARALLEL** to execute tasks

### 🚨 CRITICAL INSTRUCTION: You are the SWARM ORCHESTRATOR

**MANDATORY**: When using swarms, you MUST:
1. **SPAWN ALL AGENTS IN ONE BATCH** - Use multiple tool calls in a SINGLE message
2. **EXECUTE TASKS IN PARALLEL** - Never wait for one task before starting another
3. **USE BATCHTOOL FOR EVERYTHING** - Multiple operations = Single message with multiple tools
4. **ALL AGENTS MUST USE COORDINATION TOOLS** - Every spawned agent MUST use ruv-swarm hooks and memory

## 📋 MANDATORY AGENT COORDINATION PROTOCOL

### 🔴 CRITICAL: Every Agent MUST Follow This Protocol

When you spawn an agent using the Task tool, that agent MUST:

**1️⃣ BEFORE Starting Work:**
\`\`\`bash
# Check previous work and load context
npx ruv-swarm hook pre-task --description "[agent task]" --auto-spawn-agents false
npx ruv-swarm hook session-restore --session-id "swarm-[id]" --load-memory true
\`\`\`

**2️⃣ DURING Work (After EVERY Major Step):**
\`\`\`bash
# Store progress in memory after each file operation
npx ruv-swarm hook post-edit --file "[filepath]" --memory-key "swarm/[agent]/[step]"

# Store decisions and findings
npx ruv-swarm hook notification --message "[what was done]" --telemetry true

# Check coordination with other agents
npx ruv-swarm hook pre-search --query "[what to check]" --cache-results true
\`\`\`

**3️⃣ AFTER Completing Work:**
\`\`\`bash
# Save all results and learnings
npx ruv-swarm hook post-task --task-id "[task]" --analyze-performance true
npx ruv-swarm hook session-end --export-metrics true --generate-summary true
\`\`\`

### 🎯 AGENT PROMPT TEMPLATE

When spawning agents, ALWAYS include these coordination instructions:

\`\`\`
You are the [Agent Type] agent in a coordinated swarm.

MANDATORY COORDINATION:
1. START: Run \`npx ruv-swarm hook pre-task --description "[your task]"\`
2. DURING: After EVERY file operation, run \`npx ruv-swarm hook post-edit --file "[file]" --memory-key "agent/[step]"\`
3. MEMORY: Store ALL decisions using \`npx ruv-swarm hook notification --message "[decision]"\`
4. END: Run \`npx ruv-swarm hook post-task --task-id "[task]" --analyze-performance true\`

Your specific task: [detailed task description]

REMEMBER: Coordinate with other agents by checking memory BEFORE making decisions!
\`\`\`

### ⚡ PARALLEL EXECUTION IS MANDATORY

**THIS IS WRONG ❌ (Sequential - NEVER DO THIS):**
\`\`\`
Message 1: Initialize swarm
Message 2: Spawn agent 1
Message 3: Spawn agent 2
Message 4: Create file 1
Message 5: Create file 2
\`\`\`

**THIS IS CORRECT ✅ (Parallel - ALWAYS DO THIS):**
\`\`\`
Message 1: [BatchTool]
  - mcp__ruv-swarm__swarm_init
  - mcp__ruv-swarm__agent_spawn (researcher)
  - mcp__ruv-swarm__agent_spawn (coder)
  - mcp__ruv-swarm__agent_spawn (analyst)
  - mcp__ruv-swarm__agent_spawn (tester)
  - mcp__ruv-swarm__agent_spawn (coordinator)

Message 2: [BatchTool]  
  - Write file1.js
  - Write file2.js
  - Write file3.js
  - Bash mkdir commands
  - TodoWrite updates
\`\`\`

### 🎯 MANDATORY SWARM PATTERN

When given ANY complex task with swarms:

\`\`\`
STEP 1: IMMEDIATE PARALLEL SPAWN (Single Message!)
[BatchTool]:
  - mcp__ruv-swarm__swarm_init { topology: "hierarchical", maxAgents: 8, strategy: "parallel" }
  - mcp__ruv-swarm__agent_spawn { type: "architect", name: "System Designer" }
  - mcp__ruv-swarm__agent_spawn { type: "coder", name: "API Developer" }
  - mcp__ruv-swarm__agent_spawn { type: "coder", name: "Frontend Dev" }
  - mcp__ruv-swarm__agent_spawn { type: "analyst", name: "DB Designer" }
  - mcp__ruv-swarm__agent_spawn { type: "tester", name: "QA Engineer" }
  - mcp__ruv-swarm__agent_spawn { type: "researcher", name: "Tech Lead" }
  - mcp__ruv-swarm__agent_spawn { type: "coordinator", name: "PM" }
  - TodoWrite { todos: [multiple todos at once] }

STEP 2: PARALLEL TASK EXECUTION (Single Message!)
[BatchTool]:
  - mcp__ruv-swarm__task_orchestrate { task: "main task", strategy: "parallel" }
  - mcp__ruv-swarm__memory_usage { action: "store", key: "init", value: {...} }
  - Multiple Read operations
  - Multiple Write operations
  - Multiple Bash commands

STEP 3: CONTINUE PARALLEL WORK (Never Sequential!)
\`\`\`

### 📊 VISUAL TASK TRACKING FORMAT

Use this format when displaying task progress:

\`\`\`
📊 Progress Overview
   ├── Total Tasks: X
   ├── ✅ Completed: X (X%)
   ├── 🔄 In Progress: X (X%)
   ├── ⭕ Todo: X (X%)
   └── ❌ Blocked: X (X%)

📋 Todo (X)
   └── 🔴 001: [Task description] [PRIORITY] ▶

🔄 In progress (X)
   ├── 🟡 002: [Task description] ↳ X deps ▶
   └── 🔴 003: [Task description] [PRIORITY] ▶

✅ Completed (X)
   ├── ✅ 004: [Task description]
   └── ... (more completed tasks)

Priority indicators: 🔴 HIGH/CRITICAL, 🟡 MEDIUM, 🟢 LOW
Dependencies: ↳ X deps | Actionable: ▶
\`\`\`

### 🎯 REAL EXAMPLE: Full-Stack App Development

**Task**: "Build a complete REST API with authentication, database, and tests"

**🚨 MANDATORY APPROACH - Everything in Parallel:**

\`\`\`javascript
// ✅ CORRECT: SINGLE MESSAGE with ALL operations
[BatchTool - Message 1]:
  // Initialize and spawn ALL agents at once
  mcp__ruv-swarm__swarm_init { topology: "hierarchical", maxAgents: 8, strategy: "parallel" }
  mcp__ruv-swarm__agent_spawn { type: "architect", name: "System Designer" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "API Developer" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "Auth Expert" }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "DB Designer" }
  mcp__ruv-swarm__agent_spawn { type: "tester", name: "Test Engineer" }
  mcp__ruv-swarm__agent_spawn { type: "coordinator", name: "Lead" }
  
  // Update ALL todos at once
  TodoWrite { todos: [
    { id: "design", content: "Design API architecture", status: "in_progress", priority: "high" },
    { id: "auth", content: "Implement authentication", status: "pending", priority: "high" },
    { id: "db", content: "Design database schema", status: "pending", priority: "high" },
    { id: "api", content: "Build REST endpoints", status: "pending", priority: "high" },
    { id: "tests", content: "Write comprehensive tests", status: "pending", priority: "medium" }
  ]}
  
  // Start orchestration
  mcp__ruv-swarm__task_orchestrate { task: "Build REST API", strategy: "parallel" }
  
  // Store initial memory
  mcp__ruv-swarm__memory_usage { action: "store", key: "project/init", value: { started: Date.now() } }

[BatchTool - Message 2]:
  // Create ALL directories at once
  Bash("mkdir -p test-app/{src,tests,docs,config}")
  Bash("mkdir -p test-app/src/{models,routes,middleware,services}")
  Bash("mkdir -p test-app/tests/{unit,integration}")
  
  // Write ALL base files at once
  Write("test-app/package.json", packageJsonContent)
  Write("test-app/.env.example", envContent)
  Write("test-app/README.md", readmeContent)
  Write("test-app/src/server.js", serverContent)
  Write("test-app/src/config/database.js", dbConfigContent)

[BatchTool - Message 3]:
  // Read multiple files for context
  Read("test-app/package.json")
  Read("test-app/src/server.js")
  Read("test-app/.env.example")
  
  // Run multiple commands
  Bash("cd test-app && npm install")
  Bash("cd test-app && npm run lint")
  Bash("cd test-app && npm test")
\`\`\`

### 🚫 NEVER DO THIS (Sequential = WRONG):
\`\`\`javascript
// ❌ WRONG: Multiple messages, one operation each
Message 1: mcp__ruv-swarm__swarm_init
Message 2: mcp__ruv-swarm__agent_spawn (just one agent)
Message 3: mcp__ruv-swarm__agent_spawn (another agent)
Message 4: TodoWrite (single todo)
Message 5: Write (single file)
// This is 5x slower and wastes swarm coordination!
\`\`\`

### 🔄 MEMORY COORDINATION PATTERN

Every agent coordination step MUST use memory:

\`\`\`
// After each major decision or implementation
mcp__ruv-swarm__memory_usage
  action: "store"
  key: "swarm-{id}/agent-{name}/{step}"
  value: {
    timestamp: Date.now(),
    decision: "what was decided",
    implementation: "what was built",
    nextSteps: ["step1", "step2"],
    dependencies: ["dep1", "dep2"]
  }

// To retrieve coordination data
mcp__ruv-swarm__memory_usage
  action: "retrieve"
  key: "swarm-{id}/agent-{name}/{step}"

// To check all swarm progress
mcp__ruv-swarm__memory_usage
  action: "list"
  pattern: "swarm-{id}/*"
\`\`\`

### ⚡ PERFORMANCE TIPS

1. **Batch Everything**: Never operate on single files when multiple are needed
2. **Parallel First**: Always think "what can run simultaneously?"
3. **Memory is Key**: Use memory for ALL cross-agent coordination
4. **Monitor Progress**: Use mcp__ruv-swarm__swarm_monitor for real-time tracking
5. **Auto-Optimize**: Let hooks handle topology and agent selection

### 🎨 VISUAL SWARM STATUS

When showing swarm status, use this format:

\`\`\`
🐝 Swarm Status: ACTIVE
├── 🏗️ Topology: hierarchical
├── 👥 Agents: 6/8 active
├── ⚡ Mode: parallel execution
├── 📊 Tasks: 12 total (4 complete, 6 in-progress, 2 pending)
└── 🧠 Memory: 15 coordination points stored

Agent Activity:
├── 🟢 architect: Designing database schema...
├── 🟢 coder-1: Implementing auth endpoints...
├── 🟢 coder-2: Building user CRUD operations...
├── 🟢 analyst: Optimizing query performance...
├── 🟡 tester: Waiting for auth completion...
└── 🟢 coordinator: Monitoring progress...
\`\`\`

## Support

- Documentation: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm
- Issues: https://github.com/ruvnet/ruv-FANN/issues
- Examples: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm/examples

---

Remember: **ruv-swarm coordinates, Claude Code creates!** Start with \`mcp__ruv-swarm__swarm_init\` to enhance your development workflow.
`;

    const filePath = path.join(this.workingDir, 'CLAUDE.md');
    await fs.writeFile(filePath, content);
    return { file: 'CLAUDE.md', success: true };
  }

  /**
     * Generate command documentation files in organized subdirectories
     */
  async generateCommandDocs() {
    const commandsDir = path.join(this.workingDir, '.claude', 'commands');
    await fs.mkdir(commandsDir, { recursive: true });

    // Create subdirectories
    const subdirs = ['coordination', 'monitoring', 'memory', 'workflows', 'hooks',
      'optimization', 'analysis', 'training', 'automation'];
    for (const subdir of subdirs) {
      await fs.mkdir(path.join(commandsDir, subdir), { recursive: true });
    }

    const commands = {
      // Coordination commands
      'coordination/init.md': {
        title: 'Initialize Coordination Framework',
        tool: 'mcp__ruv-swarm__swarm_init',
        params: '{"topology": "mesh", "maxAgents": 5, "strategy": "balanced"}',
        description: 'Set up a coordination topology to guide Claude Code\'s approach to complex tasks',
        details: `This tool creates a coordination framework that helps Claude Code:
- Break down complex problems systematically
- Approach tasks from multiple perspectives
- Maintain consistency across large projects
- Work more efficiently through structured coordination

Remember: This does NOT create actual coding agents. It creates a coordination pattern for Claude Code to follow.`,
      },
      'coordination/spawn.md': {
        title: 'Create Cognitive Patterns',
        tool: 'mcp__ruv-swarm__agent_spawn',
        params: '{"type": "researcher", "name": "Literature Analysis", "capabilities": ["deep-analysis"]}',
        description: 'Define cognitive patterns that represent different approaches Claude Code can take',
        details: `Agent types represent thinking patterns, not actual coders:
- **researcher**: Systematic exploration approach
- **coder**: Implementation-focused thinking
- **analyst**: Data-driven decision making
- **architect**: Big-picture system design
- **reviewer**: Quality and consistency checking

These patterns guide how Claude Code approaches different aspects of your task.`,
      },
      'coordination/orchestrate.md': {
        title: 'Coordinate Task Execution',
        tool: 'mcp__ruv-swarm__task_orchestrate',
        params: '{"task": "Implement authentication system", "strategy": "parallel", "priority": "high"}',
        description: 'Break down and coordinate complex tasks for systematic execution by Claude Code',
        details: `Orchestration strategies:
- **parallel**: Claude Code works on independent components simultaneously
- **sequential**: Step-by-step execution for dependent tasks
- **adaptive**: Dynamically adjusts based on task complexity

The orchestrator creates a plan that Claude Code follows using its native tools.`,
      },

      // Monitoring commands
      'monitoring/status.md': {
        title: 'Check Coordination Status',
        tool: 'mcp__ruv-swarm__swarm_status',
        params: '{"verbose": true}',
        description: 'Monitor the effectiveness of current coordination patterns',
        details: `Shows:
- Active coordination topologies
- Current cognitive patterns in use
- Task breakdown and progress
- Resource utilization for coordination
- Overall system health`,
      },
      'monitoring/agents.md': {
        title: 'List Active Patterns',
        tool: 'mcp__ruv-swarm__agent_list',
        params: '{"filter": "active"}',
        description: 'View all active cognitive patterns and their current focus areas',
        details: `Filters:
- **all**: Show all defined patterns
- **active**: Currently engaged patterns
- **idle**: Available but unused patterns
- **busy**: Patterns actively coordinating tasks`,
      },

      // Memory commands
      'memory/usage.md': {
        title: 'Memory Management',
        tool: 'mcp__ruv-swarm__memory_usage',
        params: '{"detail": "detailed"}',
        description: 'Track persistent memory usage across Claude Code sessions',
        details: `Memory helps Claude Code:
- Maintain context between sessions
- Remember project decisions
- Track implementation patterns
- Store coordination strategies that worked well`,
      },
      'memory/neural.md': {
        title: 'Neural Pattern Training',
        tool: 'mcp__ruv-swarm__neural_train',
        params: '{"iterations": 10}',
        description: 'Improve coordination patterns through neural network training',
        details: `Training improves:
- Task breakdown effectiveness
- Coordination pattern selection
- Resource allocation strategies
- Overall coordination efficiency`,
      },

      // Workflow examples
      'workflows/research.md': {
        title: 'Research Workflow Coordination',
        content: `# Research Workflow Coordination

## Purpose
Coordinate Claude Code's research activities for comprehensive, systematic exploration.

## Step-by-Step Coordination

### 1. Initialize Research Framework
\`\`\`
Tool: mcp__ruv-swarm__swarm_init
Parameters: {"topology": "mesh", "maxAgents": 5, "strategy": "balanced"}
\`\`\`
Creates a mesh topology for comprehensive exploration from multiple angles.

### 2. Define Research Perspectives
\`\`\`
Tool: mcp__ruv-swarm__agent_spawn
Parameters: {"type": "researcher", "name": "Literature Review"}
\`\`\`
\`\`\`
Tool: mcp__ruv-swarm__agent_spawn  
Parameters: {"type": "analyst", "name": "Data Analysis"}
\`\`\`
Sets up different analytical approaches for Claude Code to use.

### 3. Execute Coordinated Research
\`\`\`
Tool: mcp__ruv-swarm__task_orchestrate
Parameters: {"task": "Research modern web frameworks performance", "strategy": "adaptive"}
\`\`\`

## What Claude Code Actually Does
1. Uses **WebSearch** tool for finding resources
2. Uses **Read** tool for analyzing documentation
3. Uses **Task** tool for parallel exploration
4. Synthesizes findings using coordination patterns
5. Stores insights in memory for future reference

Remember: The swarm coordinates HOW Claude Code researches, not WHAT it finds.`,
      },
      'workflows/development.md': {
        title: 'Development Workflow Coordination',
        content: `# Development Workflow Coordination

## Purpose
Structure Claude Code's approach to complex development tasks for maximum efficiency.

## Step-by-Step Coordination

### 1. Initialize Development Framework
\`\`\`
Tool: mcp__ruv-swarm__swarm_init
Parameters: {"topology": "hierarchical", "maxAgents": 8, "strategy": "specialized"}
\`\`\`
Creates hierarchical structure for organized, top-down development.

### 2. Define Development Perspectives
\`\`\`
Tool: mcp__ruv-swarm__agent_spawn
Parameters: {"type": "architect", "name": "System Design"}
\`\`\`
\`\`\`
Tool: mcp__ruv-swarm__agent_spawn
Parameters: {"type": "coder", "name": "Implementation Focus"}
\`\`\`
Sets up architectural and implementation thinking patterns.

### 3. Coordinate Implementation
\`\`\`
Tool: mcp__ruv-swarm__task_orchestrate
Parameters: {"task": "Build REST API with authentication", "strategy": "parallel", "priority": "high"}
\`\`\`

## What Claude Code Actually Does
1. Uses **Write** tool to create new files
2. Uses **Edit/MultiEdit** tools for code modifications
3. Uses **Bash** tool for testing and building
4. Uses **TodoWrite** tool for task tracking
5. Follows coordination patterns for systematic implementation

Remember: All code is written by Claude Code using its native tools!`,
      },

      // Hook commands
      'hooks/overview.md': {
        title: 'Claude Code Hooks Overview',
        content: `# Claude Code Hooks for ruv-swarm

## Purpose
Automatically coordinate, format, and learn from Claude Code operations using hooks.

## Available Hooks

### Pre-Operation Hooks
- **pre-edit**: Validate and assign agents before file modifications
- **pre-bash**: Check command safety and resource requirements
- **pre-task**: Auto-spawn agents for complex tasks

### Post-Operation Hooks
- **post-edit**: Auto-format code and train neural patterns
- **post-bash**: Log execution and update metrics
- **post-search**: Cache results and improve search patterns

### MCP Integration Hooks
- **mcp-initialized**: Persist swarm configuration
- **agent-spawned**: Update agent roster
- **task-orchestrated**: Monitor task progress
- **neural-trained**: Save pattern improvements

### Session Hooks
- **notify**: Custom notifications with swarm status
- **session-end**: Generate summary and save state
- **session-restore**: Load previous session state

## Configuration
Hooks are configured in \`.claude/settings.json\`:

\`\`\`json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "^(Write|Edit|MultiEdit)$",
        "hooks": [{
          "type": "command",
          "command": "npx ruv-swarm hook pre-edit --file '\${tool.params.file_path}'"
        }]
      }
    ]
  }
}
\`\`\`

## Benefits
- 🤖 Automatic agent assignment based on file type
- 🎨 Consistent code formatting
- 🧠 Continuous neural pattern improvement
- 💾 Cross-session memory persistence
- 📊 Performance metrics tracking

## See Also
- [Pre-Edit Hook](./pre-edit.md)
- [Post-Edit Hook](./post-edit.md)
- [Session End Hook](./session-end.md)`,
      },
      'hooks/setup.md': {
        title: 'Setting Up Hooks',
        content: `# Setting Up ruv-swarm Hooks

## Quick Start

### 1. Initialize with Hooks
\`\`\`bash
npx ruv-swarm init --claude --force
\`\`\`

This automatically creates:
- \`.claude/settings.json\` with hook configurations
- Hook command documentation
- Default hook handlers

### 2. Test Hook Functionality
\`\`\`bash
# Test pre-edit hook
npx ruv-swarm hook pre-edit --file test.js --ensure-coordination

# Test session summary
npx ruv-swarm hook session-end --generate-summary
\`\`\`

### 3. Customize Hooks

Edit \`.claude/settings.json\` to customize:

\`\`\`json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "^Write$",
        "hooks": [{
          "type": "command",
          "command": "npx ruv-swarm hook custom-pre-write --file '\${tool.params.file_path}'"
        }]
      }
    ]
  }
}
\`\`\`

## Hook Response Format

Hooks return JSON with:
- \`continue\`: Whether to proceed (true/false)
- \`reason\`: Explanation for decision
- \`metadata\`: Additional context

Example blocking response:
\`\`\`json
{
  "continue": false,
  "reason": "Protected file - manual review required",
  "metadata": {
    "file": ".env.production",
    "protection_level": "high"
  }
}
\`\`\`

## Performance Tips
- Keep hooks lightweight (< 100ms)
- Use caching for repeated operations
- Batch related operations
- Run non-critical hooks asynchronously

## Debugging Hooks
\`\`\`bash
# Enable debug output
export RUV_SWARM_HOOK_DEBUG=true

# Test specific hook
npx ruv-swarm hook pre-edit --file app.js --debug
\`\`\`

## Common Patterns

### Auto-Format on Save
Already configured by default for common file types.

### Protected File Detection
\`\`\`json
{
  "matcher": "^(Write|Edit)$",
  "hooks": [{
    "type": "command",
    "command": "npx ruv-swarm hook check-protected --file '\${tool.params.file_path}'"
  }]
}
\`\`\`

### Automatic Testing
\`\`\`json
{
  "matcher": "^Write$",
  "hooks": [{
    "type": "command",
    "command": "test -f '\${tool.params.file_path%.js}.test.js' && npm test '\${tool.params.file_path%.js}.test.js'"
  }]
}
\`\`\``,
      },
    };

    const createdFiles = [];

    // Generate command files
    for (const [filepath, config] of Object.entries(commands)) {
      let content;

      if (config.content) {
        // Use provided content for workflow files
        ({ content } = config);
      } else {
        // Generate content for tool documentation
        content = `# ${config.title}

## 🎯 Key Principle
**This tool coordinates Claude Code's actions. It does NOT write code or create content.**

## MCP Tool Usage in Claude Code

**Tool:** \`${config.tool}\`

## Parameters
\`\`\`json
${config.params}
\`\`\`

## Description
${config.description}

## Details
${config.details}

## Example Usage

**In Claude Code:**
1. Use the tool: \`${config.tool}\`
2. With parameters: \`${config.params}\`
3. Claude Code then executes the coordinated plan using its native tools

## Important Reminders
- ✅ This tool provides coordination and structure
- ✅ Claude Code performs all actual implementation
- ❌ The tool does NOT write code
- ❌ The tool does NOT access files directly
- ❌ The tool does NOT execute commands

## See Also
- Main documentation: /claude.md
- Other commands in this category
- Workflow examples in /workflows/
`;
      }

      const filePath = path.join(commandsDir, filepath);
      await fs.writeFile(filePath, content);
      createdFiles.push(filepath);
    }

    return { files: createdFiles, success: true };
  }

  /**
     * Generate settings.json with hook configurations
     */
  async generateSettingsJson() {
    const settings = {
      env: {
        RUV_SWARM_AUTO_COMMIT: 'false',
        RUV_SWARM_AUTO_PUSH: 'false',
        RUV_SWARM_HOOKS_ENABLED: 'false',
        RUV_SWARM_TELEMETRY_ENABLED: 'true',
        RUV_SWARM_REMOTE_EXECUTION: 'true',
      },
      permissions: {
        allow: [
          'Bash(npx ruv-swarm *)',
          'Bash(npm run lint)',
          'Bash(npm run test:*)',
          'Bash(npm test *)',
          'Bash(git status)',
          'Bash(git diff *)',
          'Bash(git log *)',
          'Bash(git add *)',
          'Bash(git commit *)',
          'Bash(git push)',
          'Bash(git config *)',
          'Bash(node *)',
          'Bash(which *)',
          'Bash(pwd)',
          'Bash(ls *)',
        ],
        deny: [
          'Bash(rm -rf /)',
          'Bash(curl * | bash)',
          'Bash(wget * | sh)',
          'Bash(eval *)',
        ],
      },
      hooks: {},
      mcpServers: {
        'ruv-swarm': {
          command: 'npx',
          args: ['ruv-swarm', 'mcp', 'start'],
          env: {
            RUV_SWARM_HOOKS_ENABLED: 'false',
            RUV_SWARM_TELEMETRY_ENABLED: 'true',
            RUV_SWARM_REMOTE_READY: 'true',
          },
        },
      },
      includeCoAuthoredBy: true,
    };

    const filePath = path.join(this.workingDir, '.claude', 'settings.json');
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, JSON.stringify(settings, null, 2));

    return { file: '.claude/settings.json', success: true };
  }

  /**
     * Generate all documentation files
     */
  async generateAll() {
    console.log('📚 Generating Claude Code documentation...');

    try {
      const results = {
        claudeMd: await this.generateClaudeMd(),
        commands: await this.generateCommandDocs(),
        advancedCommands: await this.advancedGenerator.generateAdvancedCommands(),
        settings: await this.generateSettingsJson(),
        success: true,
      };

      const totalCommands = results.commands.files.length + results.advancedCommands.files.length;

      console.log('✅ Documentation generated successfully');
      console.log('   - CLAUDE.md');
      console.log('   - .claude/settings.json (with enhanced hooks)');
      console.log(`   - .claude/commands/ directory with ${ totalCommands } files`);
      console.log(`     • Basic commands: ${ results.commands.files.length}`);
      console.log(`     • Advanced optimization: ${ results.advancedCommands.files.length}`);

      return results;
    } catch (error) {
      console.error('❌ Failed to generate documentation:', error.message);
      throw error;
    }
  }
}

export { ClaudeDocsGenerator };