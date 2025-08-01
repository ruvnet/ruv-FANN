# ruv-swarm User Guide

Welcome to ruv-swarm - a cognitive orchestration framework that enhances Claude Code with intelligent task coordination, persistent memory, and neural learning capabilities.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Examples](#examples)

## Getting Started

### Installation

Install ruv-swarm globally or as a project dependency:

```bash
# Global installation (recommended for Claude Code)
npm install -g ruv-swarm

# Or as a project dependency
npm install ruv-swarm --save-dev
```

### First Swarm

Create your first swarm in just a few steps:

```bash
# Initialize a swarm with default settings
npx ruv-swarm swarm init

# Or specify topology and strategy
npx ruv-swarm swarm init --topology mesh --strategy balanced
```

### Quick Start Example

```bash
# 1. Initialize a swarm
npx ruv-swarm swarm init --topology mesh

# 2. Spawn your first agent
npx ruv-swarm agent spawn --type researcher --name "Code Analyzer"

# 3. Execute a task
npx ruv-swarm task execute --description "Analyze project structure" --strategy parallel

# 4. Check results
npx ruv-swarm task results
```

## Core Concepts

### Agents

Agents are cognitive patterns that guide how Claude Code approaches tasks. They don't write code themselves - they coordinate Claude Code's actions.

**Agent Types:**
- **Researcher**: Systematic exploration and analysis
- **Developer**: Implementation-focused coordination
- **Reviewer**: Quality assurance and validation
- **Architect**: System design and structure
- **Analyst**: Data processing and insights
- **Optimizer**: Performance improvements
- **Tester**: Test coverage and validation
- **Documenter**: Documentation generation

### Coordination

ruv-swarm provides four coordination topologies:

1. **Mesh**: All agents can communicate with each other
   - Best for: Complex, interconnected tasks
   - Example: Full-stack feature implementation

2. **Hierarchical**: Tree-like structure with lead agents
   - Best for: Large projects with clear delegation
   - Example: Multi-module system development

3. **Ring**: Agents communicate in a circular pattern
   - Best for: Sequential processing tasks
   - Example: Data pipeline implementation

4. **Star**: Central coordinator with satellite agents
   - Best for: Centralized decision-making
   - Example: API development with single source of truth

### Memory

Persistent memory allows continuity across sessions:

```bash
# Store important context
npx ruv-swarm memory set --key "project/architecture" --value "microservices with REST APIs"

# Retrieve memory
npx ruv-swarm memory get --key "project/architecture"

# List all memory keys
npx ruv-swarm memory list
```

## Basic Usage

### Spawning Agents

Create agents for different aspects of your task:

```bash
# Spawn a single agent
npx ruv-swarm agent spawn --type developer --name "Backend Dev"

# Spawn multiple agents for a complex task
npx ruv-swarm agent spawn --type architect --name "System Designer"
npx ruv-swarm agent spawn --type developer --name "API Builder"
npx ruv-swarm agent spawn --type tester --name "Test Writer"

# List active agents
npx ruv-swarm agent list
```

### Executing Tasks

Tasks are broken down and coordinated across agents:

```bash
# Simple task execution
npx ruv-swarm task execute --description "Implement user authentication"

# Task with specific strategy
npx ruv-swarm task execute \
  --description "Build REST API with CRUD operations" \
  --strategy parallel \
  --timeout 3600000

# Check task status
npx ruv-swarm task status

# Get detailed results
npx ruv-swarm task results --format detailed
```

### Monitoring Progress

Keep track of swarm activity:

```bash
# Real-time monitoring
npx ruv-swarm swarm monitor

# Check swarm status
npx ruv-swarm swarm status

# View agent metrics
npx ruv-swarm agent metrics
```

## Advanced Features

### Parallel Execution Patterns

Optimize task execution with parallel strategies:

```javascript
// Example: Parallel file processing
const parallelConfig = {
  strategy: "parallel",
  maxConcurrency: 4,
  taskDistribution: "round-robin"
};

// Execute with parallel configuration
npx ruv-swarm task execute \
  --description "Process all test files" \
  --strategy parallel \
  --config ./parallel-config.json
```

**Parallel Patterns:**
1. **Map-Reduce**: Split work, process in parallel, combine results
2. **Pipeline**: Sequential stages with parallel processing within each
3. **Scatter-Gather**: Distribute subtasks, collect results
4. **Work Stealing**: Dynamic load balancing

### Hook Configuration

Hooks provide lifecycle integration:

```bash
# Pre-task hook - runs before task execution
npx ruv-swarm hook pre-task --description "Setup environment"

# Post-edit hook - runs after file modifications
npx ruv-swarm hook post-edit --file "./src/api.js" --memory-key "api/changes"

# Post-task hook - runs after task completion
npx ruv-swarm hook post-task --task-id "api-build" --analyze-performance true
```

**Available Hooks:**
- `pre-task`: Task initialization and setup
- `post-edit`: File modification tracking
- `post-task`: Cleanup and analysis
- `session-restore`: Load previous session state
- `performance-analyze`: Detailed performance metrics

### Git Integration

Seamless version control integration:

```bash
# Analyze repository before changes
npx ruv-swarm hook git-analyze

# Track changes with git awareness
npx ruv-swarm task execute \
  --description "Refactor authentication module" \
  --git-aware true

# Commit with swarm metadata
npx ruv-swarm hook git-commit \
  --message "feat: Add JWT authentication" \
  --include-metrics true
```

### Neural Training

Improve coordination patterns over time:

```bash
# Check neural network status
npx ruv-swarm neural status

# Train on successful patterns
npx ruv-swarm neural train --model attention --iterations 100

# View learned patterns
npx ruv-swarm neural patterns --model attention

# Export neural weights
npx ruv-swarm neural export --model all --output ./neural-weights.json
```

**Trainable Models:**
- Attention mechanisms for focus
- LSTM for sequence learning
- Transformer for complex patterns
- Custom models for specific workflows

### Performance Optimization

Monitor and improve swarm efficiency:

```bash
# Run comprehensive benchmark
npx ruv-swarm benchmark run --iterations 10

# Analyze performance bottlenecks
npx ruv-swarm performance analyze --task-id "recent"

# Optimize swarm configuration
npx ruv-swarm swarm optimize --target "speed"

# View optimization suggestions
npx ruv-swarm performance suggest
```

**Optimization Targets:**
- **Speed**: Minimize execution time
- **Tokens**: Reduce token usage
- **Quality**: Maximize output quality
- **Balanced**: Optimal trade-offs

## Best Practices

### 1. Choose the Right Topology

```bash
# For interconnected features
npx ruv-swarm swarm init --topology mesh

# For large, structured projects
npx ruv-swarm swarm init --topology hierarchical

# For sequential workflows
npx ruv-swarm swarm init --topology ring

# For centralized coordination
npx ruv-swarm swarm init --topology star
```

### 2. Use Memory Effectively

```bash
# Store project context
npx ruv-swarm memory set --key "project/stack" --value "React, Node.js, PostgreSQL"

# Store architectural decisions
npx ruv-swarm memory set --key "decisions/auth" --value "JWT with refresh tokens"

# Create memory snapshots
npx ruv-swarm memory snapshot --name "v1.0-release"
```

### 3. Leverage Parallel Execution

```bash
# For independent tasks
npx ruv-swarm task execute \
  --description "Write unit tests for all modules" \
  --strategy parallel \
  --max-agents 4

# For dependent tasks
npx ruv-swarm task execute \
  --description "Build and deploy microservices" \
  --strategy pipeline
```

### 4. Monitor and Optimize

```bash
# Regular monitoring
npx ruv-swarm swarm monitor --interval 5000

# Performance tracking
npx ruv-swarm hook post-task --analyze-performance true

# Continuous improvement
npx ruv-swarm neural train --auto true
```

### 5. Use Hooks for Automation

```bash
# Automate setup
echo 'npx ruv-swarm hook pre-task --setup-env true' >> .git/hooks/pre-commit

# Track all changes
echo 'npx ruv-swarm hook post-edit --auto-memory true' >> .ruv-swarm/hooks/post-edit

# Analyze after completion
echo 'npx ruv-swarm hook post-task --generate-report true' >> .ruv-swarm/hooks/post-task
```

## Troubleshooting

### Common Issues

#### Swarm Not Initializing

```bash
# Check if swarm exists
npx ruv-swarm swarm status

# Reset if needed
npx ruv-swarm swarm reset

# Reinitialize with verbose logging
npx ruv-swarm swarm init --verbose
```

#### Agent Communication Failures

```bash
# Check agent health
npx ruv-swarm agent health

# Restart specific agent
npx ruv-swarm agent restart --name "Backend Dev"

# Clear agent queue
npx ruv-swarm agent clear-queue
```

#### Memory Issues

```bash
# Check memory usage
npx ruv-swarm memory usage

# Clean old entries
npx ruv-swarm memory clean --older-than 30d

# Export before cleanup
npx ruv-swarm memory export --output ./memory-backup.json
```

#### Performance Degradation

```bash
# Run diagnostics
npx ruv-swarm diagnose

# Check resource usage
npx ruv-swarm resource status

# Optimize configuration
npx ruv-swarm config optimize
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Set debug mode
export RUV_SWARM_DEBUG=true

# Or use debug flag
npx ruv-swarm --debug task execute --description "Debug task"

# View debug logs
npx ruv-swarm logs --level debug --tail 100
```

## Examples

### Example 1: Building a REST API

```bash
# Initialize swarm for API development
npx ruv-swarm swarm init --topology hierarchical --strategy specialized

# Spawn specialized agents
npx ruv-swarm agent spawn --type architect --name "API Designer"
npx ruv-swarm agent spawn --type developer --name "Endpoint Builder"
npx ruv-swarm agent spawn --type tester --name "API Tester"
npx ruv-swarm agent spawn --type documenter --name "API Docs"

# Execute the task
npx ruv-swarm task execute \
  --description "Build REST API for user management with CRUD operations" \
  --strategy pipeline

# Monitor progress
npx ruv-swarm swarm monitor
```

### Example 2: Refactoring Legacy Code

```bash
# Setup refactoring swarm
npx ruv-swarm swarm init --topology mesh --strategy quality

# Spawn analysis agents
npx ruv-swarm agent spawn --type analyst --name "Code Analyzer"
npx ruv-swarm agent spawn --type architect --name "Refactor Planner"
npx ruv-swarm agent spawn --type developer --name "Refactor Implementer"
npx ruv-swarm agent spawn --type reviewer --name "Quality Checker"

# Analyze before refactoring
npx ruv-swarm hook pre-task --analyze-codebase true

# Execute refactoring
npx ruv-swarm task execute \
  --description "Refactor authentication module to use modern patterns" \
  --git-aware true \
  --safety-checks true
```

### Example 3: Full-Stack Feature Implementation

```bash
# Initialize full-stack swarm
npx ruv-swarm swarm init --topology mesh --max-agents 8

# Spawn full-stack team
npx ruv-swarm agent spawn --type architect --name "System Architect"
npx ruv-swarm agent spawn --type developer --name "Frontend Dev" --specialization "React"
npx ruv-swarm agent spawn --type developer --name "Backend Dev" --specialization "Node.js"
npx ruv-swarm agent spawn --type developer --name "Database Dev" --specialization "PostgreSQL"
npx ruv-swarm agent spawn --type tester --name "Integration Tester"
npx ruv-swarm agent spawn --type reviewer --name "Code Reviewer"

# Store feature requirements
npx ruv-swarm memory set --key "feature/requirements" \
  --value "User dashboard with real-time updates, charts, and notifications"

# Execute with parallel strategy
npx ruv-swarm task execute \
  --description "Implement user dashboard feature" \
  --strategy parallel \
  --coordination-mode "tight"

# Generate comprehensive report
npx ruv-swarm hook post-task --generate-report true --include-metrics true
```

### Example 4: Performance Optimization

```bash
# Setup optimization swarm
npx ruv-swarm swarm init --topology star --strategy performance

# Spawn optimization specialists
npx ruv-swarm agent spawn --type optimizer --name "Performance Analyzer"
npx ruv-swarm agent spawn --type developer --name "Optimization Implementer"
npx ruv-swarm agent spawn --type tester --name "Benchmark Runner"

# Run baseline benchmarks
npx ruv-swarm benchmark run --save-baseline true

# Execute optimization task
npx ruv-swarm task execute \
  --description "Optimize API response times and reduce memory usage" \
  --strategy adaptive \
  --target-metrics "response_time<100ms,memory<512MB"

# Compare results
npx ruv-swarm benchmark compare --baseline latest
```

### Example 5: Automated Testing Suite

```bash
# Initialize testing swarm
npx ruv-swarm swarm init --topology ring --strategy thorough

# Spawn testing agents
npx ruv-swarm agent spawn --type tester --name "Unit Tester"
npx ruv-swarm agent spawn --type tester --name "Integration Tester"
npx ruv-swarm agent spawn --type tester --name "E2E Tester"
npx ruv-swarm agent spawn --type analyst --name "Coverage Analyzer"

# Configure test strategy
cat > test-config.json << EOF
{
  "coverage": {
    "target": 80,
    "include": ["src/**/*.js"],
    "exclude": ["src/**/*.test.js"]
  },
  "parallel": true,
  "maxWorkers": 4
}
EOF

# Execute test creation
npx ruv-swarm task execute \
  --description "Create comprehensive test suite with 80% coverage" \
  --config ./test-config.json \
  --strategy parallel

# Monitor test execution
npx ruv-swarm task monitor --real-time true
```

## Conclusion

ruv-swarm transforms how Claude Code approaches complex development tasks by providing intelligent coordination, persistent memory, and adaptive learning. Start with simple swarms and gradually explore advanced features as your projects grow.

Remember: **ruv-swarm coordinates, Claude Code creates!**

For more information:
- GitHub: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm
- Issues: https://github.com/ruvnet/ruv-FANN/issues
- Examples: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm/examples