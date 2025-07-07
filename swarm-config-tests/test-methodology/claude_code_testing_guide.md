# Claude Code Testing Guide for ruv-swarm

## 🚀 Quick Reference: Essential Commands & Best Practices

This guide provides comprehensive instructions for Claude Code when running tests and benchmarks for the ruv-swarm system.

## 📋 Test Objectives (from test_purpose.md)

Compare performance between:
1. **Claude Native** (baseline)
2. **Swarm Configurations**:
   - Different agent counts (1, 3, 5, 10+)
   - Different architectures (Flat, Hierarchical, Dynamic)

### Key Metrics to Measure:
- **Accuracy/Correctness**
- **Coherence** 
- **Latency**
- **Token Efficiency**
- **Consensus Divergence**

## 🔴 CRITICAL: Parallel Execution is MANDATORY

### ⚡ The #1 Rule: BATCH EVERYTHING

**NEVER** send multiple messages for related operations. **ALWAYS** combine multiple tool calls in ONE message.

### ✅ CORRECT Pattern (Everything in ONE Message):
```javascript
[Single Message with Multiple Tools]:
  mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 5 }
  mcp__ruv-swarm__agent_spawn { type: "researcher" }
  mcp__ruv-swarm__agent_spawn { type: "coder" }
  mcp__ruv-swarm__agent_spawn { type: "analyst" }
  mcp__ruv-swarm__task_orchestrate { task: "Analyze code", strategy: "parallel" }
  TodoWrite { todos: [todo1, todo2, todo3] }
  Bash "mkdir -p test-results/{native,swarm-3,swarm-5,swarm-10}"
  Write "test-results/config.json"
```

### ❌ WRONG Pattern (Sequential - NEVER DO THIS):
```javascript
Message 1: mcp__ruv-swarm__swarm_init
Message 2: mcp__ruv-swarm__agent_spawn 
Message 3: mcp__ruv-swarm__agent_spawn
// This is 3x slower!
```

## 🧪 Test Implementation Strategy

### 1. Setup Test Environment (Single Batch)
```bash
# Create all directories and files in ONE message
[BatchTool]:
  Bash "mkdir -p test-results/{native,swarm-{1,3,5,10}}/{coding,math,research}"
  Bash "mkdir -p test-results/benchmarks/{performance,tokens,accuracy}"
  Write "test-results/test-config.json" { test configuration }
  Write "test-results/test-prompts.json" { test prompts }
  TodoWrite { todos: [setup tasks] }
```

### 2. Initialize All Test Configurations (Parallel)
```javascript
[BatchTool]:
  // Initialize all swarms at once
  mcp__ruv-swarm__swarm_init { topology: "flat", maxAgents: 3, strategy: "balanced" }
  mcp__ruv-swarm__swarm_init { topology: "hierarchical", maxAgents: 3 }
  mcp__ruv-swarm__swarm_init { topology: "dynamic", maxAgents: 5 }
  mcp__ruv-swarm__swarm_init { topology: "flat", maxAgents: 10 }
  
  // Spawn all agents for all swarms
  mcp__ruv-swarm__agent_spawn { type: "researcher", name: "R1" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "C1" }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "A1" }
  // ... repeat for all configurations
```

### 3. Run Tests in Parallel
```javascript
[BatchTool]:
  // Execute all test tasks simultaneously
  mcp__ruv-swarm__task_orchestrate { 
    task: "Generate Python function for binary search",
    strategy: "parallel",
    maxAgents: 3 
  }
  mcp__ruv-swarm__task_orchestrate {
    task: "Solve integral of x^2 from 0 to 1",
    strategy: "adaptive",
    maxAgents: 5
  }
  mcp__ruv-swarm__task_orchestrate {
    task: "Research best practices for REST API design",
    strategy: "hierarchical",
    maxAgents: 10
  }
  
  // Monitor all swarms
  mcp__ruv-swarm__swarm_monitor { duration: 30, interval: 5 }
```

## 📊 Test Matrix Implementation

### Test Configuration File Structure:
```json
{
  "test_configs": [
    {
      "name": "Claude Native",
      "mode": "native",
      "agents": 1,
      "architecture": null
    },
    {
      "name": "Swarm Config A",
      "mode": "swarm",
      "agents": 3,
      "architecture": "flat",
      "topology": "mesh"
    },
    {
      "name": "Swarm Config B", 
      "mode": "swarm",
      "agents": 3,
      "architecture": "hierarchical",
      "topology": "hierarchical"
    },
    {
      "name": "Swarm Config C",
      "mode": "swarm", 
      "agents": 5,
      "architecture": "dynamic",
      "topology": "mesh"
    },
    {
      "name": "Swarm Config D",
      "mode": "swarm",
      "agents": 10,
      "architecture": "flat",
      "topology": "star"
    }
  ]
}
```

### Test Prompts Structure:
```json
{
  "task_types": {
    "coding": [
      "Write a Python function to implement quicksort",
      "Debug this JavaScript code that calculates Fibonacci numbers",
      "Create a REST API endpoint for user authentication"
    ],
    "math": [
      "Solve the differential equation dy/dx = 2x",
      "Calculate the eigenvalues of a 3x3 matrix",
      "Prove that sqrt(2) is irrational"
    ],
    "research": [
      "Compare microservices vs monolithic architectures",
      "Summarize latest developments in quantum computing",
      "Analyze pros and cons of different ML frameworks"
    ]
  }
}
```

## 🛠️ Specific Commands for Testing

### 1. Initialize Test Environment
```bash
# Single batch command to set up everything
npx ruv-swarm test init --configs all --parallel
```

### 2. Run Benchmarks
```bash
# Run comprehensive benchmarks in parallel
npx ruv-swarm benchmark run --suite complete --iterations 50 --parallel
```

### 3. Collect Metrics
```bash
# Gather all metrics in one operation
[BatchTool]:
  mcp__ruv-swarm__agent_metrics { metric: "all" }
  mcp__ruv-swarm__memory_usage { detail: "by-agent" }
  mcp__ruv-swarm__task_results { taskId: "all", format: "detailed" }
  mcp__ruv-swarm__benchmark_run { type: "all", iterations: 20 }
```

### 4. Generate Reports
```bash
# Create all reports simultaneously
[BatchTool]:
  Write "test-results/performance-report.md"
  Write "test-results/accuracy-metrics.json"
  Write "test-results/token-usage.csv"
  Write "test-results/consensus-analysis.md"
```

## 📈 Performance Optimization Tips

### 1. Use SIMD Acceleration
```bash
export RUV_SWARM_USE_SIMD=true
```

### 2. Enable Neural Network Caching
```javascript
mcp__ruv-swarm__neural_train { 
  iterations: 100,
  cache: true,
  persistence: true 
}
```

### 3. Optimize Memory Usage
```javascript
// Monitor and adjust in real-time
mcp__ruv-swarm__memory_usage { detail: "by-agent" }
// Then adjust agent count based on results
```

## 🎯 Best Practices for Claude Code

### 1. **Always Use BatchTool**
- Combine all related operations in a single message
- Never wait for one operation before starting another
- Think "what can run simultaneously?"

### 2. **Parallel Test Execution**
```javascript
// Run all test variations at once
[BatchTool]:
  // Native baseline
  Task { prompt: "Native: Solve problem X" }
  
  // All swarm configs simultaneously  
  mcp__ruv-swarm__task_orchestrate { config: "A", task: "Solve problem X" }
  mcp__ruv-swarm__task_orchestrate { config: "B", task: "Solve problem X" }
  mcp__ruv-swarm__task_orchestrate { config: "C", task: "Solve problem X" }
  mcp__ruv-swarm__task_orchestrate { config: "D", task: "Solve problem X" }
```

### 3. **Efficient Data Collection**
```javascript
// Collect all metrics in one sweep
[BatchTool]:
  Read "test-results/*/output.json"
  Grep "pattern: 'execution_time|tokens_used|accuracy'" "test-results"
  mcp__ruv-swarm__swarm_status { verbose: true }
  mcp__ruv-swarm__agent_list { filter: "all" }
```

### 4. **Real-time Monitoring**
```javascript
// Set up continuous monitoring
mcp__ruv-swarm__swarm_monitor { 
  duration: 300,  // 5 minutes
  interval: 10,   // Update every 10 seconds
  metrics: ["latency", "tokens", "accuracy", "consensus"]
}
```

## 🔍 Debugging & Troubleshooting

### Enable Debug Mode
```bash
export RUV_SWARM_DEBUG=true
export RUV_SWARM_LOG_LEVEL=trace
```

### Check System Status
```javascript
[BatchTool]:
  mcp__ruv-swarm__features_detect { category: "all" }
  mcp__ruv-swarm__swarm_status { verbose: true }
  Bash "npx ruv-swarm doctor --check all"
```

### Common Issues & Solutions

1. **Agent Spawn Failures**
   - Check swarm capacity
   - Verify topology supports requested agent count
   - Monitor memory usage

2. **Task Timeouts**
   - Increase timeout: `--timeout 60000`
   - Check agent availability
   - Verify task complexity matches agent count

3. **Memory Issues**
   - Reduce agent count
   - Enable memory optimization flags
   - Use hierarchical topology for better resource distribution

## 📊 Expected Performance Baselines

Based on ruv-swarm benchmarks:
- **SWE-Bench Solve Rate**: 84.8%
- **Token Reduction**: 32.3%
- **Speed Improvement**: 2.8-4.4x
- **Coordination Accuracy**: 99.5%

Your tests should aim to validate these metrics across different configurations.

## 🚀 Quick Start Checklist

1. ✅ Initialize swarms with parallel BatchTool
2. ✅ Spawn all agents in ONE message
3. ✅ Execute all test tasks simultaneously
4. ✅ Monitor performance in real-time
5. ✅ Collect metrics in parallel batches
6. ✅ Generate reports concurrently
7. ✅ Never use sequential operations

## 📝 Sample Test Run Script

```javascript
// Complete test run in minimal messages
[Message 1 - Setup]:
  TodoWrite { todos: test plan }
  Bash "mkdir -p test-results/{all directories}"
  Write all configuration files
  
[Message 2 - Initialize]:
  mcp__ruv-swarm__swarm_init (all configs)
  mcp__ruv-swarm__agent_spawn (all agents)
  
[Message 3 - Execute]:
  mcp__ruv-swarm__task_orchestrate (all test tasks)
  mcp__ruv-swarm__swarm_monitor
  
[Message 4 - Collect]:
  mcp__ruv-swarm__task_results (all results)
  mcp__ruv-swarm__agent_metrics
  Write all reports
```

Remember: **Parallel execution is not optional - it's mandatory for accurate benchmarking!**