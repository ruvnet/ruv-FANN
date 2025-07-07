# Standard Swarm Configurations for Testing

## Overview
This document defines the standard swarm configurations used for benchmarking against Claude Native baseline. Each configuration tests different aspects of multi-agent collaboration.

## Configuration Naming Convention
- **Config A**: 3 agents, flat topology
- **Config B**: 3 agents, hierarchical topology  
- **Config C**: 5 agents, dynamic/adaptive strategy
- **Config D**: 10 agents, stress test (optional for simple tests)

---

## 🔷 Configuration A: Simple Parallel (3 Agents, Flat)

### Purpose
Test basic parallel processing with minimal coordination overhead.

### Setup Command
```javascript
[BatchTool - Single Message]:
  mcp__ruv-swarm__swarm_init { 
    topology: "mesh", 
    maxAgents: 3, 
    strategy: "balanced" 
  }
  mcp__ruv-swarm__agent_spawn { 
    type: "coder", 
    name: "implementation-specialist",
    capabilities: ["coding", "algorithms", "data-structures"]
  }
  mcp__ruv-swarm__agent_spawn { 
    type: "tester", 
    name: "quality-assurance",
    capabilities: ["testing", "validation", "edge-cases"]
  }
  mcp__ruv-swarm__agent_spawn { 
    type: "analyst", 
    name: "code-reviewer",
    capabilities: ["review", "optimization", "documentation"]
  }
```

### Expected Behavior
- All agents work simultaneously on their specialties
- Minimal coordination needed
- Fast execution with some integration overhead
- Good for tasks with clear separation of concerns

### Task Distribution Pattern
```
Task Input → [Parallel Processing]
           ├── Coder: Implements solution
           ├── Tester: Creates tests
           └── Analyst: Reviews and documents
           → Integration → Final Output
```

---

## 🔷 Configuration B: Hierarchical (3 Agents)

### Purpose
Test structured delegation and coordination patterns.

### Setup Command
```javascript
[BatchTool - Single Message]:
  mcp__ruv-swarm__swarm_init { 
    topology: "hierarchical", 
    maxAgents: 3, 
    strategy: "specialized" 
  }
  mcp__ruv-swarm__agent_spawn { 
    type: "coordinator", 
    name: "lead-architect",
    capabilities: ["planning", "delegation", "integration"]
  }
  mcp__ruv-swarm__agent_spawn { 
    type: "coder", 
    name: "implementation-expert",
    capabilities: ["coding", "algorithms", "optimization"]
  }
  mcp__ruv-swarm__agent_spawn { 
    type: "analyst", 
    name: "quality-specialist",
    capabilities: ["testing", "validation", "documentation"]
  }
```

### Expected Behavior
- Coordinator analyzes task and creates plan
- Delegates specific subtasks to specialists
- Integrates results into cohesive solution
- More structured but potentially slower

### Task Distribution Pattern
```
Task Input → Coordinator (Analysis & Planning)
           ├── Delegates → Coder (Implementation)
           ├── Delegates → Analyst (Quality & Docs)
           └── Integrates all results
           → Final Output
```

---

## 🔷 Configuration C: Dynamic Team (5 Agents)

### Purpose
Test adaptive collaboration with specialized expertise.

### Setup Command
```javascript
[BatchTool - Single Message]:
  mcp__ruv-swarm__swarm_init { 
    topology: "mesh", 
    maxAgents: 5, 
    strategy: "adaptive" 
  }
  mcp__ruv-swarm__agent_spawn { 
    type: "researcher", 
    name: "problem-analyzer",
    capabilities: ["analysis", "research", "requirements"]
  }
  mcp__ruv-swarm__agent_spawn { 
    type: "coder", 
    name: "core-developer",
    capabilities: ["implementation", "algorithms", "core-logic"]
  }
  mcp__ruv-swarm__agent_spawn { 
    type: "coder", 
    name: "edge-case-handler",
    capabilities: ["error-handling", "validation", "edge-cases"]
  }
  mcp__ruv-swarm__agent_spawn { 
    type: "optimizer", 
    name: "performance-tuner",
    capabilities: ["optimization", "efficiency", "refactoring"]
  }
  mcp__ruv-swarm__agent_spawn { 
    type: "tester", 
    name: "test-engineer",
    capabilities: ["testing", "validation", "coverage"]
  }
```

### Expected Behavior
- Agents dynamically coordinate based on task complexity
- More specialized handling of different aspects
- Potential for higher quality but more coordination
- Self-organizing based on task requirements

### Task Distribution Pattern
```
Task Input → [Dynamic Analysis]
           ├── Researcher: Requirements analysis
           ├── Core Dev: Main implementation
           ├── Edge Handler: Robustness
           ├── Optimizer: Performance
           └── Tester: Validation
           → Adaptive Integration → Final Output
```

---

## 🔷 Configuration D: Full Swarm (10 Agents) - Optional

### Purpose
Stress test to find diminishing returns point.

### Setup Command
```javascript
[BatchTool - Single Message]:
  mcp__ruv-swarm__swarm_init { 
    topology: "star", 
    maxAgents: 10, 
    strategy: "balanced" 
  }
  // Central coordinator
  mcp__ruv-swarm__agent_spawn { type: "coordinator", name: "central-hub" }
  
  // Specialized teams
  mcp__ruv-swarm__agent_spawn { type: "researcher", name: "requirements-analyst" }
  mcp__ruv-swarm__agent_spawn { type: "researcher", name: "solution-researcher" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "primary-developer" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "secondary-developer" }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "code-analyst" }
  mcp__ruv-swarm__agent_spawn { type: "optimizer", name: "optimization-expert" }
  mcp__ruv-swarm__agent_spawn { type: "tester", name: "unit-tester" }
  mcp__ruv-swarm__agent_spawn { type: "tester", name: "integration-tester" }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "documentation-writer" }
```

### Expected Behavior
- High coordination overhead
- Potential for very thorough solutions
- Risk of diminishing returns
- Tests scalability limits

---

## Testing Matrix

| Test Type | Config A | Config B | Config C | Config D |
|-----------|----------|----------|----------|----------|
| Simple Tests | ✅ | ✅ | ✅ | Optional |
| Moderate Tests | ✅ | ✅ | ✅ | ✅ |
| High Tests | ✅ | ✅ | ✅ | ✅ |

## Key Metrics to Compare

### 1. **Execution Time**
- Baseline: ~47.5 seconds average
- Target: Identify if parallel execution provides speedup

### 2. **Quality Score**
- Baseline: 9.69/10 average
- Target: Maintain or improve quality

### 3. **Token Efficiency**
- Baseline: ~3160 total tokens
- Formula: Quality Score / Total Tokens Used

### 4. **Coordination Overhead**
- Measure: Time spent in coordination vs actual work
- Identify: Optimal agent count for different task types

### 5. **Consensus/Divergence**
- Track: How often agents agree on approach
- Measure: Integration complexity

## Implementation Notes

### Critical: Parallel Execution
**ALL configurations MUST use BatchTool** to spawn agents in a single message:

```javascript
// ✅ CORRECT - Single Message
[BatchTool]:
  mcp__ruv-swarm__swarm_init { ... }
  mcp__ruv-swarm__agent_spawn { ... }
  mcp__ruv-swarm__agent_spawn { ... }
  mcp__ruv-swarm__agent_spawn { ... }
  mcp__ruv-swarm__task_orchestrate { ... }

// ❌ WRONG - Multiple Messages
Message 1: swarm_init
Message 2: agent_spawn
Message 3: agent_spawn
// This defeats the purpose of parallel execution!
```

### Memory & Persistence
Each configuration should utilize:
```javascript
mcp__ruv-swarm__memory_usage {
  action: "store",
  key: "test_results/config_X/test_Y",
  value: { results, metrics, observations }
}
```

### Monitoring
During execution, use:
```javascript
mcp__ruv-swarm__swarm_monitor {
  duration: 30,
  interval: 5
}
```

## Expected Outcomes by Configuration

### Config A (3 Flat)
- **Pros**: Fast, simple coordination
- **Cons**: Limited specialization
- **Best for**: Simple tasks with clear components

### Config B (3 Hierarchical)
- **Pros**: Structured approach, clear delegation
- **Cons**: Coordination overhead
- **Best for**: Tasks requiring planning

### Config C (5 Dynamic)
- **Pros**: Specialized expertise, adaptive
- **Cons**: More complex integration
- **Best for**: Moderate to complex tasks

### Config D (10 Stress)
- **Pros**: Extreme thoroughness
- **Cons**: High overhead, diminishing returns
- **Best for**: Testing scalability limits

## Success Criteria

A swarm configuration is considered successful if it:
1. Maintains quality score ≥ baseline (9.69/10)
2. Shows meaningful speedup OR quality improvement
3. Token efficiency within 150% of baseline
4. Successfully integrates all agent outputs
5. No critical coordination failures