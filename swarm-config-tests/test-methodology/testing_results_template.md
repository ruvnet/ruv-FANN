# Testing Results Template

## Overview
This template provides a standardized format for recording and comparing test results across different configurations (Claude Native vs various Swarm configurations). Use this template for each test to ensure consistent measurement and reporting.

## Test Information
- **Test Name**: [Test 1/2/3/4 - Name]
- **Test Date**: [YYYY-MM-DD HH:MM:SS]
- **Tester**: [Name/ID]
- **Environment**: [System specs, Claude version, ruv-swarm version]

## Configuration Results

### Configuration: Claude Native (Baseline)
**Setup Details**:
- Agent Count: 1
- Architecture: N/A
- Execution Mode: Direct prompt

**Metrics**:
| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Total Execution Time | | seconds | From prompt to complete response |
| First Token Time | | seconds | Time to first output token |
| Total Input Tokens | | tokens | |
| Total Output Tokens | | tokens | |
| Token Efficiency | | ratio | Output quality per token |
| Memory Usage | | MB | Peak memory consumption |

**Quality Scores** (0-10 scale):
- Accuracy/Correctness: [ ] / 10
- Coherence: [ ] / 10
- Completeness: [ ] / 10
- Code Quality: [ ] / 10
- Documentation: [ ] / 10

**Detailed Results**:
```
[Paste or summarize the actual output]
```

**Notes & Observations**:
- 
- 

---

### Configuration: Swarm Config A (3 agents, flat)
**Setup Details**:
- Agent Count: 3
- Architecture: Flat/Mesh
- Topology: [mesh/star/hierarchical/ring]
- Strategy: [balanced/specialized/adaptive]

**Metrics**:
| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
| Total Execution Time | | seconds | Including coordination overhead |
| Parallel Speedup | | ratio | vs baseline |
| Total Input Tokens | | tokens | Sum across all agents |
| Total Output Tokens | | tokens | Sum across all agents |
| Token Efficiency | | ratio | |
| Memory Usage | | MB | Peak across all agents |
| Coordination Overhead | | seconds | Time spent in coordination |

**Quality Scores** (0-10 scale):
- Accuracy/Correctness: [ ] / 10
- Coherence: [ ] / 10
- Completeness: [ ] / 10
- Code Quality: [ ] / 10
- Documentation: [ ] / 10

**Consensus Metrics**:
- Agent Agreement Level: [ ]% (How often agents agreed)
- Integration Complexity: [Low/Medium/High]
- Conflicting Solutions: [Yes/No - describe if yes]

**Per-Agent Performance**:
| Agent | Type | Tasks Completed | Tokens Used | Time Spent |
|-------|------|-----------------|-------------|------------|
| Agent 1 | | | | |
| Agent 2 | | | | |
| Agent 3 | | | | |

**Detailed Results**:
```
[Paste or summarize the combined output]
```

**Notes & Observations**:
- 
- 

---

### Configuration: Swarm Config B (3 agents, hierarchical)
[Repeat same structure as Config A]

---

### Configuration: Swarm Config C (5 agents, dynamic)
[Repeat same structure with 5 agents]

---

### Configuration: Swarm Config D (10 agents, stress test)
[Repeat same structure with 10 agents]

---

## Comparative Analysis

### Performance Comparison Table
| Configuration | Execution Time | Token Usage | Quality Score | Efficiency |
|--------------|----------------|-------------|---------------|------------|
| Claude Native | | | | baseline |
| Config A (3, flat) | | | | |
| Config B (3, hierarchical) | | | | |
| Config C (5, dynamic) | | | | |
| Config D (10, stress) | | | | |

### Speedup Analysis
```
Speedup = Baseline Time / Configuration Time
Efficiency = Speedup / Number of Agents

| Configuration | Speedup | Efficiency | Parallel Overhead |
|--------------|---------|------------|-------------------|
| Config A | | | |
| Config B | | | |
| Config C | | | |
| Config D | | | |
```

### Token Efficiency Analysis
```
Token Efficiency Score = Quality Score / Total Tokens Used

| Configuration | Total Tokens | Quality | Efficiency Score |
|--------------|--------------|---------|------------------|
| Claude Native | | | |
| Config A | | | |
| Config B | | | |
| Config C | | | |
| Config D | | | |
```

### Consensus Divergence (Multi-Agent Configurations Only)
| Configuration | Agreement % | Major Conflicts | Resolution Time |
|--------------|-------------|-----------------|-----------------|
| Config A | | | |
| Config B | | | |
| Config C | | | |
| Config D | | | |

## Quality Assessment Details

### Evaluation Criteria Breakdown
For each configuration, provide detailed scoring:

**1. Accuracy/Correctness**
- [ ] All requirements met
- [ ] No logical errors
- [ ] Handles edge cases
- [ ] Mathematically sound (if applicable)

**2. Coherence**
- [ ] Consistent approach throughout
- [ ] Well-structured solution
- [ ] Clear reasoning
- [ ] Integrated components (multi-agent)

**3. Code Quality** (if applicable)
- [ ] Follows best practices
- [ ] Efficient implementation
- [ ] Proper error handling
- [ ] Well-documented

**4. Completeness**
- [ ] All deliverables provided
- [ ] Comprehensive coverage
- [ ] Production-ready (if required)
- [ ] Testing included

## Measurement Instructions

### 1. Timing Measurements
```bash
# Use high-precision timing
START_TIME=$(date +%s.%N)
# ... execute test ...
END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc)
```

### 2. Token Counting
- Use Claude's token counter for accurate measurement
- For swarms, sum tokens across all agents
- Record both input and output tokens separately

### 3. Memory Usage
```bash
# Monitor peak memory usage during execution
/usr/bin/time -v [command] 2>&1 | grep "Maximum resident set size"
```

### 4. Quality Scoring Guidelines
- **0-2**: Fails to meet basic requirements
- **3-4**: Meets some requirements with significant issues
- **5-6**: Meets most requirements with minor issues
- **7-8**: Good solution with minor improvements possible
- **9-10**: Excellent solution exceeding expectations

### 5. Consensus Measurement (Multi-Agent)
```python
def calculate_consensus(agent_outputs):
    # Extract key decisions/approaches from each agent
    decisions = extract_decisions(agent_outputs)
    
    # Calculate agreement percentage
    total_decisions = len(decisions)
    agreed_decisions = count_agreements(decisions)
    
    consensus_percentage = (agreed_decisions / total_decisions) * 100
    return consensus_percentage
```

## Visualization Requirements

### 1. Performance Radar Chart
Create a radar chart comparing all configurations across:
- Execution Speed
- Token Efficiency  
- Quality Score
- Scalability
- Cost Effectiveness

### 2. Time Series Plot
Show execution timeline for multi-agent configurations:
- Agent spawn times
- Task execution periods
- Coordination overhead
- Integration phases

### 3. Token Usage Heatmap
Visualize token distribution across agents and tasks

## Summary and Recommendations

### Key Findings
1. 
2. 
3. 

### Performance Insights
- Best configuration for speed: 
- Best configuration for quality: 
- Best configuration for efficiency: 

### Recommendations
- For simple tasks: 
- For complex tasks: 
- For time-critical tasks: 
- For quality-critical tasks: 

### Anomalies and Interesting Observations
- 
- 

## Raw Data
[Attach or link to raw logs, full outputs, and detailed measurements]

---

**Template Version**: 1.0
**Last Updated**: 2025-01-05