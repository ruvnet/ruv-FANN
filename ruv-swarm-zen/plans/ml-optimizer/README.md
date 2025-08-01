# ML Optimizer for Claude Code Swarms

## Overview

This directory contains a comprehensive system for training, optimizing, and comparing ML-based coding swarms using Claude Code CLI. The system benchmarks performance on SWE-Bench challenges and uses stream-json output for real-time analysis and optimization.

## Directory Structure

```
ml-optimizer/
├── 01-master-plan.md          # Overall architecture and vision
├── 02-model-analysis.md       # Neuro-divergent model evaluation
├── 03-benchmarking-system.md  # Benchmarking framework design
├── 04-training-strategy.md    # ML training methodology
├── 05-implementation-roadmap.md # 4-week implementation plan
└── README.md                  # This file
```

## Key Features

### 1. Claude Code CLI Integration
- Uses stream-json output format for real-time performance analysis
- Command format: `claude "task" -p --dangerously-skip-permissions --output-format stream-json --verbose`
- Parses thinking, tool_use, and message events for training data

### 2. SWE-Bench Integration
- Real-world coding challenges from open-source repositories
- Difficulty levels: Easy, Medium, Hard
- Automatic evaluation of generated patches

### 3. Neuro-Divergent Models
- 27+ neural forecasting models
- Cognitive pattern matching (Convergent, Divergent, Lateral, Systems)
- Task-specific model selection

### 4. Performance Metrics
- Task completion time
- Token usage efficiency
- Code quality scores
- Swarm coordination metrics

## Quick Start

1. **Review Plans**: Start with `01-master-plan.md` for overall architecture
2. **Model Selection**: See `02-model-analysis.md` for choosing appropriate models
3. **Benchmarking**: Use `03-benchmarking-system.md` to set up testing
4. **Training**: Follow `04-training-strategy.md` for model optimization
5. **Implementation**: Use `05-implementation-roadmap.md` for step-by-step execution

## Implementation Timeline

- **Week 1**: Claude Code stream parser and benchmarking framework
- **Week 2**: SWE-Bench integration and baseline collection
- **Week 3**: Stream-based model training and optimization
- **Week 4**: Deployment and documentation

## Expected Outcomes

- 30-50% reduction in token usage
- 2x improvement in task completion speed
- >85% success rate on SWE-Bench challenges
- Automated model selection based on task complexity

## Next Steps

The benchmarking tool implementation is currently in progress. Once complete, we'll proceed with:
1. Training specialized models for different coding tasks
2. Optimizing swarm coordination strategies
3. Creating deployment packages with documentation

## Resources

- SWE-Bench: https://github.com/princeton-nlp/SWE-bench
- Claude Code CLI Documentation: https://docs.anthropic.com/en/docs/claude-code
- Neuro-Divergent Models: /workspaces/ruv-FANN/neuro-divergent/

Created by rUv