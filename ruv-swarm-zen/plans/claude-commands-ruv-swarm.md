# Claude Commands RUV Swarm Integration Specification

## Overview
This document defines the `.claude/commands` integration for RUV-FANN swarm operations, providing a unified interface for swarm orchestration, agent management, and distributed computation.

## Command Specifications

### 1. ruv-swarm init
Initialize swarm infrastructure with configurable topology and resource allocation.

```bash
# Command syntax
ruv-swarm init [OPTIONS]

# Options
--topology <type>        # mesh | star | hierarchical | ring | hybrid (default: mesh)
--agents <count>         # Initial agent count (default: 3)
--memory-pool <size>     # Shared memory pool size in MB (default: 512)
--cache-strategy <type>  # lru | lfu | arc | adaptive (default: adaptive)
--config <file>          # Load configuration from YAML/JSON file
--profile <name>         # Use predefined profile: dev | prod | research | analysis
```

#### Configuration Schema
```yaml
swarm:
  name: "ruv-swarm-instance-1"
  topology: "mesh"
  resource_limits:
    memory: "2GB"
    cpu_cores: 4
    gpu_enabled: true
  
  agents:
    min_count: 3
    max_count: 16
    auto_scale: true
    
  communication:
    protocol: "msgpack"  # json | msgpack | protobuf
    compression: true
    encryption: false
    
  optimization:
    simd_enabled: true
    cache_alignment: 64
    batch_size: 32
```

#### Example Usage
```bash
# Basic initialization
ruv-swarm init

# Production setup with hierarchical topology
ruv-swarm init --topology hierarchical --agents 8 --profile prod

# Custom configuration
ruv-swarm init --config swarm-config.yaml
```

### 2. ruv-swarm spawn
Create specialized agents with defined capabilities and resource allocations.

```bash
# Command syntax
ruv-swarm spawn <agent-type> [OPTIONS]

# Agent types
vision      # Computer vision processing agent
audio       # Audio analysis agent
text        # Natural language processing agent
fusion      # Multi-modal fusion agent
detector    # Lie detection specialized agent
optimizer   # Performance optimization agent
analyzer    # Data analysis agent
coordinator # Task coordination agent

# Options
--name <name>            # Agent identifier
--priority <level>       # low | medium | high | critical
--memory <size>          # Agent memory allocation
--capabilities <list>    # Comma-separated capability list
--dependencies <list>    # Required agent dependencies
--pool <name>            # Assign to specific resource pool
```

#### Agent Capability Definitions
```yaml
agent_templates:
  vision:
    capabilities:
      - face_detection
      - emotion_recognition
      - micro_expression_analysis
      - gaze_tracking
    resource_requirements:
      memory: "512MB"
      gpu_required: true
      
  audio:
    capabilities:
      - pitch_detection
      - voice_stress_analysis
      - speech_pattern_recognition
      - prosody_analysis
    resource_requirements:
      memory: "256MB"
      dsp_acceleration: true
```

#### Example Usage
```bash
# Spawn basic vision agent
ruv-swarm spawn vision --name vision-agent-1

# Spawn high-priority audio analyzer
ruv-swarm spawn audio --name audio-primary --priority high --memory 512MB

# Spawn fusion agent with dependencies
ruv-swarm spawn fusion --dependencies vision-agent-1,audio-primary --capabilities "weighted_fusion,temporal_alignment"
```

### 3. ruv-swarm orchestrate
Coordinate agent activities and manage distributed task execution.

```bash
# Command syntax
ruv-swarm orchestrate <task-definition> [OPTIONS]

# Options
--strategy <type>        # sequential | parallel | pipeline | adaptive
--timeout <seconds>      # Task timeout (default: 300)
--checkpoint <interval>  # Checkpoint interval in seconds
--retry <count>          # Max retry attempts (default: 3)
--output <format>        # json | msgpack | sqlite | parquet
--monitor               # Enable real-time monitoring
```

#### Task Definition Format
```yaml
task:
  name: "multimodal_analysis"
  description: "Analyze video for deception indicators"
  
  stages:
    - name: "preprocessing"
      agents: ["vision-agent-1", "audio-primary"]
      parallel: true
      tasks:
        - extract_frames
        - extract_audio
        
    - name: "analysis"
      agents: ["vision-agent-1", "audio-primary", "text-agent-1"]
      parallel: true
      tasks:
        - analyze_facial_expressions
        - analyze_voice_patterns
        - analyze_speech_content
        
    - name: "fusion"
      agents: ["fusion-agent-1"]
      parallel: false
      tasks:
        - combine_modalities
        - compute_confidence_scores
        
  output:
    format: "json"
    schema: "deception_report_v1"
```

#### Example Usage
```bash
# Basic orchestration
ruv-swarm orchestrate analysis-task.yaml

# Pipeline processing with monitoring
ruv-swarm orchestrate video-analysis.yaml --strategy pipeline --monitor

# Parallel execution with checkpointing
ruv-swarm orchestrate batch-process.yaml --strategy parallel --checkpoint 30 --output parquet
```

### 4. ruv-swarm monitor
Real-time swarm monitoring and performance analytics.

```bash
# Command syntax
ruv-swarm monitor [OPTIONS]

# Options
--view <type>           # dashboard | metrics | logs | trace (default: dashboard)
--agents <list>         # Monitor specific agents
--metrics <list>        # Specific metrics to track
--interval <seconds>    # Update interval (default: 1)
--export <file>         # Export monitoring data
--alert <config>        # Alert configuration file
```

#### Monitoring Metrics
```yaml
metrics:
  system:
    - cpu_usage
    - memory_usage
    - network_io
    - disk_io
    
  agent:
    - task_throughput
    - processing_latency
    - error_rate
    - queue_depth
    
  swarm:
    - total_throughput
    - agent_utilization
    - communication_overhead
    - load_balance_efficiency
```

#### Alert Configuration
```yaml
alerts:
  - name: "high_memory_usage"
    condition: "memory_usage > 90%"
    severity: "warning"
    action: "scale_up"
    
  - name: "agent_failure"
    condition: "agent_status == 'failed'"
    severity: "critical"
    action: "restart_agent"
```

#### Example Usage
```bash
# Interactive dashboard
ruv-swarm monitor

# Monitor specific agents with metrics export
ruv-swarm monitor --agents vision-agent-1,audio-primary --export metrics.csv

# Trace view with custom interval
ruv-swarm monitor --view trace --interval 0.5
```

### 5. ruv-swarm optimize
Performance tuning and resource optimization for swarm operations.

```bash
# Command syntax
ruv-swarm optimize [OPTIONS]

# Options
--target <metric>       # latency | throughput | memory | energy
--profile <name>        # Save optimization profile
--auto                  # Automatic optimization
--benchmark             # Run benchmarks before/after
--constraints <file>    # Optimization constraints
```

#### Optimization Strategies
```yaml
optimization:
  strategies:
    latency:
      - minimize_communication_overhead
      - enable_simd_operations
      - optimize_cache_usage
      - reduce_serialization_cost
      
    throughput:
      - maximize_parallelization
      - batch_processing
      - pipeline_optimization
      - load_balancing
      
    memory:
      - object_pooling
      - memory_compression
      - cache_eviction_tuning
      - reduce_allocations
```

#### Constraints Definition
```yaml
constraints:
  max_memory: "4GB"
  min_accuracy: 0.95
  max_latency_ms: 100
  min_throughput_fps: 30
  
  trade_offs:
    - priority: "latency"
      weight: 0.6
    - priority: "accuracy"
      weight: 0.4
```

#### Example Usage
```bash
# Optimize for latency
ruv-swarm optimize --target latency

# Auto-optimization with benchmarking
ruv-swarm optimize --auto --benchmark --profile optimized-config

# Constrained optimization
ruv-swarm optimize --target throughput --constraints constraints.yaml
```

## Integration with RUV-FANN

### Memory Pool Integration
```bash
# Commands automatically use RUV-FANN's memory pool
ruv-swarm init --memory-pool 1024  # 1GB shared pool

# Agents share memory efficiently
ruv-swarm spawn vision --pool shared-pool-1
```

### SIMD Optimization
```bash
# Enable SIMD operations across swarm
ruv-swarm optimize --enable-simd

# Check SIMD utilization
ruv-swarm monitor --metrics simd_usage
```

### Lie Detection Pipeline
```bash
# Complete lie detection workflow
ruv-swarm orchestrate lie-detection.yaml --monitor

# Workflow includes:
# 1. Video/audio ingestion
# 2. Parallel feature extraction
# 3. Multi-modal fusion
# 4. Confidence scoring
# 5. Report generation
```

## Advanced Features

### Dynamic Scaling
```bash
# Enable auto-scaling
ruv-swarm init --auto-scale --min-agents 3 --max-agents 16

# Manual scaling
ruv-swarm scale --agents 8
```

### Fault Tolerance
```bash
# Enable checkpointing and recovery
ruv-swarm orchestrate task.yaml --checkpoint 30 --recovery enabled

# Health checks
ruv-swarm health --detailed
```

### Distributed Training
```bash
# Distribute model training across swarm
ruv-swarm train model.yaml --distributed --agents 4
```

## Command Aliases and Shortcuts

```bash
# Aliases for common operations
alias rsi='ruv-swarm init'
alias rss='ruv-swarm spawn'
alias rso='ruv-swarm orchestrate'
alias rsm='ruv-swarm monitor'

# Quick commands
ruv-swarm quick-detect video.mp4  # Full lie detection pipeline
ruv-swarm benchmark               # Run performance benchmarks
ruv-swarm status                  # Swarm status summary
```

## Error Handling

### Common Error Patterns
```bash
# Agent spawn failure
Error: Failed to spawn agent 'vision-agent-1'
Reason: Insufficient GPU memory
Solution: ruv-swarm optimize --target memory

# Communication timeout
Error: Agent communication timeout
Reason: Network congestion
Solution: ruv-swarm optimize --target latency

# Resource exhaustion
Error: Memory pool exhausted
Reason: Too many active agents
Solution: ruv-swarm scale --down 2
```

## Best Practices

1. **Initialize with appropriate topology**
   - Use mesh for research/experimentation
   - Use hierarchical for production workloads
   - Use star for centralized coordination

2. **Monitor resource usage**
   ```bash
   ruv-swarm monitor --metrics memory_usage,cpu_usage --alert alerts.yaml
   ```

3. **Optimize before production**
   ```bash
   ruv-swarm optimize --auto --benchmark --profile production
   ```

4. **Use checkpointing for long tasks**
   ```bash
   ruv-swarm orchestrate long-task.yaml --checkpoint 60
   ```

5. **Profile and benchmark regularly**
   ```bash
   ruv-swarm benchmark --save baseline
   ruv-swarm benchmark --compare baseline
   ```