# 🚀 Agent Coordination Prompt: WASM Implementation Swarm

## 🎯 Mission
Initialize a 5-agent swarm to implement the complete WASM integration for ruv-swarm, exposing all Rust capabilities (ruv-FANN neural networks, neuro-divergent forecasting, and ruv-swarm orchestration) through WebAssembly interfaces in the NPX package.

## 📋 Pre-Implementation Checklist
- [x] Planning documents created (00-08 in /plans/wasm/)
- [x] Neural architecture research completed
- [x] Claude Code integration patterns documented
- [x] SWE-bench testing strategy defined
- [ ] Implementation phase ready to begin

## 🤖 5-Agent Swarm Initialization Command

```bash
claude "Initialize 5-agent WASM implementation swarm for ruv-swarm. Each agent should follow their respective planning documents in /workspaces/ruv-FANN/ruv-swarm/plans/wasm/. 

Agent 1 (WASM Architect - Convergent): Implement build pipeline from 01-agent1-wasm-architecture.md
Agent 2 (Neural Specialist - Divergent): Implement neural integration from 02-agent2-neural-integration.md  
Agent 3 (Forecasting Specialist - Systems): Implement forecasting models from 03-agent3-forecasting-models.md
Agent 4 (Swarm Coordinator - Critical): Implement orchestration from 04-agent4-swarm-orchestration.md
Agent 5 (Integration Specialist - Lateral): Implement NPX integration from 05-agent5-npx-integration.md

Use TodoWrite for task coordination and Memory for sharing architectural decisions. All agents must ensure:
1. Per-agent neural networks using ruv-FANN and neuro-divergent
2. WASM modules compile and optimize correctly
3. All tests pass (unit, integration, performance, SWE-bench)
4. MCP tools function properly with new WASM capabilities
5. NPX package works seamlessly with progressive loading

Begin parallel implementation following the 4-week timeline in 00-master-wasm-implementation-plan.md" -p --dangerously-skip-permissions --output-format stream-json --verbose
```

## 📊 Agent Task Breakdown

### Agent 1: WASM Architect (Convergent Thinking)
**Primary Tasks:**
1. Set up WASM build environment with wasm-pack
2. Create multi-crate build pipeline for ruv-FANN, neuro-divergent, and ruv-swarm
3. Implement memory management and SIMD optimizations
4. Configure size optimization and progressive loading
5. Set up CI/CD pipeline with automated builds

**Success Criteria:**
- All Rust crates compile to WASM successfully
- Bundle size < 5MB total (or progressive chunks < 500KB each)
- SIMD optimizations provide 2-4x performance improvement
- Memory usage < 50MB for 10-agent swarm

### Agent 2: Neural Network Specialist (Divergent Thinking)
**Primary Tasks:**
1. Implement WasmNeuralNetwork bindings for ruv-FANN
2. Create AgentNeuralNetworkManager for per-agent networks
3. Expose all 18 activation functions and 5 training algorithms
4. Implement cascade correlation and neural persistence
5. Enable real-time training and fine-tuning

**Success Criteria:**
- All ruv-FANN features accessible from JavaScript
- Support 100+ simultaneous agent neural networks
- Training performance matches native Rust (within 20%)
- Neural states persist and restore correctly

### Agent 3: Forecasting Specialist (Systems Thinking)
**Primary Tasks:**
1. Implement WasmNeuralForecast bindings for neuro-divergent
2. Create AgentForecastingManager for model assignment
3. Expose all 27+ forecasting models with WASM interfaces
4. Implement time series processing and ensemble methods
5. Enable dynamic model switching based on performance

**Success Criteria:**
- All neuro-divergent models accessible from JavaScript
- Support 50+ simultaneous agent forecasting models
- Forecasting accuracy within 5% of native implementation
- Model switching latency < 200ms

### Agent 4: Swarm Coordinator (Critical Thinking)
**Primary Tasks:**
1. Implement WasmSwarmOrchestrator with full topology support
2. Create NeuralSwarmCoordinator for distributed processing
3. Implement cognitive diversity engine with pattern matching
4. Enable knowledge synchronization between agents
5. Implement collective intelligence emergence patterns

**Success Criteria:**
- All 4 topologies (mesh, hierarchical, ring, star) functional
- Cognitive patterns properly influence agent behavior
- Knowledge sync < 100ms swarm-wide
- 3x training speedup with distributed learning

### Agent 5: Integration Specialist (Lateral Thinking)
**Primary Tasks:**
1. Create progressive WASM loading system for NPX
2. Implement comprehensive MCP tool enhancements
3. Create neural network per-agent APIs and templates
4. Integrate with Claude Code command patterns
5. Ensure backward compatibility and smooth migration

**Success Criteria:**
- Progressive loading reduces initial load by 80%
- All MCP tools enhanced with WASM capabilities
- NPX commands work seamlessly with new features
- Zero breaking changes for existing users

## 🧪 Testing Requirements

### Unit Tests
```bash
# Each agent must ensure their unit tests pass
npm test -- --coverage --testPathPattern="wasm"
cargo test --target wasm32-unknown-unknown
```

### Integration Tests
```bash
# Test cross-agent functionality
npm run test:integration
./test-mcp-tools.js --wasm-enabled
```

### Performance Tests
```bash
# Verify performance targets
npm run benchmark:wasm
cargo bench --target wasm32-unknown-unknown
```

### SWE-bench Tests
```bash
# Validate coding capabilities
claude "Run SWE-bench test suite on WASM-enabled swarm" --swe-bench-mode --output-format stream-json
```

## 🔄 Coordination Patterns

### Memory-Driven Architecture Sharing
```javascript
// Agent 1 stores build configuration
TodoWrite([{
  id: "wasm_build_config",
  content: "Store WASM build configuration in Memory",
  status: "completed",
  priority: "high"
}]);
Memory.store("wasm_build_config", buildConfig);

// Other agents retrieve and use
const buildConfig = Memory.get("wasm_build_config");
```

### Parallel Task Execution
```javascript
// All agents work simultaneously on their modules
Task("Agent 1: Build Pipeline", "Implement WASM build system from plan");
Task("Agent 2: Neural WASM", "Create neural network bindings");
Task("Agent 3: Forecasting WASM", "Implement forecasting models");
Task("Agent 4: Swarm WASM", "Build orchestration system");
Task("Agent 5: NPX Integration", "Create progressive loading");
```

### Progress Synchronization
```javascript
TodoWrite([
  { id: "week1_milestone", content: "Complete WASM setup", status: "pending" },
  { id: "week2_milestone", content: "Core features implemented", status: "pending" },
  { id: "week3_milestone", content: "Integration complete", status: "pending" },
  { id: "week4_milestone", content: "Testing and optimization", status: "pending" }
]);
```

## 📈 Success Metrics

### Performance Targets
- Initial WASM load time < 500ms
- Neural network creation < 50ms per agent
- Swarm initialization < 200ms for 10 agents
- Memory usage < 5MB per agent
- CPU usage < 70% during peak operations

### Functionality Targets
- 100% of ruv-FANN features exposed
- 100% of neuro-divergent models available
- All 4 swarm topologies functional
- All MCP tools enhanced and working
- SWE-bench performance ≥ 70% on medium problems

### Quality Targets
- Test coverage > 90%
- Zero memory leaks
- TypeScript definitions 100% complete
- Documentation coverage 100%
- Examples for all major features

## 🚦 Implementation Phases

### Week 1: Foundation (Days 1-7)
- Set up WASM build environment
- Create basic bindings for all crates
- Implement memory management
- Set up testing infrastructure

### Week 2: Core Features (Days 8-14)
- Implement neural network bindings
- Create forecasting model interfaces
- Build swarm orchestration WASM
- Develop progressive loading system

### Week 3: Integration (Days 15-21)
- Integrate all WASM modules
- Enhance MCP tools
- Create NPX interfaces
- Implement per-agent neural networks

### Week 4: Polish (Days 22-28)
- Performance optimization
- Comprehensive testing
- Documentation updates
- Production preparation

## 🎯 Final Validation

Before marking implementation complete, ensure:

1. **All Tests Pass**
   ```bash
   npm run test:all
   cargo test --all-features
   ./run-all-tests.sh
   ```

2. **Performance Benchmarks Met**
   ```bash
   npm run benchmark:all
   ./verify-performance-targets.sh
   ```

3. **MCP Tools Functional**
   ```bash
   ./test-all-mcp-tools.js
   npx ruv-swarm mcp status
   ```

4. **SWE-bench Performance**
   ```bash
   ./run-swe-bench-suite.sh
   ```

5. **Documentation Complete**
   ```bash
   npm run docs:build
   npm run docs:validate
   ```

## 🏁 Start Implementation

Execute the swarm initialization command above to begin the implementation. Each agent should:
1. Read their respective planning document
2. Use TodoWrite to track their tasks
3. Share critical decisions via Memory
4. Coordinate through the swarm orchestrator
5. Validate their work continuously

The swarm should operate with high autonomy while maintaining coordination through the established patterns. Regular status updates should be provided through the monitoring system.

**May the swarm intelligence guide us to success! 🐝**