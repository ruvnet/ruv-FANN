# Test 1: Code Generation - Implement a Rate-Limited API Client

## 🔴 Difficulty: HIGH
**Expected Duration**: 15-20 minutes per configuration (Optional Advanced Test)

## Test Overview
This test evaluates the ability to generate production-quality code for a real-world scenario: implementing a rate-limited API client with retry logic, error handling, and concurrent request management.

## Test Prompt
```
Create a Python class called RateLimitedAPIClient that implements the following requirements:

1. Support configurable rate limiting (e.g., 100 requests per minute)
2. Implement exponential backoff retry logic for failed requests
3. Handle concurrent requests using asyncio
4. Support request queuing when rate limit is reached
5. Provide detailed logging and metrics collection
6. Include proper error handling for network issues, timeouts, and API errors
7. Support both GET and POST methods
8. Include a circuit breaker pattern that opens after 5 consecutive failures

The client should be production-ready with type hints, docstrings, and comprehensive error handling.
```

## Expected Deliverables
- Complete implementation of RateLimitedAPIClient class
- Unit tests covering all functionality
- Usage examples
- Documentation of design decisions

## Test Configurations

### 1. Claude Native (Baseline)
- **Setup**: Direct prompt to Claude without swarm
- **Agent Count**: 1
- **Architecture**: N/A
- **Execution**: Single response generation

### 2. Swarm Config A: Simple Parallel (3 agents, flat)
- **Setup**: 
  ```javascript
  mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 3, strategy: "balanced" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "implementation-expert" }
  mcp__ruv-swarm__agent_spawn { type: "tester", name: "test-designer" }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "code-reviewer" }
  ```
- **Agent Count**: 3
- **Architecture**: Flat - all agents work in parallel
- **Task Distribution**:
  - Coder: Implements the main class
  - Tester: Creates unit tests
  - Analyst: Reviews code quality and suggests improvements

### 3. Swarm Config B: Hierarchical (3 agents)
- **Setup**:
  ```javascript
  mcp__ruv-swarm__swarm_init { topology: "hierarchical", maxAgents: 3, strategy: "specialized" }
  mcp__ruv-swarm__agent_spawn { type: "coordinator", name: "lead-architect" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "implementation-specialist" }
  mcp__ruv-swarm__agent_spawn { type: "tester", name: "qa-engineer" }
  ```
- **Agent Count**: 3
- **Architecture**: Hierarchical - coordinator delegates to specialists
- **Workflow**:
  1. Coordinator designs architecture and delegates tasks
  2. Coder implements based on architecture
  3. Tester validates implementation
  4. Coordinator integrates and finalizes

### 4. Swarm Config C: Dynamic (5 agents)
- **Setup**:
  ```javascript
  mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 5, strategy: "adaptive" }
  mcp__ruv-swarm__agent_spawn { type: "researcher", name: "best-practices-researcher" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "async-specialist" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "error-handling-expert" }
  mcp__ruv-swarm__agent_spawn { type: "optimizer", name: "performance-tuner" }
  mcp__ruv-swarm__agent_spawn { type: "tester", name: "integration-tester" }
  ```
- **Agent Count**: 5
- **Architecture**: Dynamic - agents coordinate based on task complexity
- **Adaptive Strategy**: Agents dynamically allocate effort based on code complexity

### 5. Swarm Config D: Stress Test (10 agents, flat)
- **Setup**:
  ```javascript
  mcp__ruv-swarm__swarm_init { topology: "star", maxAgents: 10, strategy: "balanced" }
  // Spawn 10 specialized agents for different aspects
  ```
- **Agent Count**: 10
- **Architecture**: Star topology with central coordination
- **Specializations**: Rate limiting, async handling, error management, testing, documentation, etc.

## Evaluation Metrics

### 1. Correctness (40%)
- [ ] Implements all required features
- [ ] Handles edge cases properly
- [ ] No logical errors or bugs
- [ ] Thread-safe implementation

### 2. Code Quality (25%)
- [ ] Proper use of Python idioms
- [ ] Clear and maintainable code structure
- [ ] Comprehensive type hints
- [ ] Meaningful variable/function names

### 3. Testing (15%)
- [ ] Unit test coverage > 80%
- [ ] Tests cover edge cases
- [ ] Mocking used appropriately
- [ ] Tests are maintainable

### 4. Documentation (10%)
- [ ] Clear docstrings for all public methods
- [ ] Usage examples provided
- [ ] Design decisions explained
- [ ] README or documentation file

### 5. Performance (10%)
- [ ] Efficient rate limiting implementation
- [ ] Minimal overhead for request processing
- [ ] Proper use of async/await
- [ ] Memory-efficient queue management

## Measurement Instructions

### Timing
```bash
# Start timer when prompt is submitted
START_TIME=$(date +%s.%N)

# Execute test based on configuration
# ... test execution ...

# End timer when complete response is received
END_TIME=$(date +%s.%N)
LATENCY=$(echo "$END_TIME - $START_TIME" | bc)
```

### Token Usage
- Record input tokens from prompt
- Record output tokens from response
- Calculate total tokens used
- Note any token optimization strategies employed

### Quality Assessment
1. Run automated code quality checks:
   ```bash
   pylint generated_code.py
   mypy generated_code.py
   black --check generated_code.py
   ```

2. Run unit tests:
   ```bash
   pytest test_generated_code.py -v --cov=generated_code
   ```

3. Manual review checklist:
   - [ ] All requirements implemented
   - [ ] Error handling comprehensive
   - [ ] Code is production-ready
   - [ ] Documentation complete

### Consensus Divergence (Multi-Agent Only)
For swarm configurations, measure:
- Agreement on design patterns used
- Consistency in coding style
- Conflicting implementations
- Integration challenges

## Expected Outcomes

### Claude Native (Baseline)
- Single cohesive implementation
- Consistent style throughout
- May miss some edge cases
- Linear development approach

### Swarm Configurations
- **Config A**: Parallel development of components, potential integration challenges
- **Config B**: Well-architected solution with clear separation of concerns
- **Config C**: Comprehensive solution with specialized optimizations
- **Config D**: Possibly over-engineered, but highly robust implementation

## Notes
- Ensure all configurations use the same prompt
- Measure time from prompt submission to final deliverable
- Document any coordination overhead in swarm configurations
- Note any emergent behaviors or unexpected benefits/drawbacks