# Test 3: Mathematical Problem Solving - Optimization with Constraints

## 🔴 Difficulty: HIGH
**Expected Duration**: 20-25 minutes per configuration (Optional Advanced Test)

## Test Overview
This test evaluates mathematical reasoning and problem-solving capabilities through a complex optimization problem that requires understanding of calculus, linear algebra, and algorithmic thinking.

## Test Prompt
```
Solve the following optimization problem and provide a complete solution with implementation:

A logistics company needs to optimize their delivery route system. They have:
- N delivery locations in a 2D plane with coordinates (x_i, y_i)
- M delivery trucks, each with capacity C_j
- Each location i has demand d_i (packages to deliver)
- Each truck starts from depot at (0, 0) and must return
- Time window constraints: location i must be visited between [a_i, b_i]
- Fuel cost is proportional to Euclidean distance traveled

Objectives:
1. Minimize total distance traveled by all trucks
2. Ensure all demands are met
3. Respect capacity constraints
4. Meet time window requirements
5. Balance load across trucks (minimize max - min packages per truck)

Tasks:
1. Formulate this as a mathematical optimization problem
2. Prove whether the problem is NP-hard
3. Develop an efficient approximation algorithm
4. Implement the algorithm in Python
5. Analyze time and space complexity
6. Provide bounds on the approximation ratio
7. Create visualizations of example solutions

Test with:
N = 20 locations
M = 4 trucks
Capacities: [50, 40, 45, 55]
Demands: randomly between 5-15 packages
Time windows: 2-hour windows between 8 AM - 6 PM
Average speed: 30 km/h
```

## Expected Deliverables
- Mathematical formulation (ILP or MILP)
- Complexity analysis and NP-hardness proof
- Algorithm implementation with comments
- Visualization of routes
- Performance analysis
- Theoretical guarantees

## Test Configurations

### 1. Claude Native (Baseline)
- **Setup**: Direct mathematical problem to Claude
- **Agent Count**: 1
- **Architecture**: N/A
- **Approach**: Sequential problem solving

### 2. Swarm Config A: Mathematical Team (3 agents, flat)
- **Setup**:
  ```javascript
  mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 3, strategy: "balanced" }
  mcp__ruv-swarm__agent_spawn { type: "researcher", name: "theory-expert" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "algorithm-implementer" }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "complexity-analyzer" }
  ```
- **Agent Count**: 3
- **Architecture**: Flat - parallel analysis
- **Task Distribution**:
  - Researcher: Mathematical formulation and proofs
  - Coder: Algorithm implementation
  - Analyst: Complexity analysis and bounds

### 3. Swarm Config B: Hierarchical Problem Solving (3 agents)
- **Setup**:
  ```javascript
  mcp__ruv-swarm__swarm_init { topology: "hierarchical", maxAgents: 3, strategy: "specialized" }
  mcp__ruv-swarm__agent_spawn { type: "coordinator", name: "problem-architect" }
  mcp__ruv-swarm__agent_spawn { type: "researcher", name: "math-specialist" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "implementation-expert" }
  ```
- **Agent Count**: 3
- **Architecture**: Hierarchical - structured approach
- **Workflow**:
  1. Coordinator breaks down problem
  2. Specialist handles mathematical aspects
  3. Expert implements solution
  4. Coordinator integrates and validates

### 4. Swarm Config C: Specialized Solvers (5 agents)
- **Setup**:
  ```javascript
  mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 5, strategy: "adaptive" }
  mcp__ruv-swarm__agent_spawn { type: "researcher", name: "optimization-theorist" }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "np-complexity-expert" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "heuristic-designer" }
  mcp__ruv-swarm__agent_spawn { type: "optimizer", name: "performance-engineer" }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "visualization-specialist" }
  ```
- **Agent Count**: 5
- **Architecture**: Dynamic - domain expertise
- **Specializations**: Each agent focuses on specific aspects

### 5. Swarm Config D: Full Research Team (10 agents)
- **Setup**:
  ```javascript
  mcp__ruv-swarm__swarm_init { topology: "star", maxAgents: 10, strategy: "balanced" }
  // Specialists for: ILP formulation, graph theory, approximation algorithms,
  // metaheuristics, constraint programming, visualization, testing, etc.
  ```
- **Agent Count**: 10
- **Architecture**: Star - comprehensive coverage
- **Approach**: Multiple solution strategies explored in parallel

## Evaluation Metrics

### 1. Mathematical Correctness (30%)
- [ ] Correct problem formulation
- [ ] Valid constraints representation
- [ ] Accurate objective function
- [ ] Sound mathematical reasoning

### 2. Algorithm Quality (25%)
- [ ] Efficient implementation
- [ ] Handles all constraints
- [ ] Produces valid solutions
- [ ] Good approximation quality

### 3. Theoretical Analysis (20%)
- [ ] Correct complexity analysis
- [ ] Valid NP-hardness proof (if applicable)
- [ ] Approximation ratio bounds
- [ ] Time/space complexity accurate

### 4. Implementation (15%)
- [ ] Clean, efficient code
- [ ] Proper data structures
- [ ] Handles edge cases
- [ ] Well-documented

### 5. Visualization & Testing (10%)
- [ ] Clear route visualizations
- [ ] Comprehensive test cases
- [ ] Performance benchmarks
- [ ] Solution quality metrics

## Measurement Instructions

### Solution Quality Metrics
```python
def evaluate_solution(routes, locations, demands, capacities):
    metrics = {
        "total_distance": calculate_total_distance(routes),
        "capacity_violations": check_capacity_constraints(routes, demands, capacities),
        "time_window_violations": check_time_windows(routes, time_windows),
        "demand_satisfaction": verify_all_demands_met(routes, demands),
        "load_balance": calculate_load_balance(routes),
        "computation_time": end_time - start_time
    }
    return metrics
```

### Approximation Quality
```python
# Compare against known bounds or optimal solutions for small instances
def approximation_ratio(solution_value, optimal_value):
    return solution_value / optimal_value

# For larger instances, compare against lower bounds
def compute_lower_bound():
    # MST-based lower bound
    # LP relaxation
    # Other techniques
    pass
```

### Complexity Verification
1. Run algorithm on increasing problem sizes
2. Plot runtime vs problem size
3. Verify theoretical complexity matches empirical
4. Memory usage analysis

### Consensus Metrics (Multi-Agent)
- Agreement on problem formulation
- Consistency in algorithm choice
- Convergence on approximation strategies
- Integration challenges

## Expected Outcomes

### Claude Native (Baseline)
- Single approach to the problem
- May focus on one solution strategy
- Linear development of ideas
- Consistent notation throughout

### Swarm Configurations
- **Config A**: Parallel development of theory and implementation
- **Config B**: Well-structured solution with clear hierarchy
- **Config C**: Multiple algorithmic approaches compared
- **Config D**: Comprehensive exploration of solution space

## Test Data Generator
```python
import numpy as np
import random

def generate_test_instance(n_locations=20, n_trucks=4):
    # Generate random locations
    locations = [(random.uniform(0, 100), random.uniform(0, 100)) 
                 for _ in range(n_locations)]
    
    # Generate demands
    demands = [random.randint(5, 15) for _ in range(n_locations)]
    
    # Generate time windows (2-hour windows)
    time_windows = []
    for i in range(n_locations):
        start = random.uniform(8, 16)  # 8 AM to 4 PM start
        end = start + 2
        time_windows.append((start, min(end, 18)))  # Cap at 6 PM
    
    # Truck capacities
    capacities = [50, 40, 45, 55][:n_trucks]
    
    return {
        "locations": locations,
        "demands": demands,
        "time_windows": time_windows,
        "capacities": capacities,
        "depot": (0, 0),
        "speed": 30  # km/h
    }
```

## Visualization Requirements
```python
import matplotlib.pyplot as plt

def visualize_routes(routes, locations, depot):
    plt.figure(figsize=(10, 10))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot depot
    plt.scatter(*depot, color='black', s=200, marker='s', label='Depot')
    
    # Plot locations
    for i, loc in enumerate(locations):
        plt.scatter(*loc, color='gray', s=100)
        plt.text(loc[0]+1, loc[1]+1, str(i))
    
    # Plot routes
    for truck_id, route in enumerate(routes):
        color = colors[truck_id % len(colors)]
        route_locs = [depot] + [locations[i] for i in route] + [depot]
        
        xs, ys = zip(*route_locs)
        plt.plot(xs, ys, color=color, linewidth=2, 
                label=f'Truck {truck_id}', marker='o')
    
    plt.legend()
    plt.title('Delivery Routes Visualization')
    plt.xlabel('X coordinate (km)')
    plt.ylabel('Y coordinate (km)')
    plt.grid(True, alpha=0.3)
    plt.show()
```

## Notes
- Focus on both theoretical and practical aspects
- Consider multiple solution approaches
- Document assumptions clearly
- Provide intuition behind mathematical formulations