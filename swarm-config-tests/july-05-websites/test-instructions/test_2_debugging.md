# Test 2: Debugging - Fix Complex Concurrency Bug

## 🔴 Difficulty: HIGH
**Expected Duration**: 15-20 minutes per configuration (Optional Advanced Test)

## Test Overview
This test evaluates debugging capabilities on a complex multi-threaded Python application with race conditions, deadlocks, and memory leaks. The task simulates a real-world scenario similar to SWE-bench challenges.

## Test Prompt
```
Debug and fix the following issues in a distributed task processing system:

The provided code has several critical bugs:
1. Race condition causing duplicate task processing
2. Potential deadlock between worker threads
3. Memory leak in the task queue implementation
4. Incorrect error propagation causing silent failures
5. Task results occasionally being lost or corrupted

Here's the buggy code:

```python
import threading
import queue
import time
import random
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import weakref

@dataclass
class Task:
    id: str
    payload: Dict[str, Any]
    callback: Optional[Callable] = None
    retry_count: int = 0
    max_retries: int = 3

class DistributedTaskProcessor:
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.task_queue = queue.Queue()
        self.result_store = {}
        self.processing_tasks = set()
        self.workers = []
        self.lock = threading.Lock()
        self.shutdown = False
        self._start_workers()
    
    def _start_workers(self):
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self, worker_id: int):
        while not self.shutdown:
            try:
                task = self.task_queue.get(timeout=1)
                
                # Bug: Race condition here
                if task.id in self.processing_tasks:
                    self.task_queue.put(task)
                    continue
                
                self.processing_tasks.add(task.id)
                
                # Process task
                result = self._process_task(task)
                
                # Bug: Potential deadlock
                with self.lock:
                    self.result_store[task.id] = result
                    if task.callback:
                        task.callback(result)
                
                # Bug: Memory leak - tasks never removed from processing_tasks
                
            except queue.Empty:
                continue
            except Exception as e:
                # Bug: Error not properly handled
                print(f"Worker {worker_id} error: {e}")
    
    def _process_task(self, task: Task) -> Dict[str, Any]:
        # Simulate processing
        time.sleep(random.uniform(0.1, 0.5))
        
        # Randomly fail some tasks
        if random.random() < 0.1:
            raise Exception("Task processing failed")
        
        return {"task_id": task.id, "result": "processed", "data": task.payload}
    
    def submit_task(self, task: Task) -> str:
        # Bug: No check for duplicate task IDs
        self.task_queue.put(task)
        return task.id
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        # Bug: Race condition when reading results
        return self.result_store.get(task_id)
    
    def shutdown(self):
        self.shutdown = True
        for worker in self.workers:
            worker.join()
```

Requirements:
1. Fix all identified bugs
2. Ensure thread safety throughout
3. Implement proper cleanup and resource management
4. Add comprehensive error handling
5. Maintain backward compatibility
6. Add unit tests for the fixes
```

## Expected Deliverables
- Fixed version of DistributedTaskProcessor class
- Explanation of each bug and how it was fixed
- Unit tests demonstrating the fixes work correctly
- Performance analysis showing no regression

## Test Configurations

### 1. Claude Native (Baseline)
- **Setup**: Direct debugging prompt to Claude
- **Agent Count**: 1
- **Architecture**: N/A
- **Approach**: Linear debugging process

### 2. Swarm Config A: Parallel Analysis (3 agents, flat)
- **Setup**:
  ```javascript
  mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 3, strategy: "balanced" }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "concurrency-expert" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "fix-implementer" }
  mcp__ruv-swarm__agent_spawn { type: "tester", name: "verification-specialist" }
  ```
- **Agent Count**: 3
- **Architecture**: Flat - parallel bug analysis
- **Task Distribution**:
  - Analyst: Identifies and documents all bugs
  - Coder: Implements fixes
  - Tester: Verifies fixes and writes tests

### 3. Swarm Config B: Hierarchical Debugging (3 agents)
- **Setup**:
  ```javascript
  mcp__ruv-swarm__swarm_init { topology: "hierarchical", maxAgents: 3, strategy: "specialized" }
  mcp__ruv-swarm__agent_spawn { type: "coordinator", name: "debug-lead" }
  mcp__ruv-swarm__agent_spawn { type: "researcher", name: "bug-analyzer" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "fix-developer" }
  ```
- **Agent Count**: 3
- **Architecture**: Hierarchical - structured debugging
- **Workflow**:
  1. Coordinator creates debugging plan
  2. Researcher analyzes each bug systematically
  3. Coder implements fixes based on analysis
  4. Coordinator validates complete solution

### 4. Swarm Config C: Specialized Debugging Team (5 agents)
- **Setup**:
  ```javascript
  mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 5, strategy: "adaptive" }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "race-condition-expert" }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "deadlock-specialist" }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "memory-leak-detective" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "concurrent-fix-developer" }
  mcp__ruv-swarm__agent_spawn { type: "tester", name: "stress-tester" }
  ```
- **Agent Count**: 5
- **Architecture**: Dynamic - specialized expertise
- **Specializations**: Each agent focuses on specific bug types

### 5. Swarm Config D: Comprehensive Debug Squad (10 agents)
- **Setup**:
  ```javascript
  mcp__ruv-swarm__swarm_init { topology: "star", maxAgents: 10, strategy: "balanced" }
  // 10 agents with various debugging specializations
  ```
- **Agent Count**: 10
- **Architecture**: Star - central coordination
- **Coverage**: Static analysis, dynamic testing, formal verification, etc.

## Evaluation Metrics

### 1. Bug Detection (30%)
- [ ] All 5 bugs correctly identified
- [ ] Root causes properly explained
- [ ] Additional bugs found (bonus)
- [ ] Clear documentation of issues

### 2. Fix Quality (35%)
- [ ] All bugs properly fixed
- [ ] No new bugs introduced
- [ ] Thread-safe implementation
- [ ] Maintains performance

### 3. Code Quality (15%)
- [ ] Clean, readable fixes
- [ ] Minimal code changes
- [ ] Follows Python best practices
- [ ] Proper error handling

### 4. Testing (15%)
- [ ] Comprehensive test coverage
- [ ] Tests reproduce original bugs
- [ ] Tests verify fixes
- [ ] Stress tests included

### 5. Documentation (5%)
- [ ] Clear explanation of each fix
- [ ] Performance impact documented
- [ ] Migration notes if needed

## Measurement Instructions

### Bug Detection Time
```python
# Measure time to identify each bug
bug_detection_times = {
    "race_condition": None,
    "deadlock": None,
    "memory_leak": None,
    "error_propagation": None,
    "lost_results": None
}
```

### Fix Implementation Time
- Track time from bug identification to fix implementation
- Note any iterations required
- Measure testing time separately

### Correctness Verification
1. Run original buggy code with test harness
2. Confirm all bugs manifest
3. Run fixed code with same harness
4. Verify all bugs resolved
5. Run stress tests for edge cases

### Performance Comparison
```python
# Benchmark before and after
def benchmark_throughput():
    processor = DistributedTaskProcessor(num_workers=8)
    start_time = time.time()
    
    # Submit 1000 tasks
    for i in range(1000):
        task = Task(id=f"task_{i}", payload={"data": i})
        processor.submit_task(task)
    
    # Wait for completion
    # ... measurement code ...
    
    return tasks_per_second, avg_latency
```

### Consensus Analysis (Multi-Agent)
- Compare bug identification across agents
- Analyze different fix approaches
- Measure agreement on root causes
- Document any conflicting solutions

## Expected Outcomes

### Claude Native (Baseline)
- Sequential bug analysis
- Single unified fix approach
- May miss subtle interactions
- Consistent coding style

### Swarm Configurations
- **Config A**: Quick parallel analysis, potential integration challenges
- **Config B**: Systematic approach with clear documentation
- **Config C**: Deep expertise on each bug type, comprehensive fixes
- **Config D**: Extremely thorough analysis, possible over-engineering

## Test Harness
```python
def test_race_condition():
    """Verify race condition is fixed"""
    processor = DistributedTaskProcessor(num_workers=10)
    task_ids = []
    
    # Submit same task ID from multiple threads
    def submit_duplicate():
        for i in range(100):
            task = Task(id="duplicate_task", payload={"data": i})
            processor.submit_task(task)
    
    threads = [threading.Thread(target=submit_duplicate) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Verify task processed exactly once
    assert processor.processing_count("duplicate_task") == 1

def test_memory_leak():
    """Verify memory leak is fixed"""
    import gc
    import tracemalloc
    
    tracemalloc.start()
    processor = DistributedTaskProcessor()
    
    # Process many tasks
    for i in range(10000):
        task = Task(id=f"task_{i}", payload={"data": i})
        processor.submit_task(task)
    
    # Wait for processing
    time.sleep(5)
    
    # Check memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assert current < 100 * 1024 * 1024  # Less than 100MB
```

## Notes
- Focus on correctness over optimization
- Ensure fixes are minimal and targeted
- Document any assumptions made
- Consider edge cases and error scenarios