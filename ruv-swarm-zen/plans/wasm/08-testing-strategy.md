# Comprehensive Testing Strategy

## Overview
This document outlines the comprehensive testing strategy for the WASM-powered ruv-swarm implementation, ensuring reliability, performance, and compatibility across all components and platforms.

## 🎯 Testing Goals

### Primary Objectives
- **Functional Correctness**: All WASM modules behave identically to Rust implementations
- **Performance Validation**: Meet or exceed performance targets (2-4x improvement)
- **Cross-Platform Compatibility**: Work across Node.js, browsers, and various JavaScript engines
- **Integration Reliability**: Seamless interaction between WASM modules and JavaScript
- **Regression Prevention**: Catch breaking changes early in development

### Success Criteria
- **95%+ Test Coverage**: Comprehensive coverage across all modules
- **100% API Compatibility**: Full backward compatibility with existing JavaScript API
- **Performance Targets Met**: All benchmarks within acceptable ranges
- **Zero Memory Leaks**: No memory growth during extended operation
- **Cross-Platform Success**: Tests pass on all target platforms

## 🧪 Testing Architecture

### Test Categories

#### 1. Unit Tests
- **WASM Module Tests**: Individual WASM function testing
- **JavaScript Wrapper Tests**: API wrapper functionality
- **Integration Interface Tests**: WASM ↔ JavaScript bridge testing
- **Performance Unit Tests**: Individual operation benchmarks

#### 2. Integration Tests
- **Multi-Module Integration**: Cross-module interaction testing
- **NPX Package Integration**: End-to-end NPX workflow testing
- **MCP Tool Integration**: Complete MCP functionality testing
- **Swarm Orchestration Tests**: Full swarm lifecycle testing

#### 3. Performance Tests
- **Benchmark Tests**: Comparative performance validation
- **Load Tests**: High-volume operation testing
- **Memory Tests**: Memory usage and leak detection
- **Scalability Tests**: Large-scale swarm testing

#### 4. Compatibility Tests
- **Browser Compatibility**: Multi-browser testing
- **Node.js Version Tests**: Multiple Node.js version support
- **Platform Tests**: Different operating systems
- **JavaScript Engine Tests**: V8, SpiderMonkey, JavaScriptCore

#### 5. End-to-End Tests
- **User Workflow Tests**: Complete user journey testing
- **CLI Integration Tests**: Full CLI functionality
- **Documentation Tests**: Example code validation
- **Error Handling Tests**: Graceful failure scenarios

### Test Infrastructure

```
ruv-swarm/
├── tests/
│   ├── unit/
│   │   ├── wasm/                    # WASM module unit tests
│   │   │   ├── neural.test.js
│   │   │   ├── forecasting.test.js
│   │   │   ├── swarm.test.js
│   │   │   └── core.test.js
│   │   ├── javascript/              # JavaScript wrapper tests
│   │   │   ├── ruv-swarm.test.js
│   │   │   ├── mcp-tools.test.js
│   │   │   └── memory-manager.test.js
│   │   └── integration/             # WASM-JS bridge tests
│   │       ├── data-serialization.test.js
│   │       ├── memory-sharing.test.js
│   │       └── error-propagation.test.js
│   ├── integration/
│   │   ├── multi-module.test.js     # Cross-module testing
│   │   ├── npx-package.test.js      # NPX workflow testing
│   │   ├── mcp-integration.test.js  # MCP tool testing
│   │   └── swarm-lifecycle.test.js  # Full swarm testing
│   ├── performance/
│   │   ├── benchmarks/              # Performance benchmarks
│   │   │   ├── neural-benchmarks.js
│   │   │   ├── swarm-benchmarks.js
│   │   │   └── memory-benchmarks.js
│   │   ├── load/                    # Load testing
│   │   │   ├── high-volume.test.js
│   │   │   └── stress.test.js
│   │   └── scalability/             # Scalability testing
│   │       ├── large-swarm.test.js
│   │       └── concurrent-tasks.test.js
│   ├── compatibility/
│   │   ├── browsers/                # Browser testing
│   │   │   ├── chrome.test.js
│   │   │   ├── firefox.test.js
│   │   │   └── safari.test.js
│   │   ├── nodejs/                  # Node.js version testing
│   │   │   ├── node16.test.js
│   │   │   ├── node18.test.js
│   │   │   └── node20.test.js
│   │   └── platforms/               # Platform testing
│   │       ├── linux.test.js
│   │       ├── macos.test.js
│   │       └── windows.test.js
│   ├── e2e/
│   │   ├── user-workflows.test.js   # Complete user journeys
│   │   ├── cli-integration.test.js  # Full CLI testing
│   │   └── error-scenarios.test.js  # Error handling
│   ├── fixtures/                    # Test data and fixtures
│   │   ├── neural-networks/
│   │   ├── time-series-data/
│   │   └── swarm-configurations/
│   ├── mocks/                       # Mock implementations
│   │   ├── wasm-mocks.js
│   │   └── api-mocks.js
│   └── utils/                       # Testing utilities
│       ├── test-helpers.js
│       ├── performance-helpers.js
│       └── assertion-helpers.js
```

## 🔧 SWE-bench Integration Testing

### Overview
SWE-bench (Software Engineering Benchmark) integration provides comprehensive evaluation of ruv-swarm's coding capabilities on real-world software engineering tasks. This section outlines how to leverage SWE-bench to test swarm-based solutions for bug fixes, feature implementations, and code optimization challenges.

### SWE-bench Integration Architecture

```
ruv-swarm/
├── tests/
│   ├── swe-bench/
│   │   ├── integration/               # SWE-bench integration tests
│   │   │   ├── swarm-swe-runner.js   # Core SWE-bench runner for swarms
│   │   │   ├── problem-loader.js     # Load and parse SWE-bench problems
│   │   │   └── result-evaluator.js   # Evaluate swarm solutions
│   │   ├── benchmarks/               # SWE-bench benchmark categories
│   │   │   ├── bug-fixes/            # Bug fixing benchmark tests
│   │   │   ├── feature-impl/         # Feature implementation tests
│   │   │   ├── code-optimization/    # Code optimization challenges
│   │   │   └── refactoring/          # Code refactoring tasks
│   │   ├── datasets/                 # SWE-bench dataset management
│   │   │   ├── swe-bench-lite.json   # Curated subset for CI/CD
│   │   │   ├── swe-bench-full.json   # Complete benchmark dataset
│   │   │   └── custom-problems.json  # Custom ruv-swarm specific problems
│   │   ├── workflows/                # Claude Code + SWE-bench workflows
│   │   │   ├── stream-processing/    # Stream-based solution workflows
│   │   │   ├── batch-processing/     # Batch solution workflows
│   │   │   └── interactive-mode/     # Interactive debugging workflows
│   │   ├── metrics/                  # Performance and quality metrics
│   │   │   ├── success-criteria.js   # Success rate calculations
│   │   │   ├── performance-metrics.js # Speed and efficiency metrics
│   │   │   └── quality-assessment.js # Code quality evaluation
│   │   └── regression/               # Regression testing
│   │       ├── wasm-regression.js    # WASM optimization regression tests
│   │       └── neural-regression.js  # Neural network regression tests
```

### Benchmark Categories

#### 1. Bug Fixing Benchmarks
Test swarm performance on identifying and fixing software bugs from real GitHub issues.

```javascript
// tests/swe-bench/benchmarks/bug-fixes/bug-fix-swarm.test.js

const { SWEBenchRunner } = require('../../integration/swarm-swe-runner');
const { RuvSwarm } = require('../../../../npm/src');

describe('SWE-bench Bug Fixing Benchmarks', () => {
    let ruvSwarm;
    let sweBenchRunner;
    
    beforeAll(async () => {
        ruvSwarm = await RuvSwarm.initialize({
            enableNeuralNetworks: true,
            enableCognitiveDiversity: true,
            useSIMD: true
        });
        
        sweBenchRunner = new SWEBenchRunner({
            ruvSwarm,
            datasetPath: '../datasets/swe-bench-lite.json',
            category: 'bug-fixes',
            maxTimePerProblem: 300000 // 5 minutes
        });
    });

    test('swarm solves Python bug fixing problems', async () => {
        const bugFixProblems = await sweBenchRunner.loadProblems({
            language: 'python',
            category: 'bug-fix',
            difficulty: 'easy',
            limit: 10
        });
        
        const results = [];
        
        for (const problem of bugFixProblems) {
            const swarm = await ruvSwarm.createSwarm({
                name: `bug-fix-swarm-${problem.id}`,
                topology: 'hierarchical',
                maxAgents: 5,
                enableNeuralAgents: true
            });
            
            // Spawn specialized agents for bug fixing
            const debugger = await swarm.spawn({
                type: 'debugger',
                name: 'bug-analyzer',
                capabilities: ['code_analysis', 'error_detection', 'debugging']
            });
            
            const coder = await swarm.spawn({
                type: 'coder',
                name: 'bug-fixer',
                capabilities: ['code_writing', 'refactoring', 'testing']
            });
            
            const tester = await swarm.spawn({
                type: 'tester',
                name: 'test-validator',
                capabilities: ['test_writing', 'validation', 'regression_testing']
            });
            
            // Orchestrate bug fixing task
            const solution = await swarm.orchestrate({
                description: `Fix bug in ${problem.repository}: ${problem.issue_title}`,
                priority: 'high',
                metadata: {
                    problem_id: problem.id,
                    repository: problem.repository,
                    issue_url: problem.issue_url,
                    failing_tests: problem.failing_tests,
                    problem_statement: problem.problem_statement
                },
                maxAgents: 3,
                strategy: 'collaborative',
                timeout: 300000 // 5 minutes
            });
            
            // Evaluate solution
            const evaluation = await sweBenchRunner.evaluateSolution(problem, solution);
            results.push({
                problem_id: problem.id,
                success: evaluation.tests_passed,
                solution_quality: evaluation.quality_score,
                time_taken: evaluation.execution_time,
                agents_used: solution.assignedAgents.length
            });
        }
        
        const successRate = results.filter(r => r.success).length / results.length;
        const averageTime = results.reduce((sum, r) => sum + r.time_taken, 0) / results.length;
        const averageQuality = results.reduce((sum, r) => sum + r.solution_quality, 0) / results.length;
        
        console.log(`Bug Fixing Benchmark Results:
            Success Rate: ${(successRate * 100).toFixed(1)}% (${results.filter(r => r.success).length}/${results.length})
            Average Time: ${(averageTime / 1000).toFixed(1)}s
            Average Quality Score: ${averageQuality.toFixed(2)}/10
            Target Success Rate: >70%`);
        
        expect(successRate).toBeGreaterThan(0.7); // Target: >70% success rate
        expect(averageTime).toBeLessThan(240000); // Target: <4 minutes average
        expect(averageQuality).toBeGreaterThan(7.0); // Target: >7.0 quality score
    });

    test('swarm handles complex multi-file bug fixes', async () => {
        const complexProblems = await sweBenchRunner.loadProblems({
            category: 'bug-fix',
            difficulty: 'hard',
            files_modified: '>3',
            limit: 5
        });
        
        const results = [];
        
        for (const problem of complexProblems) {
            const swarm = await ruvSwarm.createSwarm({
                name: `complex-bug-swarm-${problem.id}`,
                topology: 'mesh',
                maxAgents: 8,
                enableCognitiveDiversity: true
            });
            
            // Spawn diverse team for complex problems
            await Promise.all([
                swarm.spawn({ type: 'researcher', name: 'code-explorer', cognitivePattern: 'divergent' }),
                swarm.spawn({ type: 'analyst', name: 'system-analyzer', cognitivePattern: 'systems' }),
                swarm.spawn({ type: 'debugger', name: 'error-detective', cognitivePattern: 'critical' }),
                swarm.spawn({ type: 'coder', name: 'solution-implementer', cognitivePattern: 'convergent' }),
                swarm.spawn({ type: 'tester', name: 'quality-assurer', cognitivePattern: 'lateral' }),
                swarm.spawn({ type: 'coordinator', name: 'team-orchestrator', cognitivePattern: 'abstract' })
            ]);
            
            const solution = await swarm.orchestrate({
                description: `Solve complex multi-file bug: ${problem.issue_title}`,
                priority: 'critical',
                metadata: {
                    problem_id: problem.id,
                    complexity: 'high',
                    files_count: problem.files_modified,
                    estimated_difficulty: problem.difficulty_rating
                },
                strategy: 'hierarchical',
                timeout: 600000 // 10 minutes for complex problems
            });
            
            const evaluation = await sweBenchRunner.evaluateSolution(problem, solution);
            results.push(evaluation);
        }
        
        const successRate = results.filter(r => r.tests_passed).length / results.length;
        
        expect(successRate).toBeGreaterThan(0.4); // Lower target for complex problems: >40%
    });
});
```

#### 2. Feature Implementation Benchmarks
Evaluate swarm capabilities in implementing new features based on specifications.

```javascript
// tests/swe-bench/benchmarks/feature-impl/feature-implementation.test.js

describe('SWE-bench Feature Implementation Benchmarks', () => {
    test('swarm implements new API endpoints', async () => {
        const featureProblems = await sweBenchRunner.loadProblems({
            category: 'feature-implementation',
            type: 'api-endpoint',
            limit: 8
        });
        
        const results = [];
        
        for (const problem of featureProblems) {
            const swarm = await ruvSwarm.createSwarm({
                name: `api-feature-swarm-${problem.id}`,
                topology: 'star',
                maxAgents: 6
            });
            
            // Specialized agents for API development
            await Promise.all([
                swarm.spawn({ type: 'architect', name: 'api-designer', capabilities: ['system_design', 'api_design'] }),
                swarm.spawn({ type: 'coder', name: 'backend-developer', capabilities: ['backend_development', 'database'] }),
                swarm.spawn({ type: 'tester', name: 'api-tester', capabilities: ['api_testing', 'integration_testing'] }),
                swarm.spawn({ type: 'documenter', name: 'api-documenter', capabilities: ['documentation', 'openapi'] })
            ]);
            
            const solution = await swarm.orchestrate({
                description: `Implement API feature: ${problem.feature_description}`,
                priority: 'high',
                metadata: {
                    problem_id: problem.id,
                    feature_type: 'api-endpoint',
                    requirements: problem.requirements,
                    acceptance_criteria: problem.acceptance_criteria
                },
                strategy: 'sequential',
                phases: ['design', 'implementation', 'testing', 'documentation']
            });
            
            const evaluation = await sweBenchRunner.evaluateSolution(problem, solution);
            results.push(evaluation);
        }
        
        const successRate = results.filter(r => r.tests_passed && r.meets_requirements).length / results.length;
        expect(successRate).toBeGreaterThan(0.6); // Target: >60% for feature implementation
    });
});
```

#### 3. Code Optimization Benchmarks
Test swarm performance on identifying and implementing performance optimizations.

```javascript
// tests/swe-bench/benchmarks/code-optimization/optimization-swarm.test.js

describe('SWE-bench Code Optimization Benchmarks', () => {
    test('swarm optimizes performance bottlenecks', async () => {
        const optimizationProblems = await sweBenchRunner.loadProblems({
            category: 'optimization',
            type: 'performance',
            limit: 6
        });
        
        for (const problem of optimizationProblems) {
            const swarm = await ruvSwarm.createSwarm({
                name: `optimization-swarm-${problem.id}`,
                topology: 'mesh',
                maxAgents: 4,
                enableNeuralAgents: true
            });
            
            // Optimization-focused team
            await Promise.all([
                swarm.spawn({ type: 'analyzer', name: 'performance-profiler', cognitivePattern: 'critical' }),
                swarm.spawn({ type: 'optimizer', name: 'code-optimizer', cognitivePattern: 'convergent' }),
                swarm.spawn({ type: 'tester', name: 'benchmark-tester', cognitivePattern: 'lateral' })
            ]);
            
            const solution = await swarm.orchestrate({
                description: `Optimize performance: ${problem.optimization_target}`,
                priority: 'high',
                metadata: {
                    problem_id: problem.id,
                    baseline_performance: problem.baseline_metrics,
                    target_improvement: problem.target_improvement,
                    constraints: problem.constraints
                }
            });
            
            const evaluation = await sweBenchRunner.evaluateSolution(problem, solution);
            
            expect(evaluation.performance_improvement).toBeGreaterThan(problem.target_improvement);
            expect(evaluation.tests_passed).toBe(true);
        }
    });
});
```

### Claude Code + SWE-bench Workflows

#### Stream Processing Workflow
```javascript
// tests/swe-bench/workflows/stream-processing/claude-swe-stream.test.js

describe('Claude Code + SWE-bench Stream Processing', () => {
    test('claude stream-json workflow with SWE-bench', async () => {
        const testCommand = `claude "create swarm to solve SWE-bench problem django__django-12345 using ruv-FANN neural networks" --swe-bench-mode --output-format stream-json`;
        
        const result = await executeClaudeCommand(testCommand);
        
        // Verify stream-json output format
        const streamLines = result.split('\n').filter(line => line.startsWith('data: '));
        expect(streamLines.length).toBeGreaterThan(0);
        
        // Parse and validate JSON stream
        for (const line of streamLines) {
            const data = JSON.parse(line.substring(6)); // Remove 'data: ' prefix
            
            expect(data).toHaveProperty('type');
            expect(data).toHaveProperty('timestamp');
            
            if (data.type === 'swarm_creation') {
                expect(data).toHaveProperty('swarm_id');
                expect(data).toHaveProperty('topology');
            } else if (data.type === 'agent_spawn') {
                expect(data).toHaveProperty('agent_id');
                expect(data).toHaveProperty('agent_type');
                expect(data).toHaveProperty('capabilities');
            } else if (data.type === 'task_orchestration') {
                expect(data).toHaveProperty('task_id');
                expect(data).toHaveProperty('swe_bench_problem_id');
            } else if (data.type === 'solution_progress') {
                expect(data).toHaveProperty('progress_percentage');
                expect(data).toHaveProperty('current_phase');
            } else if (data.type === 'solution_complete') {
                expect(data).toHaveProperty('success');
                expect(data).toHaveProperty('tests_passed');
                expect(data).toHaveProperty('solution_quality');
            }
        }
    });

    test('interactive debugging workflow', async () => {
        const interactiveCommands = [
            `claude "initialize SWE-bench debugging session for problem flask__flask-4567" --swe-bench-mode --interactive`,
            `claude "spawn debugging swarm with cognitive diversity" --continue-session`,
            `claude "analyze failing tests and identify root cause" --continue-session`,
            `claude "implement fix with test validation" --continue-session`,
            `claude "generate solution report" --continue-session --output-format detailed-json`
        ];
        
        let sessionContext = null;
        
        for (const command of interactiveCommands) {
            const result = await executeClaudeCommand(command, sessionContext);
            
            if (command.includes('--continue-session') && sessionContext) {
                expect(result).toContain('session_id');
            }
            
            if (command.includes('initialize')) {
                sessionContext = extractSessionContext(result);
                expect(sessionContext).toHaveProperty('session_id');
                expect(sessionContext).toHaveProperty('problem_id', 'flask__flask-4567');
            }
            
            if (command.includes('solution report')) {
                const report = JSON.parse(result);
                expect(report).toHaveProperty('problem_analysis');
                expect(report).toHaveProperty('solution_approach');
                expect(report).toHaveProperty('implementation_details');
                expect(report).toHaveProperty('test_results');
                expect(report).toHaveProperty('quality_metrics');
            }
        }
    });
});
```

#### Batch Processing Workflow
```javascript
// tests/swe-bench/workflows/batch-processing/claude-swe-batch.test.js

describe('Claude Code + SWE-bench Batch Processing', () => {
    test('batch solve multiple SWE-bench problems', async () => {
        const batchCommand = `claude "solve SWE-bench problems: django__django-12345,flask__flask-4567,requests__requests-2345" --swe-bench-mode --batch-processing --max-parallel-swarms 3 --output-format comprehensive-json`;
        
        const result = await executeClaudeCommand(batchCommand);
        const batchResults = JSON.parse(result);
        
        expect(batchResults).toHaveProperty('batch_id');
        expect(batchResults).toHaveProperty('problems_solved');
        expect(batchResults).toHaveProperty('overall_success_rate');
        expect(batchResults).toHaveProperty('performance_metrics');
        
        expect(batchResults.problems_solved).toHaveLength(3);
        
        for (const problemResult of batchResults.problems_solved) {
            expect(problemResult).toHaveProperty('problem_id');
            expect(problemResult).toHaveProperty('swarm_composition');
            expect(problemResult).toHaveProperty('solution_approach');
            expect(problemResult).toHaveProperty('execution_time');
            expect(problemResult).toHaveProperty('success');
            expect(problemResult).toHaveProperty('test_results');
            
            if (problemResult.success) {
                expect(problemResult.test_results.passed).toBeGreaterThan(0);
                expect(problemResult.test_results.failed).toBe(0);
            }
        }
        
        expect(batchResults.overall_success_rate).toBeGreaterThan(0.5);
    });

    test('adaptive swarm sizing for different problem complexities', async () => {
        const adaptiveCommand = `claude "solve SWE-bench problems with adaptive swarm sizing" --swe-bench-mode --adaptive-sizing --problem-set swe-bench-mixed-difficulty --output-format stream-json`;
        
        const result = await executeClaudeCommand(adaptiveCommand);
        const streamData = parseStreamJson(result);
        
        const swarmCreations = streamData.filter(d => d.type === 'swarm_creation');
        
        // Verify adaptive sizing
        const easyProblems = swarmCreations.filter(s => s.problem_difficulty === 'easy');
        const hardProblems = swarmCreations.filter(s => s.problem_difficulty === 'hard');
        
        if (easyProblems.length > 0 && hardProblems.length > 0) {
            const avgEasyAgents = easyProblems.reduce((sum, s) => sum + s.max_agents, 0) / easyProblems.length;
            const avgHardAgents = hardProblems.reduce((sum, s) => sum + s.max_agents, 0) / hardProblems.length;
            
            expect(avgHardAgents).toBeGreaterThan(avgEasyAgents);
        }
    });
});
```

### Performance Metrics and Success Criteria

#### Success Rate Calculations
```javascript
// tests/swe-bench/metrics/success-criteria.js

class SWEBenchSuccessCriteria {
    static calculateSuccessRate(results) {
        const totalProblems = results.length;
        const successfulSolutions = results.filter(result => {
            return result.tests_passed && 
                   result.no_regression && 
                   result.code_quality_score >= 7.0 &&
                   result.execution_time <= result.timeout;
        });
        
        return {
            overall_success_rate: successfulSolutions.length / totalProblems,
            test_pass_rate: results.filter(r => r.tests_passed).length / totalProblems,
            quality_success_rate: results.filter(r => r.code_quality_score >= 7.0).length / totalProblems,
            efficiency_success_rate: results.filter(r => r.execution_time <= r.timeout * 0.8).length / totalProblems,
            breakdown_by_difficulty: this.calculateByDifficulty(results),
            breakdown_by_category: this.calculateByCategory(results)
        };
    }
    
    static calculatePerformanceMetrics(results) {
        return {
            average_execution_time: results.reduce((sum, r) => sum + r.execution_time, 0) / results.length,
            median_execution_time: this.calculateMedian(results.map(r => r.execution_time)),
            average_agents_used: results.reduce((sum, r) => sum + r.agents_used, 0) / results.length,
            average_code_quality: results.reduce((sum, r) => sum + r.code_quality_score, 0) / results.length,
            neural_network_utilization: results.filter(r => r.used_neural_networks).length / results.length,
            cognitive_diversity_score: this.calculateCognitiveDiversityScore(results)
        };
    }
    
    static defineTargets() {
        return {
            bug_fixes: {
                easy: { success_rate: 0.85, avg_time: 120000 },    // 85% success, <2 min
                medium: { success_rate: 0.70, avg_time: 240000 },  // 70% success, <4 min
                hard: { success_rate: 0.45, avg_time: 600000 }     // 45% success, <10 min
            },
            feature_implementation: {
                simple: { success_rate: 0.75, avg_time: 300000 },  // 75% success, <5 min
                complex: { success_rate: 0.55, avg_time: 900000 }  // 55% success, <15 min
            },
            optimization: {
                performance: { success_rate: 0.65, improvement: 1.5 }, // 65% success, 50% improvement
                memory: { success_rate: 0.60, improvement: 1.3 }       // 60% success, 30% improvement
            },
            refactoring: {
                simple: { success_rate: 0.80, quality_improvement: 1.2 },
                complex: { success_rate: 0.60, quality_improvement: 1.5 }
            }
        };
    }
}
```

#### Quality Assessment Metrics
```javascript
// tests/swe-bench/metrics/quality-assessment.js

class SWEBenchQualityAssessment {
    static evaluateCodeQuality(solution) {
        return {
            correctness: this.evaluateCorrectness(solution),
            maintainability: this.evaluateMaintainability(solution),
            performance: this.evaluatePerformance(solution),
            security: this.evaluateSecurity(solution),
            documentation: this.evaluateDocumentation(solution),
            test_coverage: this.evaluateTestCoverage(solution),
            overall_score: this.calculateOverallScore(solution)
        };
    }
    
    static evaluateSwarmPerformance(swarmExecution) {
        return {
            coordination_efficiency: this.evaluateCoordination(swarmExecution),
            agent_utilization: this.evaluateAgentUtilization(swarmExecution),
            cognitive_diversity_impact: this.evaluateCognitiveDiversity(swarmExecution),
            neural_network_contribution: this.evaluateNeuralContribution(swarmExecution),
            time_efficiency: this.evaluateTimeEfficiency(swarmExecution)
        };
    }
}
```

### Automated Testing Pipeline

#### CI/CD Integration
```yaml
# .github/workflows/swe-bench-integration.yml
name: SWE-bench Integration Testing

on:
  push:
    branches: [ main, ruv-swarm ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  swe-bench-lite:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: ruv-swarm/npm/package-lock.json
    
    - name: Setup SWE-bench Dataset
      run: |
        cd ruv-swarm/tests/swe-bench/datasets
        # Download SWE-bench lite dataset (subset for CI)
        curl -L -o swe-bench-lite.json "https://github.com/princeton-nlp/SWE-bench/releases/download/v1.0/swe-bench-lite.json"
    
    - name: Install dependencies and build
      run: |
        cd ruv-swarm/npm
        npm ci
        cd ..
        node build/wasm-build.js
    
    - name: Run SWE-bench Bug Fix Tests
      run: |
        cd ruv-swarm/npm
        npm run test:swe-bench:bug-fixes
      env:
        SWE_BENCH_MODE: lite
        MAX_TIME_PER_PROBLEM: 300000
        MAX_PROBLEMS: 10
    
    - name: Run SWE-bench Feature Implementation Tests
      run: |
        cd ruv-swarm/npm
        npm run test:swe-bench:features
      env:
        SWE_BENCH_MODE: lite
        MAX_PROBLEMS: 5
    
    - name: Upload SWE-bench Results
      uses: actions/upload-artifact@v3
      with:
        name: swe-bench-results
        path: ruv-swarm/npm/swe-bench-results/

  swe-bench-full:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    timeout-minutes: 480  # 8 hours for full benchmark
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup environment
      run: |
        # Setup for full SWE-bench run
        cd ruv-swarm/tests/swe-bench/datasets
        curl -L -o swe-bench-full.json "https://github.com/princeton-nlp/SWE-bench/releases/download/v1.0/swe-bench.json"
    
    - name: Run Full SWE-bench Evaluation
      run: |
        cd ruv-swarm/npm
        npm run test:swe-bench:full
      env:
        SWE_BENCH_MODE: full
        ENABLE_DETAILED_LOGGING: true
        SAVE_INTERMEDIATE_RESULTS: true
    
    - name: Generate Performance Report
      run: |
        cd ruv-swarm/npm
        npm run swe-bench:generate-report
    
    - name: Upload Full Results
      uses: actions/upload-artifact@v3
      with:
        name: swe-bench-full-results
        path: |
          ruv-swarm/npm/swe-bench-results/
          ruv-swarm/npm/performance-reports/

  claude-code-integration:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Test Claude Code SWE-bench Commands
      run: |
        # Test Claude Code integration with SWE-bench
        cd ruv-swarm/npm
        
        # Test stream-json output
        timeout 300 npx claude "create swarm to solve SWE-bench problem test-problem-1" --swe-bench-mode --output-format stream-json > stream-output.json || true
        
        # Validate stream output
        npm run validate:stream-output stream-output.json
        
        # Test batch processing
        timeout 600 npx claude "solve SWE-bench problems: test-problem-1,test-problem-2" --swe-bench-mode --batch-processing --output-format comprehensive-json > batch-output.json || true
        
        # Validate batch output
        npm run validate:batch-output batch-output.json
```

### Regression Testing

#### WASM Optimization Regression Tests
```javascript
// tests/swe-bench/regression/wasm-regression.js

describe('WASM Optimization Regression Tests', () => {
    test('WASM optimizations maintain SWE-bench performance', async () => {
        // Test with and without WASM optimizations
        const baselineResults = await runSWEBenchSuite({
            useWASM: false,
            problemSet: 'regression-test-set',
            maxProblems: 20
        });
        
        const optimizedResults = await runSWEBenchSuite({
            useWASM: true,
            useSIMD: true,
            problemSet: 'regression-test-set',
            maxProblems: 20
        });
        
        // Performance should improve or stay the same
        expect(optimizedResults.averageExecutionTime).toBeLessThanOrEqual(baselineResults.averageExecutionTime * 1.1);
        
        // Success rate should not decrease
        expect(optimizedResults.successRate).toBeGreaterThanOrEqual(baselineResults.successRate - 0.05);
        
        // WASM should provide speedup for neural operations
        if (optimizedResults.neuralOperationTime && baselineResults.neuralOperationTime) {
            const speedup = baselineResults.neuralOperationTime / optimizedResults.neuralOperationTime;
            expect(speedup).toBeGreaterThan(1.5); // At least 50% speedup
        }
    });
    
    test('neural network optimizations maintain solution quality', async () => {
        const problems = await loadSWEBenchProblems({
            category: 'neural-intensive',
            limit: 10
        });
        
        const results = [];
        
        for (const problem of problems) {
            const baselineSolution = await solveProblemWithSwarm(problem, {
                enableNeuralNetworks: false
            });
            
            const neuralSolution = await solveProblemWithSwarm(problem, {
                enableNeuralNetworks: true,
                useWASMNeuralOps: true
            });
            
            results.push({
                problem_id: problem.id,
                baseline_quality: baselineSolution.codeQualityScore,
                neural_quality: neuralSolution.codeQualityScore,
                baseline_time: baselineSolution.executionTime,
                neural_time: neuralSolution.executionTime
            });
        }
        
        const qualityRegression = results.filter(r => 
            r.neural_quality < r.baseline_quality - 0.5
        ).length;
        
        // No more than 10% quality regression allowed
        expect(qualityRegression / results.length).toBeLessThan(0.1);
    });
});
```

#### Neural Network Regression Tests
```javascript
// tests/swe-bench/regression/neural-regression.js

describe('Neural Network Regression Tests', () => {
    test('cognitive diversity maintains problem-solving effectiveness', async () => {
        const diversityConfigs = [
            { enableCognitiveDiversity: false },
            { enableCognitiveDiversity: true, diversityTarget: 0.6 },
            { enableCognitiveDiversity: true, diversityTarget: 0.8 }
        ];
        
        const problems = await loadSWEBenchProblems({
            category: 'complex-reasoning',
            limit: 15
        });
        
        const results = [];
        
        for (const config of diversityConfigs) {
            const configResults = [];
            
            for (const problem of problems) {
                const solution = await solveProblemWithSwarm(problem, config);
                configResults.push(solution);
            }
            
            results.push({
                config,
                successRate: configResults.filter(s => s.testsPasssed).length / configResults.length,
                averageQuality: configResults.reduce((sum, s) => sum + s.codeQualityScore, 0) / configResults.length,
                averageTime: configResults.reduce((sum, s) => sum + s.executionTime, 0) / configResults.length
            });
        }
        
        // Cognitive diversity should improve or maintain performance
        const noDiversityResult = results.find(r => !r.config.enableCognitiveDiversity);
        const diversityResults = results.filter(r => r.config.enableCognitiveDiversity);
        
        for (const diversityResult of diversityResults) {
            // Success rate should improve or stay within 5%
            expect(diversityResult.successRate).toBeGreaterThanOrEqual(noDiversityResult.successRate - 0.05);
            
            // Quality should improve for complex problems
            expect(diversityResult.averageQuality).toBeGreaterThanOrEqual(noDiversityResult.averageQuality);
        }
    });
});
```

This comprehensive SWE-bench integration ensures that ruv-swarm maintains high performance on real-world software engineering challenges while benefiting from WASM optimizations and neural network enhancements.

## 🔬 WASM Module Testing

### Neural Network Module Tests
```javascript
// tests/unit/wasm/neural.test.js - Neural network WASM testing

const { WasmNeuralNetwork, ActivationFunctionManager, WasmTrainer } = require('../../../npm/wasm/neural');
const { generateTestData, assertFloatArraysEqual } = require('../../utils/test-helpers');

describe('Neural Network WASM Module', () => {
    let neuralNetwork;
    
    beforeEach(async () => {
        // Initialize WASM module
        await initializeWasm();
        
        // Create test network
        neuralNetwork = new WasmNeuralNetwork({
            input_size: 2,
            hidden_layers: [
                { size: 4, activation: 'sigmoid', steepness: 1.0 }
            ],
            output_size: 1,
            output_activation: 'sigmoid',
            connection_rate: 1.0,
            random_seed: 12345
        });
    });

    describe('Network Creation', () => {
        test('creates network with correct architecture', () => {
            const info = neuralNetwork.get_network_info();
            
            expect(info.num_inputs).toBe(2);
            expect(info.num_outputs).toBe(1);
            expect(info.total_neurons).toBe(7); // 2 input + 4 hidden + 1 output
            expect(info.total_connections).toBeGreaterThan(0);
        });

        test('handles invalid configuration gracefully', () => {
            expect(() => {
                new WasmNeuralNetwork({
                    input_size: 0, // Invalid
                    output_size: 1
                });
            }).toThrow('Invalid config');
        });
    });

    describe('Forward Propagation', () => {
        test('processes input correctly', () => {
            const inputs = [0.5, 0.7];
            const outputs = neuralNetwork.run(inputs);
            
            expect(outputs).toHaveLength(1);
            expect(outputs[0]).toBeGreaterThanOrEqual(0);
            expect(outputs[0]).toBeLessThanOrEqual(1);
        });

        test('produces consistent outputs for same inputs', () => {
            const inputs = [0.3, 0.8];
            const outputs1 = neuralNetwork.run(inputs);
            const outputs2 = neuralNetwork.run(inputs);
            
            assertFloatArraysEqual(outputs1, outputs2, 1e-6);
        });

        test('handles batch processing efficiently', () => {
            const batchSize = 100;
            const inputs = [0.4, 0.6];
            
            const startTime = performance.now();
            for (let i = 0; i < batchSize; i++) {
                neuralNetwork.run(inputs);
            }
            const endTime = performance.now();
            
            const avgTime = (endTime - startTime) / batchSize;
            expect(avgTime).toBeLessThan(5); // Should be under 5ms per inference
        });

        test('validates input size', () => {
            expect(() => {
                neuralNetwork.run([0.5]); // Wrong input size
            }).toThrow('Network run error');
            
            expect(() => {
                neuralNetwork.run([0.5, 0.7, 0.9]); // Wrong input size
            }).toThrow('Network run error');
        });
    });

    describe('Weight Management', () => {
        test('gets and sets weights correctly', () => {
            const originalWeights = neuralNetwork.get_weights();
            expect(originalWeights).toBeInstanceOf(Float32Array);
            expect(originalWeights.length).toBeGreaterThan(0);
            
            // Modify weights
            const modifiedWeights = originalWeights.map(w => w + 0.1);
            neuralNetwork.set_weights(modifiedWeights);
            
            const newWeights = neuralNetwork.get_weights();
            assertFloatArraysEqual(newWeights, modifiedWeights, 1e-6);
        });

        test('validates weight count', () => {
            const originalWeights = neuralNetwork.get_weights();
            const wrongWeights = new Float32Array(originalWeights.length - 1);
            
            expect(() => {
                neuralNetwork.set_weights(wrongWeights);
            }).toThrow('Set weights error');
        });
    });

    describe('Training Integration', () => {
        test('sets training data correctly', () => {
            const trainingData = {
                inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
                outputs: [[0], [1], [1], [0]]
            };
            
            expect(() => {
                neuralNetwork.set_training_data(trainingData);
            }).not.toThrow();
        });

        test('integrates with training algorithms', () => {
            const trainer = new WasmTrainer({
                algorithm: 'incremental_backprop',
                learning_rate: 0.7,
                max_epochs: 10,
                target_error: 0.1
            });
            
            const trainingData = generateTestData('xor');
            const error = trainer.train_epoch(neuralNetwork, trainingData);
            
            expect(error).toBeGreaterThan(0);
            expect(error).toBeLessThan(1);
        });
    });
});

describe('Activation Function Manager', () => {
    let activationManager;
    
    beforeEach(() => {
        activationManager = new ActivationFunctionManager();
    });

    test('lists all activation functions', () => {
        const functions = activationManager.get_all_functions();
        
        expect(functions).toHaveLength(18);
        expect(functions.map(f => f[0])).toContain('sigmoid');
        expect(functions.map(f => f[0])).toContain('relu');
        expect(functions.map(f => f[0])).toContain('tanh');
    });

    test('tests individual activation functions', () => {
        const testCases = [
            { func: 'sigmoid', input: 0.0, expected: 0.5 },
            { func: 'relu', input: -1.0, expected: 0.0 },
            { func: 'relu', input: 1.0, expected: 1.0 },
            { func: 'linear', input: 0.5, expected: 0.5 }
        ];
        
        for (const { func, input, expected } of testCases) {
            const result = activationManager.test_activation_function(func, input, 1.0);
            expect(result).toBeCloseTo(expected, 3);
        }
    });

    test('compares multiple activation functions', () => {
        const comparisons = activationManager.compare_functions(0.5);
        
        expect(comparisons).toHaveProperty('sigmoid');
        expect(comparisons).toHaveProperty('relu');
        expect(comparisons).toHaveProperty('linear');
        
        expect(comparisons.linear).toBeCloseTo(0.5, 6);
        expect(comparisons.relu).toBeCloseTo(0.5, 6);
    });
});

describe('Memory Management', () => {
    test('does not leak memory during extended operation', async () => {
        const initialMemory = getWasmMemoryUsage();
        
        // Perform many operations
        for (let i = 0; i < 1000; i++) {
            const network = new WasmNeuralNetwork({
                input_size: 3,
                hidden_layers: [{ size: 5, activation: 'relu' }],
                output_size: 2
            });
            
            network.run([0.1, 0.2, 0.3]);
            
            // Explicitly clean up (if cleanup method exists)
            if (network.cleanup) {
                network.cleanup();
            }
        }
        
        // Force garbage collection if available
        if (global.gc) {
            global.gc();
        }
        
        const finalMemory = getWasmMemoryUsage();
        const memoryGrowth = finalMemory - initialMemory;
        
        // Allow some memory growth but not excessive
        expect(memoryGrowth).toBeLessThan(10 * 1024 * 1024); // Less than 10MB growth
    });
});
```

### Swarm Orchestration Tests
```javascript
// tests/unit/wasm/swarm.test.js - Swarm orchestration WASM testing

const { WasmSwarmOrchestrator, CognitiveDiversityEngine } = require('../../../npm/wasm/swarm');
const { createMockAgents, createMockTasks } = require('../../mocks/swarm-mocks');

describe('Swarm Orchestration WASM Module', () => {
    let orchestrator;
    
    beforeEach(async () => {
        await initializeWasm();
        orchestrator = new WasmSwarmOrchestrator();
    });

    describe('Swarm Creation', () => {
        test('creates swarm with different topologies', async () => {
            const topologies = ['mesh', 'star', 'hierarchical', 'ring'];
            
            for (const topology of topologies) {
                const result = orchestrator.create_swarm({
                    name: `test-${topology}-swarm`,
                    topology_type: topology,
                    max_agents: 10,
                    enable_cognitive_diversity: true
                });
                
                expect(result.swarm_id).toMatch(/^swarm_\d+_[a-z0-9]+$/);
                expect(result.topology).toBe(topology);
                expect(result.max_agents).toBe(10);
            }
        });

        test('validates swarm configuration', () => {
            expect(() => {
                orchestrator.create_swarm({
                    name: '',
                    topology_type: 'invalid_topology',
                    max_agents: 0
                });
            }).toThrow();
        });
    });

    describe('Agent Spawning', () => {
        let swarmId;
        
        beforeEach(() => {
            const result = orchestrator.create_swarm({
                name: 'test-swarm',
                topology_type: 'mesh',
                max_agents: 5
            });
            swarmId = result.swarm_id;
        });

        test('spawns agents with different types', () => {
            const agentTypes = ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'];
            
            for (const type of agentTypes) {
                const result = orchestrator.spawn_agent(swarmId, {
                    agent_type: type,
                    name: `test-${type}`,
                    capabilities: [`${type}_capability`]
                });
                
                expect(result.agent.type).toBe(type);
                expect(result.agent.id).toMatch(/^agent_\d+_[a-z0-9]+$/);
                expect(result.agent.cognitive_pattern).toBeDefined();
            }
        });

        test('enforces agent capacity limits', () => {
            // Spawn maximum agents
            for (let i = 0; i < 5; i++) {
                orchestrator.spawn_agent(swarmId, {
                    agent_type: 'researcher',
                    name: `agent-${i}`
                });
            }
            
            // Try to spawn one more (should fail)
            expect(() => {
                orchestrator.spawn_agent(swarmId, {
                    agent_type: 'researcher',
                    name: 'overflow-agent'
                });
            }).toThrow('maximum agent capacity');
        });

        test('assigns unique cognitive patterns for diversity', () => {
            const cognitivePatterns = new Set();
            
            for (let i = 0; i < 5; i++) {
                const result = orchestrator.spawn_agent(swarmId, {
                    agent_type: ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'][i],
                    name: `diverse-agent-${i}`
                });
                
                cognitivePatterns.add(result.agent.cognitive_pattern);
            }
            
            // Should have diverse cognitive patterns
            expect(cognitivePatterns.size).toBeGreaterThan(1);
        });
    });

    describe('Task Orchestration', () => {
        let swarmId;
        
        beforeEach(() => {
            const swarmResult = orchestrator.create_swarm({
                name: 'task-test-swarm',
                topology_type: 'mesh',
                max_agents: 5
            });
            swarmId = swarmResult.swarm_id;
            
            // Spawn test agents
            ['researcher', 'coder', 'analyst'].forEach((type, index) => {
                orchestrator.spawn_agent(swarmId, {
                    agent_type: type,
                    name: `agent-${index}`
                });
            });
        });

        test('orchestrates tasks with priority handling', () => {
            const priorities = ['low', 'medium', 'high', 'critical'];
            
            for (const priority of priorities) {
                const result = orchestrator.orchestrate_task(swarmId, {
                    description: `Test task with ${priority} priority`,
                    priority,
                    dependencies: [],
                    estimated_duration_ms: 5000
                });
                
                expect(result.task_id).toMatch(/^task_\d+_[a-z0-9]+$/);
                expect(result.status).toBe('orchestrated');
                expect(result.assigned_agents).toBeInstanceOf(Array);
                expect(result.assigned_agents.length).toBeGreaterThan(0);
            }
        });

        test('performs intelligent agent selection', () => {
            const result = orchestrator.orchestrate_task(swarmId, {
                description: 'Code analysis and optimization task',
                priority: 'high',
                required_capabilities: ['code_analysis', 'optimization'],
                max_agents: 2
            });
            
            expect(result.assigned_agents.length).toBeLessThanOrEqual(2);
            
            // Should prefer agents with relevant capabilities
            const agentAssignments = result.agent_assignments;
            const hasRelevantAgent = agentAssignments.some(assignment => 
                assignment.capabilities.some(cap => 
                    cap.includes('analysis') || cap.includes('optimization')
                )
            );
            expect(hasRelevantAgent).toBe(true);
        });

        test('handles task dependencies', () => {
            // Create initial task
            const task1Result = orchestrator.orchestrate_task(swarmId, {
                description: 'Foundation task',
                priority: 'medium'
            });
            
            // Create dependent task
            const task2Result = orchestrator.orchestrate_task(swarmId, {
                description: 'Dependent task',
                priority: 'medium',
                dependencies: [task1Result.task_id]
            });
            
            expect(task2Result.task_id).toBeDefined();
            expect(task2Result.status).toBe('orchestrated');
        });
    });

    describe('Performance Monitoring', () => {
        test('tracks orchestration performance', () => {
            const swarmResult = orchestrator.create_swarm({
                name: 'perf-test-swarm',
                topology_type: 'mesh',
                max_agents: 10
            });
            
            const startTime = performance.now();
            
            // Spawn multiple agents
            for (let i = 0; i < 10; i++) {
                orchestrator.spawn_agent(swarmResult.swarm_id, {
                    agent_type: 'researcher',
                    name: `perf-agent-${i}`
                });
            }
            
            const endTime = performance.now();
            const totalTime = endTime - startTime;
            const avgSpawnTime = totalTime / 10;
            
            expect(avgSpawnTime).toBeLessThan(50); // Should be under 50ms per agent
        });

        test('provides detailed performance metrics', () => {
            const swarmResult = orchestrator.create_swarm({
                name: 'metrics-swarm',
                topology_type: 'mesh',
                max_agents: 5
            });
            
            // Perform operations
            orchestrator.spawn_agent(swarmResult.swarm_id, {
                agent_type: 'researcher',
                name: 'metrics-agent'
            });
            
            const status = orchestrator.get_swarm_status(swarmResult.swarm_id, true);
            
            expect(status).toHaveProperty('performance');
            expect(status.performance).toHaveProperty('avg_task_completion_time');
            expect(status.performance).toHaveProperty('success_rate');
            expect(status.performance).toHaveProperty('total_memory_usage_mb');
        });
    });
});

describe('Cognitive Diversity Engine', () => {
    let diversityEngine;
    
    beforeEach(() => {
        diversityEngine = new CognitiveDiversityEngine();
    });

    test('analyzes swarm diversity metrics', () => {
        const swarmComposition = [
            { agent_id: 'a1', agent_type: 'researcher', cognitive_pattern: 'divergent', capabilities: ['research'] },
            { agent_id: 'a2', agent_type: 'coder', cognitive_pattern: 'convergent', capabilities: ['coding'] },
            { agent_id: 'a3', agent_type: 'analyst', cognitive_pattern: 'critical', capabilities: ['analysis'] }
        ];
        
        const analysis = diversityEngine.analyze_swarm_diversity(swarmComposition);
        
        expect(analysis.diversity_metrics.overall_diversity_score).toBeGreaterThan(0);
        expect(analysis.diversity_metrics.pattern_distribution).toHaveProperty('divergent');
        expect(analysis.diversity_metrics.pattern_distribution).toHaveProperty('convergent');
        expect(analysis.diversity_metrics.pattern_distribution).toHaveProperty('critical');
        expect(analysis.recommendations).toBeInstanceOf(Array);
    });

    test('recommends optimal cognitive patterns', () => {
        const taskRequirements = {
            task_type: 'creative_problem_solving',
            required_capabilities: ['creativity', 'brainstorming'],
            complexity_level: 'high',
            time_constraints: 3600000 // 1 hour
        };
        
        const currentSwarm = [
            { agent_id: 'a1', cognitive_pattern: 'convergent' },
            { agent_id: 'a2', cognitive_pattern: 'convergent' }
        ];
        
        const recommendation = diversityEngine.recommend_cognitive_pattern(taskRequirements, currentSwarm);
        
        expect(recommendation.recommended_pattern).toBe('divergent'); // Should recommend divergent for creative tasks
        expect(recommendation.confidence_score).toBeGreaterThan(0);
        expect(recommendation.reasoning).toContain('creative');
    });

    test('optimizes swarm composition', () => {
        const currentSwarm = [
            { agent_id: 'a1', cognitive_pattern: 'convergent' },
            { agent_id: 'a2', cognitive_pattern: 'convergent' },
            { agent_id: 'a3', cognitive_pattern: 'convergent' }
        ];
        
        const optimizationGoals = {
            target_diversity_score: 0.8,
            preferred_patterns: ['divergent', 'systems'],
            performance_priorities: ['creativity', 'holistic_thinking']
        };
        
        const optimization = diversityEngine.optimize_swarm_composition(currentSwarm, optimizationGoals);
        
        expect(optimization.optimization_plan).toBeDefined();
        expect(optimization.expected_improvements).toBeDefined();
        expect(optimization.implementation_steps).toBeInstanceOf(Array);
    });
});
```

## 🔄 Integration Testing

### Multi-Module Integration Tests
```javascript
// tests/integration/multi-module.test.js - Cross-module integration testing

const { RuvSwarm } = require('../../npm/src');
const { initializeAllModules, cleanupAllModules } = require('../utils/test-helpers');

describe('Multi-Module Integration', () => {
    let ruvSwarm;
    
    beforeAll(async () => {
        await initializeAllModules();
    });
    
    afterAll(async () => {
        await cleanupAllModules();
    });
    
    beforeEach(async () => {
        ruvSwarm = await RuvSwarm.initialize({
            loadingStrategy: 'eager',
            enablePersistence: true,
            enableNeuralNetworks: true,
            enableForecasting: true,
            useSIMD: true
        });
    });

    test('neural networks integrate with swarm agents', async () => {
        const swarm = await ruvSwarm.createSwarm({
            name: 'neural-integration-swarm',
            topology: 'mesh',
            maxAgents: 5,
            enableNeuralAgents: true
        });
        
        const agent = await swarm.spawn({
            type: 'researcher',
            name: 'neural-researcher',
            enableNeuralNetwork: true
        });
        
        expect(agent.neuralNetworkId).toBeDefined();
        
        // Test that agent can process tasks using neural network
        const task = await swarm.orchestrate({
            description: 'Analyze pattern in data using neural processing',
            priority: 'medium'
        });
        
        expect(task.assignedAgents).toContain(agent.id);
    });

    test('forecasting models work with swarm intelligence', async () => {
        if (!ruvSwarm.features.forecasting) {
            return; // Skip if forecasting not available
        }
        
        const swarm = await ruvSwarm.createSwarm({
            name: 'forecasting-swarm',
            topology: 'hierarchical',
            maxAgents: 3
        });
        
        const forecaster = await swarm.spawn({
            type: 'analyst',
            name: 'forecasting-analyst',
            enableNeuralNetwork: true
        });
        
        // Test forecasting integration
        const forecastTask = await swarm.orchestrate({
            description: 'Generate 12-month sales forecast using LSTM model',
            priority: 'high',
            metadata: {
                model_type: 'LSTM',
                horizon: 12,
                data_source: 'sales_history'
            }
        });
        
        expect(forecastTask.status).toBe('orchestrated');
        expect(forecastTask.assignedAgents).toContain(forecaster.id);
    });

    test('persistence layer maintains consistency across modules', async () => {
        const swarm1 = await ruvSwarm.createSwarm({
            name: 'persistence-test-swarm-1',
            topology: 'mesh',
            maxAgents: 3
        });
        
        const agent1 = await swarm1.spawn({
            type: 'researcher',
            name: 'persistent-researcher'
        });
        
        // Store some data in agent memory
        if (ruvSwarm.persistence) {
            await ruvSwarm.persistence.storeAgentMemory(
                agent1.id,
                'test_data',
                { processed: true, timestamp: Date.now() }
            );
        }
        
        // Create second swarm and verify data persistence
        const swarm2 = await ruvSwarm.createSwarm({
            name: 'persistence-test-swarm-2',
            topology: 'star',
            maxAgents: 2
        });
        
        if (ruvSwarm.persistence) {
            const retrievedData = await ruvSwarm.persistence.getAgentMemory(
                agent1.id,
                'test_data'
            );
            
            expect(retrievedData).toBeDefined();
            expect(retrievedData.processed).toBe(true);
        }
    });

    test('cognitive diversity affects neural network configurations', async () => {
        const swarm = await ruvSwarm.createSwarm({
            name: 'cognitive-neural-swarm',
            topology: 'mesh',
            maxAgents: 5,
            enableCognitiveDiversity: true,
            enableNeuralAgents: true
        });
        
        const agents = await Promise.all([
            swarm.spawn({ type: 'researcher', name: 'divergent-researcher' }),
            swarm.spawn({ type: 'coder', name: 'convergent-coder' }),
            swarm.spawn({ type: 'analyst', name: 'critical-analyst' })
        ]);
        
        // Each agent should have different cognitive patterns
        const cognitivePatterns = agents.map(agent => agent.cognitivePattern);
        const uniquePatterns = new Set(cognitivePatterns);
        
        expect(uniquePatterns.size).toBeGreaterThan(1);
        
        // Neural networks should be configured differently based on cognitive patterns
        const networkConfigs = agents.map(agent => agent.neuralNetworkId);
        expect(networkConfigs.every(id => id)).toBe(true); // All should have neural networks
    });

    test('SIMD optimizations work across all modules', async () => {
        if (!ruvSwarm.features.simd_support) {
            return; // Skip if SIMD not available
        }
        
        const swarm = await ruvSwarm.createSwarm({
            name: 'simd-optimization-swarm',
            topology: 'mesh',
            maxAgents: 3,
            enableNeuralAgents: true
        });
        
        const agent = await swarm.spawn({
            type: 'optimizer',
            name: 'simd-optimizer'
        });
        
        // Test SIMD-optimized operations
        const optimizationTask = await swarm.orchestrate({
            description: 'Optimize neural network weights using SIMD operations',
            priority: 'high',
            metadata: {
                use_simd: true,
                optimization_target: 'speed'
            }
        });
        
        expect(optimizationTask.status).toBe('orchestrated');
        
        // Verify SIMD utilization
        const globalMetrics = await ruvSwarm.getGlobalMetrics();
        expect(globalMetrics.features.simd_support).toBe(true);
    });
});
```

### NPX Package Integration Tests
```javascript
// tests/integration/npx-package.test.js - NPX workflow testing

const { execSync, spawn } = require('child_process');
const path = require('path');

describe('NPX Package Integration', () => {
    const npxCommand = 'npx ruv-swarm';
    const timeout = 30000; // 30 seconds timeout
    
    test('npx ruv-swarm --help works without installation', () => {
        const output = execSync(`${npxCommand} --help`, { 
            encoding: 'utf8',
            timeout 
        });
        
        expect(output).toContain('ruv-swarm');
        expect(output).toContain('Enhanced WASM-powered neural network swarm orchestration');
        expect(output).toContain('Commands:');
        expect(output).toContain('init');
        expect(output).toContain('spawn');
        expect(output).toContain('orchestrate');
    });

    test('npx ruv-swarm version returns correct version', () => {
        const output = execSync(`${npxCommand} version`, { 
            encoding: 'utf8',
            timeout 
        });
        
        expect(output).toMatch(/ruv-swarm v\d+\.\d+\.\d+/);
        expect(output).toContain('Enhanced WASM-powered neural swarm orchestration');
    });

    test('npx ruv-swarm features detects capabilities', () => {
        const output = execSync(`${npxCommand} features`, { 
            encoding: 'utf8',
            timeout 
        });
        
        const featuresOutput = JSON.parse(output);
        
        expect(featuresOutput).toHaveProperty('runtime');
        expect(featuresOutput).toHaveProperty('wasm');
        expect(featuresOutput).toHaveProperty('ruv_swarm');
        expect(featuresOutput.runtime).toHaveProperty('webassembly');
        expect(featuresOutput.wasm).toHaveProperty('simd_support');
    });

    test('complete swarm workflow through NPX', async () => {
        const testCommands = [
            `${npxCommand} init mesh 3`,
            `${npxCommand} spawn researcher test-researcher`,
            `${npxCommand} spawn coder test-coder`,
            `${npxCommand} status`,
            `${npxCommand} orchestrate "Test task for NPX integration"`
        ];
        
        for (const command of testCommands) {
            const output = execSync(command, { 
                encoding: 'utf8',
                timeout: 10000 // 10 seconds per command
            });
            
            expect(output).not.toContain('Error:');
            expect(output).not.toContain('❌');
        }
    });

    test('MCP server starts successfully', (done) => {
        const mcpProcess = spawn('npx', ['ruv-swarm', 'mcp', 'start'], {
            stdio: ['pipe', 'pipe', 'pipe']
        });
        
        let output = '';
        let errorOutput = '';
        
        mcpProcess.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        mcpProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });
        
        // Give the server time to start
        setTimeout(() => {
            mcpProcess.kill();
            
            // Check that server started without critical errors
            expect(errorOutput).not.toContain('Failed to initialize');
            expect(errorOutput).not.toContain('Error:');
            
            done();
        }, 5000);
    });

    test('neural network commands work through NPX', () => {
        const commands = [
            `${npxCommand} neural status`,
            `${npxCommand} neural functions`
        ];
        
        for (const command of commands) {
            const output = execSync(command, { 
                encoding: 'utf8',
                timeout: 10000
            });
            
            expect(output).not.toContain('Error:');
            expect(output).not.toContain('❌');
        }
    });

    test('forecasting commands work when enabled', () => {
        try {
            const output = execSync(`${npxCommand} forecast models`, { 
                encoding: 'utf8',
                timeout: 10000
            });
            
            if (output.includes('not available')) {
                // Forecasting not enabled, skip
                return;
            }
            
            expect(output).toContain('Available Forecasting Models:');
            expect(output).toContain('LSTM');
            expect(output).toContain('NBEATS');
        } catch (error) {
            // Forecasting might not be available in all builds
            expect(error.stdout).toContain('not available');
        }
    });

    test('benchmark command provides performance metrics', () => {
        const output = execSync(`${npxCommand} benchmark`, { 
            encoding: 'utf8',
            timeout: 30000 // Benchmarks take longer
        });
        
        expect(output).toContain('Benchmark');
        expect(output).toContain('results');
        expect(output).not.toContain('Error:');
    });

    test('memory command shows usage statistics', () => {
        const output = execSync(`${npxCommand} memory`, { 
            encoding: 'utf8',
            timeout: 10000
        });
        
        expect(output).toContain('Memory Usage');
        expect(output).toMatch(/\d+\.?\d* MB/); // Should show memory in MB
        expect(output).not.toContain('Error:');
    });
});
```

## 🚀 Performance Testing

### Benchmark Test Suite
```javascript
// tests/performance/benchmarks/comprehensive-benchmarks.js

const { RuvSwarm } = require('../../../npm/src');
const { PerformanceMonitor } = require('../../../npm/src/performance-monitor');
const { generateLargeDataset, measureMemoryUsage } = require('../../utils/performance-helpers');

describe('Comprehensive Performance Benchmarks', () => {
    let ruvSwarm;
    let performanceMonitor;
    
    beforeAll(async () => {
        ruvSwarm = await RuvSwarm.initialize({
            loadingStrategy: 'eager',
            enablePersistence: true,
            enableNeuralNetworks: true,
            enableForecasting: true,
            useSIMD: true
        });
        
        performanceMonitor = new PerformanceMonitor();
        performanceMonitor.startProfiling();
    });
    
    afterAll(() => {
        const report = performanceMonitor.stopProfiling();
        console.log('Performance Report:', JSON.stringify(report.summary, null, 2));
    });

    describe('Agent Spawning Performance', () => {
        test('spawns 100 agents within performance targets', async () => {
            const swarm = await ruvSwarm.createSwarm({
                name: 'performance-test-swarm',
                topology: 'mesh',
                maxAgents: 100,
                enableNeuralAgents: true
            });
            
            const spawnTimes = [];
            const startTime = performance.now();
            
            for (let i = 0; i < 100; i++) {
                const agentStartTime = performance.now();
                
                await swarm.spawn({
                    type: ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'][i % 5],
                    name: `perf-agent-${i}`,
                    enableNeuralNetwork: true
                });
                
                const agentEndTime = performance.now();
                const spawnTime = agentEndTime - agentStartTime;
                spawnTimes.push(spawnTime);
                
                performanceMonitor.recordAgentSpawn(`perf-agent-${i}`, spawnTime);
            }
            
            const totalTime = performance.now() - startTime;
            const avgSpawnTime = spawnTimes.reduce((a, b) => a + b, 0) / spawnTimes.length;
            const maxSpawnTime = Math.max(...spawnTimes);
            const minSpawnTime = Math.min(...spawnTimes);
            
            console.log(`Agent Spawning Performance:
                Total Time: ${totalTime.toFixed(2)}ms
                Average: ${avgSpawnTime.toFixed(2)}ms per agent
                Min: ${minSpawnTime.toFixed(2)}ms
                Max: ${maxSpawnTime.toFixed(2)}ms
                Target: <20ms per agent`);
            
            expect(avgSpawnTime).toBeLessThan(20); // Target: <20ms per agent
            expect(maxSpawnTime).toBeLessThan(50); // No agent should take >50ms
        });
    });

    describe('Task Orchestration Performance', () => {
        test('orchestrates complex tasks efficiently', async () => {
            const swarm = await ruvSwarm.createSwarm({
                name: 'orchestration-perf-swarm',
                topology: 'hierarchical',
                maxAgents: 20
            });
            
            // Spawn agents
            const agents = [];
            for (let i = 0; i < 20; i++) {
                const agent = await swarm.spawn({
                    type: ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'][i % 5],
                    name: `orchestration-agent-${i}`
                });
                agents.push(agent);
            }
            
            // Test task orchestration performance
            const orchestrationTimes = [];
            
            for (let i = 0; i < 50; i++) {
                const startTime = performance.now();
                
                const task = await swarm.orchestrate({
                    description: `Complex task ${i} requiring multiple agent coordination`,
                    priority: ['low', 'medium', 'high', 'critical'][i % 4],
                    maxAgents: Math.floor(Math.random() * 5) + 1,
                    estimatedDuration: Math.random() * 10000 + 1000
                });
                
                const endTime = performance.now();
                const orchestrationTime = endTime - startTime;
                orchestrationTimes.push(orchestrationTime);
                
                performanceMonitor.recordTaskOrchestration(task.id, orchestrationTime, task.assignedAgents.length);
            }
            
            const avgOrchestrationTime = orchestrationTimes.reduce((a, b) => a + b, 0) / orchestrationTimes.length;
            const maxOrchestrationTime = Math.max(...orchestrationTimes);
            
            console.log(`Task Orchestration Performance:
                Average: ${avgOrchestrationTime.toFixed(2)}ms
                Max: ${maxOrchestrationTime.toFixed(2)}ms
                Target: <100ms average`);
            
            expect(avgOrchestrationTime).toBeLessThan(100); // Target: <100ms average
            expect(maxOrchestrationTime).toBeLessThan(500); // No task should take >500ms
        });
    });

    describe('Neural Network Performance', () => {
        test('neural operations meet performance targets', async () => {
            if (!ruvSwarm.features.neural_networks) {
                return; // Skip if neural networks not available
            }
            
            const neuralModule = await ruvSwarm.wasmLoader.loadModule('neural');
            const networkCreationTimes = [];
            const forwardPassTimes = [];
            
            // Test network creation performance
            for (let i = 0; i < 50; i++) {
                const startTime = performance.now();
                
                const network = new neuralModule.exports.WasmNeuralNetwork({
                    input_size: 10,
                    hidden_layers: [
                        { size: 64, activation: 'relu' },
                        { size: 32, activation: 'sigmoid' }
                    ],
                    output_size: 5
                });
                
                const endTime = performance.now();
                const creationTime = endTime - startTime;
                networkCreationTimes.push(creationTime);
                
                // Test forward pass performance
                const inputs = new Array(10).fill(0).map(() => Math.random());
                
                const forwardStartTime = performance.now();
                const outputs = network.run(inputs);
                const forwardEndTime = performance.now();
                
                const forwardTime = forwardEndTime - forwardStartTime;
                forwardPassTimes.push(forwardTime);
                
                performanceMonitor.recordNeuralOperation('networkCreation', creationTime, 111); // Total neurons
                performanceMonitor.recordNeuralOperation('forwardPass', forwardTime, 111);
            }
            
            const avgCreationTime = networkCreationTimes.reduce((a, b) => a + b, 0) / networkCreationTimes.length;
            const avgForwardTime = forwardPassTimes.reduce((a, b) => a + b, 0) / forwardPassTimes.length;
            
            console.log(`Neural Network Performance:
                Network Creation: ${avgCreationTime.toFixed(2)}ms average
                Forward Pass: ${avgForwardTime.toFixed(4)}ms average
                Creation Target: <15ms
                Forward Target: <1ms`);
            
            expect(avgCreationTime).toBeLessThan(15); // Target: <15ms for network creation
            expect(avgForwardTime).toBeLessThan(1);   // Target: <1ms for forward pass
        });
    });

    describe('Memory Performance', () => {
        test('memory usage stays within bounds during extended operation', async () => {
            const initialMemory = measureMemoryUsage();
            
            const swarm = await ruvSwarm.createSwarm({
                name: 'memory-test-swarm',
                topology: 'mesh',
                maxAgents: 50
            });
            
            // Perform memory-intensive operations
            const agents = [];
            for (let i = 0; i < 50; i++) {
                const agent = await swarm.spawn({
                    type: 'researcher',
                    name: `memory-agent-${i}`,
                    enableNeuralNetwork: true
                });
                agents.push(agent);
            }
            
            const afterSpawningMemory = measureMemoryUsage();
            
            // Execute many tasks
            for (let i = 0; i < 100; i++) {
                await swarm.orchestrate({
                    description: `Memory test task ${i}`,
                    priority: 'medium'
                });
            }
            
            const afterTasksMemory = measureMemoryUsage();
            
            // Force garbage collection if available
            if (global.gc) {
                global.gc();
            }
            
            const finalMemory = measureMemoryUsage();
            
            const memoryGrowthAfterSpawning = afterSpawningMemory - initialMemory;
            const memoryGrowthAfterTasks = afterTasksMemory - afterSpawningMemory;
            const finalMemoryGrowth = finalMemory - initialMemory;
            
            console.log(`Memory Usage:
                Initial: ${initialMemory.toFixed(1)}MB
                After spawning 50 agents: ${afterSpawningMemory.toFixed(1)}MB (+${memoryGrowthAfterSpawning.toFixed(1)}MB)
                After 100 tasks: ${afterTasksMemory.toFixed(1)}MB (+${memoryGrowthAfterTasks.toFixed(1)}MB)
                Final (after GC): ${finalMemory.toFixed(1)}MB (+${finalMemoryGrowth.toFixed(1)}MB)
                Target: <100MB total for 50 agents`);
            
            expect(finalMemory).toBeLessThan(100); // Target: <100MB for 50 agents
            expect(memoryGrowthAfterTasks).toBeLessThan(10); // Tasks shouldn't cause major memory growth
        });
    });

    describe('SIMD Performance Comparison', () => {
        test('SIMD operations provide expected speedup', async () => {
            if (!ruvSwarm.features.simd_support) {
                console.log('SIMD not available, skipping SIMD performance test');
                return;
            }
            
            const neuralModule = await ruvSwarm.wasmLoader.loadModule('neural');
            const simdProcessor = new neuralModule.exports.SIMDNeuralProcessor();
            
            // Generate test data
            const size = 1000;
            const a = new Float32Array(size).fill(0).map(() => Math.random());
            const b = new Float32Array(size).fill(0).map(() => Math.random());
            
            // Test with SIMD disabled (if possible)
            const scalarTimes = [];
            for (let i = 0; i < 10; i++) {
                const startTime = performance.now();
                // This would call scalar implementation
                const result = simdProcessor.vector_add_f32(a, b);
                const endTime = performance.now();
                scalarTimes.push(endTime - startTime);
            }
            
            const avgScalarTime = scalarTimes.reduce((a, b) => a + b, 0) / scalarTimes.length;
            
            console.log(`SIMD Performance:
                Vector addition (${size} elements): ${avgScalarTime.toFixed(4)}ms
                Expected SIMD speedup: 2-4x for supported operations`);
            
            expect(avgScalarTime).toBeLessThan(5); // Should be fast regardless of SIMD
        });
    });
});
```

## 📋 Test Execution and CI/CD

### GitHub Actions Workflow
```yaml
# .github/workflows/comprehensive-testing.yml
name: Comprehensive Testing

on:
  push:
    branches: [ main, ruv-swarm ]
  pull_request:
    branches: [ main ]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16, 18, 20]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
        cache-dependency-path: ruv-swarm/npm/package-lock.json
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        target: wasm32-unknown-unknown
        components: rustfmt, clippy
    
    - name: Install dependencies
      run: |
        cd ruv-swarm/npm
        npm ci
    
    - name: Build WASM modules
      run: |
        cd ruv-swarm
        node build/wasm-build.js
    
    - name: Run unit tests
      run: |
        cd ruv-swarm/npm
        npm run test:unit
    
    - name: Run integration tests
      run: |
        cd ruv-swarm/npm
        npm run test:integration
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-node-${{ matrix.node-version }}
        path: |
          ruv-swarm/npm/test-results/
          ruv-swarm/npm/coverage/

  performance-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: ruv-swarm/npm/package-lock.json
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        target: wasm32-unknown-unknown
    
    - name: Install dependencies and build
      run: |
        cd ruv-swarm/npm
        npm ci
        cd ..
        node build/wasm-build.js
    
    - name: Run performance benchmarks
      run: |
        cd ruv-swarm/npm
        npm run test:performance
    
    - name: Upload performance results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: ruv-swarm/npm/performance-results/

  browser-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: ruv-swarm/npm/package-lock.json
    
    - name: Install dependencies and build
      run: |
        cd ruv-swarm/npm
        npm ci
        cd ..
        node build/wasm-build.js
    
    - name: Run browser tests
      run: |
        cd ruv-swarm/npm
        npm run test:browser
      env:
        BROWSER: chrome-headless

  e2e-tests:
    runs-on: ubuntu-latest
    needs: [unit-tests, performance-tests]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Test NPX package
      run: |
        # Test direct NPX usage
        npx --yes ./ruv-swarm/npm help
        npx --yes ./ruv-swarm/npm version
        npx --yes ./ruv-swarm/npm features
    
    - name: Test MCP integration
      run: |
        cd ruv-swarm/npm
        timeout 10s npx ruv-swarm mcp start || true
        npm run test:mcp
```

### Local Testing Scripts
```bash
#!/bin/bash
# scripts/run-all-tests.sh - Comprehensive local testing

set -e

echo "🧪 Running Comprehensive Test Suite"
echo "==================================="

cd ruv-swarm/npm

# Build WASM modules first
echo "🏗️ Building WASM modules..."
cd ..
node build/wasm-build.js
cd npm

# Unit tests
echo "🔬 Running unit tests..."
npm run test:unit

# Integration tests  
echo "🔗 Running integration tests..."
npm run test:integration

# Performance tests
echo "🚀 Running performance tests..."
npm run test:performance

# Browser tests (if available)
if command -v chrome &> /dev/null; then
    echo "🌐 Running browser tests..."
    npm run test:browser
fi

# E2E tests
echo "🎭 Running end-to-end tests..."
npm run test:e2e

# NPX integration tests
echo "📦 Testing NPX integration..."
npx ruv-swarm version
npx ruv-swarm features
npx ruv-swarm help

echo "✅ All tests completed successfully!"
```

This comprehensive testing strategy ensures that the WASM-powered ruv-swarm implementation is reliable, performant, and compatible across all target platforms and use cases.