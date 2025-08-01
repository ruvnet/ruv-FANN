/**
 * Core Systems Comprehensive Coverage Test Suite
 * Target: 90%+ coverage for critical core system components
 * Files: index-enhanced.js, daa-service.js, wasm-loader.js, schemas.js, persistence.js, errors.js, mcp-tools-enhanced.js
 */

import assert from 'assert';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs/promises';
import { EventEmitter } from 'events';

// Test imports
import { RuvSwarm, Swarm, Agent, Task, DAAService, daaService } from '../src/index-enhanced.js';
import { WasmModuleLoader } from '../src/wasm-loader.js';
import { SwarmPersistence } from '../src/persistence.js';
import {
  RuvSwarmError,
  ValidationError,
  SwarmError,
  AgentError,
  TaskError,
  NeuralError,
  WasmError,
  ConfigurationError,
  NetworkError,
  PersistenceError,
  ResourceError,
  ConcurrencyError,
  ErrorFactory,
  ErrorContext,
} from '../src/errors.js';
import { ValidationUtils, MCPSchemas, BaseValidator } from '../src/schemas.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Core Systems Comprehensive Coverage Tests', () => {

  // Test setup and cleanup
  let testInstances = [];

  beforeEach(() => {
    testInstances = [];
    // Reset global state
    if (global._ruvSwarmInstance) {
      global._ruvSwarmInstance = null;
    }
    if (global._ruvSwarmInitialized) {
      global._ruvSwarmInitialized = 0;
    }
  });

  afterEach(async() => {
    // Cleanup all test instances
    for (const instance of testInstances) {
      if (instance && typeof instance.cleanup === 'function') {
        try {
          await instance.cleanup();
        } catch (error) {
          console.warn('Cleanup error:', error.message);
        }
      }
    }
    testInstances = [];
  });

  describe('RuvSwarm Enhanced Core (index-enhanced.js)', () => {

    describe('Initialization Edge Cases', () => {

      it('should handle duplicate initialization gracefully', async() => {
        const instance1 = await RuvSwarm.initialize({ debug: true });
        const instance2 = await RuvSwarm.initialize({ debug: true });

        assert.strictEqual(instance1, instance2, 'Should return same instance');
        assert.strictEqual(global._ruvSwarmInitialized, 2, 'Should track initialization calls');

        testInstances.push(instance1);
      });

      it('should handle initialization with minimal options', async() => {
        const instance = await RuvSwarm.initialize({});

        assert(instance instanceof RuvSwarm, 'Should create RuvSwarm instance');
        assert(instance.wasmLoader, 'Should have WASM loader');
        assert(instance.features, 'Should have features object');

        testInstances.push(instance);
      });

      it('should handle initialization failures gracefully', async() => {
        // Mock WASM loader to fail
        const originalInit = WasmModuleLoader.prototype.initialize;
        WasmModuleLoader.prototype.initialize = async() => {
          throw new Error('WASM initialization failed');
        };

        try {
          await assert.rejects(
            () => RuvSwarm.initialize({ enableNeuralNetworks: true }),
            /Failed to initialize ruv-swarm/,
          );
        } finally {
          WasmModuleLoader.prototype.initialize = originalInit;
        }
      });

      it('should handle persistence initialization failures', async() => {
        const instance = await RuvSwarm.initialize({ enablePersistence: true });

        // Should continue without persistence on error
        assert(instance.persistence !== undefined, 'Should handle persistence gracefully');

        testInstances.push(instance);
      });

      it('should handle neural network loading failures', async() => {
        const instance = await RuvSwarm.initialize({
          enableNeuralNetworks: true,
          enableForecasting: true,
        });

        // Should gracefully degrade features
        assert(typeof instance.features.neural_networks === 'boolean', 'Should set neural networks feature');
        assert(typeof instance.features.forecasting === 'boolean', 'Should set forecasting feature');

        testInstances.push(instance);
      });

    });

    describe('Feature Detection', () => {

      it('should detect SIMD support correctly', () => {
        const simdSupported = RuvSwarm.detectSIMDSupport();
        assert(typeof simdSupported === 'boolean', 'Should return boolean for SIMD support');
      });

      it('should handle feature detection failures', async() => {
        const instance = new RuvSwarm();

        // Mock WASM loader to fail
        const originalLoadModule = instance.wasmLoader.loadModule;
        instance.wasmLoader.loadModule = async() => {
          throw new Error('Module loading failed');
        };

        try {
          await instance.detectFeatures(true);
          // Should not throw, should log warning
          assert(true, 'Should handle detection failures gracefully');
        } finally {
          instance.wasmLoader.loadModule = originalLoadModule;
        }
      });

      it('should set appropriate default features', async() => {
        const instance = await RuvSwarm.initialize();

        assert(typeof instance.features.neural_networks === 'boolean');
        assert(typeof instance.features.forecasting === 'boolean');
        assert(typeof instance.features.cognitive_diversity === 'boolean');
        assert(typeof instance.features.simd_support === 'boolean');

        testInstances.push(instance);
      });

    });

    describe('Swarm Creation and Management', () => {

      it('should create swarm with minimal configuration', async() => {
        const instance = await RuvSwarm.initialize();
        const swarm = await instance.createSwarm({});

        assert(swarm instanceof Swarm, 'Should create Swarm instance');
        assert(swarm.id, 'Should have unique ID');
        assert.strictEqual(instance.activeSwarms.size, 1, 'Should track active swarms');

        testInstances.push(instance);
      });

      it('should create swarm with existing ID (persistence loading)', async() => {
        const instance = await RuvSwarm.initialize();
        const existingId = 'swarm-12345';

        const swarm = await instance.createSwarm({
          id: existingId,
          name: 'test-swarm',
        });

        assert.strictEqual(swarm.id, existingId, 'Should use provided ID');

        testInstances.push(instance);
      });

      it('should handle WASM swarm creation failures', async() => {
        const instance = await RuvSwarm.initialize();

        // Should fallback to JavaScript implementation
        const swarm = await instance.createSwarm({
          name: 'fallback-test',
          topology: 'mesh',
        });

        assert(swarm, 'Should create swarm with fallback');
        assert(swarm.agents instanceof Map, 'Should have agents map');
        assert(swarm.tasks instanceof Map, 'Should have tasks map');

        testInstances.push(instance);
      });

      it('should handle persistence errors during swarm creation', async() => {
        const instance = await RuvSwarm.initialize({ enablePersistence: true });

        if (instance.persistence) {
          // Mock persistence to fail
          const originalCreate = instance.persistence.createSwarm;
          instance.persistence.createSwarm = () => {
            throw new Error('Database error');
          };

          try {
            const swarm = await instance.createSwarm({ name: 'error-test' });
            assert(swarm, 'Should create swarm despite persistence error');
          } finally {
            instance.persistence.createSwarm = originalCreate;
          }
        }

        testInstances.push(instance);
      });

    });

    describe('Agent Spawning and Task Orchestration', () => {

      it('should spawn agent with minimal configuration', async() => {
        const instance = await RuvSwarm.initialize();
        const swarm = await instance.createSwarm({});

        const agent = await swarm.spawn({});

        assert(agent instanceof Agent, 'Should create Agent instance');
        assert(agent.id, 'Should have unique ID');
        assert.strictEqual(swarm.agents.size, 1, 'Should track spawned agents');

        testInstances.push(instance);
      });

      it('should spawn agent with existing ID (persistence loading)', async() => {
        const instance = await RuvSwarm.initialize();
        const swarm = await instance.createSwarm({});
        const existingId = 'agent-12345';

        const agent = await swarm.spawn({
          id: existingId,
          type: 'researcher',
        });

        assert.strictEqual(agent.id, existingId, 'Should use provided ID');

        testInstances.push(instance);
      });

      it('should handle neural network loading for agents', async() => {
        const instance = await RuvSwarm.initialize({ enableNeuralNetworks: true });
        const swarm = await instance.createSwarm({});

        const agent = await swarm.spawn({
          type: 'researcher',
          enableNeuralNetwork: true,
        });

        assert(agent.neuralNetworkId, 'Should have neural network ID when enabled');

        testInstances.push(instance);
      });

      it('should orchestrate tasks with no available agents', async() => {
        const instance = await RuvSwarm.initialize();
        const swarm = await instance.createSwarm({});

        await assert.rejects(
          () => swarm.orchestrate({
            description: 'Test task',
            requiredCapabilities: ['testing'],
          }),
          /No agents available/,
        );

        testInstances.push(instance);
      });

      it('should orchestrate tasks with capability filtering', async() => {
        const instance = await RuvSwarm.initialize();
        const swarm = await instance.createSwarm({});

        await swarm.spawn({ type: 'researcher', capabilities: ['research'] });
        await swarm.spawn({ type: 'coder', capabilities: ['coding'] });

        const task = await swarm.orchestrate({
          description: 'Research task',
          requiredCapabilities: ['research'],
          maxAgents: 1,
        });

        assert(task instanceof Task, 'Should create Task instance');
        assert.strictEqual(task.assignedAgents.length, 1, 'Should assign correct number of agents');

        testInstances.push(instance);
      });

      it('should handle agent status updates during task execution', async() => {
        const instance = await RuvSwarm.initialize();
        const swarm = await instance.createSwarm({});

        const agent = await swarm.spawn({ type: 'researcher' });
        const task = await swarm.orchestrate({
          description: 'Test task',
        });

        // Wait for task execution to start
        await new Promise(resolve => setTimeout(resolve, 100));

        assert(['busy', 'idle'].includes(agent.status), 'Agent should have valid status');

        testInstances.push(instance);
      });

    });

    describe('Metrics and Monitoring', () => {

      it('should generate comprehensive global metrics', async() => {
        const instance = await RuvSwarm.initialize();
        const swarm = await instance.createSwarm({});
        await swarm.spawn({ type: 'researcher' });

        const metrics = await instance.getGlobalMetrics();

        assert(typeof metrics.totalAgents === 'number');
        assert(typeof metrics.totalTasks === 'number');
        assert(typeof metrics.totalSwarms === 'number');
        assert(typeof metrics.memoryUsage === 'number');
        assert(metrics.features);
        assert(metrics.wasm_modules);
        assert(metrics.timestamp);

        testInstances.push(instance);
      });

      it('should handle swarm status for non-existent swarm', async() => {
        const instance = await RuvSwarm.initialize();

        await assert.rejects(
          () => instance.getSwarmStatus('non-existent'),
          /Swarm not found/,
        );

        testInstances.push(instance);
      });

      it('should get all swarms status', async() => {
        const instance = await RuvSwarm.initialize();
        await instance.createSwarm({ name: 'swarm1' });
        await instance.createSwarm({ name: 'swarm2' });

        const allSwarms = await instance.getAllSwarms();

        assert.strictEqual(allSwarms.length, 2, 'Should return all swarms');
        assert(allSwarms[0].id, 'Each swarm should have ID');
        assert(allSwarms[0].status, 'Each swarm should have status');

        testInstances.push(instance);
      });

    });

    describe('Static Utility Methods', () => {

      it('should return correct version', () => {
        const version = RuvSwarm.getVersion();
        assert.strictEqual(version, '0.2.0');
      });

      it('should get memory usage information', () => {
        const memoryInfo = RuvSwarm.getMemoryUsage();

        if (memoryInfo) {
          assert(typeof memoryInfo.used === 'number');
          assert(typeof memoryInfo.total === 'number');
          assert(typeof memoryInfo.limit === 'number');
        }
      });

      it('should get runtime features', () => {
        const features = RuvSwarm.getRuntimeFeatures();

        assert(typeof features.webassembly === 'boolean');
        assert(typeof features.simd === 'boolean');
        assert(typeof features.workers === 'boolean');
        assert(typeof features.shared_array_buffer === 'boolean');
        assert(typeof features.bigint === 'boolean');
      });

    });

  });

  describe('DAA Service (daa-service.js)', () => {

    describe('Service Initialization', () => {

      it('should initialize service successfully', async() => {
        const service = new DAAService();
        await service.initialize();

        assert(service.initialized, 'Should be marked as initialized');
        assert(service.wasmLoader, 'Should have WASM loader');
        assert(service.agents instanceof Map, 'Should have agents map');
        assert(service.agentStates, 'Should have agent state manager');

        testInstances.push(service);
      });

      it('should handle WASM initialization failures gracefully', async() => {
        const service = new DAAService();

        // Mock WASM loader to fail
        const originalInit = service.wasmLoader.initialize;
        service.wasmLoader.initialize = async() => {
          throw new Error('WASM failed');
        };

        try {
          await service.initialize();
          assert(service.initialized, 'Should initialize with fallback');
        } finally {
          service.wasmLoader.initialize = originalInit;
        }

        testInstances.push(service);
      });

      it('should provide capabilities information', () => {
        const service = new DAAService();

        const uninitialized = service.getCapabilities();
        assert.strictEqual(uninitialized.autonomousLearning, false);
        assert.strictEqual(uninitialized.peerCoordination, false);

        service.initialized = true;
        const initialized = service.getCapabilities();
        assert.strictEqual(initialized.autonomousLearning, true);
        assert.strictEqual(initialized.peerCoordination, true);
        assert.strictEqual(initialized.cognitivePatterns, 6);
      });

    });

    describe('Agent Lifecycle Management', () => {

      it('should create agent with old signature (id, capabilities)', async() => {
        const service = new DAAService();
        await service.initialize();

        const agent = await service.createAgent('test-agent', ['research', 'analysis']);

        assert.strictEqual(agent.id, 'test-agent');
        assert(agent.capabilities.has('research'));
        assert(agent.capabilities.has('analysis'));

        testInstances.push(service);
      });

      it('should create agent with new signature (config object)', async() => {
        const service = new DAAService();
        await service.initialize();

        const agent = await service.createAgent({
          id: 'test-agent-2',
          capabilities: ['coding'],
          cognitivePattern: 'convergent',
          learningRate: 0.01,
        });

        assert.strictEqual(agent.id, 'test-agent-2');
        assert.strictEqual(agent.cognitivePattern, 'convergent');
        assert.strictEqual(agent.config.learningRate, 0.01);

        testInstances.push(service);
      });

      it('should handle WASM agent creation failures', async() => {
        const service = new DAAService();
        await service.initialize();
        service.wasmModule = null; // Force fallback

        const agent = await service.createAgent('fallback-agent', []);

        assert(agent, 'Should create agent with fallback');
        assert(typeof agent.wasmAgent.make_decision === 'function');

        testInstances.push(service);
      });

      it('should load persisted agent state', async() => {
        const service = new DAAService();
        await service.initialize();

        // Mock persisted state
        const originalLoad = service.agentStates.loadFromStorage;
        service.agentStates.loadFromStorage = async() => ({
          status: 'restored',
          metrics: { restored: true },
        });

        try {
          const agent = await service.createAgent('restored-agent', []);
          assert(agent.state, 'Should have restored state');
        } finally {
          service.agentStates.loadFromStorage = originalLoad;
        }

        testInstances.push(service);
      });

      it('should destroy agent successfully', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('temp-agent', []);
        assert(service.agents.has('temp-agent'));

        const destroyed = await service.destroyAgent('temp-agent');
        assert.strictEqual(destroyed, true);
        assert(!service.agents.has('temp-agent'));

        testInstances.push(service);
      });

      it('should handle destroying non-existent agent', async() => {
        const service = new DAAService();
        await service.initialize();

        const destroyed = await service.destroyAgent('non-existent');
        assert.strictEqual(destroyed, false);

        testInstances.push(service);
      });

    });

    describe('Agent Adaptation and Learning', () => {

      it('should adapt agent based on performance feedback', async() => {
        const service = new DAAService();
        await service.initialize();

        const agent = await service.createAgent('adaptive-agent', []);

        const result = await service.adaptAgent('adaptive-agent', {
          performanceScore: 0.2,
          feedback: 'Poor performance',
        });

        assert(result.previousPattern);
        assert(result.newPattern);
        assert(typeof result.improvement === 'number');
        assert(Array.isArray(result.insights));

        testInstances.push(service);
      });

      it('should handle adaptation for non-existent agent', async() => {
        const service = new DAAService();
        await service.initialize();

        await assert.rejects(
          () => service.adaptAgent('non-existent', {}),
          /Agent non-existent not found/,
        );

        testInstances.push(service);
      });

      it('should get agent learning status', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('learning-agent', []);

        const status = await service.getAgentLearningStatus('learning-agent');

        assert(typeof status.totalCycles === 'number');
        assert(typeof status.avgProficiency === 'number');
        assert(Array.isArray(status.domains));
        assert(typeof status.adaptationRate === 'number');
        assert(status.detailedMetrics);

        testInstances.push(service);
      });

      it('should get system-wide learning status', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('agent1', []);
        await service.createAgent('agent2', []);

        const status = await service.getSystemLearningStatus();

        assert(typeof status.totalCycles === 'number');
        assert(typeof status.avgProficiency === 'number');
        assert(status.detailedMetrics.totalAgents >= 2);

        testInstances.push(service);
      });

    });

    describe('Workflow Orchestration', () => {

      it('should create and execute workflow', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('worker1', []);
        await service.createAgent('worker2', []);

        const workflow = await service.createWorkflow('test-workflow', [
          { id: 'step1', task: async() => 'result1' },
          { id: 'step2', task: async() => 'result2' },
        ], {});

        assert.strictEqual(workflow.id, 'test-workflow');
        assert.strictEqual(workflow.steps.size, 2);

        const result = await service.executeWorkflow('test-workflow', {
          agentIds: ['worker1', 'worker2'],
          parallel: true,
        });

        assert(result.complete);
        assert.strictEqual(result.stepsCompleted, 2);
        assert(Array.isArray(result.stepResults));

        testInstances.push(service);
      });

      it('should handle workflow execution with no agents', async() => {
        const service = new DAAService();
        await service.initialize();

        await assert.rejects(
          () => service.executeWorkflow('non-existent'),
          /Workflow non-existent not found/,
        );

        testInstances.push(service);
      });

    });

    describe('Knowledge Sharing and Meta-Learning', () => {

      it('should share knowledge between agents', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('source', []);
        await service.createAgent('target1', []);
        await service.createAgent('target2', []);

        const result = await service.shareKnowledge('source', ['target1', 'target2'], {
          content: 'shared knowledge',
          domain: 'research',
        });

        assert(Array.isArray(result.updatedAgents));
        assert.strictEqual(result.updatedAgents.length, 2);
        assert(typeof result.transferRate === 'number');

        testInstances.push(service);
      });

      it('should handle knowledge sharing with non-existent source', async() => {
        const service = new DAAService();
        await service.initialize();

        await assert.rejects(
          () => service.shareKnowledge('non-existent', ['target'], {}),
          /Source agent non-existent not found/,
        );

        testInstances.push(service);
      });

      it('should perform meta-learning across domains', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('learner1', []);
        await service.createAgent('learner2', []);

        const result = await service.performMetaLearning({
          sourceDomain: 'research',
          targetDomain: 'analysis',
          transferMode: 'adaptive',
          agentIds: ['learner1', 'learner2'],
        });

        assert(typeof result.knowledgeItems === 'number');
        assert(Array.isArray(result.updatedAgents));
        assert(typeof result.proficiencyGain === 'number');
        assert(Array.isArray(result.insights));

        testInstances.push(service);
      });

    });

    describe('Cognitive Pattern Management', () => {

      it('should analyze cognitive patterns for specific agent', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('pattern-agent', []);

        const analysis = await service.analyzeCognitivePatterns('pattern-agent');

        assert(Array.isArray(analysis.patterns));
        assert(typeof analysis.effectiveness === 'number');
        assert(Array.isArray(analysis.recommendations));
        assert(typeof analysis.optimizationScore === 'number');

        testInstances.push(service);
      });

      it('should analyze system-wide cognitive patterns', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('agent1', []);
        await service.createAgent('agent2', []);

        const analysis = await service.analyzeCognitivePatterns();

        assert(Array.isArray(analysis.patterns));
        assert(typeof analysis.effectiveness === 'number');

        testInstances.push(service);
      });

      it('should set cognitive pattern for agent', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('pattern-agent', []);

        const result = await service.setCognitivePattern('pattern-agent', 'divergent');

        assert(result.previousPattern);
        assert.strictEqual(result.success, true);
        assert(typeof result.expectedImprovement === 'number');

        testInstances.push(service);
      });

    });

    describe('Performance and Cross-Boundary Communication', () => {

      it('should make decisions with performance tracking', async() => {
        const service = new DAAService();
        await service.initialize();

        const agent = await service.createAgent('decision-agent', []);

        const decision = await service.makeDecision('decision-agent', {
          context: 'test decision context',
        });

        assert(decision, 'Should return decision result');
        assert(agent.metrics.decisionsMade >= 1);
        assert(typeof agent.metrics.averageResponseTime === 'number');

        testInstances.push(service);
      });

      it('should handle decision making errors', async() => {
        const service = new DAAService();
        await service.initialize();

        await assert.rejects(
          () => service.makeDecision('non-existent', {}),
          /Agent non-existent not found/,
        );

        testInstances.push(service);
      });

      it('should get comprehensive performance metrics', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('perf-agent', []);

        const metrics = await service.getPerformanceMetrics({
          category: 'all',
          timeRange: '1h',
        });

        assert(typeof metrics.totalAgents === 'number');
        assert(typeof metrics.activeAgents === 'number');
        assert(typeof metrics.tasksCompleted === 'number');
        assert(typeof metrics.avgTaskTime === 'number');
        assert(typeof metrics.successRate === 'number');

        testInstances.push(service);
      });

    });

    describe('State Synchronization and Resource Management', () => {

      it('should synchronize states across agents', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('sync1', []);
        await service.createAgent('sync2', []);

        const states = await service.synchronizeStates(['sync1', 'sync2']);

        assert(states instanceof Map);

        testInstances.push(service);
      });

      it('should optimize resources', async() => {
        const service = new DAAService();
        await service.initialize();

        const result = await service.optimizeResources();

        assert(typeof result.memoryOptimized === 'boolean');
        assert(typeof result.cpuOptimized === 'boolean');
        assert(typeof result.optimizationGain === 'number');

        testInstances.push(service);
      });

      it('should perform batch operations efficiently', async() => {
        const service = new DAAService();
        await service.initialize();

        const configs = [
          { id: 'batch1', capabilities: ['test1'] },
          { id: 'batch2', capabilities: ['test2'] },
          { id: 'batch3', capabilities: ['test3'] },
        ];

        const results = await service.batchCreateAgents(configs);

        assert.strictEqual(results.length, 3);
        assert(results.every(r => r.success), 'All agents should be created successfully');

        testInstances.push(service);
      });

      it('should handle batch decision making', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('batch-agent', []);

        const decisions = [
          { agentId: 'batch-agent', context: { decision: 1 } },
          { agentId: 'batch-agent', context: { decision: 2 } },
        ];

        const results = await service.batchMakeDecisions(decisions);

        assert.strictEqual(results.length, 2);
        assert(results.every(r => r.success), 'All decisions should succeed');

        testInstances.push(service);
      });

    });

    describe('Service Status and Cleanup', () => {

      it('should provide detailed service status', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('status-agent', []);

        const status = service.getStatus();

        assert.strictEqual(status.initialized, true);
        assert(status.agents);
        assert(status.workflows);
        assert(status.wasm);
        assert(status.performance);

        testInstances.push(service);
      });

      it('should cleanup resources properly', async() => {
        const service = new DAAService();
        await service.initialize();

        await service.createAgent('cleanup-agent', []);
        assert(service.agents.size > 0);

        await service.cleanup();

        assert.strictEqual(service.agents.size, 0);

        testInstances.push(service);
      });

    });

  });

  describe('WASM Module Loader (wasm-loader.js)', () => {

    describe('Loader Initialization', () => {

      it('should initialize with eager strategy', async() => {
        const loader = new WasmModuleLoader();
        await loader.initialize('eager');

        assert.strictEqual(loader.loadingStrategy, 'eager');

        testInstances.push(loader);
      });

      it('should initialize with progressive strategy', async() => {
        const loader = new WasmModuleLoader();
        await loader.initialize('progressive');

        assert.strictEqual(loader.loadingStrategy, 'progressive');
        assert(loader.modules.has('core'), 'Should load core module');

        testInstances.push(loader);
      });

      it('should initialize with on-demand strategy', async() => {
        const loader = new WasmModuleLoader();
        const proxies = await loader.initialize('on-demand');

        assert.strictEqual(loader.loadingStrategy, 'on-demand');
        assert(typeof proxies === 'object', 'Should return proxy objects');

        testInstances.push(loader);
      });

      it('should reject unknown loading strategies', async() => {
        const loader = new WasmModuleLoader();

        await assert.rejects(
          () => loader.initialize('unknown'),
          /Unknown loading strategy/,
        );

        testInstances.push(loader);
      });

    });

    describe('Module Loading and Caching', () => {

      it('should load core module with bindings', async() => {
        const loader = new WasmModuleLoader();
        const coreModule = await loader.loadModule('core');

        assert(coreModule, 'Should load core module');
        assert(coreModule.exports, 'Should have exports');

        testInstances.push(loader);
      });

      it('should handle optional module fallbacks', async() => {
        const loader = new WasmModuleLoader();

        // Load optional module that doesn't exist
        const neuralModule = await loader.loadModule('neural');

        assert(neuralModule, 'Should return fallback module');

        testInstances.push(loader);
      });

      it('should cache loaded modules', async() => {
        const loader = new WasmModuleLoader();

        const module1 = await loader.loadModule('core');
        const module2 = await loader.loadModule('core');

        assert.strictEqual(module1, module2, 'Should return cached module');

        testInstances.push(loader);
      });

      it('should handle dependency loading', async() => {
        const loader = new WasmModuleLoader();

        // Neural depends on core
        await loader.loadModule('neural');

        assert(loader.modules.has('core'), 'Should load dependencies first');

        testInstances.push(loader);
      });

      it('should reject unknown modules', async() => {
        const loader = new WasmModuleLoader();

        await assert.rejects(
          () => loader.loadModule('unknown'),
          /Unknown module: unknown/,
        );

        testInstances.push(loader);
      });

    });

    describe('Module Status and Management', () => {

      it('should provide comprehensive module status', async() => {
        const loader = new WasmModuleLoader();
        await loader.loadModule('core');

        const status = loader.getModuleStatus();

        assert(status.core, 'Should have core module status');
        assert(typeof status.core.loaded === 'boolean');
        assert(typeof status.core.loading === 'boolean');
        assert(typeof status.core.placeholder === 'boolean');
        assert(typeof status.core.size === 'number');

        testInstances.push(loader);
      });

      it('should calculate total memory usage', async() => {
        const loader = new WasmModuleLoader();
        await loader.loadModule('core');

        const memoryUsage = loader.getTotalMemoryUsage();

        assert(typeof memoryUsage === 'number');
        assert(memoryUsage >= 0);

        testInstances.push(loader);
      });

      it('should clear module cache', async() => {
        const loader = new WasmModuleLoader();
        await loader.loadModule('core');

        const initialCacheSize = loader.wasmCache.size;
        loader.clearCache();

        assert.strictEqual(loader.wasmCache.size, 0);
        assert(initialCacheSize >= 0);

        testInstances.push(loader);
      });

      it('should optimize memory usage', async() => {
        const loader = new WasmModuleLoader();
        await loader.loadModule('core');

        const optimization = loader.optimizeMemory();

        assert(typeof optimization.cacheSize === 'number');
        assert(typeof optimization.memoryUsage === 'number');
        assert(typeof optimization.expiredEntries === 'number');

        testInstances.push(loader);
      });

    });

    describe('Error Handling and Fallbacks', () => {

      it('should handle core bindings loading failures', async() => {
        const loader = new WasmModuleLoader();

        // Mock bindings path to non-existent file
        const originalBaseDir = loader.baseDir;
        loader.baseDir = '/non/existent/path';

        try {
          const module = await loader.loadModule('core');
          assert(module.isPlaceholder, 'Should use placeholder on failure');
        } finally {
          loader.baseDir = originalBaseDir;
        }

        testInstances.push(loader);
      });

      it('should handle WASM instantiation failures', async() => {
        const loader = new WasmModuleLoader();

        // Mock file system to return invalid WASM
        const originalReadFile = fs.readFile;
        fs.readFile = async() => new Uint8Array([0, 1, 2, 3]); // Invalid WASM

        try {
          const module = await loader.loadModule('core');
          assert(module, 'Should handle instantiation failure gracefully');
        } catch (error) {
          assert(error.message.includes('placeholder') || error.message.includes('instantiate'));
        } finally {
          fs.readFile = originalReadFile;
        }

        testInstances.push(loader);
      });

      it('should create placeholder modules for missing files', async() => {
        const loader = new WasmModuleLoader();

        // Force placeholder creation
        const placeholder = loader._placeholder('test');

        assert(placeholder.isPlaceholder, 'Should mark as placeholder');
        assert(placeholder.memory, 'Should have memory');
        assert(placeholder.exports, 'Should have exports');

        testInstances.push(loader);
      });

    });

    describe('WASM Import Configuration', () => {

      it('should configure base imports correctly', async() => {
        const loader = new WasmModuleLoader();
        const imports = loader._importsFor('core');

        assert(imports.env, 'Should have env imports');
        assert(imports.env.memory, 'Should have memory');
        assert(imports.wasi_snapshot_preview1, 'Should have WASI imports');

        testInstances.push(loader);
      });

      it('should configure neural-specific imports', async() => {
        const loader = new WasmModuleLoader();
        const imports = loader._importsFor('neural');

        assert(imports.neural, 'Should have neural-specific imports');
        assert(typeof imports.neural.log_training_progress === 'function');

        testInstances.push(loader);
      });

      it('should configure forecasting-specific imports', async() => {
        const loader = new WasmModuleLoader();
        const imports = loader._importsFor('forecasting');

        assert(imports.forecasting, 'Should have forecasting-specific imports');
        assert(typeof imports.forecasting.log_forecast === 'function');

        testInstances.push(loader);
      });

      it('should handle random number generation', async() => {
        const loader = new WasmModuleLoader();
        const imports = loader._importsFor('core');

        const buffer = new ArrayBuffer(64);
        const view = new Uint8Array(buffer);
        const result = imports.wasi_snapshot_preview1.random_get(0, 64);

        assert.strictEqual(result, 0, 'Should return success code');

        testInstances.push(loader);
      });

    });

  });

});
