const { expect } = require('chai');
const sinon = require('sinon');
const os = require('os');
const RuvSwarm = require('../../../../src/core/ruv-swarm');
const PerformanceMonitor = require('../../../../src/performance-monitor');

describe('Performance Under Load Integration Tests', () => {
  let sandbox;
  let swarm;
  let performanceMonitor;

  beforeEach(() => {
    sandbox = sinon.createSandbox();
    performanceMonitor = new PerformanceMonitor();
  });

  afterEach(async () => {
    if (swarm) {
      await swarm.shutdown();
    }
    sandbox.restore();
  });

  describe('High Agent Count Stress Tests', () => {
    it('should handle 100+ agents efficiently', async function() {
      this.timeout(60000); // 60 second timeout

      swarm = new RuvSwarm();
      await swarm.init({ 
        topology: 'hierarchical',
        maxAgents: 150,
        performance: {
          monitoring: true,
          adaptiveScaling: true
        }
      });

      performanceMonitor.startTracking('agent-creation');

      // Spawn agents in batches
      const agents = [];
      const batchSize = 20;
      const totalAgents = 100;

      for (let batch = 0; batch < totalAgents / batchSize; batch++) {
        const batchPromises = [];
        for (let i = 0; i < batchSize; i++) {
          const agentType = ['coder', 'researcher', 'analyst', 'optimizer'][i % 4];
          batchPromises.push(swarm.spawnAgent({ 
            type: agentType,
            lightweight: true 
          }));
        }
        
        const batchAgents = await Promise.all(batchPromises);
        agents.push(...batchAgents);
        
        // Brief pause between batches
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      const creationMetrics = performanceMonitor.stopTracking('agent-creation');

      // Verify all agents created
      expect(agents).to.have.lengthOf(100);
      
      // Check creation performance
      expect(creationMetrics.totalTime).to.be.lessThan(30000); // Under 30 seconds
      expect(creationMetrics.averageTimePerOperation).to.be.lessThan(300); // Under 300ms per agent

      // Test concurrent task execution
      performanceMonitor.startTracking('task-execution');

      const tasks = [];
      for (let i = 0; i < 50; i++) {
        tasks.push(swarm.orchestrateTask({
          task: `Stress test task ${i}`,
          strategy: 'parallel',
          maxAgents: 5
        }));
      }

      const taskResults = await Promise.all(tasks);
      const executionMetrics = performanceMonitor.stopTracking('task-execution');

      // Verify task completion
      expect(taskResults).to.have.lengthOf(50);
      taskResults.forEach(result => {
        expect(result).to.have.property('id');
        expect(result).to.have.property('status');
      });

      // Check execution performance
      expect(executionMetrics.totalTime).to.be.lessThan(20000); // Under 20 seconds for 50 tasks
      
      // Monitor resource usage
      const resourceStats = await swarm.getResourceStats();
      expect(resourceStats.agentCount).to.equal(100);
      expect(resourceStats.memoryUsage).to.be.lessThan(1024 * 1024 * 1024); // Under 1GB
      expect(resourceStats.cpuUsage).to.be.lessThan(80); // Under 80% CPU
    });

    it('should maintain response times under heavy load', async function() {
      this.timeout(30000);

      swarm = new RuvSwarm();
      await swarm.init({ 
        topology: 'mesh',
        maxAgents: 50,
        performance: {
          targetResponseTime: 100, // 100ms target
          autoOptimize: true
        }
      });

      // Create worker agents
      const agents = await Promise.all(
        Array(30).fill().map(() => swarm.spawnAgent({ type: 'coder' }))
      );

      // Measure response times under increasing load
      const loadLevels = [10, 50, 100, 200];
      const responseTimesByLoad = {};

      for (const load of loadLevels) {
        performanceMonitor.startTracking(`load-${load}`);
        
        const requests = [];
        const startTime = Date.now();

        // Send concurrent requests
        for (let i = 0; i < load; i++) {
          requests.push(swarm.executeAgentTask(
            agents[i % agents.length].id,
            { task: 'Quick computation', complexity: 'low' }
          ));
        }

        const results = await Promise.all(requests);
        const metrics = performanceMonitor.stopTracking(`load-${load}`);

        responseTimesByLoad[load] = {
          avgResponseTime: metrics.averageTimePerOperation,
          p95ResponseTime: metrics.percentile95,
          p99ResponseTime: metrics.percentile99,
          successRate: results.filter(r => r.success).length / results.length
        };
      }

      // Verify performance doesn't degrade significantly
      expect(responseTimesByLoad[10].avgResponseTime).to.be.lessThan(150);
      expect(responseTimesByLoad[50].avgResponseTime).to.be.lessThan(300);
      expect(responseTimesByLoad[100].avgResponseTime).to.be.lessThan(500);
      expect(responseTimesByLoad[200].avgResponseTime).to.be.lessThan(1000);

      // Success rate should remain high
      Object.values(responseTimesByLoad).forEach(metrics => {
        expect(metrics.successRate).to.be.at.least(0.95);
      });
    });

    it('should scale dynamically based on load', async function() {
      this.timeout(30000);

      swarm = new RuvSwarm();
      await swarm.init({ 
        topology: 'elastic',
        initialAgents: 5,
        scalingPolicy: {
          enabled: true,
          minAgents: 5,
          maxAgents: 50,
          scaleUpThreshold: 0.8,
          scaleDownThreshold: 0.3,
          cooldownPeriod: 2000
        }
      });

      // Monitor scaling events
      const scalingEvents = [];
      swarm.on('scaling', (event) => scalingEvents.push(event));

      // Start with baseline load
      let currentLoad = 10;
      const loadGenerator = setInterval(async () => {
        const tasks = [];
        for (let i = 0; i < currentLoad; i++) {
          tasks.push(swarm.orchestrateTask({
            task: 'Dynamic load task',
            duration: 500 // 500ms tasks
          }));
        }
        await Promise.all(tasks);
      }, 1000);

      // Increase load after 5 seconds
      setTimeout(() => { currentLoad = 50; }, 5000);

      // Decrease load after 10 seconds
      setTimeout(() => { currentLoad = 5; }, 10000);

      // Run for 15 seconds
      await new Promise(resolve => setTimeout(resolve, 15000));
      clearInterval(loadGenerator);

      // Verify scaling occurred
      expect(scalingEvents).to.have.length.at.least(2);
      
      const scaleUpEvents = scalingEvents.filter(e => e.action === 'scale-up');
      const scaleDownEvents = scalingEvents.filter(e => e.action === 'scale-down');
      
      expect(scaleUpEvents).to.have.length.at.least(1);
      expect(scaleDownEvents).to.have.length.at.least(1);

      // Final agent count should be reasonable
      const finalStatus = await swarm.getStatus();
      expect(finalStatus.agents.length).to.be.within(5, 20);
    });
  });

  describe('Memory Usage Optimization', () => {
    it('should maintain stable memory under sustained load', async function() {
      this.timeout(45000);

      swarm = new RuvSwarm();
      await swarm.init({ 
        topology: 'mesh',
        maxAgents: 20,
        memory: {
          monitoring: true,
          gcInterval: 2000,
          maxHeapUsage: 512 * 1024 * 1024 // 512MB
        }
      });

      // Track memory over time
      const memorySnapshots = [];
      const memoryInterval = setInterval(() => {
        memorySnapshots.push({
          timestamp: Date.now(),
          heapUsed: process.memoryUsage().heapUsed,
          external: process.memoryUsage().external,
          rss: process.memoryUsage().rss
        });
      }, 1000);

      // Create agents
      const agents = await Promise.all(
        Array(20).fill().map(() => swarm.spawnAgent({ type: 'coder' }))
      );

      // Sustained workload for 30 seconds
      const startTime = Date.now();
      const workloadTasks = [];

      while (Date.now() - startTime < 30000) {
        // Create tasks with varying memory requirements
        for (let i = 0; i < 5; i++) {
          workloadTasks.push(swarm.orchestrateTask({
            task: 'Memory test task',
            data: Buffer.alloc(Math.random() * 1024 * 1024), // 0-1MB random data
            strategy: 'parallel'
          }));
        }
        
        // Process in batches
        if (workloadTasks.length >= 20) {
          await Promise.all(workloadTasks.splice(0, 20));
        }
        
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // Process remaining tasks
      await Promise.all(workloadTasks);
      clearInterval(memoryInterval);

      // Analyze memory usage
      const avgMemory = memorySnapshots.reduce((sum, s) => sum + s.heapUsed, 0) / memorySnapshots.length;
      const maxMemory = Math.max(...memorySnapshots.map(s => s.heapUsed));
      const minMemory = Math.min(...memorySnapshots.map(s => s.heapUsed));
      const memoryVariance = maxMemory - minMemory;

      // Memory should be stable
      expect(avgMemory).to.be.lessThan(300 * 1024 * 1024); // Average under 300MB
      expect(maxMemory).to.be.lessThan(400 * 1024 * 1024); // Peak under 400MB
      expect(memoryVariance).to.be.lessThan(100 * 1024 * 1024); // Variance under 100MB

      // Check for memory leaks
      const firstHalf = memorySnapshots.slice(0, memorySnapshots.length / 2);
      const secondHalf = memorySnapshots.slice(memorySnapshots.length / 2);
      
      const avgFirstHalf = firstHalf.reduce((sum, s) => sum + s.heapUsed, 0) / firstHalf.length;
      const avgSecondHalf = secondHalf.reduce((sum, s) => sum + s.heapUsed, 0) / secondHalf.length;
      
      // Second half shouldn't be significantly higher (no leak)
      expect(avgSecondHalf).to.be.lessThan(avgFirstHalf * 1.2);
    });

    it('should efficiently handle large data processing', async function() {
      this.timeout(30000);

      swarm = new RuvSwarm();
      await swarm.init({ 
        topology: 'hierarchical',
        maxAgents: 10,
        streaming: {
          enabled: true,
          chunkSize: 1024 * 1024 // 1MB chunks
        }
      });

      // Create specialized agents
      const agents = await Promise.all([
        swarm.spawnAgent({ type: 'coder', specialization: 'data-processing' }),
        swarm.spawnAgent({ type: 'analyst', specialization: 'aggregation' }),
        swarm.spawnAgent({ type: 'optimizer', specialization: 'compression' })
      ]);

      // Process large dataset
      const largeDataSize = 50 * 1024 * 1024; // 50MB
      const chunks = 50;
      
      performanceMonitor.startTracking('large-data-processing');

      const processingTasks = [];
      for (let i = 0; i < chunks; i++) {
        processingTasks.push(swarm.orchestrateTask({
          task: 'Process data chunk',
          data: {
            chunkId: i,
            size: largeDataSize / chunks,
            operation: 'transform'
          },
          streaming: true
        }));
      }

      const results = await Promise.all(processingTasks);
      const metrics = performanceMonitor.stopTracking('large-data-processing');

      // Verify all chunks processed
      expect(results).to.have.lengthOf(chunks);
      results.forEach(result => {
        expect(result.success).to.be.true;
        expect(result.processed).to.be.true;
      });

      // Check processing efficiency
      const throughput = largeDataSize / (metrics.totalTime / 1000); // bytes per second
      expect(throughput).to.be.greaterThan(10 * 1024 * 1024); // At least 10MB/s

      // Memory shouldn't spike
      const memoryStats = await swarm.getMemoryStats();
      expect(memoryStats.peakUsage).to.be.lessThan(200 * 1024 * 1024); // Peak under 200MB
    });
  });

  describe('Concurrent Operation Performance', () => {
    it('should handle mixed workload efficiently', async function() {
      this.timeout(30000);

      swarm = new RuvSwarm();
      await swarm.init({ 
        topology: 'mesh',
        maxAgents: 30,
        workloadBalancing: {
          enabled: true,
          strategy: 'weighted-round-robin'
        }
      });

      // Create diverse agent pool
      const agents = await Promise.all([
        ...Array(10).fill().map(() => swarm.spawnAgent({ type: 'coder', weight: 3 })),
        ...Array(5).fill().map(() => swarm.spawnAgent({ type: 'researcher', weight: 2 })),
        ...Array(5).fill().map(() => swarm.spawnAgent({ type: 'analyst', weight: 2 })),
        ...Array(5).fill().map(() => swarm.spawnAgent({ type: 'optimizer', weight: 1 })),
        ...Array(5).fill().map(() => swarm.spawnAgent({ type: 'tester', weight: 1 }))
      ]);

      // Define mixed workload
      const workloadTypes = [
        { type: 'coding', duration: 200, frequency: 0.4 },
        { type: 'research', duration: 500, frequency: 0.2 },
        { type: 'analysis', duration: 300, frequency: 0.2 },
        { type: 'optimization', duration: 1000, frequency: 0.1 },
        { type: 'testing', duration: 400, frequency: 0.1 }
      ];

      // Generate mixed tasks
      const tasks = [];
      const totalTasks = 200;
      
      performanceMonitor.startTracking('mixed-workload');

      for (let i = 0; i < totalTasks; i++) {
        const rand = Math.random();
        let cumulativeFreq = 0;
        let selectedWorkload;

        for (const workload of workloadTypes) {
          cumulativeFreq += workload.frequency;
          if (rand < cumulativeFreq) {
            selectedWorkload = workload;
            break;
          }
        }

        tasks.push(swarm.orchestrateTask({
          task: `${selectedWorkload.type} task ${i}`,
          type: selectedWorkload.type,
          estimatedDuration: selectedWorkload.duration,
          affinityType: selectedWorkload.type.replace('ing', 'er')
        }));
      }

      const results = await Promise.all(tasks);
      const metrics = performanceMonitor.stopTracking('mixed-workload');

      // Analyze results by type
      const resultsByType = {};
      workloadTypes.forEach(w => {
        resultsByType[w.type] = {
          count: 0,
          successful: 0,
          totalTime: 0
        };
      });

      results.forEach((result, index) => {
        const taskType = tasks[index].type;
        if (resultsByType[taskType]) {
          resultsByType[taskType].count++;
          if (result.success) resultsByType[taskType].successful++;
          if (result.duration) resultsByType[taskType].totalTime += result.duration;
        }
      });

      // Verify balanced execution
      Object.entries(resultsByType).forEach(([type, stats]) => {
        const successRate = stats.successful / stats.count;
        expect(successRate).to.be.at.least(0.95);
        
        // Average duration should be close to estimated
        const avgDuration = stats.totalTime / stats.count;
        const expectedDuration = workloadTypes.find(w => w.type === type).duration;
        expect(avgDuration).to.be.within(expectedDuration * 0.8, expectedDuration * 1.5);
      });

      // Overall performance
      expect(metrics.totalTime).to.be.lessThan(10000); // Complete in under 10 seconds
      expect(metrics.successRate).to.be.at.least(0.95);
    });

    it('should optimize throughput with pipelining', async function() {
      this.timeout(30000);

      swarm = new RuvSwarm();
      await swarm.init({ 
        topology: 'pipeline',
        stages: [
          { name: 'input', agents: 3 },
          { name: 'processing', agents: 5 },
          { name: 'analysis', agents: 3 },
          { name: 'output', agents: 2 }
        ],
        pipelining: {
          enabled: true,
          bufferSize: 10
        }
      });

      // Create pipeline stages
      const pipeline = await swarm.createPipeline({
        stages: [
          { type: 'researcher', count: 3 },
          { type: 'coder', count: 5 },
          { type: 'analyst', count: 3 },
          { type: 'optimizer', count: 2 }
        ]
      });

      // Track throughput
      let processedItems = 0;
      const startTime = Date.now();
      
      pipeline.on('complete', () => processedItems++);

      // Feed items into pipeline
      const totalItems = 100;
      const feedInterval = setInterval(async () => {
        if (processedItems >= totalItems) {
          clearInterval(feedInterval);
          return;
        }

        // Feed multiple items
        const batch = Math.min(5, totalItems - processedItems);
        for (let i = 0; i < batch; i++) {
          pipeline.process({
            id: `item-${processedItems + i}`,
            data: `Processing item ${processedItems + i}`,
            complexity: Math.random() > 0.5 ? 'high' : 'low'
          });
        }
      }, 100);

      // Wait for completion
      while (processedItems < totalItems) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      const totalTime = Date.now() - startTime;
      const throughput = (processedItems / totalTime) * 1000; // items per second

      // Verify pipeline efficiency
      expect(throughput).to.be.greaterThan(10); // At least 10 items/second
      
      const pipelineStats = await pipeline.getStats();
      expect(pipelineStats.stages).to.have.lengthOf(4);
      
      // Each stage should have processed items
      pipelineStats.stages.forEach(stage => {
        expect(stage.processed).to.be.greaterThan(0);
        expect(stage.utilization).to.be.greaterThan(0.5); // At least 50% utilized
      });

      // No significant bottlenecks
      const utilizations = pipelineStats.stages.map(s => s.utilization);
      const minUtilization = Math.min(...utilizations);
      const maxUtilization = Math.max(...utilizations);
      expect(maxUtilization - minUtilization).to.be.lessThan(0.3); // Balanced pipeline
    });
  });

  describe('Resource Cleanup Verification', () => {
    it('should clean up all resources after stress test', async function() {
      this.timeout(30000);

      // Baseline measurements
      const initialMemory = process.memoryUsage();
      const initialHandles = process._getActiveHandles().length;
      const initialRequests = process._getActiveRequests().length;

      swarm = new RuvSwarm();
      await swarm.init({ 
        topology: 'mesh',
        maxAgents: 50,
        cleanup: {
          aggressive: true,
          trackLeaks: true
        }
      });

      // Stress test
      const agents = await Promise.all(
        Array(50).fill().map(() => swarm.spawnAgent({ type: 'coder' }))
      );

      // Execute many tasks
      const tasks = [];
      for (let i = 0; i < 100; i++) {
        tasks.push(swarm.orchestrateTask({
          task: `Cleanup test task ${i}`,
          timeout: 5000
        }));
      }

      await Promise.all(tasks);

      // Proper shutdown
      await swarm.shutdown();
      swarm = null;

      // Force garbage collection
      if (global.gc) {
        global.gc();
        await new Promise(resolve => setTimeout(resolve, 1000));
        global.gc();
      }

      // Verify cleanup
      const finalMemory = process.memoryUsage();
      const finalHandles = process._getActiveHandles().length;
      const finalRequests = process._getActiveRequests().length;

      // Memory should return close to baseline
      const memoryGrowth = finalMemory.heapUsed - initialMemory.heapUsed;
      expect(memoryGrowth).to.be.lessThan(10 * 1024 * 1024); // Less than 10MB growth

      // No leaked handles or requests
      expect(finalHandles).to.be.lessThanOrEqual(initialHandles + 2); // Allow small variance
      expect(finalRequests).to.be.lessThanOrEqual(initialRequests + 2);
    });
  });
});