#!/usr/bin/env node

/**
 * Test script to demonstrate memory optimization
 */

import { NeuralCLI, MemoryOptimizer, PATTERN_MEMORY_CONFIG } from '../src/neural';
import { NeuralAgentFactory, COGNITIVE_PATTERNS } from '../src/neural-agent';

async function testMemoryOptimization() {
  console.log('🧠 Testing Memory Optimization\n');

  const neuralCLI = new NeuralCLI();
  await neuralCLI.initialize();

  // Test 1: Check memory usage before optimization
  console.log('📊 Test 1: Memory Usage Before Optimization');
  console.log('Pattern      | Memory Usage');
  console.log('-------------|-------------');

  const beforeOptimization = {};
  for (const pattern of Object.keys(PATTERN_MEMORY_CONFIG)) {
    const memory = await neuralCLI.getPatternMemoryUsage(pattern);
    beforeOptimization[pattern] = memory;
    console.log(`${pattern.padEnd(12)} | ${memory.toFixed(0)} MB`);
  }

  // Test 2: Initialize memory pools
  console.log('\n📊 Test 2: Initializing Memory Pools...');
  await neuralCLI.initializeMemoryPools();

  // Test 3: Check memory usage after optimization
  console.log('\n📊 Test 3: Memory Usage After Optimization');
  console.log('Pattern      | Before | After  | Reduction');
  console.log('-------------|--------|--------|----------');

  for (const pattern of Object.keys(PATTERN_MEMORY_CONFIG)) {
    const beforeMem = beforeOptimization[pattern];
    const afterMem = await neuralCLI.getPatternMemoryUsage(pattern);
    const reduction = ((beforeMem - afterMem) / beforeMem * 100).toFixed(1);

    console.log(
      `${pattern.padEnd(12)} | ${beforeMem.toFixed(0).padStart(6)} | ${afterMem.toFixed(0).padStart(6)} | ${reduction.padStart(8)}%`,
    );
  }

  // Test 4: Check memory pool statistics
  console.log('\n📊 Test 4: Memory Pool Statistics');
  const poolStats = neuralCLI.memoryOptimizer.getPoolStats();
  console.log(JSON.stringify(poolStats, null, 2));

  // Test 5: Create neural agents and check their memory
  console.log('\n📊 Test 5: Neural Agent Memory Usage');
  await NeuralAgentFactory.initializeFactory();

  const agents = [];
  for (const agentType of ['researcher', 'coder', 'analyst']) {
    const agent = NeuralAgentFactory.createNeuralAgent(
      { id: `test-${agentType}`, type: agentType },
      agentType,
    );
    agents.push(agent);

    const status = agent.getStatus();
    console.log(`\n${agentType} agent:`);
    console.log(`  Current Memory: ${status.neuralState.memoryUsage.current}`);
    console.log(`  Baseline Memory: ${status.neuralState.memoryUsage.baseline}`);
    console.log(`  Memory Efficiency: ${status.neuralState.memoryUsage.efficiency}`);
  }

  // Test 6: Simulate garbage collection
  console.log('\n📊 Test 6: Testing Garbage Collection');
  const collected = await neuralCLI.memoryOptimizer.garbageCollect();
  console.log(`Collected ${collected} old allocations`);

  console.log('\n✅ Memory optimization tests complete!');
}

// Run the test
testMemoryOptimization().catch(console.error);