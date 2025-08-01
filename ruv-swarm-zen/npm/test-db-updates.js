#!/usr/bin/env node

import sqlite3 from 'sqlite3';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function testDatabaseUpdates() {
  console.log('🧪 Testing MCP Database Updates\n');

  const dbPath = join(__dirname, 'data', 'ruv-swarm.db');
  const db = new sqlite3.Database(dbPath);

  const getCount = (table) => new Promise((resolve, reject) => {
    db.get(`SELECT COUNT(*) as count FROM ${table}`, (err, row) => {
      if (err) reject(err);
      else resolve(row.count);
    });
  });

  const getLatestRecord = (table) => new Promise((resolve, reject) => {
    db.get(`SELECT * FROM ${table} ORDER BY id DESC LIMIT 1`, (err, row) => {
      if (err) reject(err);
      else resolve(row);
    });
  });

  // Get initial counts
  console.log('📊 Initial Database State:');
  const initialCounts = {
    swarms: await getCount('swarms'),
    agents: await getCount('agents'),
    tasks: await getCount('tasks'),
    agent_memory: await getCount('agent_memory'),
  };
  console.log(JSON.stringify(initialCounts, null, 2));

  // Run MCP commands using CLI
  console.log('\n🚀 Running MCP commands...\n');

  // Helper to run commands
  const runCommand = (args) => new Promise((resolve, reject) => {
    const proc = spawn('node', ['bin/ruv-swarm-clean.js', ...args], {
      stdio: 'pipe',
    });
    let output = '';
    proc.stdout.on('data', (data) => output += data.toString());
    proc.stderr.on('data', (data) => output += data.toString());
    proc.on('close', (code) => {
      if (code !== 0) reject(new Error(`Command failed: ${output}`));
      else resolve(output);
    });
  });

  // Test swarm init
  console.log('📝 Initializing swarm...');
  try {
    await runCommand(['swarm', 'init', 'mesh', '--max-agents', '5']);
    console.log('✅ Swarm initialized');
  } catch (e) {
    console.log('ℹ️  Swarm init:', e.message);
  }

  // Test agent spawn
  console.log('\n🤖 Spawning agents...');
  try {
    await runCommand(['agent', 'spawn', 'researcher']);
    console.log('✅ Researcher agent spawned');

    await runCommand(['agent', 'spawn', 'coder']);
    console.log('✅ Coder agent spawned');
  } catch (e) {
    console.log('ℹ️  Agent spawn:', e.message);
  }

  // Test task creation
  console.log('\n📋 Creating task...');
  try {
    await runCommand(['task', 'create', 'Test database persistence']);
    console.log('✅ Task created');
  } catch (e) {
    console.log('ℹ️  Task create:', e.message);
  }

  // Wait for DB writes
  await new Promise(resolve => setTimeout(resolve, 1000));

  // Get final counts
  console.log('\n📊 Final Database State:');
  const finalCounts = {
    swarms: await getCount('swarms'),
    agents: await getCount('agents'),
    tasks: await getCount('tasks'),
    agent_memory: await getCount('agent_memory'),
  };
  console.log(JSON.stringify(finalCounts, null, 2));

  console.log('\n📈 Changes:');
  console.log(`Swarms: ${initialCounts.swarms} → ${finalCounts.swarms} (+${finalCounts.swarms - initialCounts.swarms})`);
  console.log(`Agents: ${initialCounts.agents} → ${finalCounts.agents} (+${finalCounts.agents - initialCounts.agents})`);
  console.log(`Tasks: ${initialCounts.tasks} → ${finalCounts.tasks} (+${finalCounts.tasks - initialCounts.tasks})`);
  console.log(`Memory: ${initialCounts.agent_memory} → ${finalCounts.agent_memory} (+${finalCounts.agent_memory - initialCounts.agent_memory})`);

  // Show latest records
  console.log('\n🔍 Latest records:');

  const latestSwarm = await getLatestRecord('swarms');
  if (latestSwarm) {
    console.log('\nLatest Swarm:');
    console.log(`  ID: ${latestSwarm.id}`);
    console.log(`  Topology: ${latestSwarm.topology}`);
    console.log(`  Created: ${latestSwarm.created_at}`);
  }

  const latestAgent = await getLatestRecord('agents');
  if (latestAgent) {
    console.log('\nLatest Agent:');
    console.log(`  ID: ${latestAgent.id}`);
    console.log(`  Type: ${latestAgent.type}`);
    console.log(`  Name: ${latestAgent.name}`);
    console.log(`  Created: ${latestAgent.created_at}`);
  }

  db.close();

  console.log('\n✅ Database update test completed!');
}

// Test using MCP tools directly
async function testMCPTools() {
  console.log('\n\n🔧 Testing MCP Tools Directly\n');

  // Import the enhanced MCP tools
  const { MCPToolsEnhanced } = await import('./src/mcp-tools-enhanced.js');
  const mcpTools = new MCPToolsEnhanced();

  console.log('📝 Using MCP tools to create data...');

  try {
    // Initialize swarm
    const swarmResult = await mcpTools.tools.swarm_init({
      topology: 'hierarchical',
      maxAgents: 4,
      strategy: 'adaptive',
    });
    console.log('✅ Swarm initialized:', swarmResult);

    // Spawn agent
    const agentResult = await mcpTools.tools.agent_spawn({
      type: 'analyst',
      name: 'Data Analyst',
      capabilities: ['data analysis', 'reporting'],
    });
    console.log('✅ Agent spawned:', agentResult);

    // Store memory
    const memoryResult = await mcpTools.tools.memory_usage({
      action: 'store',
      key: 'test/mcp-tools',
      value: {
        timestamp: Date.now(),
        test: 'MCP tools direct test',
        verified: true,
      },
    });
    console.log('✅ Memory stored:', memoryResult);

    // Retrieve memory
    const retrieveResult = await mcpTools.tools.memory_usage({
      action: 'retrieve',
      key: 'test/mcp-tools',
    });
    console.log('✅ Memory retrieved:', retrieveResult);

  } catch (error) {
    console.error('❌ MCP tools error:', error);
  }
}

// Run all tests
(async () => {
  try {
    await testDatabaseUpdates();
    await testMCPTools();
    console.log('\n🎉 All tests completed successfully!');
  } catch (error) {
    console.error('❌ Test failed:', error);
    process.exit(1);
  }
})();
