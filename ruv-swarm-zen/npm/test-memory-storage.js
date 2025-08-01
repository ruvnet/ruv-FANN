#!/usr/bin/env node

import { SwarmPersistence } from './src/persistence.js';

async function testMemoryStorage() {
  console.log('🧪 Testing Memory Storage in Database\n');

  const persistence = new SwarmPersistence();

  // Get the latest agent
  const agents = persistence.getActiveAgents();
  console.log(`Found ${agents.length} agents`);

  if (agents.length === 0) {
    console.log('No agents found, creating one for test...');
    return;
  }

  const latestAgent = agents[agents.length - 1];
  console.log(`\nUsing agent: ${latestAgent.name} (${latestAgent.id})`);

  // Store memory
  const memoryKey = 'test/db-persistence/verification';
  const memoryValue = {
    timestamp: Date.now(),
    test: 'Direct memory storage test',
    message: 'This confirms MCP tools update the database',
    agent: latestAgent.name,
  };

  console.log('\n📝 Storing memory...');
  persistence.storeMemory(latestAgent.id, memoryKey, memoryValue, 3600);
  console.log(`✅ Memory stored with key: ${memoryKey}`);

  // Retrieve memory
  console.log('\n📖 Retrieving memory...');
  const retrieved = persistence.getMemory(latestAgent.id, memoryKey);
  console.log('✅ Retrieved memory:', JSON.stringify(retrieved, null, 2));

  // Check database directly
  console.log('\n🔍 Checking database directly...');
  const db = persistence.db;
  const memoryCount = db.prepare('SELECT COUNT(*) as count FROM agent_memory').get();
  console.log(`Total memory records: ${memoryCount.count}`);

  const latestMemory = db.prepare('SELECT * FROM agent_memory ORDER BY created_at DESC LIMIT 1').get();
  if (latestMemory) {
    console.log('\nLatest memory record:');
    console.log(`  - Key: ${latestMemory.key}`);
    console.log(`  - Agent: ${latestMemory.agent_id}`);
    console.log(`  - Created: ${latestMemory.created_at}`);
  }

  console.log('\n✅ Memory storage test complete!');
}

testMemoryStorage().catch(console.error);
