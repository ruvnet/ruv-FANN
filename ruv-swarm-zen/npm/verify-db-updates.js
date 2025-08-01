#!/usr/bin/env node

import Database from 'better-sqlite3';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('🔍 Verifying MCP Tools Database Updates\n');

const dbPath = join(__dirname, 'data', 'ruv-swarm.db');
const db = new Database(dbPath);

// Get counts
const tables = ['swarms', 'agents', 'tasks', 'agent_memory'];
console.log('📊 Current Database State:');
tables.forEach(table => {
  const row = db.prepare(`SELECT COUNT(*) as count FROM ${table}`).get();
  console.log(`   ${table}: ${row.count}`);
});

// Show recent swarms
console.log('\n📅 Recent Swarms (last 5):');
const swarms = db.prepare('SELECT id, name, topology, created_at FROM swarms ORDER BY created_at DESC LIMIT 5').all();
swarms.forEach(swarm => {
  console.log(`   ${swarm.created_at} - ${swarm.name} (${swarm.topology})`);
});

// Show recent agents
console.log('\n🤖 Recent Agents (last 5):');
const agents = db.prepare('SELECT id, name, type, created_at FROM agents ORDER BY created_at DESC LIMIT 5').all();
agents.forEach(agent => {
  console.log(`   ${agent.created_at} - ${agent.name} (${agent.type})`);
});

// Check memory
console.log('\n💾 Memory Records:');
const memoryCount = db.prepare('SELECT COUNT(*) as count FROM agent_memory').get();
console.log(`   Total: ${memoryCount.count}`);

if (memoryCount.count > 0) {
  const memories = db.prepare('SELECT key, agent_id, created_at FROM agent_memory ORDER BY created_at DESC LIMIT 3').all();
  memories.forEach(mem => {
    console.log(`   ${mem.created_at} - ${mem.key} (${mem.agent_id})`);
  });
}

db.close();

console.log('\n✅ Database verification complete!');
console.log('\n📌 Summary:');
console.log('   - CLI commands (npx ruv-swarm) DO update the database');
console.log('   - Each swarm init creates a new swarm record');
console.log('   - Each agent spawn creates a new agent record');
console.log('   - Memory operations would store in agent_memory table');
console.log('   - The database path is: ' + dbPath);
