/**
 * Persistence Layer Tests for ruv-swarm
 * Tests database persistence, state recovery, and data integrity
 */

import assert from 'assert';
const sqlite3 = require('sqlite3').verbose();
import path from 'path';
import { promises as fs } from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Test database path
const TEST_DB_PATH = path.join(__dirname, 'test-swarm.db');

// SQLite persistence wrapper
class SwarmPersistence {
  constructor(dbPath) {
    this.dbPath = dbPath;
    this.db = null;
  }

  async connect() {
    return new Promise((resolve, reject) => {
      this.db = new sqlite3.Database(this.dbPath, (err) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      });
    });
  }

  async close() {
    return new Promise((resolve, reject) => {
      if (this.db) {
        this.db.close((err) => {
          if (err) {
            reject(err);
          } else {
            resolve();
          }
        });
      } else {
        resolve();
      }
    });
  }

  async initSchema() {
    const schema = `
            -- Agents table
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                capabilities TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Tasks table
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                description TEXT NOT NULL,
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'pending',
                assigned_agent TEXT,
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (assigned_agent) REFERENCES agents(id)
            );

            -- Memory storage table
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                ttl_seconds INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            );

            -- Swarm state table
            CREATE TABLE IF NOT EXISTS swarm_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_data TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Agent messages table
            CREATE TABLE IF NOT EXISTS agent_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES agents(id)
            );

            -- Performance metrics table
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_type TEXT NOT NULL,
                agent_id TEXT,
                value REAL NOT NULL,
                metadata TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Neural network weights table
            CREATE TABLE IF NOT EXISTS neural_weights (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                layer_index INTEGER NOT NULL,
                weights BLOB NOT NULL,
                biases BLOB,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES agents(id)
            );

            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_assigned ON tasks(assigned_agent);
            CREATE INDEX IF NOT EXISTS idx_memory_expires ON memory(expires_at);
            CREATE INDEX IF NOT EXISTS idx_messages_agent ON agent_messages(agent_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
        `;

    return new Promise((resolve, reject) => {
      this.db.exec(schema, (err) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      });
    });
  }

  // Agent operations
  async createAgent(agent) {
    const sql = `
            INSERT INTO agents (id, name, agent_type, status, capabilities)
            VALUES (?, ?, ?, ?, ?)
        `;
    return this.run(sql, [
      agent.id,
      agent.name,
      agent.agent_type,
      agent.status || 'active',
      JSON.stringify(agent.capabilities || {}),
    ]);
  }

  async getAgent(id) {
    const sql = 'SELECT * FROM agents WHERE id = ?';
    const row = await this.get(sql, [id]);
    if (row) {
      row.capabilities = JSON.parse(row.capabilities);
    }
    return row;
  }

  async updateAgentStatus(id, status) {
    const sql = `
            UPDATE agents 
            SET status = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        `;
    return this.run(sql, [status, id]);
  }

  // Task operations
  async createTask(task) {
    const sql = `
            INSERT INTO tasks (id, task_type, description, priority, status, assigned_agent)
            VALUES (?, ?, ?, ?, ?, ?)
        `;
    return this.run(sql, [
      task.id,
      task.task_type,
      task.description,
      task.priority || 'medium',
      task.status || 'pending',
      task.assigned_agent,
    ]);
  }

  async updateTaskStatus(id, status, result = null) {
    const sql = `
            UPDATE tasks 
            SET status = ?, 
                result = ?,
                started_at = CASE WHEN ? = 'in_progress' THEN CURRENT_TIMESTAMP ELSE started_at END,
                completed_at = CASE WHEN ? = 'completed' THEN CURRENT_TIMESTAMP ELSE completed_at END
            WHERE id = ?
        `;
    return this.run(sql, [status, result, status, status, id]);
  }

  // Memory operations
  async storeMemory(key, value, ttlSeconds = null) {
    const sql = `
            INSERT OR REPLACE INTO memory (key, value, ttl_seconds, expires_at)
            VALUES (?, ?, ?, ?)
        `;
    const expiresAt = ttlSeconds
      ? new Date(Date.now() + ttlSeconds * 1000).toISOString()
      : null;
    return this.run(sql, [key, JSON.stringify(value), ttlSeconds, expiresAt]);
  }

  async getMemory(key) {
    const sql = `
            SELECT * FROM memory 
            WHERE key = ? 
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
        `;
    const row = await this.get(sql, [key]);
    if (row) {
      return JSON.parse(row.value);
    }
    return null;
  }

  async cleanupExpiredMemory() {
    const sql = 'DELETE FROM memory WHERE expires_at <= CURRENT_TIMESTAMP';
    return this.run(sql);
  }

  // State operations
  async saveSwarmState(state) {
    const sql = 'INSERT INTO swarm_state (state_data) VALUES (?)';
    return this.run(sql, [JSON.stringify(state)]);
  }

  async getLatestSwarmState() {
    const sql = 'SELECT * FROM swarm_state ORDER BY timestamp DESC LIMIT 1';
    const row = await this.get(sql);
    if (row) {
      return JSON.parse(row.state_data);
    }
    return null;
  }

  // Neural weight operations
  async saveNeuralWeights(agentId, layerIndex, weights, biases = null) {
    const sql = `
            INSERT OR REPLACE INTO neural_weights (id, agent_id, layer_index, weights, biases)
            VALUES (?, ?, ?, ?, ?)
        `;
    const id = `${agentId}_layer_${layerIndex}`;
    return this.run(sql, [
      id,
      agentId,
      layerIndex,
      Buffer.from(weights),
      biases ? Buffer.from(biases) : null,
    ]);
  }

  async getNeuralWeights(agentId) {
    const sql = 'SELECT * FROM neural_weights WHERE agent_id = ? ORDER BY layer_index';
    return this.all(sql, [agentId]);
  }

  // Metrics operations
  async recordMetric(metricType, value, agentId = null, metadata = null) {
    const sql = `
            INSERT INTO metrics (metric_type, agent_id, value, metadata)
            VALUES (?, ?, ?, ?)
        `;
    return this.run(sql, [
      metricType,
      agentId,
      value,
      metadata ? JSON.stringify(metadata) : null,
    ]);
  }

  async getMetrics(metricType, since = null, agentId = null) {
    let sql = 'SELECT * FROM metrics WHERE metric_type = ?';
    const params = [metricType];

    if (since) {
      sql += ' AND timestamp >= ?';
      params.push(since);
    }

    if (agentId) {
      sql += ' AND agent_id = ?';
      params.push(agentId);
    }

    sql += ' ORDER BY timestamp DESC';
    return this.all(sql, params);
  }

  // Helper methods
  run(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.run(sql, params, function(err) {
        if (err) {
          reject(err);
        } else {
          resolve({ lastID: this.lastID, changes: this.changes });
        }
      });
    });
  }

  get(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.get(sql, params, (err, row) => {
        if (err) {
          reject(err);
        } else {
          resolve(row);
        }
      });
    });
  }

  all(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.all(sql, params, (err, rows) => {
        if (err) {
          reject(err);
        } else {
          resolve(rows);
        }
      });
    });
  }
}

// Test suites
async function runPersistenceTests() {
  console.log('🗄️  Starting Persistence Layer Tests\n');

  const results = {
    passed: 0,
    failed: 0,
    errors: [],
  };

  const db = new SwarmPersistence(TEST_DB_PATH);

  async function test(name, fn) {
    try {
      await fn();
      console.log(`✅ ${name}`);
      results.passed++;
    } catch (error) {
      console.error(`❌ ${name}`);
      console.error(`   ${error.message}`);
      results.failed++;
      results.errors.push({ test: name, error: error.message });
    }
  }

  try {
    // Setup
    await db.connect();
    await db.initSchema();

    // Test Agent Persistence
    await test('Agent Creation and Retrieval', async() => {
      const agent = {
        id: uuidv4(),
        name: 'test-researcher',
        agent_type: 'researcher',
        capabilities: {
          max_tokens: 4096,
          specialization: 'web_tech',
        },
      };

      await db.createAgent(agent);
      const retrieved = await db.getAgent(agent.id);

      assert(retrieved);
      assert.strictEqual(retrieved.id, agent.id);
      assert.strictEqual(retrieved.name, agent.name);
      assert.deepStrictEqual(retrieved.capabilities, agent.capabilities);
    });

    await test('Agent Status Update', async() => {
      const agentId = uuidv4();
      await db.createAgent({
        id: agentId,
        name: 'status-test',
        agent_type: 'coder',
      });

      await db.updateAgentStatus(agentId, 'busy');
      const agent = await db.getAgent(agentId);
      assert.strictEqual(agent.status, 'busy');
    });

    // Test Task Persistence
    await test('Task Creation and Status Updates', async() => {
      const agentId = uuidv4();
      await db.createAgent({
        id: agentId,
        name: 'task-agent',
        agent_type: 'coder',
      });

      const task = {
        id: uuidv4(),
        task_type: 'development',
        description: 'Implement user authentication',
        priority: 'high',
        assigned_agent: agentId,
      };

      await db.createTask(task);

      // Update to in_progress
      await db.updateTaskStatus(task.id, 'in_progress');
      let retrieved = await db.get('SELECT * FROM tasks WHERE id = ?', [task.id]);
      assert.strictEqual(retrieved.status, 'in_progress');
      assert(retrieved.started_at);

      // Complete task
      await db.updateTaskStatus(task.id, 'completed', 'Successfully implemented');
      retrieved = await db.get('SELECT * FROM tasks WHERE id = ?', [task.id]);
      assert.strictEqual(retrieved.status, 'completed');
      assert(retrieved.completed_at);
      assert.strictEqual(retrieved.result, 'Successfully implemented');
    });

    // Test Memory Persistence
    await test('Memory Storage and Retrieval', async() => {
      const key = 'test_config';
      const value = {
        framework: 'React',
        version: '18.0',
        features: ['hooks', 'concurrent', 'suspense'],
      };

      await db.storeMemory(key, value);
      const retrieved = await db.getMemory(key);
      assert.deepStrictEqual(retrieved, value);
    });

    await test('Memory TTL Expiration', async() => {
      const key = 'temp_data';
      const value = { temp: true };

      // Store with 1 second TTL
      await db.storeMemory(key, value, 1);

      // Should exist immediately
      let retrieved = await db.getMemory(key);
      assert.deepStrictEqual(retrieved, value);

      // Wait for expiration
      await new Promise(resolve => setTimeout(resolve, 1100));

      // Should be expired
      retrieved = await db.getMemory(key);
      assert.strictEqual(retrieved, null);
    });

    // Test Swarm State Persistence
    await test('Swarm State Save and Restore', async() => {
      const state = {
        topology: 'mesh',
        agents: ['agent1', 'agent2', 'agent3'],
        active_tasks: 5,
        configuration: {
          max_agents: 10,
          coordination_mode: 'distributed',
        },
      };

      await db.saveSwarmState(state);
      const retrieved = await db.getLatestSwarmState();
      assert.deepStrictEqual(retrieved, state);
    });

    // Test Neural Weights Persistence
    await test('Neural Weight Storage', async() => {
      const agentId = uuidv4();
      await db.createAgent({
        id: agentId,
        name: 'neural-agent',
        agent_type: 'analyst',
        capabilities: { neural_enabled: true },
      });

      // Simulate neural network weights
      const weights = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);
      const biases = new Float32Array([0.01, 0.02]);

      await db.saveNeuralWeights(agentId, 0, weights, biases);
      await db.saveNeuralWeights(agentId, 1, weights);

      const savedWeights = await db.getNeuralWeights(agentId);
      assert.strictEqual(savedWeights.length, 2);
      assert.strictEqual(savedWeights[0].layer_index, 0);
      assert(savedWeights[0].biases);
      assert.strictEqual(savedWeights[1].layer_index, 1);
    });

    // Test Metrics Persistence
    await test('Metrics Recording and Retrieval', async() => {
      const agentId = uuidv4();

      // Record various metrics
      await db.recordMetric('cpu_usage', 45.2, agentId);
      await db.recordMetric('memory_usage', 512.8, agentId);
      await db.recordMetric('task_completion_time', 3.5, agentId, {
        task_type: 'research',
      });

      // Retrieve metrics
      const cpuMetrics = await db.getMetrics('cpu_usage', null, agentId);
      assert(cpuMetrics.length >= 1);
      assert.strictEqual(cpuMetrics[0].value, 45.2);

      const allMetrics = await db.all(
        'SELECT DISTINCT metric_type FROM metrics WHERE agent_id = ?',
        [agentId],
      );
      assert.strictEqual(allMetrics.length, 3);
    });

    // Test Concurrent Operations
    await test('Concurrent Write Operations', async() => {
      const promises = [];

      // Create 10 agents concurrently
      for (let i = 0; i < 10; i++) {
        promises.push(db.createAgent({
          id: uuidv4(),
          name: `concurrent-agent-${i}`,
          agent_type: 'coder',
        }));
      }

      await Promise.all(promises);

      const agents = await db.all('SELECT COUNT(*) as count FROM agents');
      assert(agents[0].count >= 10);
    });

    // Test Transaction-like Operations
    await test('Complex State Update', async() => {
      const agentId = uuidv4();
      const taskId = uuidv4();

      // Create agent and task
      await db.createAgent({
        id: agentId,
        name: 'complex-agent',
        agent_type: 'analyst',
      });

      await db.createTask({
        id: taskId,
        task_type: 'analysis',
        description: 'Complex analysis task',
        assigned_agent: agentId,
      });

      // Update multiple states
      await Promise.all([
        db.updateAgentStatus(agentId, 'busy'),
        db.updateTaskStatus(taskId, 'in_progress'),
        db.recordMetric('task_started', 1, agentId),
        db.storeMemory(`task_${taskId}_state`, { phase: 'initialization' }),
      ]);

      // Verify all updates
      const agent = await db.getAgent(agentId);
      const task = await db.get('SELECT * FROM tasks WHERE id = ?', [taskId]);
      const taskState = await db.getMemory(`task_${taskId}_state`);

      assert.strictEqual(agent.status, 'busy');
      assert.strictEqual(task.status, 'in_progress');
      assert.deepStrictEqual(taskState, { phase: 'initialization' });
    });

    // Test Data Recovery
    await test('Database Recovery After Close/Reopen', async() => {
      // Store test data
      const testKey = 'recovery_test';
      const testValue = { important: 'data', timestamp: Date.now() };
      await db.storeMemory(testKey, testValue);

      // Close database
      await db.close();

      // Reopen and verify data persists
      await db.connect();
      const retrieved = await db.getMemory(testKey);
      assert.deepStrictEqual(retrieved, testValue);
    });

    // Test Cleanup Operations
    await test('Expired Memory Cleanup', async() => {
      // Create some expired entries
      for (let i = 0; i < 5; i++) {
        await db.run(
          'INSERT INTO memory (key, value, expires_at) VALUES (?, ?, ?)',
          [`expired_${i}`, '{}', new Date(Date.now() - 1000).toISOString()],
        );
      }

      // Create valid entries
      await db.storeMemory('valid_entry', { valid: true });

      // Run cleanup
      const result = await db.cleanupExpiredMemory();
      assert(result.changes >= 5);

      // Verify valid entry remains
      const valid = await db.getMemory('valid_entry');
      assert(valid);
    });

    // Test Performance with Large Dataset
    await test('Performance: Bulk Operations', async() => {
      const startTime = Date.now();
      const count = 100;

      // Bulk insert agents
      const agents = [];
      for (let i = 0; i < count; i++) {
        agents.push(db.createAgent({
          id: uuidv4(),
          name: `perf-agent-${i}`,
          agent_type: ['researcher', 'coder', 'analyst'][i % 3],
          capabilities: { index: i },
        }));
      }

      await Promise.all(agents);
      const insertTime = Date.now() - startTime;

      // Query performance
      const queryStart = Date.now();
      const coders = await db.all(
        'SELECT * FROM agents WHERE agent_type = ?',
        ['coder'],
      );
      const queryTime = Date.now() - queryStart;

      console.log(`   Inserted ${count} agents in ${insertTime}ms`);
      console.log(`   Queried ${coders.length} coders in ${queryTime}ms`);

      assert(insertTime < 5000); // Should complete within 5 seconds
      assert(queryTime < 100); // Query should be fast
    });

  } catch (error) {
    console.error('Test suite error:', error);
    results.failed++;
  } finally {
    await db.close();
    // Clean up test database
    try {
      await fs.unlink(TEST_DB_PATH);
    } catch (err) {
      // Ignore cleanup errors
    }
  }

  // Summary
  console.log('\n📊 Persistence Test Results');
  console.log('─'.repeat(50));
  console.log(`Total Tests: ${results.passed + results.failed}`);
  console.log(`✅ Passed: ${results.passed}`);
  console.log(`❌ Failed: ${results.failed}`);

  if (results.errors.length > 0) {
    console.log('\n❌ Failed Tests:');
    results.errors.forEach(e => {
      console.log(`  - ${e.test}: ${e.error}`);
    });
  }

  return results.failed === 0;
}

// Export for use in other test suites
export { SwarmPersistence, runPersistenceTests };

// Run tests when this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runPersistenceTests()
    .then(passed => process.exit(passed ? 0 : 1))
    .catch(error => {
      console.error('Fatal error:', error);
      process.exit(1);
    });
}