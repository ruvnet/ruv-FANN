/**
 * High-Availability SQLite Persistence Layer with Connection Pooling
 *
 * This is the production-ready version of the persistence layer that addresses
 * the connection pooling concerns identified in the production readiness assessment.
 *
 * Key improvements:
 * - Connection pooling for high concurrency
 * - Deadlock prevention through queuing
 * - Connection health monitoring
 * - Graceful degradation under load
 * - Proper resource lifecycle management
 */

import { SQLiteConnectionPool } from './sqlite-pool.js';
import path from 'path';
import fs from 'fs';

class SwarmPersistencePooled {
  constructor(dbPath = path.join(new URL('.', import.meta.url).pathname, '..', 'data', 'ruv-swarm.db'), options = {}) {
    this.dbPath = dbPath;
    this.options = {
      // Pool configuration
      maxReaders: options.maxReaders || 4,
      maxWorkers: options.maxWorkers || 2,

      // Performance settings
      mmapSize: options.mmapSize || 268435456, // 256MB
      cacheSize: options.cacheSize || -64000, // 64MB

      // High availability
      enableBackup: options.enableBackup || false,
      backupInterval: options.backupInterval || 3600000, // 1 hour

      ...options,
    };

    this.pool = null;
    this.initialized = false;
    this.initializing = false;

    // Statistics
    this.stats = {
      totalOperations: 0,
      totalErrors: 0,
      averageResponseTime: 0,
    };
  }

  async initialize() {
    if (this.initialized) return;
    if (this.initializing) {
      // Wait for initialization to complete
      return new Promise((resolve, reject) => {
        const checkInitialized = () => {
          if (this.initialized) {
            resolve();
          } else if (!this.initializing) {
            reject(new Error('Initialization failed'));
          } else {
            setTimeout(checkInitialized, 100);
          }
        };
        checkInitialized();
      });
    }

    this.initializing = true;

    try {
      // Ensure data directory exists
      const dataDir = path.dirname(this.dbPath);
      if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir, { recursive: true });
      }

      // Initialize connection pool
      this.pool = new SQLiteConnectionPool(this.dbPath, this.options);

      // Wait for pool to be ready
      await new Promise((resolve, reject) => {
        this.pool.once('ready', resolve);
        this.pool.once('error', reject);
      });

      // Initialize database schema
      await this.initDatabase();

      this.initialized = true;
      this.initializing = false;

    } catch (error) {
      this.initializing = false;
      throw error;
    }
  }

  async initDatabase() {
    // Create tables using write connection
    await this.pool.write(`
      CREATE TABLE IF NOT EXISTS swarms (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        topology TEXT NOT NULL,
        max_agents INTEGER NOT NULL,
        strategy TEXT,
        status TEXT DEFAULT 'active',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT
      )
    `);

    await this.pool.write(`
      CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,
        swarm_id TEXT,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        status TEXT DEFAULT 'idle',
        capabilities TEXT,
        neural_config TEXT,
        metrics TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (swarm_id) REFERENCES swarms(id)
      )
    `);

    await this.pool.write(`
      CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        swarm_id TEXT,
        description TEXT,
        priority TEXT DEFAULT 'medium',
        status TEXT DEFAULT 'pending',
        assigned_agents TEXT,
        result TEXT,
        error TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        completed_at DATETIME,
        execution_time_ms INTEGER,
        FOREIGN KEY (swarm_id) REFERENCES swarms(id)
      )
    `);

    await this.pool.write(`
      CREATE TABLE IF NOT EXISTS task_results (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        output TEXT,
        metrics TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (task_id) REFERENCES tasks(id),
        FOREIGN KEY (agent_id) REFERENCES agents(id)
      )
    `);

    await this.pool.write(`
      CREATE TABLE IF NOT EXISTS agent_memory (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT,
        ttl_secs INTEGER,
        expires_at DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (agent_id) REFERENCES agents(id),
        UNIQUE(agent_id, key)
      )
    `);

    await this.pool.write(`
      CREATE TABLE IF NOT EXISTS metrics (
        id TEXT PRIMARY KEY,
        entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        metric_value REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);

    await this.pool.write(`
      CREATE TABLE IF NOT EXISTS neural_networks (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        architecture TEXT NOT NULL,
        weights TEXT,
        training_data TEXT,
        performance_metrics TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (agent_id) REFERENCES agents(id)
      )
    `);

    await this.pool.write(`
      CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        swarm_id TEXT,
        event_type TEXT NOT NULL,
        event_data TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Create indexes for better performance
    const indexes = [
      'CREATE INDEX IF NOT EXISTS idx_agents_swarm ON agents(swarm_id)',
      'CREATE INDEX IF NOT EXISTS idx_tasks_swarm ON tasks(swarm_id)',
      'CREATE INDEX IF NOT EXISTS idx_task_results_task ON task_results(task_id)',
      'CREATE INDEX IF NOT EXISTS idx_task_results_agent ON task_results(agent_id)',
      'CREATE INDEX IF NOT EXISTS idx_agent_memory_agent ON agent_memory(agent_id)',
      'CREATE INDEX IF NOT EXISTS idx_metrics_entity ON metrics(entity_type, entity_id)',
      'CREATE INDEX IF NOT EXISTS idx_events_swarm ON events(swarm_id)',
      'CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)',
    ];

    for (const index of indexes) {
      await this.pool.write(index);
    }
  }

  async ensureInitialized() {
    if (!this.initialized) {
      await this.initialize();
    }
  }

  async withRetry(operation, maxRetries = 3) {
    let lastError;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;

        // Don't retry on certain errors
        if (error.message.includes('UNIQUE constraint failed') ||
            error.message.includes('NOT NULL constraint failed')) {
          throw error;
        }

        // Wait before retry
        if (attempt < maxRetries - 1) {
          await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 100));
        }
      }
    }

    throw lastError;
  }

  async trackOperation(operation) {
    const startTime = Date.now();

    try {
      const result = await operation();

      // Update statistics
      this.stats.totalOperations++;
      const duration = Date.now() - startTime;
      this.stats.averageResponseTime =
        (this.stats.averageResponseTime + duration) / 2;

      return result;
    } catch (error) {
      this.stats.totalErrors++;
      throw error;
    }
  }

  // Swarm operations
  async createSwarm(swarm) {
    await this.ensureInitialized();

    return this.trackOperation(() => this.withRetry(async () => {
      // Check if swarm already exists
      const existing = await this.pool.read('SELECT id FROM swarms WHERE id = ?', [swarm.id]);
      if (existing && existing.length > 0) {
        // Return existing swarm info instead of failing
        return { id: swarm.id, changes: 0, lastInsertRowid: null };
      }

      return this.pool.write(`
        INSERT INTO swarms (id, name, topology, max_agents, strategy, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
      `, [
        swarm.id,
        swarm.name,
        swarm.topology,
        swarm.maxAgents,
        swarm.strategy,
        JSON.stringify(swarm.metadata || {}),
      ]);
    }));
  }

  async getActiveSwarms() {
    await this.ensureInitialized();

    return this.trackOperation(async () => {
      const swarms = await this.pool.read('SELECT * FROM swarms WHERE status = ?', ['active']);
      return swarms.map(s => ({
        ...s,
        metadata: JSON.parse(s.metadata || '{}'),
      }));
    });
  }

  // Agent operations
  async createAgent(agent) {
    await this.ensureInitialized();

    return this.trackOperation(() => this.withRetry(() =>
      this.pool.write(`
        INSERT INTO agents (id, swarm_id, name, type, capabilities, neural_config, metrics)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `, [
        agent.id,
        agent.swarmId,
        agent.name,
        agent.type,
        JSON.stringify(agent.capabilities || []),
        JSON.stringify(agent.neuralConfig || {}),
        JSON.stringify(agent.metrics || {}),
      ]),
    ));
  }

  async updateAgentStatus(agentId, status) {
    await this.ensureInitialized();

    return this.trackOperation(() => this.withRetry(() =>
      this.pool.write('UPDATE agents SET status = ? WHERE id = ?', [status, agentId]),
    ));
  }

  async getAgent(id) {
    await this.ensureInitialized();

    return this.trackOperation(async () => {
      const agents = await this.pool.read('SELECT * FROM agents WHERE id = ?', [id]);
      if (agents.length === 0) return null;

      const agent = agents[0];
      return {
        ...agent,
        capabilities: JSON.parse(agent.capabilities || '[]'),
        neural_config: JSON.parse(agent.neural_config || '{}'),
        metrics: JSON.parse(agent.metrics || '{}'),
      };
    });
  }

  async getSwarmAgents(swarmId, filter = 'all') {
    await this.ensureInitialized();

    return this.trackOperation(async () => {
      let sql = 'SELECT * FROM agents WHERE swarm_id = ?';
      const params = [swarmId];

      if (filter !== 'all') {
        sql += ' AND status = ?';
        params.push(filter);
      }

      const agents = await this.pool.read(sql, params);
      return agents.map(a => ({
        ...a,
        capabilities: JSON.parse(a.capabilities || '[]'),
        neural_config: JSON.parse(a.neural_config || '{}'),
        metrics: JSON.parse(a.metrics || '{}'),
      }));
    });
  }

  // Task operations
  async createTask(task) {
    await this.ensureInitialized();

    return this.trackOperation(() => this.withRetry(() =>
      this.pool.write(`
        INSERT INTO tasks (id, swarm_id, description, priority, status, assigned_agents)
        VALUES (?, ?, ?, ?, ?, ?)
      `, [
        task.id,
        task.swarmId,
        task.description,
        task.priority || 'medium',
        task.status || 'pending',
        JSON.stringify(task.assignedAgents || []),
      ]),
    ));
  }

  async updateTask(taskId, updates) {
    await this.ensureInitialized();

    return this.trackOperation(() => this.withRetry(() => {
      const fields = [];
      const values = [];

      Object.entries(updates).forEach(([key, value]) => {
        if (key === 'assignedAgents' || key === 'result') {
          fields.push(`${key} = ?`);
          values.push(JSON.stringify(value));
        } else {
          fields.push(`${key} = ?`);
          values.push(value);
        }
      });

      values.push(taskId);
      return this.pool.write(`UPDATE tasks SET ${fields.join(', ')} WHERE id = ?`, values);
    }));
  }

  async getTask(id) {
    await this.ensureInitialized();

    return this.trackOperation(async () => {
      const tasks = await this.pool.read('SELECT * FROM tasks WHERE id = ?', [id]);
      if (tasks.length === 0) return null;

      const task = tasks[0];
      return {
        ...task,
        assigned_agents: JSON.parse(task.assigned_agents || '[]'),
        result: task.result ? JSON.parse(task.result) : null,
      };
    });
  }

  async getSwarmTasks(swarmId, status = null) {
    await this.ensureInitialized();

    return this.trackOperation(async () => {
      let sql = 'SELECT * FROM tasks WHERE swarm_id = ?';
      const params = [swarmId];

      if (status) {
        sql += ' AND status = ?';
        params.push(status);
      }

      const tasks = await this.pool.read(sql, params);
      return tasks.map(t => ({
        ...t,
        assigned_agents: JSON.parse(t.assigned_agents || '[]'),
        result: t.result ? JSON.parse(t.result) : null,
      }));
    });
  }

  // Memory operations
  async storeMemory(agentId, key, value, ttlSecs = null) {
    await this.ensureInitialized();

    return this.trackOperation(() => this.withRetry(() => {
      const expiresAt = ttlSecs ? new Date(Date.now() + ttlSecs * 1000).toISOString() : null;
      const id = `mem_${agentId}_${Date.now()}`;

      return this.pool.write(`
        INSERT OR REPLACE INTO agent_memory (id, agent_id, key, value, ttl_secs, expires_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
      `, [id, agentId, key, JSON.stringify(value), ttlSecs, expiresAt]);
    }));
  }

  async getMemory(agentId, key) {
    await this.ensureInitialized();

    return this.trackOperation(async () => {
      // First cleanup expired entries
      await this.cleanupExpiredMemory();

      const memories = await this.pool.read(`
        SELECT * FROM agent_memory 
        WHERE agent_id = ? AND key = ? 
        AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
      `, [agentId, key]);

      if (memories.length === 0) return null;

      const memory = memories[0];
      return {
        ...memory,
        value: JSON.parse(memory.value),
      };
    });
  }

  async getAllMemory(agentId) {
    await this.ensureInitialized();

    return this.trackOperation(async () => {
      // First cleanup expired entries
      await this.cleanupExpiredMemory();

      const memories = await this.pool.read(`
        SELECT * FROM agent_memory 
        WHERE agent_id = ? 
        AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
        ORDER BY updated_at DESC
      `, [agentId]);

      return memories.map(m => ({
        ...m,
        value: JSON.parse(m.value),
      }));
    });
  }

  async deleteMemory(agentId, key) {
    await this.ensureInitialized();

    return this.trackOperation(() => this.withRetry(() =>
      this.pool.write('DELETE FROM agent_memory WHERE agent_id = ? AND key = ?', [agentId, key]),
    ));
  }

  async cleanupExpiredMemory() {
    await this.ensureInitialized();

    return this.trackOperation(() => this.withRetry(() =>
      this.pool.write('DELETE FROM agent_memory WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP'),
    ));
  }

  // Neural network operations
  async storeNeuralNetwork(network) {
    await this.ensureInitialized();

    return this.trackOperation(() => this.withRetry(() => {
      const id = `nn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      return this.pool.write(`
        INSERT INTO neural_networks (id, agent_id, architecture, weights, training_data, performance_metrics)
        VALUES (?, ?, ?, ?, ?, ?)
      `, [
        id,
        network.agentId,
        JSON.stringify(network.architecture),
        JSON.stringify(network.weights),
        JSON.stringify(network.trainingData || {}),
        JSON.stringify(network.performanceMetrics || {}),
      ]);
    }));
  }

  async updateNeuralNetwork(id, updates) {
    await this.ensureInitialized();

    return this.trackOperation(() => this.withRetry(() => {
      const fields = [];
      const values = [];

      Object.entries(updates).forEach(([key, value]) => {
        fields.push(`${key} = ?`);
        values.push(JSON.stringify(value));
      });

      fields.push('updated_at = CURRENT_TIMESTAMP');
      values.push(id);

      return this.pool.write(`UPDATE neural_networks SET ${fields.join(', ')} WHERE id = ?`, values);
    }));
  }

  async getAgentNeuralNetworks(agentId) {
    await this.ensureInitialized();

    return this.trackOperation(async () => {
      const networks = await this.pool.read('SELECT * FROM neural_networks WHERE agent_id = ?', [agentId]);

      return networks.map(n => ({
        ...n,
        architecture: JSON.parse(n.architecture),
        weights: JSON.parse(n.weights),
        training_data: JSON.parse(n.training_data || '{}'),
        performance_metrics: JSON.parse(n.performance_metrics || '{}'),
      }));
    });
  }

  // Metrics operations
  async recordMetric(entityType, entityId, metricName, metricValue) {
    await this.ensureInitialized();

    return this.trackOperation(() => this.withRetry(() => {
      const id = `metric_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      return this.pool.write(`
        INSERT INTO metrics (id, entity_type, entity_id, metric_name, metric_value)
        VALUES (?, ?, ?, ?, ?)
      `, [id, entityType, entityId, metricName, metricValue]);
    }));
  }

  async getMetrics(entityType, entityId, metricName = null) {
    await this.ensureInitialized();

    return this.trackOperation(async () => {
      let sql = 'SELECT * FROM metrics WHERE entity_type = ? AND entity_id = ?';
      const params = [entityType, entityId];

      if (metricName) {
        sql += ' AND metric_name = ?';
        params.push(metricName);
      }

      sql += ' ORDER BY timestamp DESC LIMIT 100';

      return this.pool.read(sql, params);
    });
  }

  // Event logging
  async logEvent(swarmId, eventType, eventData) {
    await this.ensureInitialized();

    return this.trackOperation(() => this.withRetry(() =>
      this.pool.write(`
        INSERT INTO events (swarm_id, event_type, event_data)
        VALUES (?, ?, ?)
      `, [swarmId, eventType, JSON.stringify(eventData)]),
    ));
  }

  async getSwarmEvents(swarmId, limit = 100) {
    await this.ensureInitialized();

    return this.trackOperation(async () => {
      const events = await this.pool.read(`
        SELECT * FROM events 
        WHERE swarm_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
      `, [swarmId, limit]);

      return events.map(e => ({
        ...e,
        event_data: JSON.parse(e.event_data || '{}'),
      }));
    });
  }

  // Cleanup operations
  async cleanup() {
    await this.ensureInitialized();

    return this.trackOperation(async () => {
      // Delete expired memories
      await this.cleanupExpiredMemory();

      // Delete old events (older than 7 days)
      const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
      await this.pool.write('DELETE FROM events WHERE timestamp < ?', [sevenDaysAgo]);

      // Delete old metrics (older than 30 days)
      const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
      await this.pool.write('DELETE FROM metrics WHERE timestamp < ?', [thirtyDaysAgo]);

      // Vacuum to reclaim space
      await this.pool.write('VACUUM');
    });
  }

  // Get pool statistics
  getPoolStats() {
    return this.pool ? this.pool.getStats() : null;
  }

  // Get persistence statistics
  getPersistenceStats() {
    return this.stats;
  }

  // Check if pool is healthy
  isHealthy() {
    return this.pool ? this.pool.isHealthy : false;
  }

  // Close database connection
  async close() {
    if (this.pool) {
      await this.pool.close();
    }
  }
}

export { SwarmPersistencePooled };
