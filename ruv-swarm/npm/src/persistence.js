/**
 * SQLite Persistence Layer for ruv-swarm MCP
 */

import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';

class SwarmPersistence {
  constructor(dbPath = path.join(new URL('.', import.meta.url).pathname, '..', 'data', 'ruv-swarm.db')) {
    // Ensure data directory exists
    const dataDir = path.dirname(dbPath);
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    this.db = new Database(dbPath);
    this.initDatabase();
  }

  initDatabase() {
    // Enable foreign keys
    this.db.exec('PRAGMA foreign_keys = ON');

    // Create tables
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS swarms (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        topology TEXT NOT NULL,
        max_agents INTEGER NOT NULL,
        strategy TEXT,
        status TEXT DEFAULT 'active',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT
      );
      
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
      );
      
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
      );
      
      CREATE TABLE IF NOT EXISTS task_results (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        output TEXT,
        metrics TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (task_id) REFERENCES tasks(id),
        FOREIGN KEY (agent_id) REFERENCES agents(id)
      );
      
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
      );
      
      CREATE TABLE IF NOT EXISTS metrics (
        id TEXT PRIMARY KEY,
        entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        metric_value REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
      );
      
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
      );
      
      CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        swarm_id TEXT,
        event_type TEXT NOT NULL,
        event_data TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
      );
      
      -- Create indexes
      CREATE INDEX IF NOT EXISTS idx_agents_swarm ON agents(swarm_id);
      CREATE INDEX IF NOT EXISTS idx_tasks_swarm ON tasks(swarm_id);
      CREATE INDEX IF NOT EXISTS idx_task_results_task ON task_results(task_id);
      CREATE INDEX IF NOT EXISTS idx_task_results_agent ON task_results(agent_id);
      CREATE INDEX IF NOT EXISTS idx_agent_memory_agent ON agent_memory(agent_id);
      CREATE INDEX IF NOT EXISTS idx_metrics_entity ON metrics(entity_type, entity_id);
      CREATE INDEX IF NOT EXISTS idx_events_swarm ON events(swarm_id);
      CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
    `);
  }

  // Swarm operations
  createSwarm(swarm) {
    const stmt = this.db.prepare(`
      INSERT INTO swarms (id, name, topology, max_agents, strategy, metadata)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    return stmt.run(
      swarm.id,
      swarm.name,
      swarm.topology,
      swarm.maxAgents,
      swarm.strategy,
      JSON.stringify(swarm.metadata || {}),
    );
  }

  getActiveSwarms() {
    const stmt = this.db.prepare('SELECT * FROM swarms WHERE status = ?');
    const swarms = stmt.all('active');
    return swarms.map(s => {
      s.metadata = JSON.parse(s.metadata || '{}');
      return s;
    });
  }

  // Agent operations
  createAgent(agent) {
    const stmt = this.db.prepare(`
      INSERT INTO agents (id, swarm_id, name, type, capabilities, neural_config, metrics)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);
    return stmt.run(
      agent.id,
      agent.swarmId,
      agent.name,
      agent.type,
      JSON.stringify(agent.capabilities || []),
      JSON.stringify(agent.neuralConfig || {}),
      JSON.stringify(agent.metrics || {}),
    );
  }

  updateAgentStatus(agentId, status) {
    const stmt = this.db.prepare('UPDATE agents SET status = ? WHERE id = ?');
    return stmt.run(status, agentId);
  }

  getAgent(id) {
    const stmt = this.db.prepare('SELECT * FROM agents WHERE id = ?');
    const agent = stmt.get(id);
    if (agent) {
      agent.capabilities = JSON.parse(agent.capabilities || '[]');
      agent.neural_config = JSON.parse(agent.neural_config || '{}');
      agent.metrics = JSON.parse(agent.metrics || '{}');
    }
    return agent;
  }

  getSwarmAgents(swarmId, filter = 'all') {
    let query = 'SELECT * FROM agents WHERE swarm_id = ?';
    const params = [swarmId];

    if (filter !== 'all') {
      query += ' AND status = ?';
      params.push(filter);
    }

    const stmt = this.db.prepare(query);
    const agents = stmt.all(...params);

    return agents.map(a => {
      a.capabilities = JSON.parse(a.capabilities || '[]');
      a.neural_config = JSON.parse(a.neural_config || '{}');
      a.metrics = JSON.parse(a.metrics || '{}');
      return a;
    });
  }

  // Task operations
  createTask(task) {
    const stmt = this.db.prepare(`
      INSERT INTO tasks (id, swarm_id, description, priority, status, assigned_agents)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    return stmt.run(
      task.id,
      task.swarmId,
      task.description,
      task.priority || 'medium',
      task.status || 'pending',
      JSON.stringify(task.assignedAgents || []),
    );
  }

  updateTask(taskId, updates) {
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
    const stmt = this.db.prepare(`UPDATE tasks SET ${fields.join(', ')} WHERE id = ?`);
    return stmt.run(...values);
  }

  getTask(id) {
    const stmt = this.db.prepare('SELECT * FROM tasks WHERE id = ?');
    const task = stmt.get(id);
    if (task) {
      task.assigned_agents = JSON.parse(task.assigned_agents || '[]');
      task.result = task.result ? JSON.parse(task.result) : null;
    }
    return task;
  }

  getSwarmTasks(swarmId, status = null) {
    let query = 'SELECT * FROM tasks WHERE swarm_id = ?';
    const params = [swarmId];

    if (status) {
      query += ' AND status = ?';
      params.push(status);
    }

    const stmt = this.db.prepare(query);
    const tasks = stmt.all(...params);

    return tasks.map(t => {
      t.assigned_agents = JSON.parse(t.assigned_agents || '[]');
      t.result = t.result ? JSON.parse(t.result) : null;
      return t;
    });
  }

  // Memory operations
  storeAgentMemory(agentId, key, value) {
    const stmt = this.db.prepare(`
      INSERT INTO agent_memory (id, agent_id, key, value)
      VALUES (?, ?, ?, ?)
      ON CONFLICT(agent_id, key) DO UPDATE SET 
        value = excluded.value,
        updated_at = CURRENT_TIMESTAMP
    `);
    const id = `mem_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return stmt.run(id, agentId, key, JSON.stringify(value));
  }

  getAgentMemory(agentId, key = null) {
    if (key) {
      const stmt = this.db.prepare('SELECT * FROM agent_memory WHERE agent_id = ? AND key = ?');
      const memory = stmt.get(agentId, key);
      if (memory) {
        memory.value = JSON.parse(memory.value);
      }
      return memory;
    }
    const stmt = this.db.prepare('SELECT * FROM agent_memory WHERE agent_id = ?');
    const memories = stmt.all(agentId);
    return memories.map(m => {
      m.value = JSON.parse(m.value);
      return m;
    });

  }

  // Neural network operations
  storeNeuralNetwork(network) {
    const stmt = this.db.prepare(`
      INSERT INTO neural_networks (id, agent_id, architecture, weights, training_data, performance_metrics)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    const id = `nn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return stmt.run(
      id,
      network.agentId,
      JSON.stringify(network.architecture),
      JSON.stringify(network.weights),
      JSON.stringify(network.trainingData || {}),
      JSON.stringify(network.performanceMetrics || {}),
    );
  }

  updateNeuralNetwork(id, updates) {
    const fields = [];
    const values = [];

    Object.entries(updates).forEach(([key, value]) => {
      fields.push(`${key} = ?`);
      values.push(JSON.stringify(value));
    });

    fields.push('updated_at = CURRENT_TIMESTAMP');
    values.push(id);

    const stmt = this.db.prepare(`UPDATE neural_networks SET ${fields.join(', ')} WHERE id = ?`);
    return stmt.run(...values);
  }

  getAgentNeuralNetworks(agentId) {
    const stmt = this.db.prepare('SELECT * FROM neural_networks WHERE agent_id = ?');
    const networks = stmt.all(agentId);

    return networks.map(n => {
      n.architecture = JSON.parse(n.architecture);
      n.weights = JSON.parse(n.weights);
      n.training_data = JSON.parse(n.training_data || '{}');
      n.performance_metrics = JSON.parse(n.performance_metrics || '{}');
      return n;
    });
  }

  // Metrics operations
  recordMetric(entityType, entityId, metricName, metricValue) {
    const stmt = this.db.prepare(`
      INSERT INTO metrics (id, entity_type, entity_id, metric_name, metric_value)
      VALUES (?, ?, ?, ?, ?)
    `);
    const id = `metric_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return stmt.run(id, entityType, entityId, metricName, metricValue);
  }

  getMetrics(entityType, entityId, metricName = null) {
    let query = 'SELECT * FROM metrics WHERE entity_type = ? AND entity_id = ?';
    const params = [entityType, entityId];

    if (metricName) {
      query += ' AND metric_name = ?';
      params.push(metricName);
    }

    query += ' ORDER BY timestamp DESC LIMIT 100';

    const stmt = this.db.prepare(query);
    return stmt.all(...params);
  }

  // Event logging
  logEvent(swarmId, eventType, eventData) {
    const stmt = this.db.prepare(`
      INSERT INTO events (swarm_id, event_type, event_data)
      VALUES (?, ?, ?)
    `);
    return stmt.run(swarmId, eventType, JSON.stringify(eventData));
  }

  getSwarmEvents(swarmId, limit = 100) {
    const stmt = this.db.prepare(`
      SELECT * FROM events 
      WHERE swarm_id = ? 
      ORDER BY timestamp DESC 
      LIMIT ?
    `);
    const events = stmt.all(swarmId, limit);

    return events.map(e => {
      e.event_data = JSON.parse(e.event_data || '{}');
      return e;
    });
  }

  // Memory operations
  storeMemory(agentId, key, value, ttlSecs = null) {
    const expiresAt = ttlSecs ? new Date(Date.now() + ttlSecs * 1000).toISOString() : null;

    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO agent_memory (id, agent_id, key, value, ttl_secs, expires_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    `);

    const id = `mem_${agentId}_${Date.now()}`;
    return stmt.run(id, agentId, key, JSON.stringify(value), ttlSecs, expiresAt);
  }

  getMemory(agentId, key) {
    // First cleanup expired entries
    this.cleanupExpiredMemory();

    const stmt = this.db.prepare(`
      SELECT * FROM agent_memory 
      WHERE agent_id = ? AND key = ? 
      AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
    `);

    const memory = stmt.get(agentId, key);
    return memory ? {
      ...memory,
      value: JSON.parse(memory.value),
    } : null;
  }

  getAllMemory(agentId) {
    // First cleanup expired entries
    this.cleanupExpiredMemory();

    const stmt = this.db.prepare(`
      SELECT * FROM agent_memory 
      WHERE agent_id = ? 
      AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
      ORDER BY updated_at DESC
    `);

    const memories = stmt.all(agentId);
    return memories.map(m => ({
      ...m,
      value: JSON.parse(m.value),
    }));
  }

  deleteMemory(agentId, key) {
    const stmt = this.db.prepare('DELETE FROM agent_memory WHERE agent_id = ? AND key = ?');
    return stmt.run(agentId, key);
  }

  cleanupExpiredMemory() {
    const stmt = this.db.prepare('DELETE FROM agent_memory WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP');
    return stmt.run();
  }

  // Cleanup operations
  cleanup() {
    // Delete expired memories
    this.cleanupExpiredMemory();

    // Delete old events (older than 7 days)
    const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
    this.db.prepare('DELETE FROM events WHERE timestamp < ?').run(sevenDaysAgo);

    // Delete old metrics (older than 30 days)
    const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
    this.db.prepare('DELETE FROM metrics WHERE timestamp < ?').run(thirtyDaysAgo);

    // Vacuum to reclaim space
    this.db.exec('VACUUM');
  }

  // Close database connection
  close() {
    this.db.close();
  }
}

export { SwarmPersistence };