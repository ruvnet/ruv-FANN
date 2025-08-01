-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    status TEXT NOT NULL,
    capabilities TEXT NOT NULL, -- JSON array
    metadata TEXT NOT NULL, -- JSON object
    heartbeat INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    data TEXT NOT NULL -- Full JSON representation
);

CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type);
CREATE INDEX IF NOT EXISTS idx_agents_heartbeat ON agents(heartbeat);

-- Tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY NOT NULL,
    task_type TEXT NOT NULL,
    priority INTEGER NOT NULL,
    status TEXT NOT NULL,
    assigned_to TEXT,
    payload TEXT NOT NULL, -- JSON
    result TEXT, -- JSON
    error TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    dependencies TEXT, -- JSON array
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    started_at INTEGER,
    completed_at INTEGER,
    data TEXT NOT NULL, -- Full JSON representation
    FOREIGN KEY (assigned_to) REFERENCES agents(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_assigned ON tasks(assigned_to);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);

-- Events table for event sourcing
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY NOT NULL,
    event_type TEXT NOT NULL,
    agent_id TEXT,
    task_id TEXT,
    payload TEXT NOT NULL, -- JSON
    metadata TEXT NOT NULL, -- JSON
    timestamp INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    data TEXT NOT NULL, -- Full JSON representation
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent_id);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_sequence ON events(sequence);

-- Messages table for inter-agent communication
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY NOT NULL,
    from_agent TEXT NOT NULL,
    to_agent TEXT NOT NULL,
    message_type TEXT NOT NULL,
    content TEXT NOT NULL, -- JSON
    priority TEXT NOT NULL,
    read INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    read_at INTEGER,
    data TEXT NOT NULL, -- Full JSON representation
    FOREIGN KEY (from_agent) REFERENCES agents(id) ON DELETE CASCADE,
    FOREIGN KEY (to_agent) REFERENCES agents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_from ON messages(from_agent);
CREATE INDEX IF NOT EXISTS idx_messages_to ON messages(to_agent);
CREATE INDEX IF NOT EXISTS idx_messages_unread ON messages(to_agent, read);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);

-- Metrics table for performance tracking
CREATE TABLE IF NOT EXISTS metrics (
    id TEXT PRIMARY KEY NOT NULL,
    metric_type TEXT NOT NULL,
    agent_id TEXT,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    tags TEXT NOT NULL, -- JSON object
    timestamp INTEGER NOT NULL,
    data TEXT NOT NULL, -- Full JSON representation
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_metrics_agent ON metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_type_time ON metrics(metric_type, timestamp);

-- Schema version tracking for migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    applied_at INTEGER NOT NULL
);