# ruv-swarm Integration Guide

## 📚 Table of Contents

- [Quick Start](#quick-start)
- [Claude Code Integration](#claude-code-integration)
- [Node.js Integration](#nodejs-integration)
- [TypeScript Integration](#typescript-integration)
- [Browser Integration](#browser-integration)
- [Docker Integration](#docker-integration)
- [CI/CD Integration](#cicd-integration)
- [Remote Server Integration](#remote-server-integration)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### NPX (Instant Usage)

```bash
# Initialize a swarm instantly on any system with Node.js
npx ruv-swarm init mesh 5

# Spawn an agent
npx ruv-swarm spawn researcher "AI Research Assistant"

# Orchestrate a task
npx ruv-swarm orchestrate "Analyze neural architecture trends"

# Start MCP server for Claude Code
npx ruv-swarm mcp start
```

### NPM Installation

```bash
# Local installation
npm install ruv-swarm

# Global installation
npm install -g ruv-swarm

# Development installation
npm install ruv-swarm --save-dev
```

---

## Claude Code Integration

### Automatic Setup

```bash
# Initialize with Claude Code integration
npx ruv-swarm init mesh 10 --claude

# Force regenerate all integration files
npx ruv-swarm init mesh 10 --claude --force
```

### Manual Setup

1. **Add MCP Server to Claude Code**:

```bash
claude mcp add ruv-swarm npx ruv-swarm mcp start
```

2. **Verify Integration**:

```bash
claude mcp list
```

3. **Test MCP Tools**:

Use these tools in Claude Code:
- `mcp__ruv-swarm__swarm_init`
- `mcp__ruv-swarm__agent_spawn`
- `mcp__ruv-swarm__task_orchestrate`

### Available MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `swarm_init` | Initialize new swarm | `topology`, `maxAgents`, `strategy` |
| `agent_spawn` | Create new agent | `type`, `name`, `capabilities` |
| `task_orchestrate` | Orchestrate task | `task`, `strategy`, `priority` |
| `swarm_status` | Get swarm status | `verbose` |
| `agent_metrics` | Get agent metrics | `agentId`, `metric` |
| `neural_train` | Train neural models | `agentId`, `iterations` |

### Claude Code Hooks

Automatic hooks for seamless integration:

```javascript
// Pre-operation hooks
await swarm.hook('pre-task', { description: 'Build authentication system' });
await swarm.hook('pre-edit', { file: 'src/app.js' });

// Post-operation hooks
await swarm.hook('post-task', { 
  taskId: 'auth-system',
  analyzePerformance: true 
});
await swarm.hook('post-edit', { 
  file: 'src/app.js',
  memoryKey: 'edit-history/app-js'
});

// Git integration
await swarm.hook('git-commit', {
  agent: 'coder-123',
  generateReport: true
});
```

---

## Node.js Integration

### Basic Integration

```javascript
const { RuvSwarm } = require('ruv-swarm');

async function main() {
  // Initialize with options
  const swarm = await RuvSwarm.initialize({
    useSIMD: true,
    enablePersistence: true,
    enableNeuralNetworks: true,
    debug: false
  });

  // Create swarm
  const mySwarm = await swarm.createSwarm({
    topology: 'mesh',
    maxAgents: 10,
    cognitiveProfiles: true
  });

  // Use the swarm
  const agent = await mySwarm.spawn({
    type: 'researcher',
    name: 'Data Scientist'
  });

  const result = await mySwarm.orchestrate({
    task: "Analyze market trends",
    strategy: 'adaptive'
  });

  console.log('Result:', result);
}

main().catch(console.error);
```

### Express.js Integration

```javascript
const express = require('express');
const { RuvSwarm, EnhancedMCPTools } = require('ruv-swarm');

const app = express();
app.use(express.json());

let swarm;
let mcpTools;

// Initialize on startup
async function initialize() {
  swarm = await RuvSwarm.initialize({
    enablePersistence: true,
    enableNeuralNetworks: true
  });
  
  mcpTools = new EnhancedMCPTools();
  await mcpTools.initialize();
}

// API endpoints
app.post('/api/swarm/init', async (req, res) => {
  try {
    const { topology, maxAgents } = req.body;
    const result = await mcpTools.swarm_init({ topology, maxAgents });
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/agents/spawn', async (req, res) => {
  try {
    const { type, name, capabilities } = req.body;
    const result = await mcpTools.agent_spawn({ type, name, capabilities });
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/tasks/orchestrate', async (req, res) => {
  try {
    const { task, strategy, priority } = req.body;
    const result = await mcpTools.task_orchestrate({ task, strategy, priority });
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/swarm/status', async (req, res) => {
  try {
    const result = await mcpTools.swarm_status({ verbose: req.query.verbose === 'true' });
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Start server
initialize().then(() => {
  app.listen(3000, () => {
    console.log('ruv-swarm API server running on port 3000');
  });
});
```

---

## TypeScript Integration

### Installation

```bash
npm install ruv-swarm
npm install --save-dev @types/node typescript
```

### Basic TypeScript Usage

```typescript
import { 
  RuvSwarm, 
  SwarmConfig, 
  AgentConfig, 
  TaskConfig, 
  CognitiveProfile 
} from 'ruv-swarm';

interface ProjectConfig {
  domain: string;
  complexity: 'simple' | 'moderate' | 'complex';
  timeline: string;
}

class AIProjectOrchestrator {
  private swarm: RuvSwarm;
  
  async initialize(config: ProjectConfig): Promise<void> {
    this.swarm = await RuvSwarm.initialize({
      useSIMD: RuvSwarm.detectSIMDSupport(),
      enablePersistence: true,
      enableNeuralNetworks: true,
      debug: process.env.NODE_ENV === 'development'
    });
  }
  
  async createSpecializedTeam(domain: string): Promise<void> {
    const swarmConfig: SwarmConfig = {
      topology: 'hierarchical',
      maxAgents: 15,
      cognitiveProfiles: true
    };
    
    const mySwarm = await this.swarm.createSwarm(swarmConfig);
    
    // Create domain-specific agents
    const researcher = await mySwarm.spawn({
      type: 'researcher',
      name: `${domain} Research Lead`,
      cognitiveProfile: {
        analytical: 0.9,
        creative: 0.7,
        systematic: 0.8,
        intuitive: 0.6,
        collaborative: 0.8,
        independent: 0.7
      }
    });
    
    const architect = await mySwarm.spawn({
      type: 'architect',
      name: `${domain} System Architect`,
      cognitiveProfile: {
        systematic: 0.95,
        analytical: 0.85,
        creative: 0.6,
        intuitive: 0.7,
        collaborative: 0.75,
        independent: 0.8
      }
    });
    
    // Execute coordinated workflow
    const result = await mySwarm.orchestrate({
      task: `Design and implement ${domain} solution`,
      strategy: 'adaptive',
      maxAgents: 10,
      timeout: 1800000 // 30 minutes
    });
    
    console.log('Project result:', result);
  }
}

// Usage
async function main() {
  const orchestrator = new AIProjectOrchestrator();
  await orchestrator.initialize({
    domain: 'machine-learning',
    complexity: 'complex',
    timeline: '4-weeks'
  });
  
  await orchestrator.createSpecializedTeam('machine-learning');
}

main().catch(console.error);
```

### Advanced TypeScript Features

```typescript
import { 
  NeuralAgent, 
  NeuralNetworkManager,
  PerformanceMonitor,
  MCPTools 
} from 'ruv-swarm';

// Generic swarm interface
interface SwarmManager<T extends AgentConfig> {
  spawn(config: T): Promise<AgentWrapper>;
  orchestrate(task: TaskConfig): Promise<OrchestrationResult>;
  getMetrics(): Promise<SwarmMetrics>;
}

// Specialized neural swarm
class NeuralSwarmManager implements SwarmManager<AgentConfig & { neuralConfig: any }> {
  private neuralManager: NeuralNetworkManager;
  private monitor: PerformanceMonitor;
  
  constructor() {
    this.neuralManager = new NeuralNetworkManager();
    this.monitor = new PerformanceMonitor();
  }
  
  async spawn(config: AgentConfig & { neuralConfig: any }): Promise<NeuralAgent> {
    // Create neural model
    const model = await this.neuralManager.createModel(config.neuralConfig);
    
    // Spawn neural agent
    const agent = await this.swarm.spawn({
      ...config,
      neuralNetwork: {
        enabled: true,
        model: model.id
      }
    }) as NeuralAgent;
    
    return agent;
  }
  
  async orchestrate(task: TaskConfig): Promise<OrchestrationResult> {
    // Start performance monitoring
    this.monitor.startMonitoring();
    
    try {
      const result = await this.swarm.orchestrate(task);
      
      // Analyze performance
      const metrics = this.monitor.getCurrentMetrics();
      result.performanceMetrics = metrics;
      
      return result;
    } finally {
      this.monitor.stopMonitoring();
    }
  }
  
  async getMetrics(): Promise<SwarmMetrics> {
    return this.monitor.getCurrentMetrics();
  }
}
```

---

## Browser Integration

### Webpack Configuration

```javascript
// webpack.config.js
const path = require('path');

module.exports = {
  entry: './src/index.js',
  mode: 'development',
  module: {
    rules: [
      {
        test: /\.wasm$/,
        type: 'webassembly/async'
      }
    ]
  },
  experiments: {
    asyncWebAssembly: true
  },
  resolve: {
    fallback: {
      "fs": false,
      "path": require.resolve("path-browserify"),
      "crypto": require.resolve("crypto-browserify"),
      "stream": require.resolve("stream-browserify"),
      "buffer": require.resolve("buffer")
    }
  },
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'dist')
  }
};
```

### Browser Usage

```html
<!DOCTYPE html>
<html>
<head>
    <title>ruv-swarm Browser Example</title>
</head>
<body>
    <div id="app">
        <h1>ruv-swarm Browser Demo</h1>
        <button id="init-btn">Initialize Swarm</button>
        <button id="spawn-btn">Spawn Agent</button>
        <button id="task-btn">Execute Task</button>
        <div id="output"></div>
    </div>

    <script type="module">
        import { RuvSwarm } from './node_modules/ruv-swarm/src/index.js';
        
        let swarm;
        let mySwarm;
        
        document.getElementById('init-btn').addEventListener('click', async () => {
            try {
                swarm = await RuvSwarm.initialize({
                    useSIMD: RuvSwarm.detectSIMDSupport(),
                    debug: true
                });
                
                mySwarm = await swarm.createSwarm({
                    topology: 'mesh',
                    maxAgents: 5
                });
                
                document.getElementById('output').innerHTML += '<p>✅ Swarm initialized</p>';
            } catch (error) {
                document.getElementById('output').innerHTML += `<p>❌ Error: ${error.message}</p>`;
            }
        });
        
        document.getElementById('spawn-btn').addEventListener('click', async () => {
            try {
                const agent = await mySwarm.spawn({
                    type: 'researcher',
                    name: 'Browser Research Agent'
                });
                
                document.getElementById('output').innerHTML += `<p>🤖 Agent spawned: ${agent.id}</p>`;
            } catch (error) {
                document.getElementById('output').innerHTML += `<p>❌ Error: ${error.message}</p>`;
            }
        });
        
        document.getElementById('task-btn').addEventListener('click', async () => {
            try {
                const result = await mySwarm.orchestrate({
                    task: "Analyze web performance metrics",
                    strategy: 'adaptive'
                });
                
                document.getElementById('output').innerHTML += `<p>📋 Task completed: ${result.taskId}</p>`;
            } catch (error) {
                document.getElementById('output').innerHTML += `<p>❌ Error: ${error.message}</p>`;
            }
        });
    </script>
</body>
</html>
```

### React Integration

```jsx
import React, { useState, useEffect } from 'react';
import { RuvSwarm } from 'ruv-swarm';

const SwarmComponent = () => {
  const [swarm, setSwarm] = useState(null);
  const [agents, setAgents] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [status, setStatus] = useState('Initializing...');

  useEffect(() => {
    async function initializeSwarm() {
      try {
        const ruvSwarm = await RuvSwarm.initialize({
          useSIMD: RuvSwarm.detectSIMDSupport(),
          enablePersistence: false // Disabled for browser
        });
        
        const mySwarm = await ruvSwarm.createSwarm({
          topology: 'mesh',
          maxAgents: 8
        });
        
        setSwarm(mySwarm);
        setStatus('Ready');
      } catch (error) {
        setStatus(`Error: ${error.message}`);
      }
    }
    
    initializeSwarm();
  }, []);

  const spawnAgent = async (type, name) => {
    try {
      const agent = await swarm.spawn({ type, name });
      setAgents(prev => [...prev, agent]);
    } catch (error) {
      console.error('Failed to spawn agent:', error);
    }
  };

  const orchestrateTask = async (taskDescription) => {
    try {
      const result = await swarm.orchestrate({
        task: taskDescription,
        strategy: 'adaptive'
      });
      setTasks(prev => [...prev, result]);
    } catch (error) {
      console.error('Failed to orchestrate task:', error);
    }
  };

  return (
    <div className="swarm-component">
      <h2>ruv-swarm React Integration</h2>
      <p>Status: {status}</p>
      
      <div className="agents-section">
        <h3>Agents ({agents.length})</h3>
        <button onClick={() => spawnAgent('researcher', 'Research Agent')}>
          Spawn Researcher
        </button>
        <button onClick={() => spawnAgent('coder', 'Developer Agent')}>
          Spawn Coder
        </button>
        {agents.map(agent => (
          <div key={agent.id} className="agent-card">
            <strong>{agent.agentType}</strong>: {agent.id}
          </div>
        ))}
      </div>
      
      <div className="tasks-section">
        <h3>Tasks ({tasks.length})</h3>
        <button onClick={() => orchestrateTask('Analyze user interface patterns')}>
          Analyze UI Patterns
        </button>
        <button onClick={() => orchestrateTask('Optimize application performance')}>
          Optimize Performance
        </button>
        {tasks.map(task => (
          <div key={task.taskId} className="task-card">
            <strong>{task.status}</strong>: {task.description}
          </div>
        ))}
      </div>
    </div>
  );
};

export default SwarmComponent;
```

---

## Docker Integration

### Basic Dockerfile

```dockerfile
FROM node:18-alpine

# Install system dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    sqlite

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application code
COPY . .

# Install ruv-swarm globally for CLI access
RUN npm install -g ruv-swarm

# Expose MCP port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD npx ruv-swarm mcp status || exit 1

# Default command
CMD ["npx", "ruv-swarm", "mcp", "start", "--host", "0.0.0.0", "--port", "3000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  ruv-swarm:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - RUVA_SWARM_MAX_AGENTS=50
      - RUVA_SWARM_MEMORY_POOL=512MB
      - RUVA_SWARM_WASM_SIMD=true
    volumes:
      - swarm-data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "npx", "ruv-swarm", "mcp", "status"]
      interval: 30s
      timeout: 10s
      retries: 3
    
  # Optional: Redis for distributed coordination
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    
  # Optional: PostgreSQL for advanced persistence
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=ruv_swarm
      - POSTGRES_USER=swarm
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  swarm-data:
  redis-data:
  postgres-data:
```

### Multi-stage Production Dockerfile

```dockerfile
# Build stage
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --include=dev

COPY . .
RUN npm run build:all

# Production stage
FROM node:18-alpine AS production

# Install system dependencies
RUN apk add --no-cache \
    sqlite \
    dumb-init

# Create non-root user
RUN addgroup -g 1001 -S swarm && \
    adduser -S swarm -u 1001

WORKDIR /app

# Copy package files and install production dependencies
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy built application
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/wasm ./wasm
COPY --from=builder /app/bin ./bin
COPY --from=builder /app/src ./src

# Set ownership
RUN chown -R swarm:swarm /app
USER swarm

# Expose port
EXPOSE 3000

# Use dumb-init for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Start command
CMD ["node", "bin/ruv-swarm-clean.js", "mcp", "start", "--host", "0.0.0.0"]
```

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/ruv-swarm.yml
name: ruv-swarm CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16, 18, 20]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run tests with ruv-swarm
      run: |
        # Test ruv-swarm installation
        npx ruv-swarm --version
        
        # Test basic functionality
        npx ruv-swarm init mesh 3
        npx ruv-swarm spawn researcher "CI Test Agent"
        
        # Run test suite
        npm test
    
    - name: Run benchmarks
      run: |
        npx ruv-swarm benchmark run --iterations 5
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.node-version }}
        path: |
          test-results/
          coverage/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: 18
        cache: 'npm'
    
    - name: Build for production
      run: |
        npm ci
        npm run build:all
    
    - name: Deploy to production
      run: |
        # Deploy ruv-swarm to production environment
        docker build -t ruv-swarm:latest .
        
        # Push to registry or deploy directly
        echo "Deploying ruv-swarm to production..."
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    tools {
        nodejs '18'
    }
    
    environment {
        NODE_ENV = 'test'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh 'npm ci'
            }
        }
        
        stage('Test ruv-swarm') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'npm test'
                    }
                    post {
                        always {
                            publishTestResults testResultsPattern: 'test-results.xml'
                        }
                    }
                }
                
                stage('Integration Tests') {
                    steps {
                        sh '''
                            npx ruv-swarm init mesh 5
                            npx ruv-swarm spawn researcher "Jenkins Test Agent"
                            npx ruv-swarm orchestrate "Run integration tests"
                        '''
                    }
                }
                
                stage('Performance Tests') {
                    steps {
                        sh 'npx ruv-swarm benchmark run --iterations 10'
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'benchmark-results.json'
                        }
                    }
                }
            }
        }
        
        stage('Build') {
            when {
                branch 'main'
            }
            steps {
                sh 'npm run build:all'
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    docker build -t ruv-swarm:${BUILD_NUMBER} .
                    docker tag ruv-swarm:${BUILD_NUMBER} ruv-swarm:latest
                    
                    # Deploy to production
                    echo "Deploying ruv-swarm to production..."
                '''
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        failure {
            emailext (
                subject: "Failed Pipeline: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Something went wrong with ruv-swarm deployment.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

---

## Remote Server Integration

### NPX Remote Execution

```bash
# Execute on remote server via SSH
ssh user@remote-server 'npx ruv-swarm init mesh 10'

# Start MCP server on remote host
ssh user@remote-server 'npx ruv-swarm mcp start --host 0.0.0.0 --port 3000 &'

# Monitor remote swarm
ssh user@remote-server 'npx ruv-swarm monitor --duration 60000'

# Run benchmarks on remote hardware
ssh user@remote-server 'npx ruv-swarm benchmark run --test swe-bench'
```

### Ansible Playbook

```yaml
# deploy-ruv-swarm.yml
---
- name: Deploy ruv-swarm to remote servers
  hosts: swarm_nodes
  become: yes
  
  vars:
    node_version: "18"
    swarm_topology: "hierarchical"
    max_agents: 50
  
  tasks:
    - name: Install Node.js
      shell: |
        curl -fsSL https://deb.nodesource.com/setup_{{ node_version }}.x | sudo -E bash -
        apt-get install -y nodejs
      
    - name: Verify Node.js installation
      command: node --version
      register: node_version_output
      
    - name: Initialize ruv-swarm
      shell: npx ruv-swarm init {{ swarm_topology }} {{ max_agents }}
      become_user: swarm
      
    - name: Start MCP server
      shell: |
        nohup npx ruv-swarm mcp start --host 0.0.0.0 --port 3000 > /var/log/ruv-swarm.log 2>&1 &
      become_user: swarm
      
    - name: Verify swarm status
      shell: npx ruv-swarm status
      become_user: swarm
      register: swarm_status
      
    - name: Display swarm status
      debug:
        var: swarm_status.stdout
```

### Terraform Configuration

```hcl
# main.tf
provider "aws" {
  region = "us-west-2"
}

# EC2 instance for ruv-swarm
resource "aws_instance" "ruv_swarm_node" {
  count           = 3
  ami             = "ami-0c94855ba95b798c7"  # Ubuntu 22.04 LTS
  instance_type   = "c5.xlarge"  # Good for compute-intensive workloads
  key_name        = var.key_name
  security_groups = [aws_security_group.ruv_swarm.name]
  
  user_data = <<-EOF
    #!/bin/bash
    apt-get update
    
    # Install Node.js 18
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    apt-get install -y nodejs
    
    # Initialize ruv-swarm
    npx ruv-swarm init hierarchical 20
    
    # Start MCP server
    nohup npx ruv-swarm mcp start --host 0.0.0.0 --port 3000 > /var/log/ruv-swarm.log 2>&1 &
  EOF
  
  tags = {
    Name = "ruv-swarm-node-${count.index + 1}"
    Type = "swarm-node"
  }
}

# Security group
resource "aws_security_group" "ruv_swarm" {
  name_prefix = "ruv-swarm-"
  
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Output instance IPs
output "ruv_swarm_ips" {
  value = aws_instance.ruv_swarm_node[*].public_ip
}
```

---

## Production Deployment

### Environment Configuration

```bash
# Production environment variables
export NODE_ENV=production
export RUVA_SWARM_MAX_AGENTS=100
export RUVA_SWARM_TOPOLOGY=hierarchical
export RUVA_SWARM_PERSISTENCE=sqlite
export RUVA_SWARM_WASM_SIMD=true
export RUVA_SWARM_MEMORY_POOL=1GB
export RUVA_SWARM_WORKER_THREADS=8

# Logging
export RUST_LOG=info
export RUVA_SWARM_LOG_LEVEL=info
export RUVA_SWARM_LOG_FILE=/var/log/ruv-swarm.log

# Security
export RUVA_SWARM_ENABLE_CORS=false
export RUVA_SWARM_ALLOWED_ORIGINS="https://yourdomain.com"
export RUVA_SWARM_API_KEY_REQUIRED=true

# Performance
export RUVA_SWARM_ENABLE_METRICS=true
export RUVA_SWARM_METRICS_PORT=9090
export RUVA_SWARM_HEALTHCHECK_INTERVAL=30
```

### PM2 Configuration

```javascript
// ecosystem.config.js
module.exports = {
  apps: [{
    name: 'ruv-swarm-mcp',
    script: 'npx',
    args: 'ruv-swarm mcp start --host 0.0.0.0 --port 3000',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '2G',
    
    env: {
      NODE_ENV: 'production',
      RUVA_SWARM_MAX_AGENTS: 50,
      RUVA_SWARM_TOPOLOGY: 'hierarchical'
    },
    
    // Logging
    log_file: '/var/log/ruv-swarm/combined.log',
    out_file: '/var/log/ruv-swarm/out.log',
    error_file: '/var/log/ruv-swarm/error.log',
    
    // Monitoring
    min_uptime: '10s',
    max_restarts: 10,
    
    // Performance
    node_args: ['--max-old-space-size=4096']
  }]
};
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ruv-swarm
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ruv-swarm
  template:
    metadata:
      labels:
        app: ruv-swarm
    spec:
      containers:
      - name: ruv-swarm
        image: ruv-swarm:latest
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          value: "production"
        - name: RUVA_SWARM_MAX_AGENTS
          value: "50"
        - name: RUVA_SWARM_TOPOLOGY
          value: "hierarchical"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          exec:
            command:
            - npx
            - ruv-swarm
            - mcp
            - status
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - npx
            - ruv-swarm
            - mcp
            - status
          initialDelaySeconds: 5
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: ruv-swarm-service
spec:
  selector:
    app: ruv-swarm
  ports:
  - protocol: TCP
    port: 3000
    targetPort: 3000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ruv-swarm-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: ruv-swarm.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ruv-swarm-service
            port:
              number: 3000
```

### Load Balancer Configuration

```nginx
# nginx.conf
upstream ruv_swarm_backend {
    server 10.0.1.10:3000;
    server 10.0.1.11:3000;
    server 10.0.1.12:3000;
}

server {
    listen 80;
    server_name ruv-swarm.yourdomain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ruv-swarm.yourdomain.com;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/ruv-swarm.crt;
    ssl_certificate_key /etc/ssl/private/ruv-swarm.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    
    # MCP WebSocket support
    location / {
        proxy_pass http://ruv_swarm_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

---

## Troubleshooting

### Common Issues

#### WASM Module Loading Issues

```bash
# Verify WASM support
npx ruv-swarm features

# Force rebuild WASM modules
npm run build:wasm

# Check system compatibility
node -p "typeof WebAssembly"

# Debug WASM loading
DEBUG=ruv-swarm:wasm npx ruv-swarm init mesh 5
```

#### Remote Connection Issues

```bash
# Test connectivity
curl -f http://remote-server:3000/health

# Check firewall
sudo ufw status
sudo iptables -L

# Verify port binding
netstat -tlnp | grep :3000

# Test MCP protocol
echo '{"jsonrpc":"2.0","method":"initialize","id":1}' | nc remote-server 3000
```

#### Performance Issues

```bash
# Analyze performance
npx ruv-swarm performance analyze

# Check system resources
npx ruv-swarm benchmark run --test system

# Optimize configuration
npx ruv-swarm performance suggest

# Enable detailed logging
DEBUG=ruv-swarm:* npx ruv-swarm status --verbose
```

### Debug Commands

```bash
# Enable debug mode
export DEBUG=ruv-swarm:*
export RUVA_SWARM_DEBUG=true

# Verbose logging
npx ruv-swarm --verbose status

# Performance profiling
npx ruv-swarm profile init mesh 10

# Memory analysis
npx ruv-swarm memory analyze

# Network debugging
npx ruv-swarm network test --host remote-server --port 3000
```

### Error Recovery

```javascript
// Automatic retry with exponential backoff
import { retryWithBackoff } from 'ruv-swarm/utils';

const result = await retryWithBackoff(
  () => swarm.orchestrate({ task: "Complex task" }),
  {
    maxRetries: 3,
    baseDelay: 1000,
    maxDelay: 10000,
    backoffFactor: 2
  }
);

// Circuit breaker pattern
import { circuitBreaker } from 'ruv-swarm/utils';

const protectedOperation = circuitBreaker(
  () => agent.execute(task),
  {
    threshold: 5,
    timeout: 60000,
    resetTimeout: 300000
  }
);
```

---

## Support Resources

- **Documentation**: [Complete Documentation](./README.md)
- **API Reference**: [API Reference](./API_REFERENCE_COMPLETE.md)
- **Examples**: [Examples Directory](../examples/)
- **Issues**: [GitHub Issues](https://github.com/ruvnet/ruv-FANN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ruvnet/ruv-FANN/discussions)

---

*This integration guide covers all major platforms and deployment scenarios for ruv-swarm.*