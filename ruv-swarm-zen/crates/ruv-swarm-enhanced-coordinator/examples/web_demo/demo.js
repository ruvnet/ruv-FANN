// RUV-Swarm Web Demo
// Interactive demonstration of swarm intelligence using WASM

import init, {
  SwarmWasm,
  SwarmConfigWasm,
  TopologyWasm,
  AgentTypeWasm,
  CognitiveStyleWasm,
} from './ruv_swarm_wasm.js';

class SwarmDemo {
  constructor() {
    this.swarm = null;
    this.agents = new Map();
    this.canvas = null;
    this.ctx = null;
    this.animationId = null;
    this.stats = {
      activeAgents: 0,
      tasksCompleted: 0,
      messagesSent: 0,
      avgResponseTime: 0,
    };
    this.visualization3D = false;
  }

  async initialize() {
    // Initialize WASM module
    await init();

    // Setup canvas
    this.canvas = document.getElementById('visualization');
    this.ctx = this.canvas.getContext('2d');
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());

    // Setup event listeners
    this.setupEventListeners();

    // Start animation loop
    this.animate();

    this.log('Demo initialized', 'success');
  }

  setupEventListeners() {
    // Initialize swarm button
    document.getElementById('initSwarm').addEventListener('click', () => {
      this.initializeSwarm();
    });

    // Spawn agent button
    document.getElementById('spawnAgent').addEventListener('click', () => {
      this.showAgentModal();
    });

    // Create task button
    document.getElementById('createTask').addEventListener('click', () => {
      this.showTaskModal();
    });

    // Toggle visualization
    document.getElementById('toggleVisualization').addEventListener('click', () => {
      this.visualization3D = !this.visualization3D;
      this.log(`Switched to ${this.visualization3D ? '3D' : '2D'} visualization`, 'info');
    });

    // Agent modal controls
    document.getElementById('confirmSpawn').addEventListener('click', () => {
      this.spawnAgent();
    });
    document.getElementById('cancelSpawn').addEventListener('click', () => {
      this.hideModal('agentModal');
    });

    // Task modal controls
    document.getElementById('confirmTask').addEventListener('click', () => {
      this.createTask();
    });
    document.getElementById('cancelTask').addEventListener('click', () => {
      this.hideModal('taskModal');
    });
  }

  async initializeSwarm() {
    try {
      const topology = document.getElementById('topologySelect').value;
      const maxAgents = parseInt(document.getElementById('maxAgents').value);

      const config = new SwarmConfigWasm();
      config.max_agents = maxAgents;
      config.topology = this.getTopologyEnum(topology);
      config.cognitive_diversity = true;

      this.swarm = new SwarmWasm(config);

      // Enable controls
      document.getElementById('initSwarm').disabled = true;
      document.getElementById('spawnAgent').disabled = false;
      document.getElementById('createTask').disabled = false;
      document.getElementById('swarmStatus').classList.add('active');

      this.log(`Swarm initialized with ${topology} topology (max ${maxAgents} agents)`, 'success');

      // Start monitoring
      this.startMonitoring();
    } catch (error) {
      this.log(`Failed to initialize swarm: ${error}`, 'error');
    }
  }

  showAgentModal() {
    document.getElementById('agentModal').style.display = 'flex';
  }

  showTaskModal() {
    document.getElementById('taskModal').style.display = 'flex';
  }

  hideModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
  }

  async spawnAgent() {
    try {
      const agentType = document.getElementById('agentType').value;
      const cognitiveStyle = document.getElementById('cognitiveStyle').value;

      const agentId = await this.swarm.spawn_agent_with_style(
        this.getAgentTypeEnum(agentType),
        this.getCognitiveStyleEnum(cognitiveStyle),
      );

      // Store agent info
      this.agents.set(agentId, {
        id: agentId,
        type: agentType,
        style: cognitiveStyle,
        position: this.getRandomPosition(),
        velocity: { x: 0, y: 0 },
        connections: [],
      });

      this.updateAgentList();
      this.log(`Spawned ${agentType} agent with ${cognitiveStyle} style`, 'success');
      this.hideModal('agentModal');
    } catch (error) {
      this.log(`Failed to spawn agent: ${error}`, 'error');
    }
  }

  async createTask() {
    try {
      const taskName = document.getElementById('taskName').value;
      const taskType = document.getElementById('taskType').value;
      const subtasksText = document.getElementById('subtasks').value;
      const subtasks = subtasksText.split('\n').filter(s => s.trim());

      if (!taskName || subtasks.length === 0) {
        this.log('Please provide task name and subtasks', 'warning');
        return;
      }

      const taskId = await this.swarm.create_task(taskName, taskType, subtasks);
      this.log(`Created task: ${taskName} with ${subtasks.length} subtasks`, 'info');

      // Start task execution
      this.executeTask(taskId, taskName);
      this.hideModal('taskModal');

      // Clear form
      document.getElementById('taskName').value = '';
      document.getElementById('subtasks').value = '';
    } catch (error) {
      this.log(`Failed to create task: ${error}`, 'error');
    }
  }

  async executeTask(taskId, taskName) {
    try {
      this.log(`Executing task: ${taskName}`, 'info');

      // Show progress
      const progressBar = document.getElementById('taskProgress');
      let progress = 0;

      const progressInterval = setInterval(() => {
        progress += Math.random() * 20;
        if (progress > 100) progress = 100;
        progressBar.style.width = `${progress}%`;

        if (progress >= 100) {
          clearInterval(progressInterval);
        }
      }, 500);

      const result = await this.swarm.orchestrate(taskId);

      clearInterval(progressInterval);
      progressBar.style.width = '100%';

      this.stats.tasksCompleted++;
      this.updateStats();

      this.log(`Task completed: ${result}`, 'success');

      // Reset progress after delay
      setTimeout(() => {
        progressBar.style.width = '0%';
      }, 2000);
    } catch (error) {
      this.log(`Task execution failed: ${error}`, 'error');
    }
  }

  startMonitoring() {
    setInterval(() => {
      if (this.swarm) {
        const stats = this.swarm.get_statistics();
        this.stats = {
          activeAgents: stats.active_agents,
          tasksCompleted: stats.tasks_completed,
          messagesSent: stats.messages_sent,
          avgResponseTime: stats.avg_response_time,
        };
        this.updateStats();
      }
    }, 1000);
  }

  updateStats() {
    document.getElementById('activeAgents').textContent = this.stats.activeAgents;
    document.getElementById('tasksCompleted').textContent = this.stats.tasksCompleted;
    document.getElementById('messagesSent').textContent = this.stats.messagesSent;
    document.getElementById('avgResponseTime').textContent = `${this.stats.avgResponseTime}ms`;
  }

  updateAgentList() {
    const listContainer = document.getElementById('agentList');
    listContainer.innerHTML = '';

    if (this.agents.size === 0) {
      listContainer.innerHTML = '<p style="color: #666; text-align: center;">No agents spawned</p>';
      return;
    }

    this.agents.forEach(agent => {
      const item = document.createElement('div');
      item.className = 'agent-item';
      item.innerHTML = `
                <span>Agent ${agent.id.substr(0, 8)}</span>
                <span class="agent-type ${agent.type}">${agent.type}</span>
            `;
      listContainer.appendChild(item);
    });
  }

  log(message, type = 'info') {
    const logContainer = document.getElementById('logContainer');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
  }

  resizeCanvas() {
    const container = this.canvas.parentElement;
    this.canvas.width = container.clientWidth - 40;
    this.canvas.height = 400;
  }

  animate() {
    this.animationId = requestAnimationFrame(() => this.animate());
    this.draw();
  }

  draw() {
    const { width, height } = this.canvas;
    this.ctx.fillStyle = '#0a0a0a';
    this.ctx.fillRect(0, 0, width, height);

    if (this.agents.size === 0) return;

    // Update agent positions (simple physics simulation)
    this.agents.forEach(agent => {
      // Apply forces based on connections
      const force = { x: 0, y: 0 };

      this.agents.forEach(other => {
        if (agent.id === other.id) return;

        const dx = other.position.x - agent.position.x;
        const dy = other.position.y - agent.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > 0) {
          // Repulsion
          if (distance < 100) {
            force.x -= (dx / distance) * (100 - distance) * 0.01;
            force.y -= (dy / distance) * (100 - distance) * 0.01;
          }

          // Attraction for connected agents
          if (agent.connections.includes(other.id)) {
            if (distance > 150) {
              force.x += (dx / distance) * 0.02;
              force.y += (dy / distance) * 0.02;
            }
          }
        }
      });

      // Apply force
      agent.velocity.x = agent.velocity.x * 0.9 + force.x;
      agent.velocity.y = agent.velocity.y * 0.9 + force.y;

      // Update position
      agent.position.x += agent.velocity.x;
      agent.position.y += agent.velocity.y;

      // Boundary constraints
      agent.position.x = Math.max(20, Math.min(width - 20, agent.position.x));
      agent.position.y = Math.max(20, Math.min(height - 20, agent.position.y));
    });

    // Draw connections
    this.ctx.strokeStyle = '#4a9eff30';
    this.ctx.lineWidth = 1;

    this.agents.forEach(agent => {
      agent.connections.forEach(targetId => {
        const target = this.agents.get(targetId);
        if (target) {
          this.ctx.beginPath();
          this.ctx.moveTo(agent.position.x, agent.position.y);
          this.ctx.lineTo(target.position.x, target.position.y);
          this.ctx.stroke();
        }
      });
    });

    // Draw agents
    this.agents.forEach(agent => {
      const { x, y } = agent.position;

      // Agent circle
      this.ctx.beginPath();
      this.ctx.arc(x, y, 15, 0, Math.PI * 2);

      // Color based on type
      switch (agent.type) {
        case 'worker':
          this.ctx.fillStyle = '#4ade80';
          break;
        case 'coordinator':
          this.ctx.fillStyle = '#f59e0b';
          break;
        case 'analyzer':
          this.ctx.fillStyle = '#a78bfa';
          break;
        default:
          this.ctx.fillStyle = '#4a9eff';
      }

      this.ctx.fill();

      // Cognitive style indicator
      this.ctx.beginPath();
      this.ctx.arc(x, y, 20, 0, Math.PI * 2);
      this.ctx.strokeStyle = this.getCognitiveColor(agent.style);
      this.ctx.lineWidth = 2;
      this.ctx.stroke();
    });

    // Draw message animations
    // TODO: Add message passing animations
  }

  getRandomPosition() {
    const { width, height } = this.canvas;
    return {
      x: Math.random() * (width - 40) + 20,
      y: Math.random() * (height - 40) + 20,
    };
  }

  getTopologyEnum(topology) {
    const map = {
      'fully_connected': TopologyWasm.FullyConnected,
      'ring': TopologyWasm.Ring,
      'star': TopologyWasm.Star,
      'mesh': TopologyWasm.Mesh,
      'hierarchical': TopologyWasm.HierarchicalRing,
      'small_world': TopologyWasm.SmallWorld,
    };
    return map[topology] || TopologyWasm.FullyConnected;
  }

  getAgentTypeEnum(type) {
    const map = {
      'worker': AgentTypeWasm.Worker,
      'coordinator': AgentTypeWasm.Coordinator,
      'analyzer': AgentTypeWasm.Analyzer,
    };
    return map[type] || AgentTypeWasm.Worker;
  }

  getCognitiveStyleEnum(style) {
    const map = {
      'analytical': CognitiveStyleWasm.Analytical,
      'creative': CognitiveStyleWasm.Creative,
      'strategic': CognitiveStyleWasm.Strategic,
      'practical': CognitiveStyleWasm.Practical,
      'detail_oriented': CognitiveStyleWasm.DetailOriented,
    };
    return map[style] || CognitiveStyleWasm.Analytical;
  }

  getCognitiveColor(style) {
    const colors = {
      'analytical': '#4a9eff',
      'creative': '#f59e0b',
      'strategic': '#a78bfa',
      'practical': '#4ade80',
      'detail_oriented': '#ef4444',
    };
    return colors[style] || '#888';
  }
}

// Initialize demo when page loads
const demo = new SwarmDemo();
document.addEventListener('DOMContentLoaded', () => {
  demo.initialize();
});
