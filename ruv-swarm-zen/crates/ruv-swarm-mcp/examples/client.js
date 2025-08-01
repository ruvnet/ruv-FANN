#!/usr/bin/env node

/**
 * Example MCP client for RUV-Swarm
 *
 * This demonstrates how to interact with the MCP server using WebSocket
 */

const WebSocket = require('ws');

class MCPClient {
  constructor(url = 'ws://localhost:3000/mcp') {
    this.url = url;
    this.ws = null;
    this.requestId = 0;
    this.pendingRequests = new Map();
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.on('open', () => {
        console.log('Connected to MCP server');
        resolve();
      });

      this.ws.on('message', (data) => {
        const response = JSON.parse(data);
        console.log('Received:', JSON.stringify(response, null, 2));

        if (response.id && this.pendingRequests.has(response.id)) {
          const { resolve, reject } = this.pendingRequests.get(response.id);
          this.pendingRequests.delete(response.id);

          if (response.error) {
            reject(new Error(response.error.message));
          } else {
            resolve(response.result);
          }
        }
      });

      this.ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      });

      this.ws.on('close', () => {
        console.log('Disconnected from MCP server');
      });
    });
  }

  sendRequest(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.requestId;
      const request = {
        jsonrpc: '2.0',
        method,
        params,
        id,
      };

      this.pendingRequests.set(id, { resolve, reject });
      this.ws.send(JSON.stringify(request));
      console.log('Sent:', JSON.stringify(request, null, 2));
    });
  }

  async initialize() {
    return this.sendRequest('initialize');
  }

  async listTools() {
    return this.sendRequest('tools/list');
  }

  async callTool(name, args) {
    return this.sendRequest('tools/call', {
      name,
      arguments: args,
    });
  }

  close() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// Example usage
async function main() {
  const client = new MCPClient();

  try {
    // Connect to server
    await client.connect();

    // Initialize session
    const initResult = await client.initialize();
    console.log('\nInitialization result:', initResult);

    // List available tools
    const tools = await client.listTools();
    console.log('\nAvailable tools:', tools.tools.length);

    // Spawn a researcher agent
    console.log('\nSpawning researcher agent...');
    const spawnResult = await client.callTool('ruv-swarm.spawn', {
      agent_type: 'researcher',
      name: 'Research Agent 1',
      capabilities: {
        languages: ['english', 'technical'],
        frameworks: ['academic', 'industry'],
        tools: ['search', 'analysis'],
        specializations: ['AI', 'distributed systems'],
        max_concurrent_tasks: 5,
      },
    });
    console.log('Agent spawned:', spawnResult);

    // Create a task
    console.log('\nCreating research task...');
    const taskResult = await client.callTool('ruv-swarm.task.create', {
      task_type: 'research',
      description: 'Research state-of-the-art in swarm intelligence',
      priority: 'high',
    });
    console.log('Task created:', taskResult);

    // Query swarm state
    console.log('\nQuerying swarm state...');
    const stateResult = await client.callTool('ruv-swarm.query', {
      include_metrics: true,
    });
    console.log('Swarm state:', stateResult);

    // Orchestrate a task
    console.log('\nOrchestrating development task...');
    const orchestrateResult = await client.callTool('ruv-swarm.orchestrate', {
      objective: 'Implement distributed consensus algorithm',
      strategy: 'development',
      mode: 'hierarchical',
      max_agents: 5,
      parallel: true,
    });
    console.log('Orchestration started:', orchestrateResult);

    // Store data in memory
    console.log('\nStoring data in swarm memory...');
    const memoryResult = await client.callTool('ruv-swarm.memory.store', {
      key: 'research_findings',
      value: {
        topic: 'swarm intelligence',
        findings: ['distributed decision making', 'emergent behavior'],
        timestamp: new Date().toISOString(),
      },
    });
    console.log('Data stored:', memoryResult);

    // Retrieve data from memory
    console.log('\nRetrieving data from memory...');
    const getResult = await client.callTool('ruv-swarm.memory.get', {
      key: 'research_findings',
    });
    console.log('Retrieved data:', getResult);

    // Monitor events for 5 seconds
    console.log('\nStarting event monitoring...');
    const monitorResult = await client.callTool('ruv-swarm.monitor', {
      event_types: ['agent_spawned', 'task_completed'],
      duration_secs: 5,
    });
    console.log('Monitoring started:', monitorResult);

    // Wait a bit for monitoring
    await new Promise(resolve => setTimeout(resolve, 6000));

  } catch (error) {
    console.error('Error:', error);
  } finally {
    client.close();
  }
}

// Run the example
main().catch(console.error);
