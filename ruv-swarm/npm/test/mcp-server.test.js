/**
 * Test suite for MCP server implementation
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import WebSocket from 'ws';

// Mock WebSocket
jest.mock('ws');

// Mock the swarm module
jest.mock('../src/index.js', () => ({
  default: {
    init: jest.fn(),
    spawnAgent: jest.fn(),
    executeTask: jest.fn(),
    getSwarmStatus: jest.fn(),
    listAgents: jest.fn(),
    getAgentMetrics: jest.fn(),
    getTaskStatus: jest.fn(),
    getTaskResults: jest.fn(),
    runBenchmark: jest.fn(),
    detectSystemFeatures: jest.fn(),
    monitorSwarm: jest.fn(),
    storeMemory: jest.fn(),
    retrieveMemory: jest.fn(),
    listMemoryKeys: jest.fn(),
    getNeuralStatus: jest.fn(),
    trainNeuralAgent: jest.fn(),
    getNeuralPatterns: jest.fn(),
  },
}));

// Import after mocking
import { MCPServer } from '../src/mcp-server.js';
import RuvSwarm from '../src/index.js';

describe('MCPServer', () => {
  let server;
  let mockWsServer;
  let mockClient;

  beforeEach(() => {
    mockWsServer = {
      on: jest.fn(),
      close: jest.fn(),
    };
    WebSocket.Server.mockReturnValue(mockWsServer);

    mockClient = {
      on: jest.fn(),
      send: jest.fn(),
      close: jest.fn(),
      readyState: WebSocket.OPEN,
    };

    server = new MCPServer();
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('constructor', () => {
    it('should initialize with default port', () => {
      const newServer = new MCPServer();
      expect(newServer.port).toBe(3000);
      expect(newServer.clients).toBeInstanceOf(Set);
    });

    it('should accept custom port', () => {
      const customServer = new MCPServer(8080);
      expect(customServer.port).toBe(8080);
    });
  });

  describe('start', () => {
    it('should create WebSocket server and setup handlers', async() => {
      await server.start();

      expect(WebSocket.Server).toHaveBeenCalledWith({ port: 3000 });
      expect(mockWsServer.on).toHaveBeenCalledWith('connection', expect.any(Function));
      expect(mockWsServer.on).toHaveBeenCalledWith('error', expect.any(Function));
    });

    it('should handle new connections', async() => {
      await server.start();

      const connectionHandler = mockWsServer.on.mock.calls.find(
        call => call[0] === 'connection',
      )[1];

      connectionHandler(mockClient);

      expect(server.clients.has(mockClient)).toBe(true);
      expect(mockClient.on).toHaveBeenCalledWith('message', expect.any(Function));
      expect(mockClient.on).toHaveBeenCalledWith('close', expect.any(Function));
      expect(mockClient.on).toHaveBeenCalledWith('error', expect.any(Function));
    });
  });

  describe('handleMessage', () => {
    beforeEach(async() => {
      await server.start();
      const connectionHandler = mockWsServer.on.mock.calls.find(
        call => call[0] === 'connection',
      )[1];
      connectionHandler(mockClient);
    });

    it('should handle swarm_init tool', async() => {
      const message = {
        jsonrpc: '2.0',
        id: 1,
        method: 'tool_call',
        params: {
          tool: 'swarm_init',
          arguments: {
            topology: 'mesh',
            maxAgents: 10,
            strategy: 'adaptive',
          },
        },
      };

      RuvSwarm.init.mockResolvedValue({ swarmId: 'test-123' });

      await server.handleMessage(mockClient, JSON.stringify(message));

      expect(RuvSwarm.init).toHaveBeenCalledWith('mesh', 10, 'adaptive');
      expect(mockClient.send).toHaveBeenCalledWith(
        expect.stringContaining('"result":{"swarmId":"test-123"}'),
      );
    });

    it('should handle agent_spawn tool', async() => {
      const message = {
        jsonrpc: '2.0',
        id: 2,
        method: 'tool_call',
        params: {
          tool: 'agent_spawn',
          arguments: {
            type: 'researcher',
            name: 'Agent 1',
            config: { model: 'advanced' },
          },
        },
      };

      RuvSwarm.spawnAgent.mockResolvedValue({ agentId: 'agent-123' });

      await server.handleMessage(mockClient, JSON.stringify(message));

      expect(RuvSwarm.spawnAgent).toHaveBeenCalledWith('researcher', 'Agent 1', { model: 'advanced' });
    });

    it('should handle task_orchestrate tool', async() => {
      const message = {
        jsonrpc: '2.0',
        id: 3,
        method: 'tool_call',
        params: {
          tool: 'task_orchestrate',
          arguments: {
            task: 'Build REST API',
            agents: ['agent-1', 'agent-2'],
            strategy: 'parallel',
          },
        },
      };

      RuvSwarm.executeTask.mockResolvedValue({ taskId: 'task-123' });

      await server.handleMessage(mockClient, JSON.stringify(message));

      expect(RuvSwarm.executeTask).toHaveBeenCalledWith({
        task: 'Build REST API',
        agents: ['agent-1', 'agent-2'],
        strategy: 'parallel',
      });
    });

    it('should handle memory_usage tool with store action', async() => {
      const message = {
        jsonrpc: '2.0',
        id: 4,
        method: 'tool_call',
        params: {
          tool: 'memory_usage',
          arguments: {
            action: 'store',
            key: 'test/key',
            value: { data: 'test' },
          },
        },
      };

      RuvSwarm.storeMemory.mockResolvedValue({ stored: true });

      await server.handleMessage(mockClient, JSON.stringify(message));

      expect(RuvSwarm.storeMemory).toHaveBeenCalledWith('test/key', { data: 'test' });
    });

    it('should handle memory_usage tool with retrieve action', async() => {
      const message = {
        jsonrpc: '2.0',
        id: 5,
        method: 'tool_call',
        params: {
          tool: 'memory_usage',
          arguments: {
            action: 'retrieve',
            key: 'test/key',
          },
        },
      };

      RuvSwarm.retrieveMemory.mockResolvedValue({ data: 'test' });

      await server.handleMessage(mockClient, JSON.stringify(message));

      expect(RuvSwarm.retrieveMemory).toHaveBeenCalledWith('test/key');
    });

    it('should handle memory_usage tool with list action', async() => {
      const message = {
        jsonrpc: '2.0',
        id: 6,
        method: 'tool_call',
        params: {
          tool: 'memory_usage',
          arguments: {
            action: 'list',
            pattern: 'test/*',
          },
        },
      };

      RuvSwarm.listMemoryKeys.mockResolvedValue(['test/key1', 'test/key2']);

      await server.handleMessage(mockClient, JSON.stringify(message));

      expect(RuvSwarm.listMemoryKeys).toHaveBeenCalledWith('test/*');
    });

    it('should handle neural_train tool', async() => {
      const message = {
        jsonrpc: '2.0',
        id: 7,
        method: 'tool_call',
        params: {
          tool: 'neural_train',
          arguments: {
            agentId: 'agent-123',
            data: [1, 2, 3],
            epochs: 100,
          },
        },
      };

      RuvSwarm.trainNeuralAgent.mockResolvedValue({ trained: true });

      await server.handleMessage(mockClient, JSON.stringify(message));

      expect(RuvSwarm.trainNeuralAgent).toHaveBeenCalledWith('agent-123', [1, 2, 3], 100);
    });

    it('should handle benchmark_run tool', async() => {
      const message = {
        jsonrpc: '2.0',
        id: 8,
        method: 'tool_call',
        params: {
          tool: 'benchmark_run',
          arguments: {
            suite: 'full',
            iterations: 10,
          },
        },
      };

      RuvSwarm.runBenchmark.mockResolvedValue({
        results: { performance: 'excellent' },
      });

      await server.handleMessage(mockClient, JSON.stringify(message));

      expect(RuvSwarm.runBenchmark).toHaveBeenCalledWith('full', 10);
    });

    it('should handle invalid JSON', async() => {
      await server.handleMessage(mockClient, 'invalid json');

      expect(mockClient.send).toHaveBeenCalledWith(
        expect.stringContaining('"error"'),
      );
    });

    it('should handle unknown tools', async() => {
      const message = {
        jsonrpc: '2.0',
        id: 9,
        method: 'tool_call',
        params: {
          tool: 'unknown_tool',
          arguments: {},
        },
      };

      await server.handleMessage(mockClient, JSON.stringify(message));

      expect(mockClient.send).toHaveBeenCalledWith(
        expect.stringContaining('Unknown tool: unknown_tool'),
      );
    });

    it('should handle tool errors gracefully', async() => {
      const message = {
        jsonrpc: '2.0',
        id: 10,
        method: 'tool_call',
        params: {
          tool: 'swarm_init',
          arguments: {},
        },
      };

      RuvSwarm.init.mockRejectedValue(new Error('Initialization failed'));

      await server.handleMessage(mockClient, JSON.stringify(message));

      expect(mockClient.send).toHaveBeenCalledWith(
        expect.stringContaining('Initialization failed'),
      );
    });
  });

  describe('broadcast', () => {
    it('should send message to all connected clients', async() => {
      await server.start();

      const client1 = { ...mockClient, readyState: WebSocket.OPEN };
      const client2 = { ...mockClient, readyState: WebSocket.OPEN };
      const client3 = { ...mockClient, readyState: WebSocket.CLOSED };

      server.clients.add(client1);
      server.clients.add(client2);
      server.clients.add(client3);

      const message = { type: 'broadcast', data: 'test' };
      server.broadcast(message);

      expect(client1.send).toHaveBeenCalledWith(JSON.stringify(message));
      expect(client2.send).toHaveBeenCalledWith(JSON.stringify(message));
      expect(client3.send).not.toHaveBeenCalled();
    });
  });

  describe('client management', () => {
    it('should remove client on disconnect', async() => {
      await server.start();

      const connectionHandler = mockWsServer.on.mock.calls.find(
        call => call[0] === 'connection',
      )[1];

      connectionHandler(mockClient);
      expect(server.clients.has(mockClient)).toBe(true);

      const closeHandler = mockClient.on.mock.calls.find(
        call => call[0] === 'close',
      )[1];

      closeHandler();
      expect(server.clients.has(mockClient)).toBe(false);
    });

    it('should handle client errors', async() => {
      await server.start();

      const connectionHandler = mockWsServer.on.mock.calls.find(
        call => call[0] === 'connection',
      )[1];

      connectionHandler(mockClient);

      const errorHandler = mockClient.on.mock.calls.find(
        call => call[0] === 'error',
      )[1];

      const consoleError = jest.spyOn(console, 'error').mockImplementation();
      errorHandler(new Error('Client error'));

      expect(consoleError).toHaveBeenCalledWith(
        'Client error:',
        expect.any(Error),
      );

      consoleError.mockRestore();
    });
  });

  describe('stop', () => {
    it('should close server and all client connections', async() => {
      await server.start();

      const client1 = { ...mockClient };
      const client2 = { ...mockClient };

      server.clients.add(client1);
      server.clients.add(client2);

      await server.stop();

      expect(client1.close).toHaveBeenCalled();
      expect(client2.close).toHaveBeenCalled();
      expect(mockWsServer.close).toHaveBeenCalled();
      expect(server.clients.size).toBe(0);
    });
  });

  describe('error handling', () => {
    it('should handle server errors', async() => {
      await server.start();

      const errorHandler = mockWsServer.on.mock.calls.find(
        call => call[0] === 'error',
      )[1];

      const consoleError = jest.spyOn(console, 'error').mockImplementation();
      errorHandler(new Error('Server error'));

      expect(consoleError).toHaveBeenCalledWith(
        'WebSocket server error:',
        expect.any(Error),
      );

      consoleError.mockRestore();
    });

    it('should handle missing arguments in tool calls', async() => {
      await server.start();
      const connectionHandler = mockWsServer.on.mock.calls.find(
        call => call[0] === 'connection',
      )[1];
      connectionHandler(mockClient);

      const message = {
        jsonrpc: '2.0',
        id: 11,
        method: 'tool_call',
        params: {
          tool: 'swarm_init',
          // Missing arguments
        },
      };

      await server.handleMessage(mockClient, JSON.stringify(message));

      expect(mockClient.send).toHaveBeenCalledWith(
        expect.stringContaining('error'),
      );
    });
  });

  describe('integration scenarios', () => {
    it('should handle complete swarm workflow', async() => {
      await server.start();
      const connectionHandler = mockWsServer.on.mock.calls.find(
        call => call[0] === 'connection',
      )[1];
      connectionHandler(mockClient);

      // Initialize swarm
      RuvSwarm.init.mockResolvedValue({ swarmId: 'swarm-123' });
      await server.handleMessage(mockClient, JSON.stringify({
        jsonrpc: '2.0',
        id: 1,
        method: 'tool_call',
        params: {
          tool: 'swarm_init',
          arguments: { topology: 'mesh' },
        },
      }));

      // Spawn agents
      RuvSwarm.spawnAgent.mockResolvedValue({ agentId: 'agent-1' });
      await server.handleMessage(mockClient, JSON.stringify({
        jsonrpc: '2.0',
        id: 2,
        method: 'tool_call',
        params: {
          tool: 'agent_spawn',
          arguments: { type: 'researcher' },
        },
      }));

      // Execute task
      RuvSwarm.executeTask.mockResolvedValue({ taskId: 'task-1' });
      await server.handleMessage(mockClient, JSON.stringify({
        jsonrpc: '2.0',
        id: 3,
        method: 'tool_call',
        params: {
          tool: 'task_orchestrate',
          arguments: { task: 'Research topic' },
        },
      }));

      // Check status
      RuvSwarm.getSwarmStatus.mockResolvedValue({ status: 'active' });
      await server.handleMessage(mockClient, JSON.stringify({
        jsonrpc: '2.0',
        id: 4,
        method: 'tool_call',
        params: {
          tool: 'swarm_status',
          arguments: {},
        },
      }));

      expect(mockClient.send).toHaveBeenCalledTimes(4);
    });
  });
});