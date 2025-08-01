#!/usr/bin/env node
/**
 * Enhanced MCP Server Implementation for Issue #91
 * Fixes: timeout issues, ANSI escape codes, JSON parsing errors, and notifications/initialized handling
 *
 * Version: 2.0.0 - Production Grade MCP Server
 * Author: Claude Code Assistant
 * License: MIT
 */

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { randomUUID } from 'crypto';

// Get version from package.json
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function getVersion() {
  try {
    const packagePath = join(__dirname, '..', 'package.json');
    const packageJson = JSON.parse(readFileSync(packagePath, 'utf8'));
    return packageJson.version;
  } catch (error) {
    return 'unknown';
  }
}

/**
 * MCP-Safe Logger - No ANSI escape codes, JSON-RPC compliant
 */
class MCPSafeLogger {
  constructor(options = {}) {
    this.name = options.name || 'ruv-swarm-mcp';
    this.level = options.level || 'INFO';
    this.sessionId = options.sessionId || randomUUID();
    this.enableDebug = process.env.DEBUG === 'true' || process.env.MCP_DEBUG === 'true';
    this.operations = new Map();
  }

  setCorrelationId(id) {
    this.sessionId = id || randomUUID();
    return this.sessionId;
  }

  // Safe logging that never interferes with JSON-RPC stdout
  _safeLog(level, message, data = {}) {
    if (!this.enableDebug && (level === 'DEBUG' || level === 'TRACE')) {
      return;
    }

    // Create clean log entry without ANSI codes
    const logEntry = {
      timestamp: new Date().toISOString(),
      level,
      component: this.name,
      sessionId: this.sessionId,
      message,
      ...data,
    };

    // CRITICAL: Use stderr for all logging to avoid corrupting JSON-RPC stdout
    // Use plain text without any color codes
    const plainOutput = `[${logEntry.timestamp}] ${level} [${this.name}] (${this.sessionId}) ${message}`;

    try {
      // Send to stderr with structured data
      console.error(plainOutput);
      if (Object.keys(data).length > 0) {
        console.error(JSON.stringify(data, null, 2));
      }
    } catch (error) {
      // Fallback - don't crash if logging fails
      console.error(`[${new Date().toISOString()}] LOGGER_ERROR Failed to log: ${error.message}`);
    }
  }

  info(message, data = {}) { this._safeLog('ℹ️  INFO ', message, data); }
  warn(message, data = {}) { this._safeLog('⚠️  WARN ', message, data); }
  error(message, data = {}) { this._safeLog('❌ ERROR', message, data); }
  debug(message, data = {}) { this._safeLog('🔍 DEBUG', message, data); }
  trace(message, data = {}) { this._safeLog('🔎 TRACE', message, data); }
  success(message, data = {}) { this._safeLog('✅ SUCCESS', message, data); }
  fatal(message, data = {}) { this._safeLog('💀 FATAL', message, data); }

  startOperation(operationName, metadata = {}) {
    const operationId = randomUUID();
    this.operations.set(operationId, {
      name: operationName,
      startTime: Date.now(),
      metadata,
    });
    this.debug(`Starting operation: ${operationName}`, { operationId, ...metadata });
    return operationId;
  }

  endOperation(operationId, success, metadata = {}) {
    const operation = this.operations.get(operationId);
    if (!operation) return;

    const duration = Date.now() - operation.startTime;
    this.operations.delete(operationId);

    const level = success ? 'info' : 'warn';
    this[level](`${success ? '✅' : '⚠️'} Operation ${success ? 'completed' : 'failed'}: ${operation.name}`, {
      operationId,
      duration,
      success,
      ...metadata,
    });
  }

  logConnection(type, sessionId, details = {}) {
    this.info(`🔌 Connection ${type}: ${sessionId}`, {
      connection: details,
    });
  }

  logMcp(direction, method, details = {}) {
    const symbol = direction === 'in' ? '📥' : '📤';
    this.debug(`${symbol} MCP ${direction.toUpperCase()} → ${method}`, {
      mcp: {
        direction,
        method,
        ...details,
      },
    });
  }
}

/**
 * Enhanced MCP Protocol Handler
 */
class EnhancedMCPHandler {
  constructor(logger, mcpTools) {
    this.logger = logger;
    this.mcpTools = mcpTools;
    this.capabilities = {
      tools: {},
      resources: {
        list: true,
        read: true,
      },
      notifications: {
        initialized: true,  // CRITICAL: Support for notifications/initialized
      },
    };
  }

  async handleRequest(request) {
    const response = {
      jsonrpc: '2.0',
      id: request.id,
    };

    try {
      this.logger.debug('Processing MCP request', {
        method: request.method,
        hasParams: !!request.params,
        requestId: request.id,
      });

      switch (request.method) {
        case 'initialize':
          response.result = await this.handleInitialize(request.params);
          break;

        case 'notifications/initialized':
          // CRITICAL FIX: Handle the notifications/initialized method properly
          // This is sent by MCP clients after successful initialization
          response.result = await this.handleInitializedNotification(request.params);
          this.logger.info('✅ MCP client initialization notification received');
          break;

        case 'tools/list':
          response.result = await this.handleToolsList();
          break;

        case 'tools/call':
          response.result = await this.handleToolCall(request.params);
          break;

        case 'resources/list':
          response.result = await this.handleResourcesList();
          break;

        case 'resources/read':
          response.result = await this.handleResourceRead(request.params);
          break;

        default:
          // ENHANCED: Better error handling for unknown methods
          this.logger.warn(`⚡ Operation failed: ${request.method}`, {
            method: request.method,
            supported: ['initialize', 'notifications/initialized', 'tools/list', 'tools/call', 'resources/list', 'resources/read'],
            suggestion: 'Check MCP protocol documentation for valid methods',
          });

          response.error = {
            code: -32601,
            message: 'Method not found',
            data: `Unknown method: ${request.method}. Supported methods: initialize, notifications/initialized, tools/list, tools/call, resources/list, resources/read`,
          };
      }
    } catch (error) {
      this.logger.error('Request processing error', {
        error: error.message,
        stack: error.stack,
        request: request.method,
      });

      response.error = {
        code: -32603,
        message: 'Internal error',
        data: error.message,
      };
    }

    return response;
  }

  async handleInitialize(params) {
    const version = await getVersion();
    this.logger.info('🚀 MCP server initializing', {
      clientInfo: params?.clientInfo,
      protocolVersion: params?.protocolVersion,
      serverVersion: version,
    });

    return {
      protocolVersion: '2024-11-05',
      capabilities: this.capabilities,
      serverInfo: {
        name: 'ruv-swarm-enhanced',
        version: version,
        description: 'Enhanced ruv-swarm MCP server with timeout fixes and ANSI-safe logging',
      },
    };
  }

  async handleInitializedNotification(params) {
    // This is a notification, not a request-response, but we handle it gracefully
    this.logger.info('🎉 MCP client successfully initialized', {
      notificationParams: params,
      timestamp: new Date().toISOString(),
    });

    // For notifications, we should return success but no specific result
    return { status: 'acknowledged' };
  }

  async handleToolsList() {
    this.logger.debug('📋 Listing available tools');

    // Enhanced tool definitions with better descriptions
    return {
      tools: [
        {
          name: 'swarm_init',
          description: 'Initialize a new swarm with specified topology and configuration',
          inputSchema: {
            type: 'object',
            properties: {
              topology: {
                type: 'string',
                enum: ['mesh', 'hierarchical', 'ring', 'star'],
                description: 'Network topology for agent communication',
              },
              maxAgents: {
                type: 'number',
                minimum: 1,
                maximum: 100,
                default: 5,
                description: 'Maximum number of agents in the swarm',
              },
              strategy: {
                type: 'string',
                enum: ['balanced', 'specialized', 'adaptive'],
                default: 'balanced',
                description: 'Agent distribution and task assignment strategy',
              },
            },
            required: ['topology'],
          },
        },
        {
          name: 'swarm_status',
          description: 'Get comprehensive status information about active swarms and agents',
          inputSchema: {
            type: 'object',
            properties: {
              verbose: {
                type: 'boolean',
                default: false,
                description: 'Include detailed per-agent performance metrics',
              },
            },
          },
        },
        {
          name: 'agent_spawn',
          description: 'Create and deploy a new agent with specified capabilities',
          inputSchema: {
            type: 'object',
            properties: {
              type: {
                type: 'string',
                enum: ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'],
                description: 'Agent specialization type',
              },
              name: {
                type: 'string',
                description: 'Human-readable agent identifier',
              },
              capabilities: {
                type: 'array',
                items: { type: 'string' },
                description: 'List of specific capabilities this agent should have',
              },
            },
            required: ['type'],
          },
        },
        {
          name: 'task_orchestrate',
          description: 'Distribute and coordinate task execution across the swarm',
          inputSchema: {
            type: 'object',
            properties: {
              task: {
                type: 'string',
                description: 'Detailed task description or instructions',
              },
              strategy: {
                type: 'string',
                enum: ['parallel', 'sequential', 'adaptive'],
                default: 'adaptive',
                description: 'Task execution strategy',
              },
              priority: {
                type: 'string',
                enum: ['low', 'medium', 'high', 'critical'],
                default: 'medium',
              },
            },
            required: ['task'],
          },
        },
        {
          name: 'neural_train',
          description: 'Train neural networks for specific agents with validation',
          inputSchema: {
            type: 'object',
            properties: {
              agentId: {
                type: 'string',
                description: 'ID of the agent whose neural network to train (REQUIRED)',
              },
              iterations: {
                type: 'number',
                minimum: 1,
                maximum: 100,
                default: 10,
                description: 'Number of training iterations',
              },
              learningRate: {
                type: 'number',
                minimum: 0.0001,
                maximum: 1.0,
                default: 0.001,
                description: 'Learning rate for neural network training',
              },
              modelType: {
                type: 'string',
                enum: ['feedforward', 'lstm', 'transformer', 'cnn', 'attention'],
                default: 'feedforward',
                description: 'Type of neural network model to train',
              },
            },
            required: ['agentId'],
          },
        },
        // Additional tools can be added here...
      ],
    };
  }

  async handleToolCall(params) {
    const { name: toolName, arguments: toolArgs } = params;

    this.logger.debug('🔧 Executing tool', {
      tool: toolName,
      hasArgs: !!toolArgs,
      argCount: toolArgs ? Object.keys(toolArgs).length : 0,
    });

    const toolOpId = this.logger.startOperation(`tool-${toolName}`, {
      tool: toolName,
      args: toolArgs,
    });

    try {
      let result;
      let toolFound = false;

      // Try enhanced MCP tools first
      if (typeof this.mcpTools[toolName] === 'function') {
        result = await this.mcpTools[toolName](toolArgs);
        toolFound = true;
      }
      // Try DAA tools if available
      else if (this.mcpTools.daaMcpTools && typeof this.mcpTools.daaMcpTools[toolName] === 'function') {
        result = await this.mcpTools.daaMcpTools[toolName](toolArgs);
        toolFound = true;
      }

      if (toolFound) {
        this.logger.endOperation(toolOpId, true, {
          resultType: typeof result,
          resultSize: typeof result === 'string' ? result.length : JSON.stringify(result).length,
        });

        return {
          content: [{
            type: 'text',
            text: typeof result === 'string' ? result : JSON.stringify(result, null, 2),
          }],
        };
      } else {
        this.logger.endOperation(toolOpId, false, {
          error: 'Tool not found',
          availableTools: Object.keys(this.mcpTools),
        });

        throw new Error(`Unknown tool: ${toolName}. Available tools: ${Object.keys(this.mcpTools).join(', ')}`);
      }
    } catch (error) {
      this.logger.endOperation(toolOpId, false, { error: error.message });
      throw error;
    }
  }

  async handleResourcesList() {
    return {
      resources: [
        {
          uri: 'swarm://status',
          name: 'Swarm Status',
          description: 'Current swarm status and metrics',
          mimeType: 'application/json',
        },
        {
          uri: 'swarm://logs',
          name: 'Swarm Logs',
          description: 'Recent swarm operation logs',
          mimeType: 'text/plain',
        },
      ],
    };
  }

  async handleResourceRead(params) {
    const { uri } = params;

    switch (uri) {
      case 'swarm://status':
        const status = await this.mcpTools.swarm_status({ verbose: true });
        return {
          contents: [{
            uri,
            mimeType: 'application/json',
            text: JSON.stringify(status, null, 2),
          }],
        };

      case 'swarm://logs':
        return {
          contents: [{
            uri,
            mimeType: 'text/plain',
            text: 'Swarm logs would be provided here...',
          }],
        };

      default:
        throw new Error(`Unknown resource: ${uri}`);
    }
  }
}

/**
 * Enhanced MCP Server with Connection Management
 */
class EnhancedMCPServer {
  constructor(mcpTools, options = {}) {
    this.logger = new MCPSafeLogger({
      name: 'ruv-swarm-mcp',
      sessionId: options.sessionId || randomUUID(),
      level: options.logLevel || 'INFO',
    });

    this.handler = new EnhancedMCPHandler(this.logger, mcpTools);
    this.buffer = '';
    this.messageCount = 0;
    this.isRunning = false;
    this.lastActivityTime = Date.now();

    // Connection management without timeouts
    this.connectionState = {
      initialized: false,
      clientInfo: null,
      startTime: Date.now(),
      messagesProcessed: 0,
    };
  }

  async start() {
    if (this.isRunning) {
      this.logger.warn('Server already running');
      return;
    }

    this.isRunning = true;
    const sessionId = this.logger.setCorrelationId();
    const version = await getVersion();

    this.logger.info('🚀 Enhanced MCP server starting', {
      protocol: 'stdio',
      sessionId,
      nodeVersion: process.version,
      platform: process.platform,
      arch: process.arch,
      pid: process.pid,
      version: version,
    });

    this.logger.logConnection('established', sessionId, {
      protocol: 'stdio',
      transport: 'stdin/stdout',
      timestamp: new Date().toISOString(),
    });

    // Set up stdin handling
    process.stdin.setEncoding('utf8');

    // Signal readiness for testing environments
    if (process.env.MCP_TEST_MODE === 'true') {
      console.error('Enhanced MCP server ready'); // Use stderr to avoid JSON-RPC interference
    }

    // Process stdin data
    process.stdin.on('data', (chunk) => {
      this.handleData(chunk);
    });

    // Handle stdin end
    process.stdin.on('end', () => {
      this.logger.info('📤 stdin closed, shutting down server');
      this.shutdown();
    });

    // Error handling
    process.stdin.on('error', (error) => {
      this.logger.error('stdin error', { error: error.message });
      this.shutdown();
    });

    // Process error handling
    process.on('uncaughtException', (error) => {
      this.logger.fatal('Uncaught exception', { error: error.message, stack: error.stack });
      this.shutdown();
    });

    process.on('unhandledRejection', (reason, promise) => {
      this.logger.fatal('Unhandled rejection', { reason, promise });
      this.shutdown();
    });

    // Graceful shutdown signals
    process.on('SIGTERM', () => {
      this.logger.info('🛑 Received SIGTERM, shutting down gracefully');
      this.shutdown();
    });

    process.on('SIGINT', () => {
      this.logger.info('🛑 Received SIGINT, shutting down gracefully');
      this.shutdown();
    });

    this.logger.success('✅ MCP server fully initialized and ready');
  }

  handleData(chunk) {
    this.lastActivityTime = Date.now();
    this.logger.trace('Received stdin data', { bytes: chunk.length });

    this.buffer += chunk;

    // Process complete JSON messages
    const lines = this.buffer.split('\n');
    this.buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.trim()) {
        this.processMessage(line.trim());
      }
    }
  }

  async processMessage(line) {
    this.messageCount++;
    this.connectionState.messagesProcessed++;
    const messageId = `msg-${this.logger.sessionId}-${this.messageCount}`;

    try {
      // Parse JSON-RPC request
      const request = JSON.parse(line);

      this.logger.logMcp('in', request.method || 'unknown', {
        method: request.method,
        id: request.id,
        params: request.params,
        messageId,
      });

      // Process the request
      const opId = this.logger.startOperation(`mcp-${request.method}`, {
        requestId: request.id,
        messageId,
      });

      try {
        const response = await this.handler.handleRequest(request);

        this.logger.endOperation(opId, !response.error, {
          hasError: !!response.error,
          method: request.method,
        });

        this.logger.logMcp('out', request.method || 'response', {
          method: request.method,
          id: response.id,
          result: response.result,
          error: response.error,
          messageId,
        });

        // Send response to stdout (JSON-RPC)
        this.sendResponse(response);

      } catch (handlerError) {
        this.logger.endOperation(opId, false, { error: handlerError.message });

        // Send error response
        const errorResponse = {
          jsonrpc: '2.0',
          error: {
            code: -32603,
            message: 'Internal error',
            data: handlerError.message,
          },
          id: request.id,
        };
        this.sendResponse(errorResponse);
      }

    } catch (parseError) {
      this.logger.error('JSON parse error', {
        error: parseError.message,
        line: line.substring(0, 100) + (line.length > 100 ? '...' : ''),
        messageId,
      });

      // Send parse error response
      const parseErrorResponse = {
        jsonrpc: '2.0',
        error: {
          code: -32700,
          message: 'Parse error',
          data: 'Invalid JSON-RPC request',
        },
        id: null,
      };
      this.sendResponse(parseErrorResponse);
    }
  }

  sendResponse(response) {
    try {
      const responseStr = JSON.stringify(response);
      process.stdout.write(responseStr + '\n');

      // Ensure stdout is flushed
      if (process.stdout.flush) {
        process.stdout.flush();
      }

    } catch (writeError) {
      this.logger.fatal('Failed to write response to stdout', {
        writeError: writeError.message,
        responseSize: JSON.stringify(response).length,
      });
      this.shutdown();
    }
  }

  shutdown() {
    if (!this.isRunning) return;

    this.isRunning = false;

    this.logger.info('🔚 MCP server shutting down', {
      uptime: Date.now() - this.connectionState.startTime,
      messagesProcessed: this.connectionState.messagesProcessed,
      lastActivity: Date.now() - this.lastActivityTime,
    });

    // Clean shutdown
    process.exit(0);
  }

  getStatus() {
    return {
      isRunning: this.isRunning,
      connectionState: this.connectionState,
      uptime: Date.now() - this.connectionState.startTime,
      lastActivity: Date.now() - this.lastActivityTime,
      messagesProcessed: this.connectionState.messagesProcessed,
    };
  }
}

export { EnhancedMCPServer, MCPSafeLogger, EnhancedMCPHandler };
