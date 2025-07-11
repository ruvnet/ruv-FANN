#!/usr/bin/env node
/**
 * Clean, modular ruv-swarm CLI with Claude Code integration
 * Uses modular architecture for better maintainability and remote execution
 */

import { setupClaudeIntegration, invokeClaudeWithSwarm } from '../src/claude-integration/index.js';
import { RuvSwarm } from '../src/index-enhanced.js';
import { EnhancedMCPTools } from '../src/mcp-tools-enhanced.js';
import { daaMcpTools } from '../src/mcp-daa-tools.js';
import mcpToolsEnhanced from '../src/mcp-tools-enhanced.js';
import { Logger } from '../src/logger.js';

// Input validation constants and functions
const VALID_TOPOLOGIES = ['mesh', 'hierarchical', 'ring', 'star'];
const VALID_AGENT_TYPES = ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator', 'architect', 'tester'];
const MAX_AGENTS_LIMIT = 100;
const MIN_AGENTS_LIMIT = 1;

class ValidationError extends Error {
    constructor(message, parameter = null) {
        super(message);
        this.name = 'ValidationError';
        this.parameter = parameter;
    }
}

function validateTopology(topology) {
    if (!topology || typeof topology !== 'string') {
        throw new ValidationError('Topology must be a non-empty string', 'topology');
    }
    
    if (!VALID_TOPOLOGIES.includes(topology.toLowerCase())) {
        throw new ValidationError(
            `Invalid topology '${topology}'. Valid topologies are: ${VALID_TOPOLOGIES.join(', ')}`,
            'topology'
        );
    }
    
    return topology.toLowerCase();
}

function validateMaxAgents(maxAgents) {
    // Handle string input
    if (typeof maxAgents === 'string') {
        const parsed = parseInt(maxAgents, 10);
        if (isNaN(parsed)) {
            throw new ValidationError(
                `Invalid maxAgents '${maxAgents}'. Must be a number between ${MIN_AGENTS_LIMIT} and ${MAX_AGENTS_LIMIT}`,
                'maxAgents'
            );
        }
        maxAgents = parsed;
    }
    
    if (!Number.isInteger(maxAgents) || maxAgents < MIN_AGENTS_LIMIT || maxAgents > MAX_AGENTS_LIMIT) {
        throw new ValidationError(
            `Invalid maxAgents '${maxAgents}'. Must be an integer between ${MIN_AGENTS_LIMIT} and ${MAX_AGENTS_LIMIT}`,
            'maxAgents'
        );
    }
    
    return maxAgents;
}

function validateAgentType(type) {
    if (!type || typeof type !== 'string') {
        throw new ValidationError('Agent type must be a non-empty string', 'type');
    }
    
    if (!VALID_AGENT_TYPES.includes(type.toLowerCase())) {
        throw new ValidationError(
            `Invalid agent type '${type}'. Valid types are: ${VALID_AGENT_TYPES.join(', ')}`,
            'type'
        );
    }
    
    return type.toLowerCase();
}

function validateAgentName(name) {
    if (name !== null && name !== undefined) {
        if (typeof name !== 'string') {
            throw new ValidationError('Agent name must be a string', 'name');
        }
        
        if (name.length === 0) {
            throw new ValidationError('Agent name cannot be empty', 'name');
        }
        
        if (name.length > 100) {
            throw new ValidationError('Agent name cannot exceed 100 characters', 'name');
        }
        
        // Check for invalid characters
        if (!/^[a-zA-Z0-9\s\-_\.]+$/.test(name)) {
            throw new ValidationError(
                'Agent name can only contain letters, numbers, spaces, hyphens, underscores, and periods',
                'name'
            );
        }
    }
    
    return name;
}

function validateTaskDescription(task) {
    if (!task || typeof task !== 'string') {
        throw new ValidationError('Task description must be a non-empty string', 'task');
    }
    
    if (task.trim().length === 0) {
        throw new ValidationError('Task description cannot be empty or only whitespace', 'task');
    }
    
    if (task.length > 1000) {
        throw new ValidationError('Task description cannot exceed 1000 characters', 'task');
    }
    
    return task.trim();
}

function logValidationError(error, command) {
    console.log(`❌ Validation Error in '${command}' command:`);
    console.log(`   ${error.message}`);
    if (error.parameter) {
        console.log(`   Parameter: ${error.parameter}`);
    }
    console.log(`\n💡 For help with valid parameters, run: ruv-swarm help`);
}

let globalRuvSwarm = null;
let globalMCPTools = null;
let globalLogger = null;

// Initialize logger based on environment
function initializeLogger() {
    if (!globalLogger) {
        globalLogger = new Logger({
            name: 'ruv-swarm-mcp',
            level: process.env.LOG_LEVEL || (process.argv.includes('--debug') ? 'DEBUG' : 'INFO'),
            enableStderr: true, // Always use stderr in MCP mode
            enableFile: process.env.LOG_TO_FILE === 'true',
            formatJson: process.env.LOG_FORMAT === 'json',
            logDir: process.env.LOG_DIR || './logs',
            metadata: {
                pid: process.pid,
                version: '1.0.11',
                mode: 'mcp-stdio'
            }
        });
        
        // Set up global error handlers
        process.on('uncaughtException', (error) => {
            globalLogger.fatal('Uncaught exception', { error });
            process.exit(1);
        });
        
        process.on('unhandledRejection', (reason, promise) => {
            globalLogger.fatal('Unhandled rejection', { reason, promise });
            process.exit(1);
        });
    }
    return globalLogger;
}

async function initializeSystem() {
    if (!globalRuvSwarm) {
        // RuvSwarm.initialize already prints initialization messages
        // Use faster initialization in MCP mode
        const isMcpMode = process.env.MCP_TEST_MODE === 'true';
        globalRuvSwarm = await RuvSwarm.initialize({
            loadingStrategy: 'progressive',
            enablePersistence: true,
            enableNeuralNetworks: !isMcpMode, // Disable neural networks in test mode for faster startup
            enableForecasting: !isMcpMode,
            useSIMD: !isMcpMode && RuvSwarm.detectSIMDSupport(),
            debug: process.argv.includes('--debug')
        });
    }
    
    if (!globalMCPTools) {
        // Pass the already initialized RuvSwarm instance to avoid duplicate initialization
        globalMCPTools = new EnhancedMCPTools(globalRuvSwarm);
        await globalMCPTools.initialize(globalRuvSwarm);
        
        // Initialize DAA MCP tools with the same instance
        daaMcpTools.mcpTools = globalMCPTools;
        await daaMcpTools.ensureInitialized();
        
        // Add DAA tool methods to the MCP tools object
        const daaToolNames = [
            'daa_init', 'daa_agent_create', 'daa_agent_adapt', 'daa_workflow_create',
            'daa_workflow_execute', 'daa_knowledge_share', 'daa_learning_status',
            'daa_cognitive_pattern', 'daa_meta_learning', 'daa_performance_metrics'
        ];
        
        for (const toolName of daaToolNames) {
            if (typeof daaMcpTools[toolName] === 'function') {
                globalMCPTools[toolName] = daaMcpTools[toolName].bind(daaMcpTools);
            }
        }
    }
    
    return { ruvSwarm: globalRuvSwarm, mcpTools: globalMCPTools };
}

async function handleInit(args) {
    try {
        const { mcpTools } = await initializeSystem();
        
        // Filter out flags to get positional arguments
        const positionalArgs = args.filter(arg => !arg.startsWith('--'));
        const rawTopology = positionalArgs[0] || 'mesh';
        const rawMaxAgents = positionalArgs[1] || '5';
        const setupClaude = args.includes('--claude') || args.includes('--setup-claude');
        const forceSetup = args.includes('--force');
        const mergeSetup = args.includes('--merge');
        const noInteractive = args.includes('--no-interactive');
        const noBackup = args.includes('--no-backup');
        
        // Validate inputs
        const topology = validateTopology(rawTopology);
        const maxAgents = validateMaxAgents(rawMaxAgents);
        
        console.log('🚀 Initializing ruv-swarm...');
        
        const result = await mcpTools.swarm_init({
            topology,
            maxAgents,
            strategy: 'balanced',
            enableCognitiveDiversity: true,
            enableNeuralAgents: true,
            enableForecasting: args.includes('--forecasting')
        });
        
        console.log('🐝 Swarm initialized:');
        console.log('   ID: ' + result.id);
        console.log('   Topology: ' + result.topology);
        console.log('   Max Agents: ' + result.maxAgents);
        console.log('   Features: ' + Object.entries(result.features).filter(([k,v]) => v).map(([k,v]) => k).join(', '));
        console.log('   Performance: ' + result.performance.initialization_time_ms.toFixed(1) + 'ms');
        
        // Setup Claude integration using modular approach
        if (setupClaude || forceSetup || mergeSetup) {
            console.log('\n📚 Setting up modular Claude Code integration...');
            try {
                await setupClaudeIntegration({
                    autoSetup: setupClaude,
                    forceSetup: forceSetup,
                    mergeSetup: mergeSetup,
                    noBackup: noBackup,
                    interactive: !noInteractive,
                    workingDir: process.cwd(),
                    packageName: 'ruv-swarm'
                });
            } catch (error) {
                console.log('⚠️  Claude integration setup had issues:', error.message);
                console.log('💡 Manual setup: claude mcp add ruv-swarm npx ruv-swarm mcp start');
            }
        }
        
        console.log('\n✅ Initialization complete!');
        console.log('\n🔗 Next steps:');
        console.log('   1. Test with MCP tools: mcp__ruv-swarm__agent_spawn');
        console.log('   2. Use wrapper scripts for remote execution');
        console.log('   3. Check .claude/commands/ for detailed guides');
        
        if (forceSetup) {
            console.log('\n🔄 Files regenerated with --force flag');
        } else if (mergeSetup) {
            console.log('\n🔄 Configuration merged with existing files');
        }
    } catch (error) {
        if (error instanceof ValidationError) {
            logValidationError(error, 'init');
            return;
        }
        throw error;
    }
}

async function handleSpawn(args) {
    try {
        const { mcpTools } = await initializeSystem();
        
        const rawType = args[0] || 'researcher';
        const rawName = args[1] || null;
        
        // Validate inputs
        const type = validateAgentType(rawType);
        const name = validateAgentName(rawName);
    
        const result = await mcpTools.agent_spawn({
            type,
            name,
            enableNeuralNetwork: !args.includes('--no-neural')
        });
        
        console.log('🤖 Agent spawned:');
        console.log('   ID: ' + result.agent.id);
        console.log('   Name: ' + result.agent.name);
        console.log('   Type: ' + result.agent.type);
        console.log('   Cognitive Pattern: ' + result.agent.cognitive_pattern);
        if (result.agent.neural_network_id) {
            console.log('   Neural Network: ' + result.agent.neural_network_id);
        }
        console.log('   Swarm Capacity: ' + result.swarm_info.capacity);
    } catch (error) {
        if (error instanceof ValidationError) {
            logValidationError(error, 'spawn');
            return;
        }
        throw error;
    }
}

async function handleOrchestrate(args) {
    try {
        const { mcpTools } = await initializeSystem();
        
        const rawTask = args.join(' ');
        if (!rawTask) {
            console.log('❌ No task provided');
            console.log('Usage: ruv-swarm orchestrate "task description"');
            return;
        }
        
        // Validate task description
        const task = validateTaskDescription(rawTask);
    
        const result = await mcpTools.task_orchestrate({
            task: task,
            strategy: 'adaptive'
        });
        
        console.log('📋 Task orchestrated:');
        console.log('   ID: ' + result.taskId);
        console.log('   Description: ' + result.description);
        console.log('   Assigned Agents: ' + result.assigned_agents.length);
        console.log('   Status: ' + result.status);
        console.log('   Estimated Completion: ' + result.performance.estimated_completion_ms + 'ms');
    } catch (error) {
        if (error instanceof ValidationError) {
            logValidationError(error, 'orchestrate');
            return;
        }
        throw error;
    }
}

async function handleClaudeInvoke(args) {
    const filteredArgs = args.filter(arg => 
        !arg.includes('--skip-permissions') && 
        !arg.includes('--dangerously-skip-permissions')
    );
    
    const prompt = filteredArgs.join(' ');
    
    if (!prompt.trim()) {
        console.log('❌ No prompt provided');
        console.log('Usage: ruv-swarm claude-invoke "your swarm prompt"');
        console.log('Note: --dangerously-skip-permissions is automatically included');
        return;
    }
    
    console.log('🚀 Invoking Claude Code with ruv-swarm integration...');
    console.log('Prompt: ' + prompt.trim());
    console.log('⚠️  Automatically using --dangerously-skip-permissions for seamless execution');
    
    try {
        await invokeClaudeWithSwarm(prompt, {
            workingDir: process.cwd()
        });
    } catch (error) {
        console.error('❌ Claude invocation failed:', error.message);
        console.error('Make sure Claude Code CLI is installed and in your PATH');
        process.exit(1);
    }
}

async function handleStatus(args) {
    const { mcpTools } = await initializeSystem();
    
    const verbose = args.includes('--verbose') || args.includes('-v');
    const swarmId = args.find(arg => !arg.startsWith('-'));
    
    const result = await mcpTools.swarm_status({ verbose });
    
    if (swarmId) {
        console.log(`🐝 Swarm Status (${swarmId}):`);
        console.log(`   Agents: ${result.agents.total} (${result.agents.active} active, ${result.agents.idle} idle)`);
        console.log(`   Tasks: ${result.tasks.total} (${result.tasks.pending} pending, ${result.tasks.in_progress} in progress)`);
    } else {
        console.log('🌐 Global Status:');
        console.log(`   Active Swarms: ${result.active_swarms}`);
        console.log(`   Total Agents: ${result.global_metrics.totalAgents}`);
        console.log(`   Total Tasks: ${result.global_metrics.totalTasks}`);
        console.log(`   Memory Usage: ${result.global_metrics.memoryUsage / (1024 * 1024)}MB`);
        
        if (verbose) {
            console.log('\n📊 WASM Modules:');
            Object.entries(result.runtime_info.wasm_modules).forEach(([name, status]) => {
                console.log(`   ${name}: ${status.loaded ? '✅ Loaded' : '⏳ Not loaded'} (${(status.size / 1024).toFixed(0)}KB)`);
            });
        }
    }
}

async function handleMonitor(args) {
    const { mcpTools } = await initializeSystem();
    
    const duration = parseInt(args.find(arg => arg.match(/^\d+$/))) || 10000;
    
    console.log(`📊 Monitoring for ${duration}ms...`);
    console.log('Press Ctrl+C to stop\n');
    
    const interval = setInterval(async () => {
        const status = await mcpTools.swarm_status({ verbose: false });
        process.stdout.write('\r');
        process.stdout.write(`Swarms: ${status.active_swarms} | Agents: ${status.global_metrics.totalAgents} | Tasks: ${status.global_metrics.totalTasks} | Memory: ${(status.global_metrics.memoryUsage / (1024 * 1024)).toFixed(1)}MB`);
    }, 1000);
    
    setTimeout(() => {
        clearInterval(interval);
        console.log('\n\n✅ Monitoring complete');
    }, duration);
}

async function handleMcp(args) {
    const subcommand = args[0] || 'help';
    
    switch (subcommand) {
        case 'start':
            await startMcpServer(args.slice(1));
            break;
        case 'status':
            await getMcpStatus();
            break;
        case 'stop':
            await stopMcpServer();
            break;
        case 'tools':
            await listMcpTools();
            break;
        case 'config':
            await configureMcp(args.slice(1));
            break;
        case 'help':
        default:
            showMcpHelp();
    }
}

async function startMcpServer(args) {
    const protocol = args.find(arg => arg.startsWith('--protocol='))?.split('=')[1] || 'stdio';
    const port = args.find(arg => arg.startsWith('--port='))?.split('=')[1] || '3000';
    const host = args.find(arg => arg.startsWith('--host='))?.split('=')[1] || 'localhost';
    
    // Initialize logger first
    const logger = initializeLogger();
    const sessionId = logger.setCorrelationId();
    
    try {
        if (protocol === 'stdio') {
            // In stdio mode, only JSON-RPC messages should go to stdout
            logger.info('ruv-swarm MCP server starting in stdio mode', {
                protocol,
                sessionId,
                nodeVersion: process.version,
                platform: process.platform,
                arch: process.arch
            });
            
            // Log connection establishment
            logger.logConnection('established', sessionId, {
                protocol: 'stdio',
                transport: 'stdin/stdout',
                timestamp: new Date().toISOString()
            });
            
            // Initialize WASM if needed with timeout protection
            const initOpId = logger.startOperation('initialize-system');
            logger.info('Initializing ruv-swarm system...');
            
            // Set a reasonable timeout for initialization
            const initTimeout = setTimeout(() => {
                logger.error('System initialization timeout - forcing startup');
                if (process.env.MCP_TEST_MODE === 'true') {
                    console.error('MCP server ready'); // Signal readiness even if init is slow
                }
            }, 15000); // 15 second timeout
            
            let mcpTools = null;
            try {
                const { ruvSwarm, mcpTools: initializedMcpTools } = await initializeSystem();
                mcpTools = initializedMcpTools;
                clearTimeout(initTimeout);
                logger.endOperation(initOpId, true, { modulesLoaded: true });
                logger.info('System initialization completed successfully');
            } catch (error) {
                clearTimeout(initTimeout);
                logger.error('System initialization failed, continuing with basic functionality', { error });
                // Continue with basic functionality even if full init fails
                mcpTools = null; // Ensure mcpTools is null if initialization fails
            }
            
            // Start stdio MCP server loop
            process.stdin.setEncoding('utf8');
            
            // Log successful startup
            logger.info('MCP server successfully started in stdio mode');
            if (process.env.MCP_TEST_MODE === 'true') {
                console.error('Server initialization complete - ready for requests');
            }
            
            // Signal server readiness for testing
            if (process.env.MCP_TEST_MODE === 'true') {
                console.error('MCP server ready'); // Use stderr so it doesn't interfere with JSON-RPC
                console.error('ruv-swarm MCP server initialized in stdio mode');
                console.error('Listening on stdin/stdout for JSON-RPC messages');
            }
            
            let buffer = '';
            let messageCount = 0;
            
            process.stdin.on('data', (chunk) => {
                logger.trace('Received stdin data', { bytes: chunk.length });
                buffer += chunk;
                
                // Process complete JSON messages
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (line.trim()) {
                        messageCount++;
                        const messageId = `msg-${sessionId}-${messageCount}`;
                        
                        try {
                            const request = JSON.parse(line);
                            logger.logMcp('in', request.method || 'unknown', {
                                method: request.method,
                                id: request.id,
                                params: request.params,
                                messageId
                            });
                            
                            const opId = logger.startOperation(`mcp-${request.method}`, {
                                requestId: request.id,
                                messageId
                            });
                            
                            handleMcpRequest(request, mcpTools, logger).then(response => {
                                logger.endOperation(opId, !response.error, {
                                    hasError: !!response.error
                                });
                                
                                logger.logMcp('out', request.method || 'response', {
                                    method: request.method,
                                    id: response.id,
                                    result: response.result,
                                    error: response.error,
                                    messageId
                                });
                                
                                try {
                                    process.stdout.write(JSON.stringify(response) + '\n');
                                } catch (writeError) {
                                    logger.error('Failed to write response to stdout', { writeError, response });
                                    process.exit(1);
                                }
                            }).catch(error => {
                                logger.endOperation(opId, false, { error });
                                logger.error('Request handler error', { error, request });
                                
                                const errorResponse = {
                                    jsonrpc: '2.0',
                                    error: {
                                        code: -32603,
                                        message: 'Internal error',
                                        data: error.message
                                    },
                                    id: request.id
                                };
                                process.stdout.write(JSON.stringify(errorResponse) + '\n');
                            });
                        } catch (error) {
                            logger.error('JSON parse error', { 
                                error, 
                                line: line.substring(0, 100),
                                messageId 
                            });
                            
                            const errorResponse = {
                                jsonrpc: '2.0',
                                error: {
                                    code: -32700,
                                    message: 'Parse error',
                                    data: error.message
                                },
                                id: null
                            };
                            process.stdout.write(JSON.stringify(errorResponse) + '\n');
                        }
                    }
                }
            });
            
            // Set up connection monitoring
            const monitorInterval = setInterval(() => {
                logger.logMemoryUsage('mcp-server');
                logger.debug('Connection metrics', logger.getConnectionMetrics());
            }, 60000); // Every minute
            
            // Handle stdin close
            process.stdin.on('end', () => {
                logger.logConnection('closed', sessionId, {
                    messagesProcessed: messageCount,
                    uptime: process.uptime()
                });
                logger.info('MCP: stdin closed, shutting down...');
                clearInterval(monitorInterval);
                process.exit(0);
            });
            
            process.stdin.on('error', (error) => {
                logger.logConnection('failed', sessionId, { error });
                logger.error('MCP: stdin error, shutting down...', { error });
                clearInterval(monitorInterval);
                process.exit(1);
            });
            
            // Handle process termination signals
            process.on('SIGTERM', () => {
                logger.info('MCP: Received SIGTERM, shutting down gracefully...');
                clearInterval(monitorInterval);
                process.exit(0);
            });
            
            process.on('SIGINT', () => {
                logger.info('MCP: Received SIGINT, shutting down gracefully...');
                clearInterval(monitorInterval);
                process.exit(0);
            });
            
            // Send initialization message
            const initMessage = {
                jsonrpc: '2.0',
                method: 'server.initialized',
                params: {
                    serverInfo: {
                        name: 'ruv-swarm',
                        version: '1.0.8',
                        capabilities: {
                            tools: true,
                            prompts: false,
                            resources: true
                        }
                    }
                }
            };
            process.stdout.write(JSON.stringify(initMessage) + '\n');
            
            // Implement heartbeat mechanism
            let lastActivity = Date.now();
            const heartbeatInterval = 30000; // 30 seconds
            const heartbeatTimeout = 90000; // 90 seconds
            
            // Update activity on any received message
            const originalOnData = process.stdin._events.data;
            process.stdin.on('data', () => {
                lastActivity = Date.now();
            });
            
            // Check for connection health
            const heartbeatChecker = setInterval(() => {
                const timeSinceLastActivity = Date.now() - lastActivity;
                
                if (timeSinceLastActivity > heartbeatTimeout) {
                    logger.error('MCP: Connection timeout - no activity for', timeSinceLastActivity, 'ms');
                    logger.logConnection('timeout', sessionId, {
                        lastActivity: new Date(lastActivity).toISOString(),
                        timeout: heartbeatTimeout
                    });
                    clearInterval(monitorInterval);
                    clearInterval(heartbeatChecker);
                    process.exit(1);
                } else if (timeSinceLastActivity > heartbeatInterval) {
                    logger.debug('MCP: Connection idle for', timeSinceLastActivity, 'ms');
                }
            }, 5000); // Check every 5 seconds
            
            // Clean up heartbeat on exit
            process.on('exit', () => {
                clearInterval(heartbeatChecker);
            });
            
        } else {
            logger.error('WebSocket protocol not yet implemented', { protocol });
            console.log('❌ WebSocket protocol not yet implemented in clean version');
            console.log('Use stdio mode for Claude Code integration');
        }
    } catch (error) {
        logger.fatal('Failed to start MCP server', { error, protocol });
        console.error('❌ Failed to start MCP server:', error.message);
        process.exit(1);
    }
}

async function getMcpStatus() {
    console.log('🔍 MCP Server Status:');
    console.log('   Protocol: stdio (for Claude Code integration)');
    console.log('   Status: Ready to start');
    console.log('   Usage: npx ruv-swarm mcp start');
}

async function stopMcpServer() {
    console.log('✅ MCP server stopped (stdio mode exits automatically)');
}

async function listMcpTools() {
    console.log('🛠️  Available MCP Tools:');
    console.log('\n📊 Core Swarm Tools:');
    console.log('   mcp__ruv-swarm__swarm_init - Initialize a new swarm');
    console.log('   mcp__ruv-swarm__agent_spawn - Spawn new agents');
    console.log('   mcp__ruv-swarm__task_orchestrate - Orchestrate tasks');
    console.log('   mcp__ruv-swarm__swarm_status - Get swarm status');
    console.log('   ... and 11 more core tools');
    console.log('\n🤖 DAA (Decentralized Autonomous Agents) Tools:');
    console.log('   mcp__ruv-swarm__daa_init - Initialize DAA service');
    console.log('   mcp__ruv-swarm__daa_agent_create - Create autonomous agents');
    console.log('   mcp__ruv-swarm__daa_workflow_create - Create DAA workflows');
    console.log('   mcp__ruv-swarm__daa_learning_status - Get learning progress');
    console.log('   ... and 6 more DAA tools');
    console.log('\nFor full documentation, run: ruv-swarm init --claude');
}

function showMcpHelp() {
    console.log(`
🔌 MCP (Model Context Protocol) Commands

Usage: ruv-swarm mcp <subcommand> [options]

Subcommands:
  start [--protocol=stdio]    Start MCP server (stdio for Claude Code)
  status                      Show MCP server status
  stop                        Stop MCP server
  tools                       List available MCP tools
  help                        Show this help message

Examples:
  ruv-swarm mcp start                    # Start stdio MCP server
  ruv-swarm mcp tools                    # List available tools
  
For Claude Code integration:
  claude mcp add ruv-swarm npx ruv-swarm mcp start
`);
}

async function configureMcp(args) {
    console.log('🔧 MCP configuration is managed through Claude Code');
    console.log('Run: ruv-swarm init --claude');
}

async function getResourceContent(uri) {
    const resources = {
        'swarm://docs/getting-started': {
            contents: [{
                uri,
                mimeType: 'text/markdown',
                text: `# Getting Started with ruv-swarm

## Introduction
ruv-swarm is a powerful WASM-powered neural swarm orchestration system that enhances Claude Code's capabilities through intelligent agent coordination.

## Quick Start

1. **Initialize a swarm:**
   \`\`\`bash
   mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 5 }
   \`\`\`

2. **Spawn agents:**
   \`\`\`bash
   mcp__ruv-swarm__agent_spawn { type: "researcher", name: "Doc Analyzer" }
   mcp__ruv-swarm__agent_spawn { type: "coder", name: "Implementation Expert" }
   \`\`\`

3. **Orchestrate tasks:**
   \`\`\`bash
   mcp__ruv-swarm__task_orchestrate { task: "Build a REST API", strategy: "adaptive" }
   \`\`\`

## Key Concepts

- **Agents**: Cognitive patterns that guide Claude Code's approach
- **Topologies**: Organizational structures for agent coordination
- **Memory**: Persistent state across sessions
- **Neural Training**: Continuous improvement through learning

## Best Practices

1. Always batch operations in a single message
2. Use memory for cross-agent coordination
3. Monitor progress with status tools
4. Train neural patterns for better results`
            }]
        },
        'swarm://docs/topologies': {
            contents: [{
                uri,
                mimeType: 'text/markdown',
                text: `# Swarm Topologies

## Available Topologies

### 1. Mesh
- **Description**: Fully connected network where all agents communicate
- **Best for**: Complex problems requiring diverse perspectives
- **Characteristics**: High coordination, maximum information sharing

### 2. Hierarchical
- **Description**: Tree-like structure with clear command chain
- **Best for**: Large projects with clear subtasks
- **Characteristics**: Efficient delegation, clear responsibilities

### 3. Ring
- **Description**: Circular arrangement with sequential processing
- **Best for**: Pipeline tasks, sequential workflows
- **Characteristics**: Low overhead, predictable flow

### 4. Star
- **Description**: Central coordinator with peripheral agents
- **Best for**: Simple coordination tasks
- **Characteristics**: Minimal complexity, central control

## Choosing a Topology

Consider:
- Task complexity
- Number of agents
- Communication needs
- Performance requirements`
            }]
        },
        'swarm://docs/agent-types': {
            contents: [{
                uri,
                mimeType: 'text/markdown',
                text: `# Agent Types Guide

## Available Agent Types

### 1. Researcher
- **Cognitive Pattern**: Divergent thinking
- **Capabilities**: Information gathering, analysis, exploration
- **Best for**: Research tasks, documentation review, learning

### 2. Coder
- **Cognitive Pattern**: Convergent thinking
- **Capabilities**: Implementation, debugging, optimization
- **Best for**: Writing code, fixing bugs, refactoring

### 3. Analyst
- **Cognitive Pattern**: Systems thinking
- **Capabilities**: Pattern recognition, data analysis, insights
- **Best for**: Architecture design, performance analysis

### 4. Optimizer
- **Cognitive Pattern**: Critical thinking
- **Capabilities**: Performance tuning, efficiency improvements
- **Best for**: Optimization tasks, bottleneck resolution

### 5. Coordinator
- **Cognitive Pattern**: Lateral thinking
- **Capabilities**: Task management, delegation, synthesis
- **Best for**: Project management, integration tasks

### 6. Architect
- **Cognitive Pattern**: Abstract thinking
- **Capabilities**: System design, high-level planning
- **Best for**: Architecture decisions, design patterns

### 7. Tester
- **Cognitive Pattern**: Critical evaluation
- **Capabilities**: Quality assurance, edge case finding
- **Best for**: Testing, validation, quality control`
            }]
        },
        'swarm://docs/daa-guide': {
            contents: [{
                uri,
                mimeType: 'text/markdown',
                text: `# DAA Integration Guide

## Decentralized Autonomous Agents

DAA extends ruv-swarm with autonomous learning and adaptation capabilities.

## Key Features

1. **Autonomous Learning**: Agents learn from experience
2. **Knowledge Sharing**: Cross-agent knowledge transfer
3. **Adaptive Workflows**: Self-optimizing execution
4. **Meta-Learning**: Transfer learning across domains

## Using DAA Tools

### Initialize DAA
\`\`\`javascript
mcp__ruv-swarm__daa_init {
  enableLearning: true,
  enableCoordination: true,
  persistenceMode: "auto"
}
\`\`\`

### Create Autonomous Agent
\`\`\`javascript
mcp__ruv-swarm__daa_agent_create {
  id: "auto-001",
  capabilities: ["learning", "optimization"],
  cognitivePattern: "adaptive",
  learningRate: 0.001
}
\`\`\`

### Execute Workflow
\`\`\`javascript
mcp__ruv-swarm__daa_workflow_execute {
  workflowId: "api-development",
  agentIds: ["auto-001", "auto-002"],
  parallelExecution: true
}
\`\`\`

## Best Practices

1. Start with low learning rates
2. Enable knowledge sharing for complex tasks
3. Monitor performance metrics regularly
4. Use meta-learning for cross-domain tasks`
            }]
        },
        'swarm://examples/rest-api': {
            contents: [{
                uri,
                mimeType: 'text/markdown',
                text: `# REST API Example

## Building a Complete REST API with ruv-swarm

### Step 1: Initialize Swarm
\`\`\`javascript
[BatchTool]:
  mcp__ruv-swarm__swarm_init { topology: "hierarchical", maxAgents: 6 }
  mcp__ruv-swarm__agent_spawn { type: "architect", name: "API Designer" }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "Backend Dev" }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "DB Expert" }
  mcp__ruv-swarm__agent_spawn { type: "tester", name: "QA Engineer" }
  mcp__ruv-swarm__agent_spawn { type: "coordinator", name: "Project Lead" }
\`\`\`

### Step 2: Design Architecture
\`\`\`javascript
TodoWrite { todos: [
  { id: "design", content: "Design API architecture", priority: "high" },
  { id: "auth", content: "Implement authentication", priority: "high" },
  { id: "crud", content: "Build CRUD endpoints", priority: "medium" },
  { id: "tests", content: "Write tests", priority: "medium" }
]}
\`\`\`

### Step 3: Implementation
\`\`\`javascript
[BatchTool]:
  Bash "mkdir -p api/{src,tests,docs}"
  Write "api/package.json" { ... }
  Write "api/src/server.js" { ... }
  Write "api/src/routes/auth.js" { ... }
\`\`\`

### Step 4: Testing
\`\`\`javascript
mcp__ruv-swarm__task_orchestrate {
  task: "Run comprehensive tests",
  strategy: "parallel"
}
\`\`\`

## Complete Working Example

See the full implementation in the ruv-swarm examples directory.`
            }]
        },
        'swarm://examples/neural-training': {
            contents: [{
                uri,
                mimeType: 'text/markdown',
                text: `# Neural Training Example

## Training Neural Agents for Specific Tasks

### Step 1: Initialize Neural Network
\`\`\`javascript
mcp__ruv-swarm__neural_status { agentId: "coder-001" }
\`\`\`

### Step 2: Prepare Training Data
\`\`\`javascript
mcp__ruv-swarm__neural_train {
  agentId: "coder-001",
  iterations: 50
}
\`\`\`

### Step 3: Monitor Training Progress
\`\`\`javascript
mcp__ruv-swarm__swarm_monitor {
  duration: 30,
  interval: 1
}
\`\`\`

### Step 4: Analyze Patterns
\`\`\`javascript
mcp__ruv-swarm__neural_patterns {
  pattern: "all"
}
\`\`\`

## Training Tips

1. Start with small iteration counts
2. Monitor performance metrics
3. Adjust learning rates based on results
4. Use cognitive patterns that match your task

## Advanced Training

For complex tasks, combine multiple cognitive patterns:
- Convergent for focused problem-solving
- Divergent for creative solutions
- Systems for architectural decisions`
            }]
        },
        'swarm://schemas/swarm-config': {
            contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "title": "Swarm Configuration",
                    "type": "object",
                    "properties": {
                        "topology": {
                            "type": "string",
                            "enum": ["mesh", "hierarchical", "ring", "star"],
                            "description": "Swarm topology type"
                        },
                        "maxAgents": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 5,
                            "description": "Maximum number of agents"
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["balanced", "specialized", "adaptive"],
                            "default": "balanced",
                            "description": "Distribution strategy"
                        },
                        "enableNeuralNetworks": {
                            "type": "boolean",
                            "default": true,
                            "description": "Enable neural network features"
                        },
                        "memoryPersistence": {
                            "type": "boolean",
                            "default": true,
                            "description": "Enable persistent memory"
                        }
                    },
                    "required": ["topology"]
                }, null, 2)
            }]
        },
        'swarm://schemas/agent-config': {
            contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "title": "Agent Configuration",
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["researcher", "coder", "analyst", "optimizer", "coordinator", "architect", "tester"],
                            "description": "Agent type"
                        },
                        "name": {
                            "type": "string",
                            "maxLength": 100,
                            "description": "Custom agent name"
                        },
                        "capabilities": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Agent capabilities"
                        },
                        "cognitivePattern": {
                            "type": "string",
                            "enum": ["convergent", "divergent", "lateral", "systems", "critical", "abstract"],
                            "description": "Cognitive thinking pattern"
                        },
                        "learningRate": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.001,
                            "description": "Learning rate for neural network"
                        }
                    },
                    "required": ["type"]
                }, null, 2)
            }]
        },
        'swarm://performance/benchmarks': {
            contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({
                    "benchmarks": {
                        "wasm_load_time": {
                            "target": "200ms",
                            "achieved": "98ms",
                            "improvement": "51%"
                        },
                        "agent_spawn_time": {
                            "target": "50ms",
                            "achieved": "12ms",
                            "improvement": "76%"
                        },
                        "memory_usage_10_agents": {
                            "target": "50MB",
                            "achieved": "18.5MB",
                            "improvement": "63%"
                        },
                        "cross_boundary_latency": {
                            "target": "0.5ms",
                            "achieved": "0.15ms",
                            "improvement": "70%"
                        },
                        "token_processing": {
                            "target": "10K/sec",
                            "achieved": "42.5K/sec",
                            "improvement": "325%"
                        }
                    },
                    "swe_bench_solve_rate": "84.8%",
                    "token_reduction": "32.3%",
                    "speed_improvement": "2.8-4.4x"
                }, null, 2)
            }]
        },
        'swarm://hooks/available': {
            contents: [{
                uri,
                mimeType: 'text/markdown',
                text: `# Available Claude Code Hooks

## Pre-Operation Hooks

### pre-task
- **Purpose**: Initialize agent context before tasks
- **Usage**: \`npx ruv-swarm hook pre-task --description "task"\`
- **Features**: Auto-spawn agents, load context, optimize topology

### pre-edit
- **Purpose**: Prepare for file edits
- **Usage**: \`npx ruv-swarm hook pre-edit --file "path"\`
- **Features**: Auto-assign agents, validate permissions

### pre-search
- **Purpose**: Optimize search operations
- **Usage**: \`npx ruv-swarm hook pre-search --query "search"\`
- **Features**: Cache results, suggest alternatives

## Post-Operation Hooks

### post-edit
- **Purpose**: Process file after editing
- **Usage**: \`npx ruv-swarm hook post-edit --file "path"\`
- **Features**: Auto-format, update memory, train neural patterns

### post-task
- **Purpose**: Finalize task execution
- **Usage**: \`npx ruv-swarm hook post-task --task-id "id"\`
- **Features**: Analyze performance, update metrics

### notification
- **Purpose**: Share updates across swarm
- **Usage**: \`npx ruv-swarm hook notification --message "update"\`
- **Features**: Broadcast to agents, update memory

## Session Hooks

### session-start
- **Purpose**: Initialize session
- **Usage**: \`npx ruv-swarm hook session-start\`
- **Features**: Restore context, load memory

### session-end
- **Purpose**: Clean up session
- **Usage**: \`npx ruv-swarm hook session-end\`
- **Features**: Save state, generate summary

### session-restore
- **Purpose**: Restore previous session
- **Usage**: \`npx ruv-swarm hook session-restore --session-id "id"\`
- **Features**: Load memory, restore agent states`
            }]
        }
    };

    const resource = resources[uri];
    if (!resource) {
        throw new Error(`Resource not found: ${uri}`);
    }
    
    return resource;
}

async function handleMcpRequest(request, mcpTools, logger = null) {
    const response = {
        jsonrpc: '2.0',
        id: request.id
    };
    
    // Use default logger if not provided
    if (!logger) {
        logger = initializeLogger();
    }
    
    // Handle case where mcpTools is null (initialization failed)
    if (!mcpTools && request.method !== 'initialize' && request.method !== 'resources/list' && request.method !== 'resources/read') {
        logger.warn('MCP tools not initialized, returning limited functionality');
        response.error = {
            code: -32603,
            message: 'MCP tools not initialized - server started with limited functionality',
            data: { method: request.method }
        };
        return response;
    }
    
    try {
        logger.debug('Processing MCP request', { 
            method: request.method, 
            hasParams: !!request.params,
            requestId: request.id 
        });
        
        switch (request.method) {
            case 'initialize':
                response.result = {
                    protocolVersion: '2024-11-05',
                    capabilities: {
                        tools: {},
                        resources: {
                            list: true,
                            read: true
                        }
                    },
                    serverInfo: {
                        name: 'ruv-swarm',
                        version: '1.0.8'
                    }
                };
                break;
                
            case 'tools/list':
                response.result = {
                    tools: [
                        {
                            name: 'swarm_init',
                            description: 'Initialize a new swarm with specified topology',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    topology: { type: 'string', enum: ['mesh', 'hierarchical', 'ring', 'star'], description: 'Swarm topology type' },
                                    maxAgents: { type: 'number', minimum: 1, maximum: 100, default: 5, description: 'Maximum number of agents' },
                                    strategy: { type: 'string', enum: ['balanced', 'specialized', 'adaptive'], default: 'balanced', description: 'Distribution strategy' }
                                },
                                required: ['topology']
                            }
                        },
                        {
                            name: 'swarm_status',
                            description: 'Get current swarm status and agent information',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    verbose: { type: 'boolean', default: false, description: 'Include detailed agent information' }
                                }
                            }
                        },
                        {
                            name: 'swarm_monitor',
                            description: 'Monitor swarm activity in real-time',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    duration: { type: 'number', default: 10, description: 'Monitoring duration in seconds' },
                                    interval: { type: 'number', default: 1, description: 'Update interval in seconds' }
                                }
                            }
                        },
                        {
                            name: 'agent_spawn',
                            description: 'Spawn a new agent in the swarm',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    type: { type: 'string', enum: ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'], description: 'Agent type' },
                                    name: { type: 'string', description: 'Custom agent name' },
                                    capabilities: { type: 'array', items: { type: 'string' }, description: 'Agent capabilities' }
                                },
                                required: ['type']
                            }
                        },
                        {
                            name: 'agent_list',
                            description: 'List all active agents in the swarm',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    filter: { type: 'string', enum: ['all', 'active', 'idle', 'busy'], default: 'all', description: 'Filter agents by status' }
                                }
                            }
                        },
                        {
                            name: 'agent_metrics',
                            description: 'Get performance metrics for agents',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    agentId: { type: 'string', description: 'Specific agent ID (optional)' },
                                    metric: { type: 'string', enum: ['all', 'cpu', 'memory', 'tasks', 'performance'], default: 'all' }
                                }
                            }
                        },
                        {
                            name: 'task_orchestrate',
                            description: 'Orchestrate a task across the swarm',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    task: { type: 'string', description: 'Task description or instructions' },
                                    strategy: { type: 'string', enum: ['parallel', 'sequential', 'adaptive'], default: 'adaptive', description: 'Execution strategy' },
                                    priority: { type: 'string', enum: ['low', 'medium', 'high', 'critical'], default: 'medium', description: 'Task priority' },
                                    maxAgents: { type: 'number', minimum: 1, maximum: 10, description: 'Maximum agents to use' }
                                },
                                required: ['task']
                            }
                        },
                        {
                            name: 'task_status',
                            description: 'Check progress of running tasks',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    taskId: { type: 'string', description: 'Specific task ID (optional)' },
                                    detailed: { type: 'boolean', default: false, description: 'Include detailed progress' }
                                }
                            }
                        },
                        {
                            name: 'task_results',
                            description: 'Retrieve results from completed tasks',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    taskId: { type: 'string', description: 'Task ID to retrieve results for' },
                                    format: { type: 'string', enum: ['summary', 'detailed', 'raw'], default: 'summary', description: 'Result format' }
                                },
                                required: ['taskId']
                            }
                        },
                        {
                            name: 'benchmark_run',
                            description: 'Execute performance benchmarks',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    type: { type: 'string', enum: ['all', 'wasm', 'swarm', 'agent', 'task'], default: 'all', description: 'Benchmark type' },
                                    iterations: { type: 'number', minimum: 1, maximum: 100, default: 10, description: 'Number of iterations' }
                                }
                            }
                        },
                        {
                            name: 'features_detect',
                            description: 'Detect runtime features and capabilities',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    category: { type: 'string', enum: ['all', 'wasm', 'simd', 'memory', 'platform'], default: 'all', description: 'Feature category' }
                                }
                            }
                        },
                        {
                            name: 'memory_usage',
                            description: 'Get current memory usage statistics',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    detail: { type: 'string', enum: ['summary', 'detailed', 'by-agent'], default: 'summary', description: 'Detail level' }
                                }
                            }
                        },
                        {
                            name: 'neural_status',
                            description: 'Get neural agent status and performance metrics',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    agentId: { type: 'string', description: 'Specific agent ID (optional)' }
                                }
                            }
                        },
                        {
                            name: 'neural_train',
                            description: 'Train neural agents with sample tasks',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    agentId: { type: 'string', description: 'Specific agent ID to train (optional)' },
                                    iterations: { type: 'number', minimum: 1, maximum: 100, default: 10, description: 'Number of training iterations' }
                                }
                            }
                        },
                        {
                            name: 'neural_patterns',
                            description: 'Get cognitive pattern information',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    pattern: { type: 'string', enum: ['all', 'convergent', 'divergent', 'lateral', 'systems', 'critical', 'abstract'], default: 'all', description: 'Cognitive pattern type' }
                                }
                            }
                        },
                        // Add DAA tools
                        ...daaMcpTools.getToolDefinitions()
                    ]
                };
                break;
                
            case 'tools/call':
                const toolName = request.params.name;
                const toolArgs = request.params.arguments || {};
                
                logger.info('Tool call requested', { 
                    tool: toolName, 
                    hasArgs: Object.keys(toolArgs).length > 0,
                    requestId: request.id
                });
                
                let result = null;
                let toolFound = false;
                const toolOpId = logger.startOperation(`tool-${toolName}`, {
                    tool: toolName,
                    requestId: request.id
                });
                
                // Try regular MCP tools first (use mcpToolsEnhanced.tools)
                if (mcpToolsEnhanced.tools && typeof mcpToolsEnhanced.tools[toolName] === 'function') {
                    try {
                        logger.debug('Executing MCP tool', { tool: toolName, args: toolArgs });
                        result = await mcpToolsEnhanced.tools[toolName](toolArgs);
                        toolFound = true;
                        logger.endOperation(toolOpId, true, { resultType: typeof result });
                    } catch (error) {
                        logger.endOperation(toolOpId, false, { error });
                        logger.error('MCP tool execution failed', { 
                            tool: toolName, 
                            error,
                            args: toolArgs 
                        });
                        response.error = {
                            code: -32603,
                            message: `MCP tool error: ${error.message}`,
                            data: { tool: toolName, error: error.message }
                        };
                        break;
                    }
                }
                // Try DAA tools if not found in regular tools
                else if (typeof daaMcpTools[toolName] === 'function') {
                    try {
                        logger.debug('Executing DAA tool', { tool: toolName, args: toolArgs });
                        result = await daaMcpTools[toolName](toolArgs);
                        toolFound = true;
                        logger.endOperation(toolOpId, true, { resultType: typeof result });
                    } catch (error) {
                        logger.endOperation(toolOpId, false, { error });
                        logger.error('DAA tool execution failed', { 
                            tool: toolName, 
                            error,
                            args: toolArgs 
                        });
                        response.error = {
                            code: -32603,
                            message: `DAA tool error: ${error.message}`,
                            data: { tool: toolName, error: error.message }
                        };
                        break;
                    }
                }
                
                if (toolFound) {
                    // Format response with content array as required by Claude Code
                    response.result = {
                        content: [{
                            type: 'text',
                            text: typeof result === 'string' ? result : JSON.stringify(result, null, 2)
                        }]
                    };
                } else {
                    response.error = {
                        code: -32601,
                        message: 'Method not found',
                        data: `Unknown tool: ${toolName}`
                    };
                }
                break;
                
            case 'resources/list':
                response.result = {
                    resources: [
                        {
                            uri: 'swarm://docs/getting-started',
                            name: 'Getting Started Guide',
                            description: 'Introduction to ruv-swarm and basic usage',
                            mimeType: 'text/markdown'
                        },
                        {
                            uri: 'swarm://docs/topologies',
                            name: 'Swarm Topologies',
                            description: 'Understanding mesh, hierarchical, ring, and star topologies',
                            mimeType: 'text/markdown'
                        },
                        {
                            uri: 'swarm://docs/agent-types',
                            name: 'Agent Types Guide',
                            description: 'Detailed guide on all agent types and their capabilities',
                            mimeType: 'text/markdown'
                        },
                        {
                            uri: 'swarm://docs/daa-guide',
                            name: 'DAA Integration Guide',
                            description: 'Using Decentralized Autonomous Agents effectively',
                            mimeType: 'text/markdown'
                        },
                        {
                            uri: 'swarm://examples/rest-api',
                            name: 'REST API Example',
                            description: 'Complete example of building a REST API with ruv-swarm',
                            mimeType: 'text/markdown'
                        },
                        {
                            uri: 'swarm://examples/neural-training',
                            name: 'Neural Training Example',
                            description: 'How to train neural agents for specific tasks',
                            mimeType: 'text/markdown'
                        },
                        {
                            uri: 'swarm://schemas/swarm-config',
                            name: 'Swarm Configuration Schema',
                            description: 'JSON schema for swarm configuration',
                            mimeType: 'application/json'
                        },
                        {
                            uri: 'swarm://schemas/agent-config',
                            name: 'Agent Configuration Schema',
                            description: 'JSON schema for agent configuration',
                            mimeType: 'application/json'
                        },
                        {
                            uri: 'swarm://performance/benchmarks',
                            name: 'Performance Benchmarks',
                            description: 'Latest performance benchmark results',
                            mimeType: 'application/json'
                        },
                        {
                            uri: 'swarm://hooks/available',
                            name: 'Available Hooks',
                            description: 'List of all available Claude Code hooks',
                            mimeType: 'text/markdown'
                        }
                    ]
                };
                break;
                
            case 'resources/read':
                const resourceUri = request.params.uri;
                response.result = await getResourceContent(resourceUri);
                break;
                
            default:
                response.error = {
                    code: -32601,
                    message: 'Method not found',
                    data: `Unknown method: ${request.method}`
                };
        }
    } catch (error) {
        response.error = {
            code: -32603,
            message: 'Internal error',
            data: error.message
        };
    }
    
    return response;
}

async function handleHook(args) {
    // Hook handler for Claude Code integration
    const { main: hooksCLIMain } = await import('../src/hooks/cli.js');
    
    // Pass through to hooks CLI with 'hook' already consumed
    process.argv = ['node', 'ruv-swarm', 'hook', ...args];
    
    return hooksCLIMain();
}

async function handleNeural(args) {
    const { neuralCLI } = await import('../src/neural.js');
    const subcommand = args[0] || 'help';
    
    try {
        switch (subcommand) {
            case 'status':
                return await neuralCLI.status(args.slice(1));
            case 'train':
                return await neuralCLI.train(args.slice(1));
            case 'patterns':
                return await neuralCLI.patterns(args.slice(1));
            case 'export':
                return await neuralCLI.export(args.slice(1));
            case 'help':
            default:
                console.log(`Neural Network Commands:
  neural status                    Show neural network status
  neural train [options]           Train neural models
  neural patterns [model]          View learned patterns
  neural export [options]          Export neural weights

Examples:
  ruv-swarm neural status
  ruv-swarm neural train --model attention --iterations 100
  ruv-swarm neural patterns --model attention
  ruv-swarm neural export --model all --output ./weights.json`);
                break;
        }
    } catch (error) {
        console.error('❌ Neural command error:', error.message);
        process.exit(1);
    }
}

async function handleBenchmark(args) {
    const { benchmarkCLI } = await import('../src/benchmark.js');
    const subcommand = args[0] || 'help';
    
    try {
        switch (subcommand) {
            case 'run':
                return await benchmarkCLI.run(args.slice(1));
            case 'compare':
                return await benchmarkCLI.compare(args.slice(1));
            case 'help':
            default:
                console.log(`Benchmark Commands:
  benchmark run [options]          Run performance benchmarks
  benchmark compare [files]        Compare benchmark results

Examples:
  ruv-swarm benchmark run --iterations 10
  ruv-swarm benchmark run --test swarm-coordination
  ruv-swarm benchmark compare results-1.json results-2.json`);
                break;
        }
    } catch (error) {
        console.error('❌ Benchmark command error:', error.message);
        process.exit(1);
    }
}

async function handlePerformance(args) {
    const { performanceCLI } = await import('../src/performance.js');
    const subcommand = args[0] || 'help';
    
    try {
        switch (subcommand) {
            case 'analyze':
                return await performanceCLI.analyze(args.slice(1));
            case 'optimize':
                return await performanceCLI.optimize(args.slice(1));
            case 'suggest':
                return await performanceCLI.suggest(args.slice(1));
            case 'help':
            default:
                console.log(`Performance Commands:
  performance analyze [options]    Analyze performance bottlenecks
  performance optimize [target]    Optimize swarm configuration
  performance suggest             Get optimization suggestions

Examples:
  ruv-swarm performance analyze --task-id recent
  ruv-swarm performance optimize --target speed
  ruv-swarm performance suggest`);
                break;
        }
    } catch (error) {
        console.error('❌ Performance command error:', error.message);
        process.exit(1);
    }
}

async function handleDiagnose(args) {
    const { diagnosticsCLI } = await import('../src/cli-diagnostics.js');
    return diagnosticsCLI(args);
}

function showHelp() {
    console.log(`
🐝 ruv-swarm - Enhanced WASM-powered neural swarm orchestration

Usage: ruv-swarm <command> [options]

Commands:
  init [topology] [maxAgents]     Initialize swarm (--claude for integration)
    Options for --claude:
      --force                       Overwrite existing CLAUDE.md (creates backup)
      --merge                       Merge with existing CLAUDE.md content
      --no-backup                   Disable automatic backup creation
      --no-interactive              Skip interactive prompts (fail on conflicts)
  spawn <type> [name]             Spawn an agent (researcher, coder, analyst, etc.)
  orchestrate <task>              Orchestrate a task across agents
  status [--verbose]              Show swarm status
  monitor [duration]              Monitor swarm activity
  mcp <subcommand>                MCP server management
  hook <type> [options]           Claude Code hooks integration
  claude-invoke <prompt>          Invoke Claude with swarm integration
  neural <subcommand>             Neural network training and analysis
  benchmark <subcommand>          Performance benchmarking tools
  performance <subcommand>        Performance analysis and optimization
  diagnose <subcommand>           Run diagnostics and analyze logs
  version                         Show version information
  help                            Show this help message

Examples:
  ruv-swarm init mesh 5 --claude                    # Create CLAUDE.md (fails if exists)
  ruv-swarm init mesh 5 --claude --force            # Overwrite CLAUDE.md (creates backup)
  ruv-swarm init mesh 5 --claude --merge            # Merge with existing CLAUDE.md
  ruv-swarm init mesh 5 --claude --force --no-backup # Overwrite CLAUDE.md (no backup)
  ruv-swarm init mesh 5 --claude --no-interactive   # Non-interactive mode
  ruv-swarm spawn researcher "AI Research Specialist"
  ruv-swarm orchestrate "Build a REST API with authentication"
  ruv-swarm mcp start
  ruv-swarm hook pre-edit --file app.js --ensure-coordination
  ruv-swarm claude-invoke "Create a development swarm for my project"
  ruv-swarm neural status
  ruv-swarm benchmark run --iterations 10
  ruv-swarm performance analyze --task-id recent

Validation Rules:
  Topologies: mesh, hierarchical, ring, star
  Max Agents: 1-100 (integers only)
  Agent Types: researcher, coder, analyst, optimizer, coordinator, architect, tester
  Agent Names: 1-100 characters, alphanumeric + spaces/hyphens/underscores/periods
  Task Descriptions: 1-1000 characters, non-empty

Modular Features:
  📚 Automatic documentation generation
  🌐 Cross-platform remote execution support
  🤖 Seamless Claude Code MCP integration
  🔧 Advanced hooks for automation
  🧠 Neural pattern learning
  💾 Cross-session memory persistence

For detailed documentation, check .claude/commands/ after running init --claude
`);
}

async function main() {
    const args = process.argv.slice(2);
    
    // Handle --version flag
    if (args.includes('--version') || args.includes('-v')) {
        try {
            const fs = await import('fs');
            const path = await import('path');
            const { fileURLToPath } = await import('url');
            const __filename = fileURLToPath(import.meta.url);
            const __dirname = path.dirname(__filename);
            const packagePath = path.join(__dirname, '..', 'package.json');
            const packageJson = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
            console.log(packageJson.version);
        } catch (error) {
            console.log('1.0.8');
        }
        return;
    }
    
    const command = args[0] || 'help';

    try {
        switch (command) {
            case 'init':
                await handleInit(args.slice(1));
                break;
            case 'spawn':
                await handleSpawn(args.slice(1));
                break;
            case 'orchestrate':
                await handleOrchestrate(args.slice(1));
                break;
            case 'mcp':
                await handleMcp(args.slice(1));
                break;
            case 'status':
                await handleStatus(args.slice(1));
                break;
            case 'monitor':
                await handleMonitor(args.slice(1));
                break;
            case 'hook':
                await handleHook(args.slice(1));
                break;
            case 'claude-invoke':
            case 'claude':
                await handleClaudeInvoke(args.slice(1));
                break;
            case 'neural':
                await handleNeural(args.slice(1));
                break;
            case 'benchmark':
                await handleBenchmark(args.slice(1));
                break;
            case 'performance':
                await handlePerformance(args.slice(1));
                break;
            case 'diagnose':
                await handleDiagnose(args.slice(1));
                break;
            case 'version':
                try {
                    // Try to read version from package.json
                    const fs = await import('fs');
                    const path = await import('path');
                    const { fileURLToPath } = await import('url');
                    const __filename = fileURLToPath(import.meta.url);
                    const __dirname = path.dirname(__filename);
                    const packagePath = path.join(__dirname, '..', 'package.json');
                    const packageJson = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
                    console.log('ruv-swarm v' + packageJson.version);
                } catch (error) {
                    console.log('ruv-swarm v1.0.8');
                }
                console.log('Enhanced WASM-powered neural swarm orchestration');
                console.log('Modular Claude Code integration with remote execution support');
                console.log('DAA (Decentralized Autonomous Agents) Integration');
                break;
            case 'help':
            default:
                showHelp();
                break;
        }
    } catch (error) {
        console.error('❌ Error:', error.message);
        if (process.argv.includes('--debug')) {
            console.error(error.stack);
        }
        process.exit(1);
    }
}

// Error handling
process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught Exception:', error.message);
    if (process.argv.includes('--debug')) {
        console.error(error.stack);
    }
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Unhandled Rejection:', reason);
    if (process.argv.includes('--debug')) {
        console.error('Promise:', promise);
    }
    process.exit(1);
});

// In ES modules, this file is always the main module when run directly
main();

export { main, initializeSystem };