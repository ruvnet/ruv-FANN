#!/usr/bin/env node
/**
 * Clean, modular ruv-swarm CLI with Claude Code integration
 * Uses modular architecture for better maintainability and remote execution
 */

import { setupClaudeIntegration, invokeClaudeWithSwarm } from '../src/claude-integration/index.js';
import { RuvSwarm } from '../src/index-enhanced.js';
import { EnhancedMCPTools } from '../src/mcp-tools-enhanced.js';

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

async function initializeSystem() {
    if (!globalRuvSwarm) {
        // RuvSwarm.initialize already prints initialization messages
        globalRuvSwarm = await RuvSwarm.initialize({
            loadingStrategy: 'progressive',
            enablePersistence: true,
            enableNeuralNetworks: true,
            enableForecasting: true,
            useSIMD: RuvSwarm.detectSIMDSupport(),
            debug: process.argv.includes('--debug')
        });
    }
    
    if (!globalMCPTools) {
        // Pass the already initialized RuvSwarm instance to avoid duplicate initialization
        globalMCPTools = new EnhancedMCPTools(globalRuvSwarm);
        await globalMCPTools.initialize(globalRuvSwarm);
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
        if (setupClaude || forceSetup) {
            console.log('\n📚 Setting up modular Claude Code integration...');
            try {
                await setupClaudeIntegration({
                    autoSetup: setupClaude,
                    forceSetup: forceSetup,
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
    
    try {
        if (protocol === 'stdio') {
            // In stdio mode, only JSON-RPC messages should go to stdout
            console.error('🚀 ruv-swarm MCP server starting in stdio mode...');
            
            // Initialize WASM if needed
            const { ruvSwarm, mcpTools } = await initializeSystem();
            
            // Start stdio MCP server loop
            process.stdin.setEncoding('utf8');
            
            let buffer = '';
            process.stdin.on('data', (chunk) => {
                buffer += chunk;
                
                // Process complete JSON messages
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const request = JSON.parse(line);
                            handleMcpRequest(request, mcpTools).then(response => {
                                process.stdout.write(JSON.stringify(response) + '\n');
                            });
                        } catch (error) {
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
            
            // Send initialization message
            const initMessage = {
                jsonrpc: '2.0',
                method: 'server.initialized',
                params: {
                    serverInfo: {
                        name: 'ruv-swarm',
                        version: '0.2.0',
                        capabilities: {
                            tools: true,
                            prompts: false,
                            resources: false
                        }
                    }
                }
            };
            process.stdout.write(JSON.stringify(initMessage) + '\n');
            
        } else {
            console.log('❌ WebSocket protocol not yet implemented in clean version');
            console.log('Use stdio mode for Claude Code integration');
        }
    } catch (error) {
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
    console.log('   mcp__ruv-swarm__swarm_init - Initialize a new swarm');
    console.log('   mcp__ruv-swarm__agent_spawn - Spawn new agents');
    console.log('   mcp__ruv-swarm__task_orchestrate - Orchestrate tasks');
    console.log('   mcp__ruv-swarm__swarm_status - Get swarm status');
    console.log('   ... and 12 more tools');
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

async function handleMcpRequest(request, mcpTools) {
    const response = {
        jsonrpc: '2.0',
        id: request.id
    };
    
    try {
        switch (request.method) {
            case 'initialize':
                response.result = {
                    protocolVersion: '2024-11-05',
                    capabilities: {
                        tools: {},
                        resources: {}
                    },
                    serverInfo: {
                        name: 'ruv-swarm',
                        version: '0.2.0'
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
                        }
                    ]
                };
                break;
                
            case 'tools/call':
                const toolName = request.params.name;
                const toolArgs = request.params.arguments || {};
                
                // Call the appropriate mcpTools method
                if (mcpTools[toolName]) {
                    const result = await mcpTools[toolName](toolArgs);
                    // Format response with content array as required by Claude Code
                    // Each content item must have a type (text, image, audio, resource)
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

function showHelp() {
    console.log(`
🐝 ruv-swarm - Enhanced WASM-powered neural swarm orchestration

Usage: ruv-swarm <command> [options]

Commands:
  init [topology] [maxAgents]     Initialize swarm (--claude for integration)
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
  version                         Show version information
  help                            Show this help message

Examples:
  ruv-swarm init mesh 5 --claude --force
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
            case 'version':
                console.log('ruv-swarm v' + (RuvSwarm.getVersion ? RuvSwarm.getVersion() : '0.2.0'));
                console.log('Enhanced WASM-powered neural swarm orchestration');
                console.log('Modular Claude Code integration with remote execution support');
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