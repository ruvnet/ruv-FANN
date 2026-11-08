#!/usr/bin/env node
import { createStdioRpcServer } from '../src/stdio-rpc.js';
/** Node CLI and stdio MCP entry point for persisted metadata and checked WASM operations. */

import { setupClaudeIntegration, invokeClaudeWithSwarm } from '../src/claude-integration/index.js';
import { RuvSwarm } from '../src/index.js';
import { parseClaudeInvocationArgs } from '../src/claude-integration/core.js';
import { EnhancedMCPTools } from '../src/mcp-tools-enhanced.js';
import { Logger } from '../src/logger.js';
import { readFileSync } from 'node:fs';

async function getVersion() {
  return JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')).version;
}

const cleanupRegistry = new Map();
let cleanupNeeded = false;

function registerCleanup(key, handler) {
  if (!handler || cleanupRegistry.has(key)) {
    return;
  }
  cleanupRegistry.set(key, handler);
  cleanupNeeded = true;
}

async function cleanupResources() {
  for (const [key, handler] of cleanupRegistry.entries()) {
    try {
      await handler();
    } catch (error) {
      console.warn(`⚠️ Cleanup handler '${key}' failed:`, error?.message || error);
    }
  }
  cleanupRegistry.clear();
  cleanupNeeded = false;
}

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
        globalRuvSwarm = await RuvSwarm.initialize({
            useSIMD: RuvSwarm.detectSIMDSupport(), debug: process.argv.includes('--debug')
        });
    }
    if (!globalMCPTools) {
        globalMCPTools = new EnhancedMCPTools(globalRuvSwarm);
        registerCleanup('mcp-tools', () => globalMCPTools.destroy());
        await globalMCPTools.initialize();
    }
    return { ruvSwarm: globalRuvSwarm, mcpTools: globalMCPTools };
}

async function handleInit(args) {
    const { mcpTools } = await initializeSystem();
    const positional = args.filter(arg => !arg.startsWith('--'));
    const result = await mcpTools.swarm_init({ topology: validateTopology(positional[0] || 'mesh'), maxAgents: validateMaxAgents(positional[1] || '5') });
    console.log(JSON.stringify(result, null, 2));
    if (args.some(arg => ['--claude', '--setup-claude', '--force', '--merge'].includes(arg))) {
        await setupClaudeIntegration({ autoSetup: args.includes('--claude') || args.includes('--setup-claude'), forceSetup: args.includes('--force'), mergeSetup: args.includes('--merge'),
            noBackup: args.includes('--no-backup'), interactive: !args.includes('--no-interactive'), workingDir: process.cwd(), packageName: 'ruv-swarm' });
    }
}

async function handleSpawn(args) {
    const { mcpTools } = await initializeSystem();
    const result = await mcpTools.agent_spawn({ type: validateAgentType(args[0] || 'researcher'), name: validateAgentName(args[1] || null) });
    console.log(JSON.stringify(result, null, 2));
}

async function handleOrchestrate(args) {
    const { mcpTools } = await initializeSystem();
    const task = validateTaskDescription(args.join(' '));
    console.log(JSON.stringify(await mcpTools.task_orchestrate({ task }), null, 2));
}

async function handleClaudeInvoke(args) {
    const { skipPermissions, prompt } = parseClaudeInvocationArgs(args);
    if (!prompt || Buffer.byteLength(prompt, 'utf8') > 128 * 1024) {
        console.error('Provide a non-empty prompt of at most 128 KiB.');
        return;
    }
    try {
        await invokeClaudeWithSwarm(prompt, {
            workingDir: process.cwd(), secure: !skipPermissions
        });
    } catch (error) {
        console.error('Claude invocation failed:', error.message);
        process.exitCode = 1;
    }
}

async function handleStatus(args) {
    const { mcpTools } = await initializeSystem();
    console.log(JSON.stringify(await mcpTools.swarm_status({ swarmId: args.find(arg => !arg.startsWith('-')) }), null, 2));
}

async function handleMonitor(args) {
    const { mcpTools } = await initializeSystem();
    const duration = Number(args.find(arg => /^\d+$/.test(arg)) || 1000);
    console.log(JSON.stringify(await mcpTools.swarm_monitor({ duration }), null, 2));
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
    if (protocol === 'stdio') {
        // stdout is exclusively the JSON-RPC transport; library diagnostics use stderr.
        console.log = console.error.bind(console);
        console.info = console.error.bind(console);
    }
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

            // Initialize WASM if needed
            const initOpId = logger.startOperation('initialize-system');
            const { ruvSwarm, mcpTools } = await initializeSystem();
            logger.endOperation(initOpId, true, { modulesLoaded: true });

            // Start stdio MCP server loop
            process.stdin.setEncoding('utf8');

            // Signal server readiness for testing
            if (process.env.MCP_TEST_MODE === 'true') {
                console.error('MCP server ready'); // Use stderr so it doesn't interfere with JSON-RPC
            }

            const transport = createStdioRpcServer({ input: process.stdin, output: process.stdout,
                handle: request => handleMcpRequest(request, mcpTools, logger) });

            // Send initialization message
            const initMessage = {
                jsonrpc: '2.0',
                method: 'server.initialized',
                params: {
                    serverInfo: {
                        name: 'ruv-swarm',
                        version: await getVersion(),
                        capabilities: {
                            tools: true,
                            prompts: false,
                            resources: true
                        }
                    }
                }
            };
            await transport.send(initMessage);
            await transport.done;
            logger.info('MCP input closed; all pending replies drained');

        } else {
            throw new Error('WebSocket transport is not supported by the Node distribution; use stdio');
        }
    } catch (error) {
        logger.fatal('Failed to start MCP server', { error, protocol });
        console.error('❌ Failed to start MCP server:', error.message);
        process.exit(1);
    }
}

async function getMcpStatus() {
    console.log('Node MCP transport: stdio');
    console.log('Start with: ruv-swarm mcp start');
    console.log('This command does not inspect servers owned by another process.');
}

async function stopMcpServer() {
    if (!childProcess) throw new Error('No supervised MCP child belongs to this process; stop the owning client or process');
    childProcess.kill('SIGTERM');
    childProcess = null;
    console.log('Sent SIGTERM to the supervised MCP child');
}

async function listMcpTools() {
    const { getSupportedToolDefinitions } = await import('../src/mcp-tools-enhanced.js');
    console.log(JSON.stringify({ tools: getSupportedToolDefinitions() }, null, 2));
}

async function configureMcp(args) {
    console.log('🔧 MCP configuration is managed through Claude Code');
    console.log('Run: ruv-swarm init --claude');
}

async function getResourceContent(uri) {
    if (!['swarm://docs/getting-started', 'swarm://docs/stability'].includes(uri)) throw new Error('Resource not found');
    const { readFile } = await import('node:fs/promises');
    return { contents: [{ uri, mimeType: 'text/markdown', text: await readFile(new URL('../README.md', import.meta.url), 'utf8') }] };
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

            case 'ping': { response.result = {}; break; }
            case 'tools/list': {
                response.result = { tools: mcpTools.getAllToolDefinitions() };
                break;
            }
            case 'tools/call': {
                const name = request.params?.name;
                const args = request.params?.arguments || {};
                if (typeof name !== 'string' || !Object.hasOwn(mcpTools.tools, name)) {
                    response.error = { code: -32601, message: `Unknown or unsupported tool: ${String(name)}` };
                    break;
                }
                try {
                    const result = await mcpTools.tools[name](args);
                    response.result = { content: [{ type: 'text', text: JSON.stringify(result) }] };
                } catch (error) {
                    response.error = { code: error instanceof TypeError || error instanceof RangeError ? -32602 : -32603,
                        message: error.message || String(error), data: { code: error.code || 'TOOL_ERROR', tool: name } };
                }
                break;
            }
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
    const command = args[0] || 'help';
    if (['status', 'infer', 'forecast', 'train', 'patterns', 'export'].includes(command)) {
        return neuralCLI[command](args.slice(1));
    }
    console.log(`Neural commands:
  neural status
  neural infer --layers 2,1 --weights 2,-1 --input 3,4 --activation Linear
  neural forecast --model linear --input 1,3,5
Training and learned-pattern export require an external implementation.`);
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
  ruv-swarm benchmark run --test dense-inference
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
    console.log(`ruv-swarm: local agents, SQLite task state, and WASM inference

  init [mesh|ring|star|hierarchical] [maxAgents] [--claude]
  spawn [type] [name]              Register a local agent
  orchestrate "task description"   Assign idle matching agents
  status                          Read persisted metadata
  monitor [milliseconds]          Sample local state for up to 60 seconds
  mcp start                       Start the stdio JSON-RPC server
  mcp tools                       List supported tools and input schemas
  neural status                   Report actual inference capabilities
  neural infer --layers 2,1 --weights 2,-1 --input 3,4
  neural forecast --model linear --input 1,3,5
  benchmark run --iterations 1000  Measure local dense inference
  performance analyze             Measure inference and process memory
  claude-invoke "prompt"           Invoke installed Claude with normal permissions
  --version                       Print the package version

Local assignments do not execute an external model. Training, learned-pattern
analysis, autonomous DAA execution and automatic optimization are unavailable.
Use --dangerously-skip-permissions only as an explicit Claude invocation opt-in.
`);
}

// Set by the print-and-return paths (--version, version, help). Imports at the
// top of this file open handles (e.g. the pooled SQLite persistence layer), so
// returning from main() is not enough for Node to exit.
let oneShot = false;

async function main() {
    const args = process.argv.slice(2);
    let exitCode = 0;

    // Handle --version flag
    if (args.includes('--version') || args.includes('-v')) {
        oneShot = true;
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
                console.log('ruv-swarm v' + await getVersion());
                console.log('Local metadata, SQLite persistence, dense WASM inference and statistical forecasts');
                break;
            case 'help':
            default:
                oneShot = true;
                showHelp();
                break;
        }
    } catch (error) {
        exitCode = 1;
        console.error('❌ Error:', error.message);
        if (process.argv.includes('--debug')) {
            console.error(error.stack);
        }
    } finally {
        if (cleanupNeeded) {
            await cleanupResources();
        }
    }

    if (exitCode !== 0) {
        process.exit(exitCode);
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
main().then(() => {
    if (oneShot) {
        // Flush first: stdout to a pipe is asynchronous on macOS.
        process.stdout.write('', () => process.exit(0));
    }
});

export { main, initializeSystem };
