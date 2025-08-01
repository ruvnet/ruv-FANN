#!/bin/bash
# Build script for ruv-swarm-daa WASM module

set -e

echo "🚀 Building ruv-swarm-daa WASM module..."

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Install wasm-pack if not present
if ! command -v wasm-pack &> /dev/null; then
    echo "📦 Installing wasm-pack..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf pkg target/wasm32-unknown-unknown

# Build with wasm-pack
echo "🔨 Building WASM module..."
wasm-pack build \
    --target web \
    --out-dir pkg \
    --release \
    -- --features wasm

# Optimize WASM binary
if command -v wasm-opt &> /dev/null; then
    echo "⚡ Optimizing WASM binary with wasm-opt..."
    wasm-opt -Oz \
        -o pkg/ruv_swarm_daa_bg_optimized.wasm \
        pkg/ruv_swarm_daa_bg.wasm
    mv pkg/ruv_swarm_daa_bg_optimized.wasm pkg/ruv_swarm_daa_bg.wasm
else
    echo "⚠️  wasm-opt not found, skipping optimization"
fi

# Create JavaScript bindings with progressive loading
echo "📝 Creating enhanced JavaScript bindings..."
cat > pkg/ruv_swarm_daa_enhanced.js << 'EOF'
import init, * as wasm from './ruv_swarm_daa.js';

let wasmModule = null;
let initPromise = null;

// Progressive loading with SIMD detection
export async function initializeDAA(options = {}) {
    if (initPromise) return initPromise;
    
    initPromise = (async () => {
        try {
            // Check for WebAssembly support
            if (typeof WebAssembly === 'undefined') {
                throw new Error('WebAssembly is not supported in this environment');
            }
            
            // Check for SIMD support
            const simdSupported = WebAssembly.validate(new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x04, 0x01, 0x60, 0x00, 0x00, 0x03, 0x02,
                0x01, 0x00, 0x0a, 0x09, 0x01, 0x07, 0x00, 0x41,
                0x00, 0xfd, 0x0f, 0x0b
            ]));
            
            console.log('SIMD support:', simdSupported ? 'enabled' : 'disabled');
            
            // Load WASM module
            const startTime = performance.now();
            wasmModule = await init();
            const loadTime = performance.now() - startTime;
            
            console.log(`DAA WASM module loaded in ${loadTime.toFixed(2)}ms`);
            
            // Initialize utils
            if (wasm.WasmUtils) {
                wasm.WasmUtils.init();
            }
            
            return {
                module: wasmModule,
                loadTime,
                simdSupported,
                ...wasm
            };
        } catch (error) {
            console.error('Failed to initialize DAA WASM:', error);
            throw error;
        }
    })();
    
    return initPromise;
}

// Export all WASM exports
export * from './ruv_swarm_daa.js';

// Helper functions for easier usage
export function createAgent(id) {
    if (!wasmModule) {
        throw new Error('DAA WASM not initialized. Call initializeDAA() first.');
    }
    return new wasm.WasmAutonomousAgent(id);
}

export function createCoordinator() {
    if (!wasmModule) {
        throw new Error('DAA WASM not initialized. Call initializeDAA() first.');
    }
    return new wasm.WasmCoordinator();
}

export function createResourceManager(maxMemoryMB) {
    if (!wasmModule) {
        throw new Error('DAA WASM not initialized. Call initializeDAA() first.');
    }
    return new wasm.WasmResourceManager(maxMemoryMB);
}

// Performance monitoring
export function getPerformanceInfo() {
    if (!wasmModule || !wasm.WasmUtils) {
        return { error: 'WASM not initialized' };
    }
    return JSON.parse(wasm.WasmUtils.get_performance_info());
}

// System capabilities check
export function getSystemCapabilities() {
    if (!wasmModule || !wasm.WasmUtils) {
        return [];
    }
    return wasm.WasmUtils.get_system_capabilities();
}
EOF

# Create TypeScript definitions
echo "📝 Creating TypeScript definitions..."
cat > pkg/ruv_swarm_daa.d.ts << 'EOF'
/* tslint:disable */
/* eslint-disable */

export interface InitOptions {
    simd?: boolean;
}

export interface DAAModule {
    module: WebAssembly.Module;
    loadTime: number;
    simdSupported: boolean;
}

export function initializeDAA(options?: InitOptions): Promise<DAAModule>;

export class WasmAutonomousAgent {
    constructor(id: string);
    readonly id: string;
    autonomy_level: number;
    learning_rate: number;
    
    add_capability(capability_type: string): boolean;
    remove_capability(capability_type: string): boolean;
    has_capability(capability_type: string): boolean;
    get_capabilities(): string[];
    make_decision(context_json: string): Promise<string>;
    adapt(feedback_json: string): Promise<string>;
    is_autonomous(): boolean;
    get_status(): string;
    optimize_resources(): Promise<string>;
}

export class WasmCoordinator {
    constructor();
    add_agent(agent: WasmAutonomousAgent): void;
    remove_agent(agent_id: string): boolean;
    agent_count(): number;
    set_strategy(strategy: string): void;
    get_strategy(): string;
    set_frequency(frequency_ms: number): void;
    get_frequency(): number;
    coordinate(): Promise<string>;
    get_status(): string;
}

export class WasmResourceManager {
    constructor(max_memory_mb: number);
    allocate_memory(size_mb: number): boolean;
    deallocate_memory(size_mb: number): boolean;
    get_memory_usage(): number;
    get_allocated_memory(): number;
    get_max_memory(): number;
    set_cpu_usage(usage: number): void;
    get_cpu_usage(): number;
    enable_optimization(): void;
    disable_optimization(): void;
    is_optimization_enabled(): boolean;
    optimize(): Promise<string>;
    get_status(): string;
}

export class WasmUtils {
    static init(): void;
    static get_system_capabilities(): string[];
    static check_wasm_support(): boolean;
    static get_performance_info(): string;
    static log(message: string): void;
    static create_context(js_object: any): string;
    static create_feedback(performance: number, efficiency: number): string;
}

export function createAgent(id: string): WasmAutonomousAgent;
export function createCoordinator(): WasmCoordinator;
export function createResourceManager(maxMemoryMB: number): WasmResourceManager;
export function getPerformanceInfo(): any;
export function getSystemCapabilities(): string[];
EOF

# Generate package.json for the WASM package
echo "📦 Creating package.json..."
cat > pkg/package.json << EOF
{
  "name": "@ruv-swarm/daa-wasm",
  "version": "1.0.4",
  "description": "WebAssembly bindings for ruv-swarm Decentralized Autonomous Agents",
  "main": "ruv_swarm_daa_enhanced.js",
  "types": "ruv_swarm_daa.d.ts",
  "module": "ruv_swarm_daa_enhanced.js",
  "sideEffects": false,
  "files": [
    "ruv_swarm_daa_bg.wasm",
    "ruv_swarm_daa.js",
    "ruv_swarm_daa_enhanced.js",
    "ruv_swarm_daa.d.ts"
  ],
  "keywords": [
    "wasm",
    "webassembly",
    "daa",
    "autonomous-agents",
    "swarm",
    "ai"
  ],
  "author": "ruvnet <ruv@ruvnet.com>",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/ruvnet/ruv-FANN.git"
  },
  "bugs": {
    "url": "https://github.com/ruvnet/ruv-FANN/issues"
  },
  "homepage": "https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm"
}
EOF

# Check if build was successful
if [ ! -f pkg/ruv_swarm_daa_bg.wasm ]; then
    echo "❌ Build failed. Using simplified WASM module..."
    exit 1
fi

# Create example HTML file
echo "📄 Creating example HTML..."
cat > pkg/example.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>ruv-swarm DAA WASM Example</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .status {
            background: #e3f2fd;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
        }
        button {
            background: #2196f3;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover {
            background: #1976d2;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin: 10px 0;
        }
        .metric {
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
        }
        .log {
            background: #263238;
            color: #aed581;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <h1>🤖 ruv-swarm DAA WASM Demo</h1>
    
    <div class="container">
        <h2>System Status</h2>
        <div id="systemStatus" class="status">Initializing...</div>
        <div id="loadTime" class="metric"></div>
    </div>

    <div class="container">
        <h2>Autonomous Agents</h2>
        <button onclick="createNewAgent()">Create Agent</button>
        <button onclick="coordinateAgents()">Coordinate All</button>
        <div id="agentList" class="metrics"></div>
    </div>

    <div class="container">
        <h2>Resource Management</h2>
        <button onclick="allocateMemory()">Allocate 10MB</button>
        <button onclick="deallocateMemory()">Deallocate 10MB</button>
        <button onclick="optimizeResources()">Optimize</button>
        <div id="resourceStatus" class="metrics"></div>
    </div>

    <div class="container">
        <h2>Console Log</h2>
        <div id="console" class="log"></div>
    </div>

    <script type="module">
        import { initializeDAA, createAgent, createCoordinator, createResourceManager, getPerformanceInfo, getSystemCapabilities } from './ruv_swarm_daa_enhanced.js';
        
        let coordinator;
        let resourceManager;
        let agents = [];
        let agentCounter = 0;
        
        function log(message) {
            const console = document.getElementById('console');
            const timestamp = new Date().toLocaleTimeString();
            console.innerHTML += `[${timestamp}] ${message}<br>`;
            console.scrollTop = console.scrollHeight;
        }
        
        async function init() {
            try {
                log('Initializing DAA WASM module...');
                const result = await initializeDAA();
                
                document.getElementById('systemStatus').innerHTML = 
                    `✅ DAA System Ready | SIMD: ${result.simdSupported ? '✓' : '✗'}`;
                document.getElementById('loadTime').innerHTML = 
                    `Load time: ${result.loadTime.toFixed(2)}ms`;
                
                // Initialize coordinator and resource manager
                coordinator = createCoordinator();
                resourceManager = createResourceManager(1024); // 1GB max
                
                log('System initialized successfully');
                
                // Display capabilities
                const capabilities = getSystemCapabilities();
                log(`System capabilities: ${capabilities.join(', ')}`);
                
                // Display performance info
                const perfInfo = getPerformanceInfo();
                log(`Performance info: ${JSON.stringify(perfInfo)}`);
                
                updateResourceStatus();
            } catch (error) {
                log(`Error: ${error.message}`);
                document.getElementById('systemStatus').innerHTML = '❌ Failed to initialize';
            }
        }
        
        window.createNewAgent = function() {
            try {
                const id = `agent-${++agentCounter}`;
                const agent = createAgent(id);
                
                // Set random properties
                agent.autonomy_level = Math.random();
                agent.learning_rate = Math.random() * 0.1;
                
                // Add random capabilities
                const capabilities = ['learning', 'prediction', 'coordination', 'self_healing'];
                capabilities.forEach(cap => {
                    if (Math.random() > 0.5) {
                        agent.add_capability(cap);
                    }
                });
                
                coordinator.add_agent(agent);
                agents.push(agent);
                
                log(`Created ${id} with autonomy level ${agent.autonomy_level.toFixed(2)}`);
                updateAgentList();
            } catch (error) {
                log(`Error creating agent: ${error.message}`);
            }
        };
        
        window.coordinateAgents = async function() {
            try {
                log('Starting agent coordination...');
                const result = await coordinator.coordinate();
                log(`Coordination result: ${result}`);
                
                // Update agent metrics
                for (const agent of agents) {
                    const decision = await agent.make_decision('{"available_actions": ["explore", "learn", "adapt"]}');
                    log(`${agent.id} decision: ${decision}`);
                }
            } catch (error) {
                log(`Error coordinating agents: ${error.message}`);
            }
        };
        
        window.allocateMemory = function() {
            try {
                const success = resourceManager.allocate_memory(10);
                log(success ? 'Allocated 10MB' : 'Failed to allocate memory');
                updateResourceStatus();
            } catch (error) {
                log(`Error allocating memory: ${error.message}`);
            }
        };
        
        window.deallocateMemory = function() {
            try {
                const success = resourceManager.deallocate_memory(10);
                log(success ? 'Deallocated 10MB' : 'Failed to deallocate memory');
                updateResourceStatus();
            } catch (error) {
                log(`Error deallocating memory: ${error.message}`);
            }
        };
        
        window.optimizeResources = async function() {
            try {
                log('Optimizing resources...');
                const result = await resourceManager.optimize();
                log(`Optimization result: ${result}`);
                updateResourceStatus();
            } catch (error) {
                log(`Error optimizing resources: ${error.message}`);
            }
        };
        
        function updateAgentList() {
            const container = document.getElementById('agentList');
            container.innerHTML = agents.map(agent => {
                const status = JSON.parse(agent.get_status());
                return `
                    <div class="metric">
                        <strong>${status.id}</strong><br>
                        Autonomy: ${(status.autonomy_level * 100).toFixed(0)}%<br>
                        Learning: ${status.learning_rate.toFixed(3)}<br>
                        Capabilities: ${status.capabilities_count}
                    </div>
                `;
            }).join('');
        }
        
        function updateResourceStatus() {
            const status = JSON.parse(resourceManager.get_status());
            const container = document.getElementById('resourceStatus');
            container.innerHTML = `
                <div class="metric">
                    <strong>Memory</strong><br>
                    ${status.allocated_memory}MB / ${status.max_memory}MB<br>
                    Usage: ${(status.memory_usage * 100).toFixed(1)}%
                </div>
                <div class="metric">
                    <strong>CPU</strong><br>
                    Usage: ${(status.cpu_usage * 100).toFixed(1)}%<br>
                    Optimization: ${status.optimization_enabled ? '✓' : '✗'}
                </div>
            `;
        }
        
        // Initialize on load
        init();
    </script>
</body>
</html>
EOF

# Calculate sizes
echo "📊 Build complete! Analyzing sizes..."
WASM_SIZE=$(wc -c < pkg/ruv_swarm_daa_bg.wasm)
JS_SIZE=$(wc -c < pkg/ruv_swarm_daa.js)

echo "✅ WASM module built successfully!"
echo "   WASM size: $(numfmt --to=iec-i --suffix=B $WASM_SIZE)"
echo "   JS size: $(numfmt --to=iec-i --suffix=B $JS_SIZE)"
echo "   Output directory: ./pkg"
echo ""
echo "🚀 To test the module, open pkg/example.html in a web browser"
echo "📦 To publish: cd pkg && npm publish"