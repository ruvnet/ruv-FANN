#!/bin/bash
# Minimal WASM build script for ruv-swarm-daa

set -e

echo "🚀 Building minimal ruv-swarm-daa WASM module..."

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf pkg-minimal target/wasm32-unknown-unknown

# Create a minimal Cargo.toml for WASM
cat > Cargo-wasm.toml << 'EOF'
[package]
name = "ruv-swarm-daa-wasm"
version = "1.0.4"
edition = "2021"

[lib]
name = "ruv_swarm_daa_wasm"
crate-type = ["cdylib"]
path = "src-wasm/lib.rs"

[dependencies]
wasm-bindgen = "0.2"
web-sys = { version = "0.3", features = ["console"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
console_error_panic_hook = "0.1"

[target.'cfg(target_arch = "wasm32")'.dependencies]
getrandom = { version = "0.2", features = ["js"] }
EOF

# Create minimal source file
mkdir -p src-wasm
cat > src-wasm/lib.rs << 'EOF'
use wasm_bindgen::prelude::*;
use web_sys::console;
use serde::{Deserialize, Serialize};

#[wasm_bindgen]
pub struct DAAAgent {
    id: String,
    autonomy_level: f64,
}

#[wasm_bindgen]
impl DAAAgent {
    #[wasm_bindgen(constructor)]
    pub fn new(id: &str) -> Self {
        console::log_1(&format!("Creating DAA agent: {}", id).into());
        DAAAgent {
            id: id.to_string(),
            autonomy_level: 1.0,
        }
    }
    
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }
    
    #[wasm_bindgen]
    pub fn make_decision(&self, context: &str) -> String {
        format!("Agent {} decision for context: {}", self.id, context)
    }
    
    #[wasm_bindgen]
    pub fn get_status(&self) -> String {
        let status = Status {
            id: self.id.clone(),
            autonomy_level: self.autonomy_level,
        };
        serde_json::to_string(&status).unwrap_or_else(|_| "{}".to_string())
    }
}

#[wasm_bindgen]
pub fn init_daa() {
    console_error_panic_hook::set_once();
    console::log_1(&"DAA WASM initialized".into());
}

#[derive(Serialize, Deserialize)]
struct Status {
    id: String,
    autonomy_level: f64,
}
EOF

# Build with wasm-pack using the minimal config
echo "🔨 Building minimal WASM module..."
CARGO_TARGET_DIR=target-wasm wasm-pack build \
    --target web \
    --out-dir pkg-minimal \
    --release \
    --manifest-path Cargo-wasm.toml

# Create enhanced JS wrapper
echo "📝 Creating enhanced JavaScript bindings..."
cat > pkg-minimal/ruv_swarm_daa_minimal.js << 'EOF'
import init, * as wasm from './ruv_swarm_daa_wasm.js';

let initialized = false;

export async function initializeDAA() {
    if (!initialized) {
        await init();
        wasm.init_daa();
        initialized = true;
        console.log('✅ DAA WASM module initialized');
    }
    return wasm;
}

export { DAAAgent } from './ruv_swarm_daa_wasm.js';

export function createAgent(id) {
    if (!initialized) {
        throw new Error('DAA not initialized. Call initializeDAA() first.');
    }
    return new wasm.DAAAgent(id);
}
EOF

# Create example HTML
echo "📄 Creating minimal example..."
cat > pkg-minimal/example.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Minimal DAA WASM Example</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .output { background: #f0f0f0; padding: 10px; margin: 10px 0; }
        button { padding: 10px; margin: 5px; }
    </style>
</head>
<body>
    <h1>Minimal DAA WASM Module</h1>
    <button onclick="createAgent()">Create Agent</button>
    <button onclick="makeDecision()">Make Decision</button>
    <div id="output" class="output">Loading...</div>
    
    <script type="module">
        import { initializeDAA, createAgent } from './ruv_swarm_daa_minimal.js';
        
        let agent = null;
        
        async function init() {
            try {
                await initializeDAA();
                document.getElementById('output').textContent = '✅ DAA initialized';
            } catch (error) {
                document.getElementById('output').textContent = '❌ Error: ' + error.message;
            }
        }
        
        window.createAgent = function() {
            agent = createAgent('agent-' + Date.now());
            const status = JSON.parse(agent.get_status());
            document.getElementById('output').textContent = 
                `Created agent: ${status.id} (autonomy: ${status.autonomy_level})`;
        };
        
        window.makeDecision = function() {
            if (!agent) {
                document.getElementById('output').textContent = 'Create an agent first!';
                return;
            }
            const decision = agent.make_decision('test-context');
            document.getElementById('output').textContent = `Decision: ${decision}`;
        };
        
        init();
    </script>
</body>
</html>
EOF

# Clean up temporary files
rm -f Cargo-wasm.toml
rm -rf src-wasm

echo "✅ Minimal WASM module built successfully!"
echo "   Output directory: ./pkg-minimal"
echo "   To test: open pkg-minimal/example.html in a web browser"