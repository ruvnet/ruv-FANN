#!/usr/bin/env node

/**
 * Test script to validate WASM loading fix
 * This script tests the specific changes needed to fix issue #41
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🔧 Testing WASM Loading Fix for Issue #41\n');

const tests = {
  passed: [],
  failed: [],
  suggestions: [],
};

// Test 1: Check current WASM loading behavior
console.log('1️⃣ Testing current WASM loading...');
try {
  const testCode = `
    const { WasmModuleLoader } = require('ruv-swarm/src/wasm-loader.js');
    const loader = new WasmModuleLoader();
    
    (async () => {
      await loader.initialize('progressive');
      const status = loader.getModuleStatus();
      console.log(JSON.stringify({ 
        isPlaceholder: status.core.placeholder,
        loaded: status.core.loaded 
      }));
    })();
  `;

  const result = execSync(`node -e '${testCode}'`, { encoding: 'utf8' });
  const parsed = JSON.parse(result);

  if (parsed.isPlaceholder) {
    tests.failed.push('WASM loads as placeholder');
    tests.suggestions.push('Need to fix path resolution in wasm-loader.js');
  } else {
    tests.passed.push('WASM loads correctly');
  }
} catch (error) {
  tests.failed.push(`WASM loading error: ${error.message}`);
}

// Test 2: Check WASM file accessibility
console.log('2️⃣ Checking WASM file paths...');
try {
  const possiblePaths = [
    path.join(process.cwd(), 'node_modules/ruv-swarm/wasm/ruv_swarm_wasm_bg.wasm'),
    path.join(__dirname, '../node_modules/ruv-swarm/wasm/ruv_swarm_wasm_bg.wasm'),
    path.join(require.resolve('ruv-swarm'), '../../wasm/ruv_swarm_wasm_bg.wasm'),
  ];

  let found = false;
  for (const wasmPath of possiblePaths) {
    if (fs.existsSync(wasmPath)) {
      console.log(`  ✅ Found WASM at: ${wasmPath}`);
      found = true;

      // Check if we can read it
      const stats = fs.statSync(wasmPath);
      console.log(`  📏 Size: ${stats.size} bytes`);

      // Verify it's a valid WASM file
      const buffer = fs.readFileSync(wasmPath);
      const magic = buffer.slice(0, 4).toString();
      if (magic === '\0asm') {
        tests.passed.push('WASM file is valid');
      } else {
        tests.failed.push('Invalid WASM magic number');
      }
      break;
    }
  }

  if (!found) {
    tests.failed.push('WASM file not found in expected locations');
    tests.suggestions.push('Check npm package structure');
  }
} catch (error) {
  tests.failed.push(`Path checking error: ${error.message}`);
}

// Test 3: Test proposed fix approach
console.log('3️⃣ Testing fix approach...');
try {
  // This simulates what the fix should do
  const fixCode = `
    const path = require('path');
    const fs = require('fs');
    
    // Better path resolution using require.resolve
    try {
      const packagePath = require.resolve('ruv-swarm/package.json');
      const packageDir = path.dirname(packagePath);
      const wasmPath = path.join(packageDir, 'wasm', 'ruv_swarm_wasm_bg.wasm');
      
      if (fs.existsSync(wasmPath)) {
        console.log(JSON.stringify({ 
          success: true, 
          path: wasmPath,
          exists: true 
        }));
      } else {
        console.log(JSON.stringify({ 
          success: false, 
          error: 'WASM not found' 
        }));
      }
    } catch (e) {
      console.log(JSON.stringify({ 
        success: false, 
        error: e.message 
      }));
    }
  `;

  const result = execSync(`node -e '${fixCode}'`, { encoding: 'utf8' });
  const parsed = JSON.parse(result);

  if (parsed.success) {
    tests.passed.push('Fix approach works');
    tests.suggestions.push(`Use require.resolve() for reliable path resolution`);
  } else {
    tests.failed.push('Fix approach failed');
  }
} catch (error) {
  tests.failed.push(`Fix test error: ${error.message}`);
}

// Test 4: Check if wasm-bindgen JS can be loaded
console.log('4️⃣ Testing wasm-bindgen module loading...');
try {
  const bindgenTest = `
    try {
      // Try to require the wasm-bindgen generated JS
      const wasmModule = require('ruv-swarm/wasm/ruv_swarm_wasm.js');
      console.log(JSON.stringify({ 
        success: true, 
        hasExports: Object.keys(wasmModule).length > 0 
      }));
    } catch (e) {
      console.log(JSON.stringify({ 
        success: false, 
        error: e.message 
      }));
    }
  `;

  const result = execSync(`node -e '${bindgenTest}'`, { encoding: 'utf8' });
  const parsed = JSON.parse(result);

  if (parsed.success) {
    tests.passed.push('wasm-bindgen JS module loads');
  } else {
    tests.failed.push(`wasm-bindgen JS error: ${parsed.error}`);
    tests.suggestions.push('May need to update import/require strategy');
  }
} catch (error) {
  tests.failed.push(`Bindgen test error: ${error.message}`);
}

// Generate fix recommendations
console.log('\n📋 Fix Recommendations for wasm-loader.js:\n');

const fixCode = `
// Recommended fix for #getWasmPathCandidates() method:

#getWasmPathCandidates() {
  const candidates = [];
  
  // 1. Use require.resolve for npm installations
  try {
    const packagePath = require.resolve('ruv-swarm/package.json');
    const packageDir = path.dirname(packagePath);
    candidates.push({
      description: 'NPM installation via require.resolve',
      wasmDir: path.join(packageDir, 'wasm'),
      loaderPath: path.join(packageDir, 'wasm', 'wasm-bindings-loader.mjs'),
      wasmBinary: path.join(packageDir, 'wasm', 'ruv_swarm_wasm_bg.wasm'),
      jsBindings: path.join(packageDir, 'wasm', 'ruv_swarm_wasm.js')
    });
  } catch (e) {
    // Not in npm installation
  }
  
  // 2. Check relative to current file (for local dev)
  candidates.push({
    description: 'Local development (relative to src/)',
    wasmDir: path.join(this.baseDir, '..', 'wasm'),
    // ... rest of paths
  });
  
  // 3. Environment variable override
  if (process.env.RUV_SWARM_WASM_PATH) {
    candidates.push({
      description: 'Environment variable override',
      wasmDir: process.env.RUV_SWARM_WASM_PATH,
      // ... rest of paths
    });
  }
  
  // Filter out non-existent paths
  return candidates.filter(candidate => {
    try {
      return fs.existsSync(candidate.wasmDir);
    } catch {
      return false;
    }
  });
}

// Also update #loadDirectWasm to handle CommonJS properly:

async #loadDirectWasm(wasmPath) {
  const jsBindingsPath = wasmPath.replace('_bg.wasm', '.js');
  
  try {
    // For CommonJS environments
    if (typeof require !== 'undefined') {
      const wasmModule = require(jsBindingsPath);
      // Initialize the module
      if (wasmModule.default || wasmModule.init) {
        const initFn = wasmModule.default || wasmModule.init;
        await initFn(wasmPath);
      }
      this.loadedWasm = wasmModule;
      return { instance: { exports: wasmModule }, module: wasmModule, exports: wasmModule };
    }
    
    // For ESM environments (existing code)
    // ... existing ESM code ...
  } catch (error) {
    console.error('Failed to load wasm-bindgen module:', error);
    throw error; // Don't fall back to placeholder silently
  }
}
`;

console.log(fixCode);

// Print test summary
console.log('\n📊 Test Summary:');
console.log(`✅ Passed: ${tests.passed.length}`);
console.log(`❌ Failed: ${tests.failed.length}`);

if (tests.failed.length > 0) {
  console.log('\nFailed tests:');
  tests.failed.forEach(t => console.log(`  - ${t}`));
}

if (tests.suggestions.length > 0) {
  console.log('\nSuggestions:');
  tests.suggestions.forEach(s => console.log(`  - ${s}`));
}

// Save detailed report
const report = {
  timestamp: new Date().toISOString(),
  tests: {
    passed: tests.passed,
    failed: tests.failed,
  },
  suggestions: tests.suggestions,
  fixRequired: tests.failed.length > 0,
};

fs.writeFileSync('wasm-fix-test-report.json', JSON.stringify(report, null, 2));
console.log('\n📄 Detailed report saved to: wasm-fix-test-report.json');

process.exit(tests.failed.length > 0 ? 1 : 0);
