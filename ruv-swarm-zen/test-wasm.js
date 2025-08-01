#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

console.log('Testing WASM build...\n');

// Check if pkg directory exists
const pkgDir = path.join(__dirname, 'crates/ruv-swarm-wasm/pkg');
if (!fs.existsSync(pkgDir)) {
  console.error('❌ pkg directory not found at:', pkgDir);
  process.exit(1);
}

// Check for required files
const requiredFiles = [
  'ruv_swarm_wasm_bg.wasm',
  'ruv_swarm_wasm.js',
  'ruv_swarm_wasm.d.ts',
  'ruv_swarm_wasm_bg.wasm.d.ts',
  'package.json',
];

let allFilesExist = true;
console.log('Checking required files:');
for (const file of requiredFiles) {
  const filePath = path.join(pkgDir, file);
  if (fs.existsSync(filePath)) {
    const stats = fs.statSync(filePath);
    console.log(`✅ ${file} (${(stats.size / 1024).toFixed(1)}KB)`);
  } else {
    console.log(`❌ ${file} - NOT FOUND`);
    allFilesExist = false;
  }
}

if (!allFilesExist) {
  console.error('\n❌ Some required files are missing');
  process.exit(1);
}

// Check WASM file size
const wasmPath = path.join(pkgDir, 'ruv_swarm_wasm_bg.wasm');
const wasmStats = fs.statSync(wasmPath);
const wasmSizeKB = wasmStats.size / 1024;

console.log(`\n📊 WASM Size: ${wasmSizeKB.toFixed(1)}KB`);

// Check performance targets
const loadTarget = 500; // ms
const spawnTarget = 100; // ms
const memoryTarget = 50; // MB

console.log('\n🎯 Performance Targets:');
console.log(`   Load Time: < ${loadTarget}ms`);
console.log(`   Spawn Time: < ${spawnTarget}ms`);
console.log(`   Memory Usage: < ${memoryTarget}MB`);

// Read package.json
const packageJsonPath = path.join(pkgDir, 'package.json');
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));

console.log('\n📦 Package Info:');
console.log(`   Name: ${packageJson.name}`);
console.log(`   Version: ${packageJson.version}`);
console.log(`   Module: ${packageJson.module}`);
console.log(`   Types: ${packageJson.types}`);

console.log('\n✅ WASM build validation complete!');
console.log('\n📍 Next steps:');
console.log('   1. cd crates/ruv-swarm-wasm/pkg && python3 -m http.server 8000');
console.log('   2. Open http://localhost:8000/test.html in browser');
console.log('   3. Or test examples: cd examples/browser && python3 -m http.server 8001');
