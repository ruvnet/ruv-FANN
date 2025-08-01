import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function checkWasmFiles() {
  console.log('Checking WASM files...\n');

  // Check global installation
  const globalPaths = [
    '/home/codespace/nvm/current/lib/node_modules/ruv-swarm/wasm',
    '/usr/local/lib/node_modules/ruv-swarm/wasm',
    '/usr/lib/node_modules/ruv-swarm/wasm',
  ];

  for (const path of globalPaths) {
    try {
      const files = await fs.readdir(path);
      console.log(`✅ Found global installation at: ${path}`);
      console.log(`   Files: ${files.join(', ')}`);

      // Check for specific WASM files
      const wasmFiles = files.filter(f => f.endsWith('.wasm'));
      if (wasmFiles.length === 0) {
        console.log('   ❌ No .wasm files found!');
      } else {
        console.log(`   ✅ WASM files: ${wasmFiles.join(', ')}`);
      }
      console.log('');
    } catch (error) {
      console.log(`❌ ${path}: ${error.message}`);
    }
  }

  // Check local installation
  console.log('\nChecking local installation...');
  const localPath = join(__dirname, 'node_modules/ruv-swarm/wasm');
  try {
    const files = await fs.readdir(localPath);
    console.log(`✅ Found local installation at: ${localPath}`);
    console.log(`   Files: ${files.join(', ')}`);
  } catch (error) {
    console.log(`❌ No local installation found`);
  }

  // Try to load the WASM module directly
  console.log('\nTrying to load WASM module...');
  try {
    const { WasmModuleLoader } = await import('/home/codespace/nvm/current/lib/node_modules/ruv-swarm/src/wasm-loader.js');
    const loader = new WasmModuleLoader();

    console.log('✅ WasmModuleLoader imported successfully');
    console.log(`   Base directory: ${loader.baseDir}`);

    // Try to initialize
    await loader.initialize('progressive');
    console.log('✅ Loader initialized successfully');

    const status = loader.getModuleStatus();
    console.log('\nModule Status:', JSON.stringify(status, null, 2));
  } catch (error) {
    console.log(`❌ Failed to load WASM module: ${error.message}`);
    console.log(`   Stack: ${error.stack}`);
  }
}

checkWasmFiles().catch(console.error);
