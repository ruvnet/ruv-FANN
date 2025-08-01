import fs from 'fs/promises';

const wasmPath = '/home/codespace/nvm/current/lib/node_modules/ruv-swarm/wasm/ruv_swarm_wasm_bg.wasm';

async function testDirectWasmLoading() {
  try {
    console.log('Testing direct WASM loading...\n');

    // Check if file exists
    await fs.access(wasmPath);
    console.log('✅ WASM file exists:', wasmPath);

    // Read the file
    const wasmBuffer = await fs.readFile(wasmPath);
    console.log(`✅ WASM file read successfully, size: ${wasmBuffer.length} bytes`);

    // Try to instantiate
    const imports = {
      env: {
        memory: new WebAssembly.Memory({ initial: 256, maximum: 4096 }),
      },
      wasi_snapshot_preview1: {
        proc_exit: (code) => {
          throw new Error(`WASI exit ${code}`);
        },
        fd_write: () => 0,
        random_get: (ptr, len) => {
          return 0;
        },
      },
    };

    console.log('\nInstantiating WASM module...');
    const { instance, module } = await WebAssembly.instantiate(wasmBuffer, imports);
    console.log('✅ WASM module instantiated successfully!');
    console.log('   Exports:', Object.keys(instance.exports));

  } catch (error) {
    console.error('❌ Error:', error.message);
    if (error.stack) {
      console.error('\nStack trace:');
      console.error(error.stack);
    }
  }
}

testDirectWasmLoading();
