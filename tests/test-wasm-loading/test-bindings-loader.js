import { fileURLToPath, pathToFileURL } from 'url';
import path from 'path';
import fs from 'fs/promises';

async function testBindingsLoader() {
  console.log('Testing wasm-bindings-loader.mjs...\n');

  const loaderPath = '/home/codespace/nvm/current/lib/node_modules/ruv-swarm/wasm/wasm-bindings-loader.mjs';

  try {
    // Check if file exists
    await fs.access(loaderPath);
    console.log('✅ Loader file exists:', loaderPath);

    // Try to import it
    const loaderURL = pathToFileURL(loaderPath).href;
    console.log('   URL:', loaderURL);

    const loaderModule = await import(loaderURL);
    console.log('✅ Loader module imported successfully');
    console.log('   Module keys:', Object.keys(loaderModule));

    if (loaderModule.default) {
      const bindingsLoader = loaderModule.default;
      console.log('\n✅ Found default export');
      console.log('   Type:', typeof bindingsLoader);

      if (typeof bindingsLoader.initialize === 'function') {
        console.log('\n   Initializing bindings loader...');
        await bindingsLoader.initialize();
        console.log('✅ Bindings loader initialized!');

        // Check what functions are available
        console.log('\n   Available functions:');
        for (const key in bindingsLoader) {
          if (typeof bindingsLoader[key] === 'function' && !key.startsWith('_')) {
            console.log(`     - ${key}`);
          }
        }
      }
    }

  } catch (error) {
    console.error('❌ Error:', error.message);
    if (error.stack) {
      console.error('\nStack trace:');
      console.error(error.stack.split('\n').slice(0, 5).join('\n'));
    }
  }
}

testBindingsLoader().catch(console.error);
