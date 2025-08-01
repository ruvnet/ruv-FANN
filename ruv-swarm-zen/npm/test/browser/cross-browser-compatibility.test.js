/**
 * Cross-Browser Compatibility Tests
 * Tests WASM functionality across different browser environments
 */

import { describe, it, expect, beforeAll, afterAll } from '@playwright/test';
import { chromium, firefox, webkit } from 'playwright';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer } from 'http';
import { readFile } from 'fs/promises';
import handler from 'serve-handler';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Cross-Browser WASM Compatibility', () => {
  let server;
  let serverUrl;
  const browsers = [];

  beforeAll(async() => {
    // Start local server to serve test files
    server = createServer((request, response) => {
      return handler(request, response, {
        public: path.join(__dirname, '../../'),
        headers: [
          {
            source: '**/*.wasm',
            headers: [{
              key: 'Content-Type',
              value: 'application/wasm',
            }],
          },
          {
            source: '**/*',
            headers: [{
              key: 'Cross-Origin-Embedder-Policy',
              value: 'require-corp',
            }, {
              key: 'Cross-Origin-Opener-Policy',
              value: 'same-origin',
            }],
          },
        ],
      });
    });

    await new Promise((resolve) => {
      server.listen(0, '127.0.0.1', () => {
        const { port } = server.address();
        serverUrl = `http://127.0.0.1:${port}`;
        console.log(`Test server running at ${serverUrl}`);
        resolve();
      });
    });

    // Launch browsers
    const browserTypes = [
      { name: 'chromium', launcher: chromium },
      { name: 'firefox', launcher: firefox },
      { name: 'webkit', launcher: webkit },
    ];

    for (const { name, launcher } of browserTypes) {
      const browser = await launcher.launch({
        headless: true,
        args: name === 'chromium' ? ['--enable-unsafe-webgpu'] : [],
      });
      browsers.push({ name, browser });
    }
  });

  afterAll(async() => {
    // Close all browsers
    for (const { browser } of browsers) {
      await browser.close();
    }

    // Close server
    await new Promise((resolve) => {
      server.close(resolve);
    });
  });

  describe('Basic WASM Support', () => {
    for (const { name, browser } of browsers) {
      it(`should detect WASM support in ${name}`, async() => {
        const context = await browser.newContext();
        const page = await context.newPage();

        await page.goto(`${serverUrl}/test/browser/wasm-test.html`);

        const wasmSupported = await page.evaluate(() => {
          return typeof WebAssembly !== 'undefined' &&
                 typeof WebAssembly.instantiate === 'function';
        });

        expect(wasmSupported).toBe(true);

        await context.close();
      });

      it(`should load WASM module in ${name}`, async() => {
        const context = await browser.newContext();
        const page = await context.newPage();

        await page.goto(`${serverUrl}/test/browser/wasm-test.html`);

        const moduleLoaded = await page.evaluate(async() => {
          try {
            const response = await fetch('/wasm/ruv_swarm_wasm_bg.wasm');
            const buffer = await response.arrayBuffer();
            const module = await WebAssembly.compile(buffer);
            return module instanceof WebAssembly.Module;
          } catch (error) {
            console.error('WASM load error:', error);
            return false;
          }
        });

        expect(moduleLoaded).toBe(true);

        await context.close();
      });
    }
  });

  describe('SIMD Support', () => {
    for (const { name, browser } of browsers) {
      it(`should detect SIMD support in ${name}`, async() => {
        const context = await browser.newContext();
        const page = await context.newPage();

        await page.goto(`${serverUrl}/test/browser/wasm-test.html`);

        const simdInfo = await page.evaluate(async() => {
          // Load RuvSwarm module
          const { RuvSwarm } = await import('/src/index-enhanced.js');
          const swarm = await RuvSwarm.initialize({ debug: true });

          return {
            simdDetected: swarm.features.simd,
            simdInWasm: await swarm.wasmLoader.detectSIMDSupport(),
          };
        });

        console.log(`${name} SIMD support:`, simdInfo);

        // SIMD is optional, so we just log the result
        if (simdInfo.simdDetected) {
          expect(simdInfo.simdInWasm).toBe(true);
        }

        await context.close();
      });
    }
  });

  describe('Memory Management', () => {
    for (const { name, browser } of browsers) {
      it(`should handle SharedArrayBuffer in ${name}`, async() => {
        const context = await browser.newContext();
        const page = await context.newPage();

        await page.goto(`${serverUrl}/test/browser/wasm-test.html`);

        const sharedMemoryTest = await page.evaluate(() => {
          try {
            // Check if SharedArrayBuffer is available
            if (typeof SharedArrayBuffer === 'undefined') {
              return { available: false, reason: 'SharedArrayBuffer not defined' };
            }

            // Try to create one
            const buffer = new SharedArrayBuffer(1024);
            const view = new Int32Array(buffer);
            view[0] = 42;

            // Test Atomics if available
            if (typeof Atomics !== 'undefined') {
              Atomics.store(view, 0, 123);
              const value = Atomics.load(view, 0);
              return {
                available: true,
                atomicsSupported: true,
                testValue: value,
              };
            }

            return { available: true, atomicsSupported: false };
          } catch (error) {
            return {
              available: false,
              reason: error.message,
            };
          }
        });

        console.log(`${name} SharedArrayBuffer:`, sharedMemoryTest);

        // SharedArrayBuffer requires specific headers, may not be available
        if (sharedMemoryTest.available && sharedMemoryTest.atomicsSupported) {
          expect(sharedMemoryTest.testValue).toBe(123);
        }

        await context.close();
      });

      it(`should handle memory growth in ${name}`, async() => {
        const context = await browser.newContext();
        const page = await context.newPage();

        await page.goto(`${serverUrl}/test/browser/wasm-test.html`);

        const memoryTest = await page.evaluate(async() => {
          const { RuvSwarm } = await import('/src/index-enhanced.js');
          const swarm = await RuvSwarm.initialize({ debug: false });

          const initialMemory = swarm.wasmLoader.getMemorySize();

          // Allocate large amount of memory
          const allocations = [];
          for (let i = 0; i < 10; i++) {
            const data = new Float32Array(100000); // ~400KB each
            allocations.push(await swarm.wasmLoader.allocateAndStore(data));
          }

          const afterAllocMemory = swarm.wasmLoader.getMemorySize();

          // Clean up
          for (const ptr of allocations) {
            await swarm.wasmLoader.deallocate(ptr);
          }

          return {
            initial: initialMemory,
            afterAlloc: afterAllocMemory,
            grew: afterAllocMemory > initialMemory,
          };
        });

        expect(memoryTest.grew).toBe(true);
        console.log(`${name} memory growth: ${memoryTest.initial} -> ${memoryTest.afterAlloc}`);

        await context.close();
      });
    }
  });

  describe('Neural Network Operations', () => {
    for (const { name, browser } of browsers) {
      it(`should run neural network in ${name}`, async() => {
        const context = await browser.newContext();
        const page = await context.newPage();

        await page.goto(`${serverUrl}/test/browser/wasm-test.html`);

        const nnTest = await page.evaluate(async() => {
          const { RuvSwarm } = await import('/src/index-enhanced.js');
          const swarm = await RuvSwarm.initialize({
            enableNeuralNetworks: true,
            debug: false,
          });

          // Create simple neural network
          const network = await swarm.neuralManager.createNetwork({
            type: 'mlp',
            layers: [
              { units: 10, activation: 'relu' },
              { units: 5, activation: 'softmax' },
            ],
          });

          // Run inference
          const input = new Float32Array(10).fill(0.5);
          const startTime = performance.now();
          const output = await network.predict(input);
          const inferenceTime = performance.now() - startTime;

          return {
            success: true,
            outputLength: output.length,
            inferenceTime,
            outputSum: Array.from(output).reduce((a, b) => a + b, 0),
          };
        });

        expect(nnTest.success).toBe(true);
        expect(nnTest.outputLength).toBe(5);
        expect(nnTest.outputSum).toBeCloseTo(1.0, 1); // Softmax should sum to ~1
        console.log(`${name} NN inference time: ${nnTest.inferenceTime.toFixed(2)}ms`);

        await context.close();
      });
    }
  });

  describe('Swarm Operations', () => {
    for (const { name, browser } of browsers) {
      it(`should create and manage swarm in ${name}`, async() => {
        const context = await browser.newContext();
        const page = await context.newPage();

        await page.goto(`${serverUrl}/test/browser/wasm-test.html`);

        const swarmTest = await page.evaluate(async() => {
          const { RuvSwarm } = await import('/src/index-enhanced.js');
          const ruvSwarm = await RuvSwarm.initialize({ debug: false });

          // Create swarm
          const swarm = await ruvSwarm.createSwarm({
            name: 'browser-test-swarm',
            topology: 'mesh',
            maxAgents: 5,
          });

          // Spawn agents
          const agents = [];
          for (let i = 0; i < 3; i++) {
            const agent = await swarm.spawn({
              type: ['researcher', 'coder', 'analyst'][i],
            });
            agents.push({
              id: agent.id,
              type: agent.type,
              status: agent.status,
            });
          }

          // Execute task
          const result = await swarm.orchestrate({
            task: 'test-task',
            strategy: 'parallel',
          });

          return {
            swarmCreated: true,
            agentCount: agents.length,
            agents,
            taskCompleted: result.completed,
          };
        });

        expect(swarmTest.swarmCreated).toBe(true);
        expect(swarmTest.agentCount).toBe(3);
        expect(swarmTest.taskCompleted).toBe(true);

        await context.close();
      });
    }
  });

  describe('Performance Characteristics', () => {
    for (const { name, browser } of browsers) {
      it(`should measure WASM performance in ${name}`, async() => {
        const context = await browser.newContext();
        const page = await context.newPage();

        await page.goto(`${serverUrl}/test/browser/wasm-test.html`);

        const perfTest = await page.evaluate(async() => {
          const { RuvSwarm } = await import('/src/index-enhanced.js');
          const swarm = await RuvSwarm.initialize({
            useSIMD: true,
            debug: false,
          });

          const benchmarks = {};

          // Benchmark 1: Vector operations
          const vectorSize = 100000;
          const a = new Float32Array(vectorSize).fill(1.0);
          const b = new Float32Array(vectorSize).fill(2.0);

          const vectorStart = performance.now();
          const vectorResult = await swarm.wasmLoader.vectorAdd(a, b);
          benchmarks.vectorOps = performance.now() - vectorStart;

          // Benchmark 2: Matrix multiplication
          const matrixSize = 100;
          const matA = new Float32Array(matrixSize * matrixSize).fill(1.0);
          const matB = new Float32Array(matrixSize * matrixSize).fill(2.0);

          const matrixStart = performance.now();
          const matrixResult = await swarm.wasmLoader.matrixMultiply(
            matA, matB, matrixSize, matrixSize, matrixSize,
          );
          benchmarks.matrixOps = performance.now() - matrixStart;

          // Benchmark 3: Neural network
          const network = await swarm.neuralManager.createNetwork({
            type: 'mlp',
            layers: [
              { units: 100, activation: 'relu' },
              { units: 50, activation: 'relu' },
              { units: 10, activation: 'softmax' },
            ],
          });

          const nnInput = new Float32Array(100).fill(0.5);
          const iterations = 100;

          const nnStart = performance.now();
          for (let i = 0; i < iterations; i++) {
            await network.predict(nnInput);
          }
          benchmarks.nnInference = (performance.now() - nnStart) / iterations;

          return benchmarks;
        });

        console.log(`${name} Performance Benchmarks:`);
        console.log(`  Vector ops (100k elements): ${perfTest.vectorOps.toFixed(2)}ms`);
        console.log(`  Matrix mult (100x100): ${perfTest.matrixOps.toFixed(2)}ms`);
        console.log(`  NN inference (avg): ${perfTest.nnInference.toFixed(2)}ms`);

        // Performance should be reasonable across browsers
        expect(perfTest.vectorOps).toBeLessThan(50);
        expect(perfTest.matrixOps).toBeLessThan(100);
        expect(perfTest.nnInference).toBeLessThan(5);

        await context.close();
      });
    }
  });

  describe('Error Handling', () => {
    for (const { name, browser } of browsers) {
      it(`should handle WASM errors gracefully in ${name}`, async() => {
        const context = await browser.newContext();
        const page = await context.newPage();

        await page.goto(`${serverUrl}/test/browser/wasm-test.html`);

        const errorTest = await page.evaluate(async() => {
          const { RuvSwarm } = await import('/src/index-enhanced.js');
          const swarm = await RuvSwarm.initialize({ debug: false });

          const errors = [];

          // Test 1: Invalid agent type
          try {
            await swarm.createSwarm({ name: 'test' })
              .then(s => s.spawn({ type: 'invalid-type' }));
          } catch (error) {
            errors.push({
              test: 'invalid-agent-type',
              caught: true,
              message: error.message,
            });
          }

          // Test 2: Memory allocation failure
          try {
            await swarm.wasmLoader.allocate(Number.MAX_SAFE_INTEGER);
          } catch (error) {
            errors.push({
              test: 'memory-allocation',
              caught: true,
              message: error.message,
            });
          }

          // Test 3: Invalid neural network config
          try {
            await swarm.neuralManager.createNetwork({
              type: 'invalid',
              layers: [],
            });
          } catch (error) {
            errors.push({
              test: 'invalid-network',
              caught: true,
              message: error.message,
            });
          }

          return errors;
        });

        expect(errorTest).toHaveLength(3);
        errorTest.forEach(error => {
          expect(error.caught).toBe(true);
          expect(error.message).toBeTruthy();
        });

        await context.close();
      });
    }
  });

  describe('WebWorker Integration', () => {
    for (const { name, browser } of browsers) {
      it(`should run WASM in WebWorker in ${name}`, async() => {
        const context = await browser.newContext();
        const page = await context.newPage();

        await page.goto(`${serverUrl}/test/browser/wasm-test.html`);

        const workerTest = await page.evaluate(async() => {
          // Create worker script
          const workerScript = `
            importScripts('/src/index.js');
            
            self.onmessage = async (e) => {
              const { RuvSwarm } = self;
              const swarm = await RuvSwarm.initialize({ debug: false });
              
              const result = await swarm.wasmLoader.vectorAdd(
                e.data.a,
                e.data.b
              );
              
              self.postMessage({ result: Array.from(result) });
            };
          `;

          const blob = new Blob([workerScript], { type: 'application/javascript' });
          const worker = new Worker(URL.createObjectURL(blob));

          const result = await new Promise((resolve) => {
            worker.onmessage = (e) => resolve(e.data);
            worker.postMessage({
              a: new Float32Array([1, 2, 3, 4]),
              b: new Float32Array([5, 6, 7, 8]),
            });
          });

          worker.terminate();

          return result;
        });

        expect(workerTest.result).toEqual([6, 8, 10, 12]);

        await context.close();
      });
    }
  });
});

// Create test HTML file
export async function createTestHTML() {
  const html = `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>WASM Browser Test</title>
    <meta http-equiv="Cross-Origin-Embedder-Policy" content="require-corp">
    <meta http-equiv="Cross-Origin-Opener-Policy" content="same-origin">
</head>
<body>
    <h1>RuvSwarm WASM Browser Test</h1>
    <div id="status">Loading...</div>
    <div id="results"></div>
    
    <script type="module">
        import { RuvSwarm } from '/src/index-enhanced.js';
        
        window.RuvSwarm = RuvSwarm;
        
        async function runTests() {
            const status = document.getElementById('status');
            const results = document.getElementById('results');
            
            try {
                status.textContent = 'Initializing RuvSwarm...';
                const swarm = await RuvSwarm.initialize({
                    debug: true,
                    enableNeuralNetworks: true,
                    useSIMD: true
                });
                
                status.textContent = 'RuvSwarm initialized successfully!';
                
                // Display features
                results.innerHTML = \`
                    <h2>Detected Features:</h2>
                    <ul>
                        <li>WASM Support: ✓</li>
                        <li>SIMD Support: \${swarm.features.simd ? '✓' : '✗'}</li>
                        <li>SharedArrayBuffer: \${typeof SharedArrayBuffer !== 'undefined' ? '✓' : '✗'}</li>
                        <li>Atomics: \${typeof Atomics !== 'undefined' ? '✓' : '✗'}</li>
                        <li>WebWorkers: \${typeof Worker !== 'undefined' ? '✓' : '✗'}</li>
                    </ul>
                \`;
                
                window.swarm = swarm;
            } catch (error) {
                status.textContent = 'Error: ' + error.message;
                console.error(error);
            }
        }
        
        runTests();
    </script>
</body>
</html>`;

  await writeFile(
    path.join(__dirname, 'wasm-test.html'),
    html,
  );
}
