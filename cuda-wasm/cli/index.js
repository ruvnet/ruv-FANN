#!/usr/bin/env node

const { Command } = require('commander');
const chalk = require('chalk').default || require('chalk');
const fs = require('fs').promises;
const path = require('path');
const semver = require('semver');
const { transpileCuda, analyzeKernel, benchmark, getVersion } = require('../dist');

// Simple spinner replacement for ora
const createSpinner = (text) => ({
  start() {
    console.log(text);
    return this;
  },
  succeed(text) {
    console.log(chalk.green('✓'), text || 'Done!');
  },
  fail(text) {
    console.log(chalk.red('✗'), text || 'Failed!');
  },
});

const program = new Command();

program
  .name('cuda-wasm')
  .description('High-performance CUDA to WebAssembly/WebGPU transpiler')
  .version('1.1.0');

program
  .command('transpile <input>')
  .description('Transpile CUDA code to WebAssembly/WebGPU')
  .option('-o, --output <path>', 'Output file path')
  .option('-t, --target <target>', 'Target platform (wasm|webgpu)', 'wasm')
  .option('-O, --optimize', 'Enable optimizations', false)
  .option('--profile', 'Generate profiling data', false)
  .action(async (input, options) => {
    const spinner = createSpinner('🚀 Transpiling CUDA code...').start();

    try {
      // Read input file
      const cudaCode = await fs.readFile(input, 'utf8');

      // Transpile code
      const result = transpileCuda(input, {
        output: options.output,
        target: options.target,
        optimize: options.optimize,
        profile: options.profile,
      });

      // Determine output path
      const outputPath = options.output || input.replace(/\.(cu|cuh)$/, '.wasm');

      // Output path is handled by transpileCuda

      spinner.succeed(chalk.green(`✓ Transpiled successfully to ${outputPath}`));

      // Show results
      console.log(chalk.blue('\nTranspilation Results:'));
      console.log(`  Input: ${result.inputFile}`);
      console.log(`  Output: ${result.outputFile}`);
      console.log(`  Size: ${result.size} bytes`);
      console.log(`  Optimizations: ${result.optimizations.join(', ')}`);
      if (result.kernels) {
        console.log(`  Kernels: ${result.kernels.join(', ')}`);
      }

      if (result.warnings && result.warnings.length > 0) {
        console.log(chalk.yellow('\nWarnings:'));
        result.warnings.forEach(warning => console.log(`  - ${warning}`));
      }
    } catch (error) {
      spinner.fail(chalk.red(`✗ Transpilation failed: ${error.message}`));
      process.exit(1);
    }
  });

program
  .command('analyze <input>')
  .description('Analyze CUDA kernel for optimization opportunities')
  .action(async (input) => {
    const spinner = createSpinner('🔍 Analyzing CUDA kernel...').start();

    try {
      const cudaCode = await fs.readFile(input, 'utf8');
      const analysis = analyzeKernel(input);

      spinner.succeed(chalk.green('✓ Analysis complete'));

      console.log(chalk.blue('\nKernel Analysis:'));
      console.log(chalk.yellow('Kernel Name:'), analysis.kernelName);
      console.log(chalk.yellow('Complexity:'), analysis.complexity);
      console.log(chalk.yellow('Memory Access:'), analysis.memoryAccess);

      if (analysis.metrics) {
        console.log(chalk.blue('\nPerformance Metrics:'));
        console.log(chalk.yellow('Thread Utilization:'), analysis.metrics.threadUtilization);
        console.log(chalk.yellow('Shared Memory Usage:'), analysis.metrics.sharedMemoryUsage);
        console.log(chalk.yellow('Register Usage:'), analysis.metrics.estimatedRegisterUsage);
      }

      if (analysis.optimization_suggestions.length > 0) {
        console.log(chalk.blue('\nOptimization Suggestions:'));
        analysis.optimization_suggestions.forEach((suggestion, i) => {
          console.log(chalk.yellow(`${i + 1}.`), suggestion);
        });
      }
    } catch (error) {
      spinner.fail(chalk.red(`✗ Analysis failed: ${error.message}`));
      process.exit(1);
    }
  });

program
  .command('benchmark <input>')
  .description('Benchmark CUDA kernel performance')
  .option('-i, --iterations <n>', 'Number of iterations', '100')
  .action(async (input, options) => {
    const spinner = createSpinner('⚡ Running benchmarks...').start();

    try {
      const cudaCode = await fs.readFile(input, 'utf8');
      const iterations = parseInt(options.iterations);

      // Run benchmarks
      const results = await benchmark(input, { iterations });

      spinner.succeed(chalk.green('✓ Benchmarks complete'));

      console.log(chalk.blue('\nBenchmark Results:'));
      console.log(chalk.yellow('Native execution time:'), `${results.nativeTime}ms`);
      console.log(chalk.yellow('WASM execution time:'), `${results.wasmTime}ms`);
      console.log(chalk.yellow('Speedup:'), `${results.speedup}x`);
      console.log(chalk.yellow('Throughput:'), results.throughput);
      console.log(chalk.yellow('Efficiency:'), results.efficiency);
    } catch (error) {
      spinner.fail(chalk.red(`✗ Benchmark failed: ${error.message}`));
      process.exit(1);
    }
  });

program
  .command('init')
  .description('Initialize a new CUDA-Rust-WASM project')
  .option('-n, --name <name>', 'Project name', 'my-cuda-wasm-project')
  .action(async (options) => {
    const spinner = createSpinner('📦 Initializing project...').start();

    try {
      const projectPath = path.join(process.cwd(), options.name);

      // Create project structure
      await fs.mkdir(projectPath, { recursive: true });
      await fs.mkdir(path.join(projectPath, 'src'), { recursive: true });
      await fs.mkdir(path.join(projectPath, 'kernels'), { recursive: true });

      // Create package.json
      const packageJson = {
        name: options.name,
        version: '1.0.0',
        description: 'A CUDA-Rust-WASM project',
        main: 'dist/index.js',
        scripts: {
          build: 'cuda-wasm transpile kernels/*.cu -o dist/',
          test: 'jest',
          benchmark: 'cuda-wasm benchmark kernels/*.cu',
        },
        dependencies: {
          'cuda-wasm': '^1.0.1',
        },
      };

      await fs.writeFile(
        path.join(projectPath, 'package.json'),
        JSON.stringify(packageJson, null, 2),
      );

      // Create example kernel
      const exampleKernel = `// Example CUDA kernel
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}`;

      await fs.writeFile(
        path.join(projectPath, 'kernels', 'vector_add.cu'),
        exampleKernel,
      );

      // Create README
      const readme = `# ${options.name}

A CUDA-Rust-WASM project for high-performance GPU computing in the browser.

## Getting Started

1. Install dependencies:
   \`\`\`bash
   npm install
   \`\`\`

2. Build the project:
   \`\`\`bash
   npm run build
   \`\`\`

3. Run benchmarks:
   \`\`\`bash
   npm run benchmark
   \`\`\`

## Project Structure

- \`kernels/\` - CUDA kernel source files
- \`src/\` - JavaScript/TypeScript source files
- \`dist/\` - Transpiled WebAssembly output

## Documentation

For more information, visit: https://github.com/ruvnet/ruv-FANN/tree/main/cuda-wasm
`;

      await fs.writeFile(path.join(projectPath, 'README.md'), readme);

      spinner.succeed(chalk.green(`✓ Project initialized at ${projectPath}`));
      console.log(chalk.blue('\nNext steps:'));
      console.log(chalk.yellow('1.'), `cd ${options.name}`);
      console.log(chalk.yellow('2.'), 'npm install');
      console.log(chalk.yellow('3.'), 'npm run build');
    } catch (error) {
      spinner.fail(chalk.red(`✗ Initialization failed: ${error.message}`));
      process.exit(1);
    }
  });

program.parse(process.argv);
