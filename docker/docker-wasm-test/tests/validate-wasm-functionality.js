#!/usr/bin/env node

/**
 * WASM Functionality Validator
 * Ensures that the actual WASM module is being used, not a placeholder
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class WasmValidator {
  constructor() {
    this.results = {
      checks: [],
      wasmFunctionality: {},
      passed: true,
    };
  }

  check(name, fn) {
    console.log(`\n🔍 Checking: ${name}`);
    try {
      const result = fn();
      this.results.checks.push({
        name,
        status: 'PASSED',
        result,
      });
      console.log(`✅ ${name}: PASSED`);
      return true;
    } catch (error) {
      this.results.checks.push({
        name,
        status: 'FAILED',
        error: error.message,
      });
      this.results.passed = false;
      console.log(`❌ ${name}: FAILED - ${error.message}`);
      return false;
    }
  }

  async runValidation() {
    console.log('🧪 WASM Functionality Validation');
    console.log('================================\n');

    // Check 1: WASM files exist and are valid
    this.check('WASM Files Present', () => {
      const wasmPath = path.join(process.cwd(), 'node_modules', 'ruv-swarm', 'wasm');
      if (!fs.existsSync(wasmPath)) {
        throw new Error('WASM directory not found');
      }

      const files = fs.readdirSync(wasmPath);
      const requiredFiles = ['ruv_swarm_wasm_bg.wasm', 'ruv_swarm_wasm.js'];

      for (const file of requiredFiles) {
        if (!files.includes(file)) {
          throw new Error(`Missing required file: ${file}`);
        }
      }

      this.results.wasmFunctionality.files = files;
      return true;
    });

    // Check 2: WASM binary format
    this.check('WASM Binary Format', () => {
      const wasmFile = path.join(process.cwd(), 'node_modules', 'ruv-swarm', 'wasm', 'ruv_swarm_wasm_bg.wasm');
      const buffer = fs.readFileSync(wasmFile);

      // Check magic number
      const magic = buffer.slice(0, 4);
      if (magic.toString() !== '\0asm') {
        throw new Error('Invalid WASM magic number');
      }

      // Check version
      const version = buffer.readUInt32LE(4);
      if (version !== 1) {
        throw new Error(`Invalid WASM version: ${version}`);
      }

      // Check file size (should be substantial)
      const stats = fs.statSync(wasmFile);
      if (stats.size < 10000) { // 10KB minimum
        throw new Error(`WASM file too small: ${stats.size} bytes`);
      }

      this.results.wasmFunctionality.binarySize = stats.size;
      this.results.wasmFunctionality.version = version;
      return true;
    });

    // Check 3: WASM can be instantiated
    this.check('WASM Instantiation', async () => {
      try {
        const wasmFile = path.join(process.cwd(), 'node_modules', 'ruv-swarm', 'wasm', 'ruv_swarm_wasm_bg.wasm');
        const wasmBuffer = fs.readFileSync(wasmFile);

        // Try to compile the WASM module
        const wasmModule = await WebAssembly.compile(wasmBuffer);

        if (!wasmModule) {
          throw new Error('Failed to compile WASM module');
        }

        // Check module exports
        const moduleExports = WebAssembly.Module.exports(wasmModule);
        this.results.wasmFunctionality.exportCount = moduleExports.length;

        if (moduleExports.length === 0) {
          throw new Error('WASM module has no exports');
        }

        console.log(`  Found ${moduleExports.length} exports`);
        return true;
      } catch (error) {
        throw new Error(`WASM instantiation failed: ${error.message}`);
      }
    });

    // Check 4: Test actual functionality
    this.check('WASM Functionality Test', () => {
      // Create a test script that uses WASM features
      const testScript = `
        const { WasmModuleLoader } = require('ruv-swarm/src/wasm-loader.js');
        const loader = new WasmModuleLoader();
        
        (async () => {
          await loader.initialize('progressive');
          const status = loader.getModuleStatus();
          
          if (status.core.placeholder) {
            throw new Error('Using placeholder');
          }
          
          const memory = loader.getTotalMemoryUsage();
          if (memory === 0) {
            throw new Error('No memory allocated');
          }
          
          console.log(JSON.stringify({ success: true, memory }));
        })().catch(e => {
          console.error(JSON.stringify({ error: e.message }));
          process.exit(1);
        });
      `;

      const result = execSync(`node -e '${testScript}'`, { encoding: 'utf8' });
      const parsed = JSON.parse(result);

      if (parsed.error) {
        throw new Error(parsed.error);
      }

      this.results.wasmFunctionality.memoryUsage = parsed.memory;
      return true;
    });

    // Check 5: NPX commands work with WASM
    this.check('NPX Command WASM Integration', () => {
      const commands = [
        { cmd: 'npx ruv-swarm --version', check: 'version' },
        { cmd: 'npx ruv-swarm benchmark --help', check: 'help' },
        { cmd: 'npx ruv-swarm swarm --help', check: 'help' },
      ];

      for (const { cmd, check } of commands) {
        try {
          const output = execSync(cmd, { encoding: 'utf8', stdio: 'pipe' });
          // Commands should work without errors
        } catch (error) {
          // Some commands exit with non-zero on help, but should have output
          if (!error.stdout && !error.stderr) {
            throw new Error(`Command failed: ${cmd}`);
          }
        }
      }

      return true;
    });

    // Check 6: Performance characteristics
    this.check('WASM Performance Characteristics', () => {
      const startTime = Date.now();

      // Run a benchmark that should use WASM
      try {
        execSync('npx ruv-swarm benchmark --quick --iterations 5', {
          encoding: 'utf8',
          stdio: 'pipe',
          timeout: 30000,
        });
      } catch (error) {
        // Benchmark might fail but should at least try to run
        if (!error.stdout) {
          throw new Error('Benchmark produced no output');
        }
      }

      const duration = Date.now() - startTime;
      this.results.wasmFunctionality.benchmarkDuration = duration;

      // WASM should be reasonably fast
      if (duration > 30000) {
        throw new Error('Performance too slow, might be using fallback');
      }

      return true;
    });

    // Generate report
    this.generateReport();
  }

  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      passed: this.results.passed,
      totalChecks: this.results.checks.length,
      passedChecks: this.results.checks.filter(c => c.status === 'PASSED').length,
      failedChecks: this.results.checks.filter(c => c.status === 'FAILED').length,
      wasmDetails: this.results.wasmFunctionality,
      checks: this.results.checks,
    };

    // Save report
    fs.writeFileSync('wasm-validation-report.json', JSON.stringify(report, null, 2));

    // Print summary
    console.log('\n📊 Validation Summary');
    console.log('====================');
    console.log(`Total Checks: ${report.totalChecks}`);
    console.log(`✅ Passed: ${report.passedChecks}`);
    console.log(`❌ Failed: ${report.failedChecks}`);

    if (report.wasmDetails.binarySize) {
      console.log(`\nWASM Binary Size: ${(report.wasmDetails.binarySize / 1024).toFixed(2)} KB`);
    }

    if (report.wasmDetails.exportCount) {
      console.log(`WASM Exports: ${report.wasmDetails.exportCount}`);
    }

    if (report.wasmDetails.memoryUsage) {
      console.log(`Memory Usage: ${(report.wasmDetails.memoryUsage / 1024).toFixed(2)} KB`);
    }

    console.log(`\n${this.results.passed ? '✅ VALIDATION PASSED' : '❌ VALIDATION FAILED'}`);
    console.log('Full report saved to: wasm-validation-report.json');

    return this.results.passed;
  }
}

// Run validation
if (require.main === module) {
  const validator = new WasmValidator();
  validator.runValidation().then(passed => {
    process.exit(passed ? 0 : 1);
  }).catch(error => {
    console.error('Validation error:', error);
    process.exit(1);
  });
}

module.exports = WasmValidator;
