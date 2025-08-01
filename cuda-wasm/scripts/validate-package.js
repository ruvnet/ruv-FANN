#!/usr/bin/env node

/**
 * Package validation script for @cuda-wasm/core
 * Validates the package is ready for NPM publication
 */

const fs = require('fs');
const path = require('path');

// Colors for output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function log(message, color = 'reset') {
  console.log(colors[color] + message + colors.reset);
}

class PackageValidator {
  constructor() {
    this.errors = [];
    this.warnings = [];
    this.checks = [];
    this.basePath = process.cwd();
  }

  addError(message) {
    this.errors.push(message);
    log(`❌ ERROR: ${message}`, 'red');
  }

  addWarning(message) {
    this.warnings.push(message);
    log(`⚠️  WARNING: ${message}`, 'yellow');
  }

  addSuccess(message) {
    log(`✅ ${message}`, 'green');
  }

  fileExists(filePath) {
    const fullPath = path.join(this.basePath, filePath);
    return fs.existsSync(fullPath);
  }

  readJsonFile(filePath) {
    try {
      const fullPath = path.join(this.basePath, filePath);
      const content = fs.readFileSync(fullPath, 'utf8');
      return JSON.parse(content);
    } catch (error) {
      return null;
    }
  }

  validatePackageJson() {
    log('\n📦 Validating package.json...', 'cyan');

    const pkg = this.readJsonFile('package.json');
    if (!pkg) {
      this.addError('package.json not found or invalid');
      return;
    }

    // Required fields
    const requiredFields = ['name', 'version', 'description', 'main', 'types', 'license'];
    for (const field of requiredFields) {
      if (!pkg[field]) {
        this.addError(`package.json missing required field: ${field}`);
      }
    }

    // Validate scoped package name
    if (pkg.name !== '@cuda-wasm/core') {
      this.addError(`Expected name '@cuda-wasm/core', got '${pkg.name}'`);
    } else {
      this.addSuccess('Package name is correctly scoped');
    }

    // Validate version format
    if (!/^\d+\.\d+\.\d+/.test(pkg.version)) {
      this.addError('Version should follow semantic versioning');
    } else {
      this.addSuccess(`Version ${pkg.version} follows semver`);
    }

    // Check for proper exports field
    if (!pkg.exports) {
      this.addWarning('package.json should include exports field for modern Node.js');
    } else {
      this.addSuccess('Package has proper exports configuration');
    }

    // Check engines
    if (!pkg.engines || !pkg.engines.node) {
      this.addWarning('Should specify Node.js engine requirements');
    } else {
      this.addSuccess(`Node.js engine requirement: ${pkg.engines.node}`);
    }

    // Check scripts
    const requiredScripts = ['build', 'test', 'prepublishOnly'];
    for (const script of requiredScripts) {
      if (!pkg.scripts || !pkg.scripts[script]) {
        this.addWarning(`Missing recommended script: ${script}`);
      } else {
        this.addSuccess(`Script '${script}' is defined`);
      }
    }

    // Check publishConfig
    if (!pkg.publishConfig) {
      this.addWarning('Missing publishConfig - needed for scoped packages');
    } else if (pkg.publishConfig.access !== 'public') {
      this.addError('publishConfig.access should be "public" for scoped package');
    } else {
      this.addSuccess('publishConfig is correctly set for public scoped package');
    }
  }

  validateFiles() {
    log('\n📁 Validating required files...', 'cyan');

    const requiredFiles = [
      'LICENSE',
      'CHANGELOG.md',
      'dist/index.js',
      'dist/index.d.ts',
      'dist/index.esm.js',
      'dist/index.browser.js',
      'cli/index.js',
    ];

    for (const file of requiredFiles) {
      if (this.fileExists(file)) {
        this.addSuccess(`Required file exists: ${file}`);
      } else {
        this.addError(`Missing required file: ${file}`);
      }
    }

    // Check for files that shouldn't be included
    const excludedFiles = [
      'Cargo.toml',
      'Cargo.lock',
      'src/',
      'target/',
      'tests/',
      '.git/',
    ];

    for (const file of excludedFiles) {
      if (this.fileExists(file)) {
        this.addWarning(`Development file present (should be excluded): ${file}`);
      }
    }
  }

  validateTypeScriptDefinitions() {
    log('\n📝 Validating TypeScript definitions...', 'cyan');

    if (!this.fileExists('dist/index.d.ts')) {
      this.addError('TypeScript definitions file missing');
      return;
    }

    try {
      const dtsContent = fs.readFileSync(path.join(this.basePath, 'dist/index.d.ts'), 'utf8');

      const requiredExports = [
        'transpileCuda',
        'analyzeKernel',
        'benchmark',
        'createWebGPUKernel',
        'TranspileOptions',
        'TranspileResult',
        'KernelAnalysis',
      ];

      for (const exportName of requiredExports) {
        if (dtsContent.includes(exportName)) {
          this.addSuccess(`TypeScript export found: ${exportName}`);
        } else {
          this.addError(`Missing TypeScript export: ${exportName}`);
        }
      }
    } catch (error) {
      this.addError(`Could not read TypeScript definitions: ${error.message}`);
    }
  }

  validateNpmIgnore() {
    log('\n🙈 Validating .npmignore...', 'cyan');

    if (!this.fileExists('.npmignore')) {
      this.addWarning('.npmignore file missing - all files will be included');
      return;
    }

    try {
      const npmignoreContent = fs.readFileSync(path.join(this.basePath, '.npmignore'), 'utf8');

      const shouldExclude = ['src/', 'target/', 'tests/', 'Cargo.toml', '.git/'];
      for (const pattern of shouldExclude) {
        if (npmignoreContent.includes(pattern)) {
          this.addSuccess(`Excludes development files: ${pattern}`);
        } else {
          this.addWarning(`Should exclude: ${pattern}`);
        }
      }
    } catch (error) {
      this.addError(`Could not read .npmignore: ${error.message}`);
    }
  }

  validateDistDirectory() {
    log('\n📦 Validating dist/ directory...', 'cyan');

    if (!this.fileExists('dist/')) {
      this.addError('dist/ directory missing - run npm run build');
      return;
    }

    const distFiles = [
      'index.js',
      'index.d.ts',
      'index.esm.js',
      'index.browser.js',
    ];

    for (const file of distFiles) {
      const filePath = `dist/${file}`;
      if (this.fileExists(filePath)) {
        this.addSuccess(`Dist file exists: ${file}`);

        // Check file size
        const stats = fs.statSync(path.join(this.basePath, filePath));
        if (stats.size === 0) {
          this.addError(`Dist file is empty: ${file}`);
        } else if (stats.size > 5 * 1024 * 1024) { // 5MB
          this.addWarning(`Dist file is large (${Math.round(stats.size / 1024 / 1024)}MB): ${file}`);
        }
      } else {
        this.addError(`Missing dist file: ${file}`);
      }
    }
  }

  validateCLI() {
    log('\n⌨️  Validating CLI...', 'cyan');

    if (!this.fileExists('cli/index.js')) {
      this.addError('CLI entry point missing');
      return;
    }

    try {
      const cliContent = fs.readFileSync(path.join(this.basePath, 'cli/index.js'), 'utf8');

      if (cliContent.startsWith('#!/usr/bin/env node')) {
        this.addSuccess('CLI has proper shebang');
      } else {
        this.addError('CLI missing shebang line');
      }

      if (cliContent.includes('require(\'../dist\')')) {
        this.addSuccess('CLI imports from dist/');
      } else {
        this.addWarning('CLI should import from dist/ directory');
      }
    } catch (error) {
      this.addError(`Could not read CLI file: ${error.message}`);
    }
  }

  validateLicense() {
    log('\n📄 Validating license...', 'cyan');

    if (!this.fileExists('LICENSE')) {
      this.addError('LICENSE file missing');
      return;
    }

    try {
      const licenseContent = fs.readFileSync(path.join(this.basePath, 'LICENSE'), 'utf8');

      if (licenseContent.includes('MIT License')) {
        this.addSuccess('MIT License detected');
      } else {
        this.addWarning('License type unclear - should be MIT');
      }

      if (licenseContent.includes('2024')) {
        this.addSuccess('License year is current');
      } else {
        this.addWarning('License year should be updated');
      }
    } catch (error) {
      this.addError(`Could not read LICENSE file: ${error.message}`);
    }
  }

  validateChangelog() {
    log('\n📋 Validating changelog...', 'cyan');

    if (!this.fileExists('CHANGELOG.md')) {
      this.addWarning('CHANGELOG.md missing - recommended for releases');
      return;
    }

    try {
      const changelogContent = fs.readFileSync(path.join(this.basePath, 'CHANGELOG.md'), 'utf8');

      if (changelogContent.includes('[1.0.0]')) {
        this.addSuccess('Changelog includes version 1.0.0');
      } else {
        this.addWarning('Changelog should include current version');
      }

      if (changelogContent.includes('## [Unreleased]') || changelogContent.includes('## [1.0.0]')) {
        this.addSuccess('Changelog follows conventional format');
      } else {
        this.addWarning('Changelog should follow conventional format');
      }
    } catch (error) {
      this.addError(`Could not read CHANGELOG.md: ${error.message}`);
    }
  }

  async run() {
    log('🔍 @cuda-wasm/core Package Validation\n', 'cyan');

    this.validatePackageJson();
    this.validateFiles();
    this.validateTypeScriptDefinitions();
    this.validateNpmIgnore();
    this.validateDistDirectory();
    this.validateCLI();
    this.validateLicense();
    this.validateChangelog();

    // Summary
    log('\n📊 Validation Summary:', 'cyan');
    log(`   ✅ Checks passed: ${this.checks.length}`, 'green');
    log(`   ⚠️  Warnings: ${this.warnings.length}`, this.warnings.length > 0 ? 'yellow' : 'green');
    log(`   ❌ Errors: ${this.errors.length}`, this.errors.length > 0 ? 'red' : 'green');

    if (this.errors.length === 0 && this.warnings.length === 0) {
      log('\n🎉 Package is ready for publication!', 'green');
      log('\nNext steps:', 'cyan');
      log('1. npm run build', 'blue');
      log('2. npm test', 'blue');
      log('3. npm run test:package', 'blue');
      log('4. npm publish', 'blue');
      return true;
    } else if (this.errors.length === 0) {
      log('\n⚠️  Package has warnings but is ready for publication', 'yellow');
      log('\nConsider addressing warnings before publishing:', 'yellow');
      this.warnings.forEach(warning => log(`   • ${warning}`, 'yellow'));
      return true;
    } else {
      log('\n💥 Package has errors and is NOT ready for publication', 'red');
      log('\nPlease fix these errors:', 'red');
      this.errors.forEach(error => log(`   • ${error}`, 'red'));
      return false;
    }
  }
}

// Run validation if this file is executed directly
if (require.main === module) {
  const validator = new PackageValidator();
  validator.run().then(success => {
    process.exit(success ? 0 : 1);
  }).catch(error => {
    log(`\n💥 Validation failed: ${error.message}`, 'red');
    process.exit(1);
  });
}

module.exports = PackageValidator;
