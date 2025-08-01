/**
 * Security module for ruv-swarm
 * Provides integrity verification and security controls
 */

import crypto from 'crypto';
import fs from 'fs/promises';
import path from 'path';

/**
 * WASM module integrity verification
 */
export class WasmIntegrityVerifier {
  constructor(hashesPath = path.join(new URL('.', import.meta.url).pathname, '..', 'wasm', 'checksums.json')) {
    this.hashesPath = hashesPath;
    this.knownHashes = new Map();
    this.initialized = false;
  }

  async loadKnownHashes() {
    try {
      const data = await fs.readFile(this.hashesPath, 'utf8');
      const hashes = JSON.parse(data);
      Object.entries(hashes).forEach(([file, hash]) => {
        this.knownHashes.set(file, hash);
      });
    } catch (error) {
      // Silently initialize with empty checksums if file doesn't exist
      // This is expected on first run or if checksums.json is gitignored
    }
  }

  async saveKnownHashes() {
    try {
      // Ensure directory exists
      const dir = path.dirname(this.hashesPath);
      await fs.mkdir(dir, { recursive: true });

      const hashes = Object.fromEntries(this.knownHashes);
      await fs.writeFile(this.hashesPath, JSON.stringify(hashes, null, 2));
    } catch (error) {
      // Silently fail if unable to save checksums
      // This prevents errors when checksums.json is gitignored
    }
  }

  /**
   * Verify WASM module integrity before loading
   * @param {string} wasmPath - Path to WASM module
   * @param {boolean} updateHash - Update the known hash if verification fails
   * @returns {Promise<boolean>} - True if verification passes
   */
  async verifyWasmModule(wasmPath, updateHash = false) {
    // Ensure hashes are loaded
    if (!this.initialized) {
      await this.loadKnownHashes();
      this.initialized = true;
    }

    try {
      const wasmData = await fs.readFile(wasmPath);
      const hash = crypto.createHash('sha256').update(wasmData).digest('hex');

      const filename = path.basename(wasmPath);
      const knownHash = this.knownHashes.get(filename);

      if (!knownHash) {
        if (updateHash) {
          this.knownHashes.set(filename, hash);
          await this.saveKnownHashes();
          return true;
        }
        throw new SecurityError(`Unknown WASM module: ${filename}`);
      }

      if (hash !== knownHash) {
        throw new SecurityError(`WASM module integrity check failed for ${filename}`);
      }

      return true;
    } catch (error) {
      if (error instanceof SecurityError) {
        throw error;
      }
      throw new SecurityError(`Failed to verify WASM module: ${error.message}`);
    }
  }

  /**
   * Load and verify WASM module
   * @param {string} wasmPath - Path to WASM module
   * @returns {Promise<ArrayBuffer>} - Verified WASM data
   */
  async loadVerifiedWasm(wasmPath) {
    // Ensure hashes are loaded
    if (!this.initialized) {
      await this.loadKnownHashes();
      this.initialized = true;
    }

    await this.verifyWasmModule(wasmPath);
    return await fs.readFile(wasmPath);
  }
}

/**
 * Command injection prevention
 */
export class CommandSanitizer {
  static validateArgument(arg) {
    // Only allow alphanumeric, dash, underscore, and common safe characters
    const safePattern = /^[a-zA-Z0-9\-_./=:\s]+$/;

    if (!safePattern.test(arg)) {
      throw new SecurityError(`Invalid argument contains unsafe characters: ${arg}`);
    }

    // Check for command injection patterns
    const dangerousPatterns = [
      /[;&|`$(){}[\]<>]/, // Shell metacharacters
      /\.\./, // Path traversal
      /^-/, // Argument injection
    ];

    for (const pattern of dangerousPatterns) {
      if (pattern.test(arg)) {
        throw new SecurityError(`Potentially dangerous pattern detected in argument: ${arg}`);
      }
    }

    return arg;
  }

  static sanitizeCommand(command, args = []) {
    // Validate command is from allowed list
    const allowedCommands = ['claude', 'npm', 'node', 'npx'];

    if (!allowedCommands.includes(command)) {
      throw new SecurityError(`Command not in allowlist: ${command}`);
    }

    // Validate all arguments
    const sanitizedArgs = args.map(arg => this.validateArgument(arg));

    return { command, args: sanitizedArgs };
  }
}

/**
 * Dependency integrity verification
 */
export class DependencyVerifier {
  static async verifyNpmPackage(packageName, expectedVersion) {
    try {
      const packageJsonPath = path.join(
        new URL('.', import.meta.url).pathname,
        '..',
        'node_modules',
        packageName,
        'package.json',
      );

      const packageData = JSON.parse(await fs.readFile(packageJsonPath, 'utf8'));

      if (packageData.version !== expectedVersion) {
        throw new SecurityError(
          `Package version mismatch for ${packageName}: expected ${expectedVersion}, got ${packageData.version}`,
        );
      }

      return true;
    } catch (error) {
      if (error instanceof SecurityError) {
        throw error;
      }
      throw new SecurityError(`Failed to verify package ${packageName}: ${error.message}`);
    }
  }
}

/**
 * Security error class
 */
export class SecurityError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SecurityError';
  }
}

/**
 * Security middleware for MCP server
 */
export function createSecurityMiddleware() {
  const wasmVerifier = new WasmIntegrityVerifier();

  return {
    async verifyWasmBeforeLoad(wasmPath) {
      return await wasmVerifier.verifyWasmModule(wasmPath);
    },

    sanitizeCommand(command, args) {
      return CommandSanitizer.sanitizeCommand(command, args);
    },

    async verifyDependency(packageName, version) {
      return await DependencyVerifier.verifyNpmPackage(packageName, version);
    },
  };
}

export default {
  WasmIntegrityVerifier,
  CommandSanitizer,
  DependencyVerifier,
  SecurityError,
  createSecurityMiddleware,
};
