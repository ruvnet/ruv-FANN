/**
 * Vitest Configuration for ruv-swarm
 * Alternative test runner configuration for ES modules
 */

import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Test environment
    environment: 'node',

    // Test file patterns
    include: [
      'npm/test/**/*.test.js',
      'npm/test/**/*.spec.js',
      'tests/**/*.test.js',
      'test/**/*.test.js',
    ],

    // Exclude patterns
    exclude: [
      'node_modules/**',
      'target/**',
      'wasm/**',
      'examples/**',
      'scripts/**',
    ],

    // Global test setup
    setupFiles: [
      'npm/test/setup/vitest.setup.js',
    ],

    // Test timeout
    testTimeout: 30000,
    hookTimeout: 10000,

    // Coverage configuration
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      reportsDirectory: 'coverage',
      include: [
        'npm/src/**/*.js',
      ],
      exclude: [
        'npm/src/**/*.test.js',
        'npm/src/**/*.spec.js',
        'npm/wasm/**',
        'npm/examples/**',
        'npm/scripts/**',
        'npm/bin/**',
      ],
      thresholds: {
        lines: 80,
        functions: 75,
        branches: 70,
        statements: 80,
      },
    },

    // Parallel execution
    pool: 'threads',
    poolOptions: {
      threads: {
        maxThreads: 4,
        minThreads: 1,
      },
    },

    // Reporter configuration
    reporter: ['verbose', 'json', 'html'],
    outputFile: {
      json: './coverage/vitest-report.json',
      html: './coverage/vitest-report.html',
    },

    // Global variables
    globals: true,

    // Watch mode
    watch: false,

    // Isolation
    isolate: true,

    // Mock configuration
    clearMocks: true,
    restoreMocks: true,
    mockReset: true,

    // Dependencies handling
    deps: {
      external: ['better-sqlite3', 'worker_threads'],
    },
  },

  // Resolve configuration
  resolve: {
    alias: {
      '@': 'npm/src',
      '@test': 'npm/test',
    },
  },

  // Define globals
  define: {
    __TEST__: true,
    __VERSION__: '"0.2.1"',
  },
});
