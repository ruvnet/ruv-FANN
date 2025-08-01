/**
 * Vitest configuration for comprehensive test suite
 * Optimized for WASM testing and high coverage targets
 */

import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    // Test environment
    environment: 'node',
    globals: true,

    // Coverage configuration
    coverage: {
      enabled: true,
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      reportsDirectory: './coverage',

      // Coverage thresholds
      thresholds: {
        statements: 90,
        branches: 90,
        functions: 90,
        lines: 90,
      },

      // Include/exclude patterns
      include: [
        'src/**/*.{js,ts}',
        'crates/*/pkg/**/*.js',
      ],
      exclude: [
        'node_modules',
        'test',
        'dist',
        'coverage',
        '**/*.d.ts',
        '**/*.test.{js,ts}',
        '**/*.spec.{js,ts}',
        '**/test-*.{js,ts}',
      ],

      // Clean coverage between runs
      clean: true,
      cleanOnRerun: true,

      // All files for accurate coverage
      all: true,
    },

    // Test execution configuration
    testTimeout: 30000, // 30 seconds for complex tests
    hookTimeout: 10000, // 10 seconds for setup/teardown

    // Parallelization
    pool: 'threads',
    poolOptions: {
      threads: {
        singleThread: false,
        maxThreads: 4,
        minThreads: 1,
      },
    },

    // Reporter configuration
    reporters: ['verbose', 'json', 'html'],
    outputFile: {
      json: './test-reports/test-results.json',
      html: './test-reports/test-results.html',
    },

    // Setup files
    setupFiles: ['./test/setup.js'],

    // Global test utilities
    globalSetup: './test/global-setup.js',

    // Retry flaky tests
    retry: 2,

    // Watch mode configuration
    watchExclude: ['**/node_modules/**', '**/dist/**', '**/coverage/**'],

    // Transform configuration for WASM
    transformMode: {
      web: [/\.[jt]sx?$/],
      ssr: [/\.wasm$/],
    },

    // Dependencies optimization
    deps: {
      inline: [
        // Inline WASM modules for testing
        /\.wasm$/,
        // Inline local dependencies
        /^(?!.*node_modules)/,
      ],
    },

    // Benchmark configuration
    benchmark: {
      outputFile: './test-reports/benchmark-results.json',
      reporters: ['verbose', 'json'],
    },
  },

  // Build configuration
  build: {
    target: 'esnext',
    lib: {
      entry: path.resolve(__dirname, '../src/index.js'),
      formats: ['es', 'cjs'],
    },
  },

  // Resolve configuration
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '../src'),
      '@wasm': path.resolve(__dirname, '../wasm'),
      '@test': path.resolve(__dirname, '.'),
    },
  },

  // Server configuration for browser tests
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },

  // Define global constants
  define: {
    __TEST__: true,
    __VITEST__: true,
  },
});
