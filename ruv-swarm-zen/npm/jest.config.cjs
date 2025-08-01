/**
 * Jest Configuration for ruv-swarm
 * Handles ES modules, CommonJS compatibility, and WebAssembly testing
 */

module.exports = {
  // Basic setup
  testEnvironment: 'node',
  rootDir: '.',

  // Test file patterns
  testMatch: [
    '<rootDir>/test/**/*.test.js',
    '<rootDir>/test/**/*.spec.js',
  ],

  // Transform settings for ES modules and other file types
  transform: {
    '^.+\\.m?js$': ['babel-jest', {
      presets: [
        ['@babel/preset-env', {
          targets: { node: '14' },
          modules: 'auto', // Let Babel decide based on file type
        }],
      ],
      plugins: [
        '@babel/plugin-syntax-dynamic-import',
        '@babel/plugin-proposal-optional-chaining',
        '@babel/plugin-proposal-nullish-coalescing-operator',
      ],
    }],
    '^.+\\.wasm$': '<rootDir>/jest-wasm-transformer.cjs',
  },

  // ES module handling
  globals: {
    'ts-jest': {
      useESM: true,
    },
  },

  // Module resolution
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },

  // Test setup
  setupFilesAfterEnv: [
    '<rootDir>/test/setup/jest.setup.cjs',
  ],

  // Coverage configuration
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: [
    'text',
    'lcov',
    'html',
    'json-summary',
  ],

  // Coverage thresholds
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 75,
      lines: 80,
      statements: 80,
    },
  },

  // Files to collect coverage from
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/**/*.test.js',
    '!src/**/*.spec.js',
    '!src/index.d.ts',
    '!wasm/**/*',
    '!examples/**/*',
    '!scripts/**/*',
    '!bin/**/*',
  ],

  // Test environment setup
  testEnvironmentOptions: {
    node: {
      experimental: ['wasm-modules'],
    },
  },

  // Timeout settings
  testTimeout: 30000,

  // Module paths that should be ignored
  modulePathIgnorePatterns: [
    '<rootDir>/wasm/',
    '<rootDir>/node_modules/',
    '<rootDir>/../target/',
    '<rootDir>/../crates/*/target/',
  ],

  // Files to ignore during testing
  testPathIgnorePatterns: [
    '/node_modules/',
    '/target/',
    '/wasm/',
    '/examples/',
    '/scripts/',
  ],

  // Reporters for detailed output
  reporters: [
    'default',
    ['jest-html-reporters', {
      publicPath: './coverage/html-report',
      filename: 'report.html',
      openReport: false,
      expand: true,
    }],
  ],

  // Handle WebAssembly and other binary files
  moduleFileExtensions: [
    'js',
    'mjs',
    'cjs',
    'json',
    'wasm',
  ],

  // Clear mocks between tests
  clearMocks: true,
  restoreMocks: true,
  resetMocks: true,

  // Error handling
  errorOnDeprecated: true,
  verbose: true,

  // Parallel execution
  maxWorkers: '50%',

  // Cache settings
  cache: true,
  cacheDirectory: '<rootDir>/.jest-cache',

  // Watch mode
  watchPathIgnorePatterns: [
    '/node_modules/',
    '/coverage/',
    '/target/',
    '/.git/',
  ],
};
