module.exports = {
  testEnvironment: 'node',
  testMatch: [
    '**/__tests__/**/*.test.js',
    '**/?(*.)+(spec|test).js',
  ],
  collectCoverageFrom: [
    'dist/**/*.js',
    'cli/**/*.js',
    '!dist/**/*.test.js',
    '!dist/**/*.spec.js',
    '!**/node_modules/**',
  ],
  coverageDirectory: 'coverage',
  coverageReporters: [
    'text',
    'lcov',
    'html',
  ],
  setupFilesAfterEnv: [
    '<rootDir>/tests/setup.js',
  ],
  testTimeout: 30000,
  verbose: true,
  forceExit: true,
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true,
};
