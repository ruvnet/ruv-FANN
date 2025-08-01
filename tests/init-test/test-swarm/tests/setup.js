// Test setup
process.env.NODE_ENV = 'test';
process.env.JWT_SECRET = 'test-secret-key';
process.env.DATABASE_URL = ':memory:';
process.env.LOG_LEVEL = 'error';

// Suppress console logs during tests
if (process.env.NODE_ENV === 'test') {
  console.log = jest.fn();
  console.error = jest.fn();
  console.warn = jest.fn();
}
