/**
 * Test Suite for Singleton Container - IoC Pattern
 * Tests memory safety, concurrent access, and proper lifecycle management
 */

import { SingletonContainer, getContainer, resetContainer } from '../src/singleton-container.js';

// Test class for dependency injection
class TestService {
  constructor(name = 'test') {
    this.name = name;
    this.isDestroyed = false;
    this.createdAt = Date.now();
  }

  destroy() {
    this.isDestroyed = true;
  }
}

class DependentService {
  constructor(testService) {
    this.testService = testService;
    this.isDestroyed = false;
  }

  destroy() {
    this.isDestroyed = true;
  }
}

async function runTests() {
  console.log('🧪 Running Singleton Container Tests...\n');

  let testsPassed = 0;
  let testsTotal = 0;

  function test(name, testFn) {
    testsTotal++;
    try {
      testFn();
      console.log(`✅ ${name}`);
      testsPassed++;
    } catch (error) {
      console.error(`❌ ${name}: ${error.message}`);
    }
  }

  function asyncTest(name, testFn) {
    testsTotal++;
    return testFn()
      .then(() => {
        console.log(`✅ ${name}`);
        testsPassed++;
      })
      .catch(error => {
        console.error(`❌ ${name}: ${error.message}`);
      });
  }

  // Basic Container Tests
  test('Container creation and registration', () => {
    const container = new SingletonContainer();
    container.register('test', () => new TestService('basic'));

    if (!container.has('test')) {
      throw new Error('Service not registered');
    }

    container.destroy();
  });

  test('Singleton instance creation', () => {
    const container = new SingletonContainer();
    container.register('test', () => new TestService('singleton'));

    const instance1 = container.get('test');
    const instance2 = container.get('test');

    if (instance1 !== instance2) {
      throw new Error('Singleton instances are different');
    }

    if (instance1.name !== 'singleton') {
      throw new Error('Instance not created correctly');
    }

    container.destroy();
  });

  test('Dependency injection', () => {
    const container = new SingletonContainer();

    container.register('testService', () => new TestService('dependency'));
    container.register('dependent', (testService) => new DependentService(testService), {
      dependencies: ['testService'],
    });

    const dependent = container.get('dependent');

    if (!dependent.testService || dependent.testService.name !== 'dependency') {
      throw new Error('Dependency not injected correctly');
    }

    container.destroy();
  });

  test('Error handling for missing factory', () => {
    const container = new SingletonContainer();

    try {
      container.get('nonexistent');
      throw new Error('Should have thrown error for missing factory');
    } catch (error) {
      if (!error.message.includes('No factory registered')) {
        throw new Error('Wrong error message: ' + error.message);
      }
    }

    container.destroy();
  });

  test('Instance cleanup and destruction', () => {
    const container = new SingletonContainer();
    container.register('test', () => new TestService('cleanup'));

    const instance = container.get('test');

    if (instance.isDestroyed) {
      throw new Error('Instance should not be destroyed yet');
    }

    container.destroy();

    if (!instance.isDestroyed) {
      throw new Error('Instance should be destroyed after container destruction');
    }
  });

  test('Global container management', () => {
    resetContainer(); // Ensure clean state

    const container1 = getContainer();
    const container2 = getContainer();

    if (container1 !== container2) {
      throw new Error('Global container should return same instance');
    }

    resetContainer();
  });

  test('Memory leak prevention', () => {
    const container = new SingletonContainer();

    // Create many instances
    for (let i = 0; i < 100; i++) {
      container.register(`test${i}`, () => new TestService(`test${i}`));
      container.get(`test${i}`);
    }

    const stats = container.getStats();
    if (stats.registeredServices !== 100 || stats.activeInstances !== 100) {
      throw new Error(`Expected 100 services and instances, got ${stats.registeredServices}/${stats.activeInstances}`);
    }

    container.destroy();

    const statsAfter = container.getStats();
    if (statsAfter.activeInstances !== 0) {
      throw new Error(`Memory leak detected: ${statsAfter.activeInstances} instances remaining`);
    }
  });

  // Concurrent Access Tests
  await asyncTest('Concurrent instance creation', async () => {
    const container = new SingletonContainer();
    container.register('concurrent', () => new TestService('concurrent'));

    // Simulate concurrent access
    const promises = Array.from({ length: 10 }, () =>
      Promise.resolve(container.get('concurrent')),
    );

    const instances = await Promise.all(promises);

    // All should be the same instance
    const firstInstance = instances[0];
    for (const instance of instances) {
      if (instance !== firstInstance) {
        throw new Error('Concurrent access created different instances');
      }
    }

    container.destroy();
  });

  // Container State Tests
  test('Container destruction prevention', () => {
    const container = new SingletonContainer();
    container.register('test', () => new TestService('destruction'));

    container.destroy();

    try {
      container.get('test');
      throw new Error('Should not allow instance creation during destruction');
    } catch (error) {
      if (!error.message.includes('during container destruction')) {
        throw new Error('Wrong error for destruction state: ' + error.message);
      }
    }

    // Test reset functionality
    container.reset();
    container.register('test2', () => new TestService('after-reset'));
    const instance = container.get('test2');

    if (!instance || instance.name !== 'after-reset') {
      throw new Error('Container should work after reset');
    }

    container.destroy();
  });

  test('Non-singleton instances', () => {
    const container = new SingletonContainer();
    container.register('nonSingleton', () => new TestService('non-singleton'), {
      singleton: false,
    });

    const instance1 = container.get('nonSingleton');
    const instance2 = container.get('nonSingleton');

    if (instance1 === instance2) {
      throw new Error('Non-singleton instances should be different');
    }

    container.destroy();
  });

  test('Factory error handling', () => {
    const container = new SingletonContainer();
    container.register('failing', () => {
      throw new Error('Factory failure');
    });

    try {
      container.get('failing');
      throw new Error('Should have thrown factory error');
    } catch (error) {
      if (!error.message.includes('Failed to create instance')) {
        throw new Error('Wrong error handling: ' + error.message);
      }
    }

    container.destroy();
  });

  // Performance Tests
  await asyncTest('Performance under load', async () => {
    const container = new SingletonContainer();
    container.register('performance', () => new TestService('performance'));

    const startTime = Date.now();

    // Get instance 1000 times (should be fast due to caching)
    for (let i = 0; i < 1000; i++) {
      container.get('performance');
    }

    const endTime = Date.now();
    const duration = endTime - startTime;

    if (duration > 100) { // Should be very fast
      throw new Error(`Performance test too slow: ${duration}ms`);
    }

    container.destroy();
  });

  // Global State Replacement Tests
  test('Global state replacement validation', () => {
    resetContainer();

    const container = getContainer();
    container.register('RuvSwarm', () => ({
      initialized: true,
      id: 'test-instance',
      destroy: () => {},
    }));

    const instance1 = container.get('RuvSwarm');
    const instance2 = container.get('RuvSwarm');

    if (instance1 !== instance2) {
      throw new Error('RuvSwarm instances should be singleton');
    }

    if (!instance1.initialized) {
      throw new Error('RuvSwarm instance not properly configured');
    }

    resetContainer();
  });

  // Final Results
  console.log(`\n📊 Test Results: ${testsPassed}/${testsTotal} passed`);

  if (testsPassed === testsTotal) {
    console.log('🎉 All tests passed! Singleton Container is ready for production.');
    return true;
  } else {
    console.log('❌ Some tests failed. Review implementation before deployment.');
    return false;
  }
}

// Run tests if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTests().then(success => {
    process.exit(success ? 0 : 1);
  }).catch(error => {
    console.error('Test runner error:', error);
    process.exit(1);
  });
}

export { runTests };
