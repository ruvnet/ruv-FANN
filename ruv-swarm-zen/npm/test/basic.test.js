/**
 * Basic Jest test to verify the test infrastructure
 */

describe('Test Infrastructure', () => {
  test('should run basic JavaScript test', () => {
    expect(1 + 1).toBe(2);
  });

  test('should handle async operations', async() => {
    const promise = Promise.resolve('test');
    const result = await promise;
    expect(result).toBe('test');
  });

  test('should have access to Node.js globals', () => {
    expect(global).toBeDefined();
    expect(process).toBeDefined();
    expect(console).toBeDefined();
  });

  test('should work with modern JavaScript features', () => {
    const obj = { a: 1, b: 2 };
    const { a, ...rest } = obj;
    expect(a).toBe(1);
    expect(rest).toEqual({ b: 2 });
  });
});
