# Mock Migration Guide - TDD London Style

## Overview
We've migrated from separate mock files to inline mocks following TDD London principles. This provides better test isolation, clearer dependencies, and easier customization per test.

## Changes Made

### 1. Removed `__mocks__` Directory
- Deleted `/test/__mocks__/wasmMock.js`
- All mocks are now inline within test files

### 2. Updated Jest Configuration
- Removed `moduleNameMapper` for `.wasm` files from `jest.config.js`
- Fixed `setupFilesAfterEnv` path to point to correct location

### 3. Created Test Helpers
- New file: `/test/helpers/wasm-test-helpers.js`
- Provides utilities for creating inline WASM mocks

## Migration Steps for Existing Tests

### Before (using __mocks__):
```javascript
// The mock was automatically applied via Jest's moduleNameMapper
import { WasmModule } from '../src/wasm-loader.js';

describe('My Test', () => {
  it('should work with WASM', () => {
    // Mock was implicitly available
    const result = WasmModule.someMethod();
  });
});
```

### After (inline mocks):
```javascript
import { createMockWasmModule } from '../helpers/wasm-test-helpers.js';

describe('My Test', () => {
  it('should work with WASM', () => {
    // Explicit inline mock
    const mockWasm = createMockWasmModule({
      someMethod: jest.fn().mockReturnValue('custom result')
    });
    
    const result = mockWasm.someMethod();
    
    // Behavior verification
    expect(mockWasm.someMethod).toHaveBeenCalled();
  });
});
```

## Available Mock Helpers

### 1. `createMockWasmModule(overrides)`
Creates a basic WASM module mock with common methods.

### 2. `createMockWasmLoader(moduleOverrides)`
Creates a mock WASM loader for testing module loading.

### 3. `createWasmModuleSpy()`
Creates a spy for tracking and verifying interaction sequences.

### 4. `WasmModuleTestDouble`
Advanced test double with strict expectation verification.

## Benefits

1. **Explicit Dependencies**: Each test clearly shows its mocks
2. **Customization**: Easy to customize mocks per test
3. **Behavior Verification**: Focus on interactions, not state
4. **Better Isolation**: Tests don't share mock state
5. **TDD London Compliance**: Follows mock-driven development principles

## Example Test File
See `/test/examples/tdd-london-wasm-example.test.js` for comprehensive examples.

## Running Tests
```bash
npm test
# or
npm run test:unit
```

## Troubleshooting

### If tests fail after migration:
1. Check that you're importing mock helpers correctly
2. Ensure you're creating mocks before using them
3. Verify mock method names match your usage
4. Use behavior verification (toHaveBeenCalled, etc.)

### Common Issues:
- **"Cannot find module '*.wasm'"**: Create inline mock instead
- **"Mock not defined"**: Import and create mock explicitly
- **"Unexpected mock behavior"**: Check your mock overrides