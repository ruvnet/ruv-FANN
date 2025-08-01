/**
 * Example: TDD London Style WASM Testing
 * Shows how to use inline mocks instead of separate mock files
 */

import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import {
  createMockWasmModule,
  createMockWasmLoader,
  createWasmModuleSpy,
  WasmModuleTestDouble,
} from '../helpers/wasm-test-helpers.js';

describe('TDD London Style WASM Testing Examples', () => {
  describe('Example 1: Simple Mock Usage', () => {
    it('should use inline WASM mock for basic operations', async () => {
      // Arrange - Create inline mock
      const mockWasmModule = createMockWasmModule({
        createSwarm: jest.fn().mockReturnValue(42),
        getVersion: jest.fn().mockReturnValue('2.0.0'),
      });

      // Act
      const swarmId = mockWasmModule.createSwarm('mesh', { maxAgents: 10 });
      const version = mockWasmModule.getVersion();

      // Assert - Behavior verification
      expect(mockWasmModule.createSwarm).toHaveBeenCalledWith('mesh', { maxAgents: 10 });
      expect(swarmId).toBe(42);
      expect(version).toBe('2.0.0');
    });
  });

  describe('Example 2: Mock Loader for Integration Tests', () => {
    it('should mock WASM loader for testing loading behavior', async () => {
      // Arrange
      const mockLoader = createMockWasmLoader({
        getVersion: jest.fn().mockReturnValue('custom-version'),
      });

      // Act
      const module = await mockLoader.loadModule('core');
      const version = module.exports.getVersion();

      // Assert
      expect(mockLoader.loadModule).toHaveBeenCalledWith('core');
      expect(version).toBe('custom-version');
      expect(mockLoader.isLoaded()).toBe(true);
    });
  });

  describe('Example 3: Behavior Verification with Spy', () => {
    it('should verify interaction sequence using spy pattern', async () => {
      // Arrange
      const wasmSpy = createWasmModuleSpy();
      const module = wasmSpy.module;

      // Act
      await module.init();
      const swarmId = module.createSwarm('hierarchical');
      module.addAgent(swarmId, 'researcher');
      module.assignTask(swarmId, 1, { type: 'analyze' });
      const state = module.getState(swarmId);
      module.destroy(swarmId);

      // Assert - Verify interaction sequence
      expect(wasmSpy.verifyInteractionSequence([
        'init',
        'createSwarm',
        'addAgent',
        'assignTask',
        'getState',
        'destroy',
      ])).toBe(true);

      // Verify no unexpected interactions
      expect(wasmSpy.verifyNoUnexpectedInteractions([
        'init', 'createSwarm', 'addAgent', 'assignTask', 'getState', 'destroy',
      ])).toBe(true);

      // Check specific interactions
      const interactions = wasmSpy.getInteractions();
      expect(interactions[1].args).toEqual(['hierarchical']);
      expect(interactions[2].args).toEqual([swarmId, 'researcher']);
    });
  });

  describe('Example 4: Test Double with Expectations', () => {
    it('should use test double for strict behavior verification', () => {
      // Arrange
      const testDouble = new WasmModuleTestDouble();

      // Set expectations
      testDouble.expect('createSwarm', 'mesh', { maxAgents: 5 }).returns(1);
      testDouble.expect('addAgent', 1, 'coder').returns(2);
      testDouble.expect('assignTask', 1, 2, { type: 'implement' });

      const module = testDouble.createModule();

      // Act
      const swarmId = module.createSwarm('mesh', { maxAgents: 5 });
      const agentId = module.addAgent(swarmId, 'coder');
      module.assignTask(swarmId, agentId, { type: 'implement' });

      // Assert - Will throw if expectations not met
      testDouble.verify();
    });
  });

  describe('Example 5: Mocking WASM in Component Tests', () => {
    // Example component that uses WASM
    class SwarmManager {
      constructor(wasmModule) {
        this.wasm = wasmModule;
        this.swarms = new Map();
      }

      async createSwarm(topology, options) {
        await this.wasm.init();
        const id = this.wasm.createSwarm(topology, options);
        this.swarms.set(id, { topology, options, agents: [] });
        return id;
      }

      addAgentToSwarm(swarmId, agentType) {
        const agentId = this.wasm.addAgent(swarmId, agentType);
        const swarm = this.swarms.get(swarmId);
        if (swarm) {
          swarm.agents.push({ id: agentId, type: agentType });
        }
        return agentId;
      }

      getSwarmInfo(swarmId) {
        const wasmState = this.wasm.getState(swarmId);
        const localInfo = this.swarms.get(swarmId);
        return { ...wasmState, ...localInfo };
      }
    }

    it('should test SwarmManager with mocked WASM', async () => {
      // Arrange - Inline mock for WASM dependency
      const mockWasm = createMockWasmModule({
        createSwarm: jest.fn().mockReturnValue(123),
        addAgent: jest.fn().mockReturnValue(456),
        getState: jest.fn().mockReturnValue({
          topology: 'mesh',
          metrics: { totalTasks: 0 },
        }),
      });

      const manager = new SwarmManager(mockWasm);

      // Act
      const swarmId = await manager.createSwarm('mesh', { maxAgents: 3 });
      const agentId = manager.addAgentToSwarm(swarmId, 'analyst');
      const info = manager.getSwarmInfo(swarmId);

      // Assert - Behavior verification
      expect(mockWasm.init).toHaveBeenCalled();
      expect(mockWasm.createSwarm).toHaveBeenCalledWith('mesh', { maxAgents: 3 });
      expect(mockWasm.addAgent).toHaveBeenCalledWith(123, 'analyst');
      expect(info).toMatchObject({
        topology: 'mesh',
        options: { maxAgents: 3 },
        agents: [{ id: 456, type: 'analyst' }],
      });
    });
  });

  describe('Example 6: Error Handling with Mocks', () => {
    it('should handle WASM initialization errors', async () => {
      // Arrange
      const mockWasm = createMockWasmModule({
        init: jest.fn().mockRejectedValue(new Error('WASM init failed')),
      });

      // Act & Assert
      await expect(mockWasm.init()).rejects.toThrow('WASM init failed');
      expect(mockWasm.init).toHaveBeenCalled();
    });
  });
});

/**
 * Migration Guide: Converting from __mocks__ to inline mocks
 *
 * 1. Remove moduleNameMapper from jest.config.js
 * 2. Delete __mocks__/wasmMock.js
 * 3. Import mock helpers in your test files:
 *    import { createMockWasmModule } from '../helpers/wasm-test-helpers.js';
 *
 * 4. Create mocks inline in your tests:
 *    const mockWasm = createMockWasmModule({ overrides: {} });
 *
 * 5. Benefits:
 *    - Clear test dependencies
 *    - Easier to customize per test
 *    - Better behavior verification
 *    - Follows TDD London principles
 */
