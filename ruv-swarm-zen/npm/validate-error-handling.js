#!/usr/bin/env node

/**
 * Quick Error Handling System Validation
 * Validates that all components are working correctly
 */

async function validateErrorHandling() {
  console.log('🔍 Validating Error Handling System...\n');

  let passed = 0;
  let failed = 0;

  // Test 1: Error Classes
  try {
    const { ValidationError, SwarmError, ErrorFactory } = await import('./src/errors.js');

    // Test ValidationError
    const validationError = new ValidationError('Test error', 'field', 'value', 'string');
    if (validationError.name !== 'ValidationError') throw new Error('ValidationError name incorrect');
    if (validationError.getSuggestions().length === 0) throw new Error('No suggestions generated');

    // Test SwarmError
    const swarmError = new SwarmError('Swarm failed', 'test-id', 'init');
    if (swarmError.swarmId !== 'test-id') throw new Error('SwarmError context not preserved');

    // Test ErrorFactory
    const factoryError = ErrorFactory.createError('agent', 'Test agent error', { agentId: 'test' });
    if (factoryError.name !== 'AgentError') throw new Error('ErrorFactory not creating correct type');

    console.log('✅ Error Classes: All tests passed');
    passed++;
  } catch (error) {
    console.log('❌ Error Classes:', error.message);
    failed++;
  }

  // Test 2: Validation System
  try {
    const { ValidationUtils } = await import('./src/schemas.js');

    // Test valid parameters
    const validResult = ValidationUtils.validateParams({ topology: 'mesh' }, 'swarm_init');
    if (validResult.topology !== 'mesh') throw new Error('Valid param validation failed');

    // Test invalid parameters
    let errorCaught = false;
    try {
      ValidationUtils.validateParams({ topology: 'invalid' }, 'swarm_init');
    } catch (error) {
      errorCaught = true;
    }
    if (!errorCaught) throw new Error('Invalid param validation should have failed');

    // Test schema count
    const schemas = ValidationUtils.getAllSchemas();
    if (schemas.length < 25) throw new Error('Not enough tool schemas defined');

    console.log('✅ Validation System: All tests passed');
    passed++;
  } catch (error) {
    console.log('❌ Validation System:', error.message);
    failed++;
  }

  // Test 3: Enhanced MCP Tools Integration
  try {
    const { EnhancedMCPTools } = await import('./src/mcp-tools-enhanced.js');

    const tools = new EnhancedMCPTools();

    // Test error context
    if (!tools.errorContext) throw new Error('Error context not initialized');
    if (!Array.isArray(tools.errorLog)) throw new Error('Error log not initialized');

    // Test error handling method
    const testError = new Error('Test error');
    const handledError = tools.handleError(testError, 'test_tool', 'test_operation');
    if (tools.errorLog.length === 0) throw new Error('Error not logged');

    // Test parameter validation
    try {
      tools.validateToolParams({ topology: 'invalid' }, 'swarm_init');
      throw new Error('Should have thrown validation error');
    } catch (validationError) {
      if (validationError.name !== 'ValidationError') {
        throw new Error('Wrong error type thrown');
      }
    }

    console.log('✅ Enhanced MCP Tools: All tests passed');
    passed++;
  } catch (error) {
    console.log('❌ Enhanced MCP Tools:', error.message);
    failed++;
  }

  // Test 4: File Integrity
  try {
    const fs = await import('fs');
    const path = await import('path');

    const requiredFiles = [
      './src/errors.js',
      './src/schemas.js',
      './test/error-handling-validation.test.js',
      './test/run-error-handling-tests.js',
    ];

    for (const file of requiredFiles) {
      if (!fs.existsSync(file)) {
        throw new Error(`Required file missing: ${file}`);
      }
    }

    console.log('✅ File Integrity: All required files present');
    passed++;
  } catch (error) {
    console.log('❌ File Integrity:', error.message);
    failed++;
  }

  // Final Report
  console.log('\n' + '='.repeat(50));
  console.log('🧪 ERROR HANDLING VALIDATION COMPLETE');
  console.log('='.repeat(50));
  console.log(`✅ Passed: ${passed}`);
  console.log(`❌ Failed: ${failed}`);
  console.log(`📊 Success Rate: ${((passed / (passed + failed)) * 100).toFixed(1)}%`);

  if (failed === 0) {
    console.log('\n🎉 All validations passed! Error handling system is ready for use.');
    console.log('\n📚 Quick Start:');
    console.log('   import { EnhancedMCPTools } from "./src/mcp-tools-enhanced.js";');
    console.log('   const tools = new EnhancedMCPTools();');
    console.log('   // All MCP tools now have robust error handling!');
  } else {
    console.log('\n⚠️  Some validations failed. Please check the error messages above.');
    process.exit(1);
  }
}

validateErrorHandling().catch(console.error);
