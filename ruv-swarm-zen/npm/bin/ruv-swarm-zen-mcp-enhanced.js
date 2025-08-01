#!/usr/bin/env node
/**
 * Enhanced MCP Server Entry Point for Issue #91
 * Comprehensive fix for MCP server timeout and JSON parsing issues
 *
 * Fixes Applied:
 * 1. ANSI escape code elimination from stdout
 * 2. Proper notifications/initialized handling
 * 3. Enhanced error handling and connection management
 * 4. JSON-RPC compliant logging
 * 5. Timeout issue resolution
 *
 * Version: 2.0.0 - Production Grade
 * Author: Claude Code Assistant
 * License: MIT
 */

import { EnhancedMCPServer } from '../src/mcp-server-enhanced.js';
import { EnhancedMCPTools } from '../src/mcp-tools-enhanced.js';
import { daaMcpTools } from '../src/mcp-daa-tools.js';
import { RuvSwarm } from '../src/index-enhanced.js';

/**
 * Initialize the system and start enhanced MCP server
 */
async function initializeAndStartServer() {
  try {
    // Initialize WASM and core systems
    const ruvSwarm = await RuvSwarm.initialize({
      loadingStrategy: 'progressive',
      enablePersistence: false, // Disable for MCP server
      enableNeuralNetworks: true,
      enableCognitiveDiversity: true,
      enableForecasting: false,
      useSIMD: false,
      debug: process.env.DEBUG === 'true',
    });

    // Initialize MCP tools
    const mcpTools = new EnhancedMCPTools();
    await mcpTools.initialize();

    // Add DAA tools
    mcpTools.daaMcpTools = daaMcpTools;

    // Create and start enhanced MCP server
    const server = new EnhancedMCPServer(mcpTools, {
      logLevel: process.env.LOG_LEVEL || 'INFO',
      sessionId: process.env.MCP_SESSION_ID,
    });

    // Start the server
    await server.start();

    return server;

  } catch (error) {
    // Use plain console.error to avoid any formatting issues
    console.error(`[${new Date().toISOString()}] FATAL Enhanced MCP server failed to start: ${error.message}`);
    console.error(error.stack);
    process.exit(1);
  }
}

/**
 * Main execution
 */
async function main() {
  // Check if this is being called as the main script
  if (process.argv[1] === import.meta.url.replace('file://', '')) {
    await initializeAndStartServer();
  }
}

// Handle uncaught errors at the top level
process.on('uncaughtException', (error) => {
  console.error(`[${new Date().toISOString()}] UNCAUGHT_EXCEPTION ${error.message}`);
  console.error(error.stack);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error(`[${new Date().toISOString()}] UNHANDLED_REJECTION ${reason}`);
  console.error(promise);
  process.exit(1);
});

// Start the server
main().catch(error => {
  console.error(`[${new Date().toISOString()}] MAIN_ERROR ${error.message}`);
  console.error(error.stack);
  process.exit(1);
});

export { initializeAndStartServer };
