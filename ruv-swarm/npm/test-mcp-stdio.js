#!/usr/bin/env node
/**
 * Test script to validate MCP server stdio functionality
 * This simulates how Claude Code would interact with the MCP server
 */

import { spawn } from 'child_process';
import { promises as fs } from 'fs';

const TEST_TIMEOUT = 30000; // 30 seconds timeout

class McpStdioTest {
    constructor() {
        this.testResults = {
            testSuite: 'mcp-server-validation',
            version: '1.0.17',
            timestamp: new Date().toISOString(),
            tests: []
        };
    }

    async runTest(name, testFunc) {
        console.log(`\n🧪 Running test: ${name}`);
        const startTime = Date.now();
        
        try {
            await testFunc();
            const duration = Date.now() - startTime;
            console.log(`✅ ${name} - PASSED (${duration}ms)`);
            this.testResults.tests.push({
                name,
                status: 'passed',
                duration: `${duration}ms`,
                timestamp: new Date().toISOString()
            });
        } catch (error) {
            const duration = Date.now() - startTime;
            console.error(`❌ ${name} - FAILED (${duration}ms): ${error.message}`);
            this.testResults.tests.push({
                name,
                status: 'failed',
                message: error.message,
                duration: `${duration}ms`,
                timestamp: new Date().toISOString()
            });
        }
    }

    async testMcpServerStart() {
        return new Promise((resolve, reject) => {
            const server = spawn('npx', ['ruv-swarm', 'mcp', 'start'], {
                stdio: ['pipe', 'pipe', 'pipe'],
                timeout: TEST_TIMEOUT
            });

            let serverOutput = '';
            let serverReady = false;
            
            server.stdout.on('data', (data) => {
                const output = data.toString();
                serverOutput += output;
                
                // Check for server initialization message
                if (output.includes('{"jsonrpc":"2.0","method":"server.initialized"')) {
                    serverReady = true;
                    server.kill();
                    resolve();
                }
            });

            server.stderr.on('data', (data) => {
                const output = data.toString();
                if (output.includes('MCP server starting in stdio mode')) {
                    // This is good - server is starting
                }
            });

            server.on('close', (code) => {
                if (serverReady) {
                    resolve();
                } else {
                    reject(new Error(`Server exited with code ${code}, output: ${serverOutput}`));
                }
            });

            server.on('error', (error) => {
                reject(new Error(`Server startup error: ${error.message}`));
            });

            // Timeout check
            setTimeout(() => {
                if (!serverReady) {
                    server.kill();
                    reject(new Error('Server startup timeout'));
                }
            }, TEST_TIMEOUT);
        });
    }

    async testMcpToolsList() {
        return new Promise((resolve, reject) => {
            const server = spawn('npx', ['ruv-swarm', 'mcp', 'start'], {
                stdio: ['pipe', 'pipe', 'pipe'],
                timeout: TEST_TIMEOUT
            });

            let receivedResponse = false;
            let serverOutput = '';

            server.stdout.on('data', (data) => {
                const output = data.toString();
                serverOutput += output;
                
                // Check for tools list response
                if (output.includes('"tools":') && output.includes('swarm_init')) {
                    receivedResponse = true;
                    server.kill();
                    resolve();
                }
            });

            server.stderr.on('data', (data) => {
                // Log stderr for debugging
                console.log('STDERR:', data.toString());
            });

            server.on('close', (code) => {
                if (receivedResponse) {
                    resolve();
                } else {
                    reject(new Error(`No tools list response received, output: ${serverOutput}`));
                }
            });

            server.on('error', (error) => {
                reject(new Error(`Server error: ${error.message}`));
            });

            // Send tools/list request after server starts
            setTimeout(() => {
                const request = '{"jsonrpc":"2.0","method":"tools/list","id":1}\n';
                server.stdin.write(request);
                server.stdin.end();
            }, 2000);

            // Timeout check
            setTimeout(() => {
                if (!receivedResponse) {
                    server.kill();
                    reject(new Error('Tools list request timeout'));
                }
            }, TEST_TIMEOUT);
        });
    }

    async testMcpToolCall() {
        return new Promise((resolve, reject) => {
            const server = spawn('npx', ['ruv-swarm', 'mcp', 'start'], {
                stdio: ['pipe', 'pipe', 'pipe'],
                timeout: TEST_TIMEOUT
            });

            let receivedResponse = false;
            let serverOutput = '';

            server.stdout.on('data', (data) => {
                const output = data.toString();
                serverOutput += output;
                
                // Check for successful tool call response
                if (output.includes('"result":') && output.includes('swarm')) {
                    receivedResponse = true;
                    server.kill();
                    resolve();
                }
            });

            server.stderr.on('data', (data) => {
                // Log stderr for debugging
                console.log('STDERR:', data.toString());
            });

            server.on('close', (code) => {
                if (receivedResponse) {
                    resolve();
                } else {
                    reject(new Error(`No tool call response received, output: ${serverOutput}`));
                }
            });

            server.on('error', (error) => {
                reject(new Error(`Server error: ${error.message}`));
            });

            // Send tool call request after server starts
            setTimeout(() => {
                const request = '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"swarm_status","arguments":{"verbose":false}},"id":2}\n';
                server.stdin.write(request);
                server.stdin.end();
            }, 2000);

            // Timeout check
            setTimeout(() => {
                if (!receivedResponse) {
                    server.kill();
                    reject(new Error('Tool call request timeout'));
                }
            }, TEST_TIMEOUT);
        });
    }

    async generateReport() {
        const summary = {
            total: this.testResults.tests.length,
            passed: this.testResults.tests.filter(t => t.status === 'passed').length,
            failed: this.testResults.tests.filter(t => t.status === 'failed').length
        };
        
        summary.passRate = ((summary.passed / summary.total) * 100).toFixed(2) + '%';
        this.testResults.summary = summary;

        // Write results to file
        const resultsPath = './docker/test-results/mcp-reliability/mcp-validation.json';
        await fs.writeFile(resultsPath, JSON.stringify(this.testResults, null, 2));

        console.log('\n📊 Test Results Summary:');
        console.log(`   Total Tests: ${summary.total}`);
        console.log(`   Passed: ${summary.passed}`);
        console.log(`   Failed: ${summary.failed}`);
        console.log(`   Pass Rate: ${summary.passRate}`);
        console.log(`   Results saved to: ${resultsPath}`);

        return summary;
    }
}

async function main() {
    console.log('🧪 Starting MCP Server Stdio Validation Tests');
    console.log('==================================================');
    
    const tester = new McpStdioTest();
    
    // Run tests
    await tester.runTest('MCP Server Start', () => tester.testMcpServerStart());
    await tester.runTest('Tools List Request', () => tester.testMcpToolsList());
    await tester.runTest('Tool Call Request', () => tester.testMcpToolCall());
    
    // Generate report
    const summary = await tester.generateReport();
    
    // Exit with appropriate code
    process.exit(summary.failed > 0 ? 1 : 0);
}

main().catch(console.error);