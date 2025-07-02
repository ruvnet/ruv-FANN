#!/usr/bin/env node

/**
 * Regression Testing Pipeline for ruv-swarm
 * Automated CI/CD integration with performance regression detection
 */

const { RuvSwarm } = require('../src/index-enhanced');
const fs = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');

class RegressionTestingPipeline {
    constructor() {
        this.pipelineResults = {
            timestamp: new Date().toISOString(),
            buildInfo: {
                commit: process.env.GITHUB_SHA || 'local',
                branch: process.env.GITHUB_REF_NAME || 'local',
                buildNumber: process.env.GITHUB_RUN_NUMBER || '0',
                environment: process.env.NODE_ENV || 'test'
            },
            baseline: null,
            currentResults: null,
            regressions: [],
            improvements: [],
            stages: [],
            overallStatus: 'UNKNOWN'
        };
        this.baselineFile = '/workspaces/ruv-FANN/ruv-swarm/npm/test/baseline-performance.json';
        this.thresholds = {
            performance: 0.05, // 5% degradation threshold
            memory: 0.10,      // 10% memory increase threshold
            errorRate: 0.02,   // 2% error rate increase threshold
            coverage: 0.01     // 1% coverage decrease threshold
        };
    }

    async runRegressionPipeline() {
        console.log('🔄 Starting Regression Testing Pipeline');
        console.log('=======================================\n');

        this.logBuildInfo();

        try {
            // Stage 1: Environment Setup
            await this.runStage('Environment Setup', this.setupEnvironment.bind(this));

            // Stage 2: Code Quality Checks
            await this.runStage('Code Quality Checks', this.runCodeQualityChecks.bind(this));

            // Stage 3: Unit Tests with Coverage
            await this.runStage('Unit Tests', this.runUnitTests.bind(this));

            // Stage 4: Integration Tests
            await this.runStage('Integration Tests', this.runIntegrationTests.bind(this));

            // Stage 5: Performance Benchmarks
            await this.runStage('Performance Benchmarks', this.runPerformanceBenchmarks.bind(this));

            // Stage 6: Load Testing
            await this.runStage('Load Testing', this.runLoadTests.bind(this));

            // Stage 7: Security Scanning
            await this.runStage('Security Scanning', this.runSecurityScans.bind(this));

            // Stage 8: Cross-Platform Testing
            await this.runStage('Cross-Platform Testing', this.runCrossPlatformTests.bind(this));

            // Stage 9: Regression Analysis
            await this.runStage('Regression Analysis', this.performRegressionAnalysis.bind(this));

            // Stage 10: Report Generation
            await this.runStage('Report Generation', this.generateRegressionReport.bind(this));

        } catch (error) {
            console.error('❌ Pipeline failed:', error);
            this.pipelineResults.overallStatus = 'FAILED';
            throw error;
        }

        return this.pipelineResults;
    }

    logBuildInfo() {
        console.log('🏗️  Build Information:');
        console.log(`   Commit: ${this.pipelineResults.buildInfo.commit}`);
        console.log(`   Branch: ${this.pipelineResults.buildInfo.branch}`);
        console.log(`   Build: #${this.pipelineResults.buildInfo.buildNumber}`);
        console.log(`   Environment: ${this.pipelineResults.buildInfo.environment}\n`);
    }

    async runStage(stageName, stageFunction) {
        console.log(`📋 Stage: ${stageName}`);
        
        const stage = {
            name: stageName,
            startTime: Date.now(),
            passed: false,
            duration: 0,
            output: [],
            errors: []
        };

        try {
            const result = await stageFunction();
            stage.passed = result.passed !== false;
            stage.output = result.output || [];
            stage.data = result.data || {};
            
            console.log(`   ${stage.passed ? '✅' : '❌'} ${stageName} ${stage.passed ? 'passed' : 'failed'}`);
            
        } catch (error) {
            stage.passed = false;
            stage.errors.push(error.message);
            console.log(`   ❌ ${stageName} failed: ${error.message}`);
        }

        stage.duration = Date.now() - stage.startTime;
        this.pipelineResults.stages.push(stage);
        console.log('');

        return stage;
    }

    async setupEnvironment() {
        const setupResult = {
            passed: true,
            output: [],
            data: {}
        };

        try {
            // Check Node.js version
            const nodeVersion = process.version;
            setupResult.output.push(`Node.js version: ${nodeVersion}`);

            // Check dependencies
            const packageJson = JSON.parse(
                await fs.readFile('/workspaces/ruv-FANN/ruv-swarm/npm/package.json', 'utf8')
            );
            setupResult.output.push(`Package version: ${packageJson.version}`);

            // Initialize test database
            const testDbPath = '/workspaces/ruv-FANN/ruv-swarm/npm/test/regression-test.db';
            try {
                await fs.unlink(testDbPath);
            } catch (error) {
                // File doesn't exist, that's fine
            }

            // Set environment variables
            process.env.NODE_ENV = 'test';
            process.env.RUV_SWARM_TEST_MODE = 'regression';

            setupResult.data = {
                nodeVersion,
                packageVersion: packageJson.version,
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            setupResult.passed = false;
            setupResult.errors = [error.message];
        }

        return setupResult;
    }

    async runCodeQualityChecks() {
        const qualityResult = {
            passed: true,
            output: [],
            data: {
                linting: { passed: false, issues: 0 },
                formatting: { passed: false, issues: 0 },
                typeChecking: { passed: false, issues: 0 }
            }
        };

        try {
            // Run ESLint
            const lintResult = await this.runCommand('npm run lint:check', { cwd: '/workspaces/ruv-FANN/ruv-swarm/npm' });
            qualityResult.data.linting.passed = lintResult.success;
            qualityResult.data.linting.issues = this.countLintIssues(lintResult.output);
            qualityResult.output.push(`Linting: ${lintResult.success ? 'PASSED' : 'FAILED'} (${qualityResult.data.linting.issues} issues)`);

            // Check if TypeScript definitions exist
            const tsFiles = await this.findFiles('/workspaces/ruv-FANN/ruv-swarm/npm/src', '.d.ts');
            qualityResult.data.typeChecking.passed = tsFiles.length > 0;
            qualityResult.output.push(`Type definitions: ${tsFiles.length} files found`);

            qualityResult.passed = qualityResult.data.linting.passed && 
                                 qualityResult.data.linting.issues < 10; // Allow up to 10 linting issues

        } catch (error) {
            qualityResult.passed = false;
            qualityResult.output.push(`Quality check error: ${error.message}`);
        }

        return qualityResult;
    }

    async runUnitTests() {
        const unitTestResult = {
            passed: true,
            output: [],
            data: {
                testsRun: 0,
                testsPassed: 0,
                testsFailed: 0,
                coverage: {
                    lines: 0,
                    branches: 0,
                    functions: 0,
                    statements: 0
                }
            }
        };

        try {
            // Run unit tests with coverage
            const testResult = await this.runCommand('npm run test:coverage', { 
                cwd: '/workspaces/ruv-FANN/ruv-swarm/npm',
                timeout: 120000 
            });

            // Parse test results
            const testOutput = testResult.output;
            unitTestResult.data.testsRun = this.extractTestCount(testOutput, 'total');
            unitTestResult.data.testsPassed = this.extractTestCount(testOutput, 'passed');
            unitTestResult.data.testsFailed = this.extractTestCount(testOutput, 'failed');

            // Parse coverage results
            const coverageData = await this.parseCoverageResults();
            unitTestResult.data.coverage = coverageData;

            unitTestResult.passed = testResult.success && unitTestResult.data.testsFailed === 0;
            unitTestResult.output.push(`Tests: ${unitTestResult.data.testsPassed}/${unitTestResult.data.testsRun} passed`);
            unitTestResult.output.push(`Coverage: ${coverageData.lines}% lines, ${coverageData.branches}% branches`);

        } catch (error) {
            unitTestResult.passed = false;
            unitTestResult.output.push(`Unit test error: ${error.message}`);
        }

        return unitTestResult;
    }

    async runIntegrationTests() {
        const integrationResult = {
            passed: true,
            output: [],
            data: {
                scenarios: [],
                totalTime: 0,
                errorRate: 0
            }
        };

        try {
            // Run integration test suite
            const testResult = await this.runCommand('node test/integration/run-integration-tests.js', {
                cwd: '/workspaces/ruv-FANN/ruv-swarm/npm',
                timeout: 300000 // 5 minutes
            });

            integrationResult.passed = testResult.success;
            integrationResult.output.push(`Integration tests: ${testResult.success ? 'PASSED' : 'FAILED'}`);

            // Parse integration test results if available
            try {
                const resultsFile = '/workspaces/ruv-FANN/ruv-swarm/npm/test-results/integration-results.json';
                const results = JSON.parse(await fs.readFile(resultsFile, 'utf8'));
                
                integrationResult.data.scenarios = results.suites || [];
                integrationResult.data.totalTime = results.duration || 0;
                integrationResult.data.errorRate = this.calculateErrorRate(results);
                
            } catch (error) {
                // Results file not found, use basic data
                integrationResult.output.push('Integration results file not found, using basic metrics');
            }

        } catch (error) {
            integrationResult.passed = false;
            integrationResult.output.push(`Integration test error: ${error.message}`);
        }

        return integrationResult;
    }

    async runPerformanceBenchmarks() {
        const perfResult = {
            passed: true,
            output: [],
            data: {
                simdPerformance: null,
                speedOptimization: null,
                memoryUsage: null,
                throughput: null
            }
        };

        try {
            // Run performance validation
            const perfTestResult = await this.runCommand('node test/comprehensive-performance-validation.test.js', {
                cwd: '/workspaces/ruv-FANN/ruv-swarm/npm',
                timeout: 600000 // 10 minutes
            });

            perfResult.passed = perfTestResult.success;

            // Load performance results
            try {
                const resultsFile = '/workspaces/ruv-FANN/ruv-swarm/npm/test/validation-report.json';
                const results = JSON.parse(await fs.readFile(resultsFile, 'utf8'));
                
                perfResult.data = {
                    simdPerformance: results.performance?.simd?.actual,
                    speedOptimization: results.performance?.speed?.actual,
                    memoryUsage: results.performance?.memoryEfficiency?.actual,
                    throughput: this.calculateThroughput(results)
                };

                perfResult.output.push(`SIMD Performance: ${perfResult.data.simdPerformance || 'N/A'}`);
                perfResult.output.push(`Speed Optimization: ${perfResult.data.speedOptimization || 'N/A'}`);
                perfResult.output.push(`Memory Usage: ${perfResult.data.memoryUsage || 'N/A'}`);

            } catch (error) {
                perfResult.output.push('Performance results file not found');
            }

        } catch (error) {
            perfResult.passed = false;
            perfResult.output.push(`Performance benchmark error: ${error.message}`);
        }

        return perfResult;
    }

    async runLoadTests() {
        const loadResult = {
            passed: true,
            output: [],
            data: {
                maxAgents: 0,
                avgResponseTime: 0,
                memoryPeak: 0,
                errorRate: 0
            }
        };

        try {
            // Run load testing suite
            const loadTestResult = await this.runCommand('node test/load-testing-suite.test.js', {
                cwd: '/workspaces/ruv-FANN/ruv-swarm/npm',
                timeout: 1800000 // 30 minutes
            });

            loadResult.passed = loadTestResult.success;

            // Load test results
            try {
                const resultsFile = '/workspaces/ruv-FANN/ruv-swarm/npm/test/load-test-report.json';
                const results = JSON.parse(await fs.readFile(resultsFile, 'utf8'));
                
                loadResult.data = {
                    maxAgents: results.performance?.maxConcurrentAgents || 0,
                    avgResponseTime: results.performance?.avgResponseTime || 0,
                    memoryPeak: results.performance?.memoryPeak || 0,
                    errorRate: parseFloat(results.performance?.errorRate) || 0
                };

                loadResult.output.push(`Max Agents: ${loadResult.data.maxAgents}`);
                loadResult.output.push(`Avg Response: ${loadResult.data.avgResponseTime}ms`);
                loadResult.output.push(`Memory Peak: ${loadResult.data.memoryPeak}MB`);
                loadResult.output.push(`Error Rate: ${loadResult.data.errorRate}%`);

            } catch (error) {
                loadResult.output.push('Load test results file not found');
            }

        } catch (error) {
            loadResult.passed = false;
            loadResult.output.push(`Load test error: ${error.message}`);
        }

        return loadResult;
    }

    async runSecurityScans() {
        const securityResult = {
            passed: true,
            output: [],
            data: {
                securityScore: 0,
                vulnerabilities: 0,
                memoryLeaks: 0,
                securityLevel: 'UNKNOWN'
            }
        };

        try {
            // Run security audit
            const secTestResult = await this.runCommand('node test/security-audit.test.js', {
                cwd: '/workspaces/ruv-FANN/ruv-swarm/npm',
                timeout: 600000 // 10 minutes
            });

            // Load security results
            try {
                const resultsFile = '/workspaces/ruv-FANN/ruv-swarm/npm/test/security-audit-report.json';
                const results = JSON.parse(await fs.readFile(resultsFile, 'utf8'));
                
                securityResult.data = {
                    securityScore: results.overallSecurity?.score || 0,
                    vulnerabilities: results.vulnerabilities?.length || 0,
                    memoryLeaks: results.memoryTests?.filter(t => !t.passed).length || 0,
                    securityLevel: results.overallSecurity?.level || 'UNKNOWN'
                };

                securityResult.passed = securityResult.data.securityLevel !== 'CRITICAL' && 
                                      securityResult.data.securityScore >= 70;

                securityResult.output.push(`Security Score: ${securityResult.data.securityScore}/100`);
                securityResult.output.push(`Security Level: ${securityResult.data.securityLevel}`);
                securityResult.output.push(`Vulnerabilities: ${securityResult.data.vulnerabilities}`);

            } catch (error) {
                securityResult.output.push('Security results file not found');
                securityResult.passed = false;
            }

        } catch (error) {
            securityResult.passed = false;
            securityResult.output.push(`Security scan error: ${error.message}`);
        }

        return securityResult;
    }

    async runCrossPlatformTests() {
        const platformResult = {
            passed: true,
            output: [],
            data: {
                platform: process.platform,
                arch: process.arch,
                nodeVersion: process.version,
                wasmSupport: false,
                sqliteSupport: false
            }
        };

        try {
            // Test WASM support
            const ruvSwarm = await RuvSwarm.initialize();
            platformResult.data.wasmSupport = await ruvSwarm.detectSIMDSupport() !== undefined;

            // Test SQLite support
            try {
                const { PersistenceManager } = require('/workspaces/ruv-FANN/ruv-swarm/npm/src/persistence');
                const pm = new PersistenceManager(':memory:');
                await pm.initialize();
                platformResult.data.sqliteSupport = true;
            } catch (error) {
                platformResult.data.sqliteSupport = false;
            }

            platformResult.passed = platformResult.data.wasmSupport && platformResult.data.sqliteSupport;

            platformResult.output.push(`Platform: ${platformResult.data.platform} ${platformResult.data.arch}`);
            platformResult.output.push(`WASM Support: ${platformResult.data.wasmSupport ? 'YES' : 'NO'}`);
            platformResult.output.push(`SQLite Support: ${platformResult.data.sqliteSupport ? 'YES' : 'NO'}`);

        } catch (error) {
            platformResult.passed = false;
            platformResult.output.push(`Cross-platform test error: ${error.message}`);
        }

        return platformResult;
    }

    async performRegressionAnalysis() {
        const regressionResult = {
            passed: true,
            output: [],
            data: {
                baselineLoaded: false,
                regressionCount: 0,
                improvementCount: 0,
                significantChanges: []
            }
        };

        try {
            // Load baseline performance data
            this.pipelineResults.baseline = await this.loadBaseline();
            regressionResult.data.baselineLoaded = this.pipelineResults.baseline !== null;

            if (this.pipelineResults.baseline) {
                // Collect current results
                this.pipelineResults.currentResults = this.collectCurrentResults();

                // Analyze for regressions
                this.analyzeRegressions();

                regressionResult.data.regressionCount = this.pipelineResults.regressions.length;
                regressionResult.data.improvementCount = this.pipelineResults.improvements.length;
                regressionResult.data.significantChanges = [
                    ...this.pipelineResults.regressions,
                    ...this.pipelineResults.improvements
                ];

                regressionResult.passed = this.pipelineResults.regressions.length === 0;

                regressionResult.output.push(`Regressions detected: ${regressionResult.data.regressionCount}`);
                regressionResult.output.push(`Improvements detected: ${regressionResult.data.improvementCount}`);

                if (regressionResult.data.regressionCount > 0) {
                    regressionResult.output.push('❌ Regression analysis FAILED - performance degradation detected');
                } else {
                    regressionResult.output.push('✅ Regression analysis PASSED - no performance degradation');
                }

            } else {
                regressionResult.output.push('No baseline found - saving current results as baseline');
                await this.saveBaseline(this.collectCurrentResults());
            }

        } catch (error) {
            regressionResult.passed = false;
            regressionResult.output.push(`Regression analysis error: ${error.message}`);
        }

        return regressionResult;
    }

    async generateRegressionReport() {
        const reportResult = {
            passed: true,
            output: [],
            data: {}
        };

        try {
            // Determine overall pipeline status
            const passedStages = this.pipelineResults.stages.filter(s => s.passed).length;
            const totalStages = this.pipelineResults.stages.length;
            const successRate = (passedStages / totalStages) * 100;

            this.pipelineResults.overallStatus = successRate >= 80 ? 'PASSED' : 'FAILED';

            // Generate comprehensive report
            const report = {
                ...this.pipelineResults,
                summary: {
                    totalStages,
                    passedStages,
                    failedStages: totalStages - passedStages,
                    successRate: successRate.toFixed(1) + '%',
                    duration: this.pipelineResults.stages.reduce((sum, stage) => sum + stage.duration, 0)
                },
                cicdIntegration: {
                    shouldDeploy: this.pipelineResults.overallStatus === 'PASSED' && 
                                 this.pipelineResults.regressions.length === 0,
                    deploymentReady: this.checkDeploymentReadiness(),
                    recommendations: this.generateCICDRecommendations()
                }
            };

            // Save reports
            const reportPath = '/workspaces/ruv-FANN/ruv-swarm/npm/test/regression-pipeline-report.json';
            await fs.writeFile(reportPath, JSON.stringify(report, null, 2));

            // Generate CI/CD compatible outputs
            await this.generateCICDOutputs(report);

            reportResult.output.push(`Overall Status: ${this.pipelineResults.overallStatus}`);
            reportResult.output.push(`Success Rate: ${report.summary.successRate}`);
            reportResult.output.push(`Duration: ${Math.round(report.summary.duration / 1000)}s`);
            reportResult.output.push(`Report saved to: ${reportPath}`);

        } catch (error) {
            reportResult.passed = false;
            reportResult.output.push(`Report generation error: ${error.message}`);
        }

        return reportResult;
    }

    // Helper methods
    async runCommand(command, options = {}) {
        return new Promise((resolve) => {
            const [cmd, ...args] = command.split(' ');
            const process = spawn(cmd, args, {
                stdio: 'pipe',
                cwd: options.cwd || '/workspaces/ruv-FANN/ruv-swarm/npm',
                ...options
            });

            let output = '';
            let error = '';

            process.stdout.on('data', (data) => output += data.toString());
            process.stderr.on('data', (data) => error += data.toString());

            process.on('close', (code) => {
                resolve({
                    success: code === 0,
                    output: output || error,
                    exitCode: code
                });
            });

            if (options.timeout) {
                setTimeout(() => {
                    process.kill();
                    resolve({ success: false, output: 'Command timeout', exitCode: -1 });
                }, options.timeout);
            }
        });
    }

    async findFiles(directory, extension) {
        try {
            const files = await fs.readdir(directory, { recursive: true });
            return files.filter(file => file.endsWith(extension));
        } catch {
            return [];
        }
    }

    countLintIssues(output) {
        const matches = output.match(/\d+ problems? \(\d+ errors?, \d+ warnings?\)/);
        if (matches) {
            const numbers = matches[0].match(/\d+/g);
            return parseInt(numbers[0]) || 0;
        }
        return 0;
    }

    extractTestCount(output, type) {
        const patterns = {
            total: /Tests completed:\s*(\d+)/,
            passed: /(\d+)\s+passed/,
            failed: /(\d+)\s+failed/
        };
        
        const match = output.match(patterns[type]);
        return match ? parseInt(match[1]) : 0;
    }

    async parseCoverageResults() {
        try {
            const coveragePath = '/workspaces/ruv-FANN/ruv-swarm/npm/coverage/coverage-summary.json';
            const coverage = JSON.parse(await fs.readFile(coveragePath, 'utf8'));
            
            return {
                lines: coverage.total?.lines?.pct || 0,
                branches: coverage.total?.branches?.pct || 0,
                functions: coverage.total?.functions?.pct || 0,
                statements: coverage.total?.statements?.pct || 0
            };
        } catch {
            return { lines: 0, branches: 0, functions: 0, statements: 0 };
        }
    }

    calculateErrorRate(results) {
        if (!results.suites) return 0;
        
        const total = results.suites.length;
        const failed = results.suites.filter(s => s.status !== 'PASSED').length;
        
        return total > 0 ? (failed / total) * 100 : 0;
    }

    calculateThroughput(results) {
        if (!results.tests) return 0;
        
        const totalDuration = results.tests.reduce((sum, test) => sum + (test.duration || 0), 0);
        const totalTests = results.tests.length;
        
        return totalDuration > 0 ? Math.round((totalTests / totalDuration) * 1000) : 0; // tests per second
    }

    async loadBaseline() {
        try {
            const baseline = await fs.readFile(this.baselineFile, 'utf8');
            return JSON.parse(baseline);
        } catch {
            return null;
        }
    }

    collectCurrentResults() {
        const perfStage = this.pipelineResults.stages.find(s => s.name === 'Performance Benchmarks');
        const loadStage = this.pipelineResults.stages.find(s => s.name === 'Load Testing');
        const unitStage = this.pipelineResults.stages.find(s => s.name === 'Unit Tests');

        return {
            timestamp: new Date().toISOString(),
            commit: this.pipelineResults.buildInfo.commit,
            performance: perfStage?.data || {},
            loadTesting: loadStage?.data || {},
            unitTests: unitStage?.data || {},
            systemInfo: {
                platform: process.platform,
                arch: process.arch,
                nodeVersion: process.version
            }
        };
    }

    analyzeRegressions() {
        const baseline = this.pipelineResults.baseline;
        const current = this.pipelineResults.currentResults;

        // Performance regression checks
        this.checkPerformanceRegression('SIMD Performance', 
            baseline.performance?.simdPerformance, 
            current.performance?.simdPerformance);

        this.checkPerformanceRegression('Speed Optimization',
            baseline.performance?.speedOptimization,
            current.performance?.speedOptimization);

        // Memory regression checks
        this.checkNumericRegression('Memory Usage',
            this.parseMemoryValue(baseline.loadTesting?.memoryPeak),
            this.parseMemoryValue(current.loadTesting?.memoryPeak),
            this.thresholds.memory);

        // Response time regression checks
        this.checkNumericRegression('Response Time',
            baseline.loadTesting?.avgResponseTime,
            current.loadTesting?.avgResponseTime,
            this.thresholds.performance);

        // Coverage regression checks
        this.checkNumericRegression('Test Coverage',
            baseline.unitTests?.coverage?.lines,
            current.unitTests?.coverage?.lines,
            this.thresholds.coverage,
            true); // Lower is worse for coverage
    }

    checkPerformanceRegression(metric, baselineValue, currentValue) {
        if (!baselineValue || !currentValue) return;

        const baselineMultiplier = parseFloat(baselineValue.replace('x', ''));
        const currentMultiplier = parseFloat(currentValue.replace('x', ''));

        if (currentMultiplier < baselineMultiplier * (1 - this.thresholds.performance)) {
            this.pipelineResults.regressions.push({
                metric,
                baseline: baselineValue,
                current: currentValue,
                change: `${((currentMultiplier / baselineMultiplier - 1) * 100).toFixed(1)}%`,
                severity: 'HIGH'
            });
        } else if (currentMultiplier > baselineMultiplier * (1 + this.thresholds.performance)) {
            this.pipelineResults.improvements.push({
                metric,
                baseline: baselineValue,
                current: currentValue,
                change: `+${((currentMultiplier / baselineMultiplier - 1) * 100).toFixed(1)}%`
            });
        }
    }

    checkNumericRegression(metric, baselineValue, currentValue, threshold, lowerIsBetter = false) {
        if (baselineValue === undefined || currentValue === undefined) return;

        const change = (currentValue - baselineValue) / baselineValue;
        const isRegression = lowerIsBetter ? change < -threshold : change > threshold;
        const isImprovement = lowerIsBetter ? change > threshold : change < -threshold;

        if (isRegression) {
            this.pipelineResults.regressions.push({
                metric,
                baseline: baselineValue,
                current: currentValue,
                change: `${(change * 100).toFixed(1)}%`,
                severity: Math.abs(change) > threshold * 2 ? 'HIGH' : 'MEDIUM'
            });
        } else if (isImprovement) {
            this.pipelineResults.improvements.push({
                metric,
                baseline: baselineValue,
                current: currentValue,
                change: `${(change * 100).toFixed(1)}%`
            });
        }
    }

    parseMemoryValue(memoryString) {
        if (!memoryString) return 0;
        return parseFloat(memoryString.replace('MB', ''));
    }

    async saveBaseline(results) {
        await fs.writeFile(this.baselineFile, JSON.stringify(results, null, 2));
    }

    checkDeploymentReadiness() {
        const criticalStages = ['Unit Tests', 'Integration Tests', 'Security Scanning'];
        const criticalPassed = criticalStages.every(stageName => 
            this.pipelineResults.stages.find(s => s.name === stageName)?.passed
        );

        return criticalPassed && this.pipelineResults.regressions.length === 0;
    }

    generateCICDRecommendations() {
        const recommendations = [];

        if (this.pipelineResults.regressions.length > 0) {
            recommendations.push('Fix performance regressions before deployment');
        }

        const failedStages = this.pipelineResults.stages.filter(s => !s.passed);
        if (failedStages.length > 0) {
            recommendations.push(`Fix failing stages: ${failedStages.map(s => s.name).join(', ')}`);
        }

        if (this.pipelineResults.overallStatus === 'PASSED') {
            recommendations.push('All checks passed - ready for deployment');
        }

        return recommendations;
    }

    async generateCICDOutputs(report) {
        // Generate GitHub Actions outputs
        const githubOutput = `
deployment_ready=${report.cicdIntegration.deploymentReady}
overall_status=${report.overallStatus}
regression_count=${report.regressions.length}
success_rate=${report.summary.successRate}
`;

        await fs.writeFile('/workspaces/ruv-FANN/ruv-swarm/npm/test/github-outputs.txt', githubOutput);

        // Generate JUnit XML for test reporting
        const junitXml = this.generateJUnitXML(report);
        await fs.writeFile('/workspaces/ruv-FANN/ruv-swarm/npm/test/regression-results.xml', junitXml);
    }

    generateJUnitXML(report) {
        const testcases = report.stages.map(stage => `
    <testcase name="${stage.name}" time="${stage.duration / 1000}" classname="RegressionPipeline">
        ${stage.passed ? '' : `<failure message="${stage.errors.join('; ')}">${stage.errors.join('\n')}</failure>`}
    </testcase>`).join('');

        return `<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="Regression Testing Pipeline" tests="${report.stages.length}" failures="${report.summary.failedStages}" time="${report.summary.duration / 1000}">
${testcases}
</testsuite>`;
    }
}

// Main execution
async function runRegressionPipeline() {
    try {
        const pipeline = new RegressionTestingPipeline();
        const results = await pipeline.runRegressionPipeline();
        
        console.log('\n🎯 REGRESSION PIPELINE SUMMARY');
        console.log('===============================');
        console.log(`Overall Status: ${results.overallStatus}`);
        console.log(`Stages: ${results.summary?.passedStages}/${results.summary?.totalStages} passed`);
        console.log(`Regressions: ${results.regressions.length}`);
        console.log(`Improvements: ${results.improvements.length}`);
        console.log(`Deployment Ready: ${results.cicdIntegration?.deploymentReady ? 'YES' : 'NO'}`);

        process.exit(results.overallStatus === 'PASSED' ? 0 : 1);
    } catch (error) {
        console.error('💥 Regression pipeline failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    runRegressionPipeline();
}

module.exports = { RegressionTestingPipeline };