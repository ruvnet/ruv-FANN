#!/usr/bin/env node
/**
 * Diagnostic CLI for ruv-swarm
 * Usage: npx ruv-swarm diagnose [options]
 */

import { diagnostics } from './diagnostics.js';
import { loggingConfig } from './logging-config.js';
import fs from 'fs';
import path from 'path';

async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'help';

  // Initialize diagnostics logger
  const logger = loggingConfig.getLogger('cli-diagnostics', { level: 'INFO' });

  try {
    switch (command) {
      case 'test':
        await runDiagnosticTests(logger);
        break;

      case 'report':
        await generateReport(args.slice(1), logger);
        break;

      case 'monitor':
        await startMonitoring(args.slice(1), logger);
        break;

      case 'logs':
        await analyzeLogs(args.slice(1), logger);
        break;

      case 'config':
        showLoggingConfig(logger);
        break;

      case 'help':
      default:
        showHelp();
        break;
    }
  } catch (error) {
    logger.error('Diagnostic command failed', { error, command });
    process.exit(1);
  }
}

async function runDiagnosticTests(logger) {
  logger.info('Running diagnostic tests...');

  const results = await diagnostics.runDiagnosticTests();

  console.log('\n📋 Diagnostic Test Results:');
  console.log('═══════════════════════════\n');

  results.tests.forEach(test => {
    const icon = test.success ? '✅' : '❌';
    console.log(`${icon} ${test.name}`);
    if (!test.success) {
      console.log(`   Error: ${test.error}`);
    } else if (test.allocated) {
      console.log(`   Allocated: ${test.allocated}`);
    } else if (test.path) {
      console.log(`   Path: ${test.path}`);
    }
  });

  console.log('\n📊 Summary:');
  console.log(`   Total Tests: ${results.summary.total}`);
  console.log(`   ✅ Passed: ${results.summary.passed}`);
  console.log(`   ❌ Failed: ${results.summary.failed}`);

  if (results.summary.failed > 0) {
    process.exit(1);
  }
}

async function generateReport(args, logger) {
  const outputPath = args.find(arg => arg.startsWith('--output='))?.split('=')[1];
  const format = args.find(arg => arg.startsWith('--format='))?.split('=')[1] || 'json';

  logger.info('Generating diagnostic report...');

  // Enable diagnostics temporarily
  diagnostics.enableAll();

  // Wait a bit to collect some samples
  await new Promise(resolve => setTimeout(resolve, 5000));

  const report = await diagnostics.generateFullReport();

  if (outputPath) {
    const reportPath = path.resolve(outputPath);

    if (format === 'json') {
      fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    } else if (format === 'markdown') {
      fs.writeFileSync(reportPath, formatReportAsMarkdown(report));
    }

    console.log(`\n📄 Report saved to: ${reportPath}`);
  } else {
    console.log('\n📊 Diagnostic Report:');
    console.log('═══════════════════════');
    console.log(formatReportForConsole(report));
  }

  // Disable diagnostics
  diagnostics.disableAll();
}

async function startMonitoring(args, logger) {
  const duration = parseInt(args.find(arg => arg.startsWith('--duration='))?.split('=')[1] || '60', 10);
  const interval = parseInt(args.find(arg => arg.startsWith('--interval='))?.split('=')[1] || '1000', 10);

  logger.info('Starting system monitoring...', { duration, interval });

  console.log('\n🔍 System Monitoring');
  console.log('═══════════════════════');
  console.log(`Duration: ${duration} seconds`);
  console.log(`Interval: ${interval}ms`);
  console.log('\nPress Ctrl+C to stop\n');

  diagnostics.enableAll();
  diagnostics.system.startMonitoring(interval);

  // Update display periodically
  const displayInterval = setInterval(() => {
    const health = diagnostics.system.getSystemHealth();
    const connection = diagnostics.connection.getConnectionSummary();

    console.clear();
    console.log('🔍 System Monitoring');
    console.log('═══════════════════════');
    console.log(`\n📊 System Health: ${health.status.toUpperCase()}`);

    if (health.issues.length > 0) {
      console.log('\n⚠️  Issues:');
      health.issues.forEach(issue => console.log(`   - ${issue}`));
    }

    console.log('\n💾 Metrics:');
    Object.entries(health.metrics).forEach(([key, value]) => {
      console.log(`   ${key}: ${value}`);
    });

    console.log('\n🔌 Connections:');
    console.log(`   Active: ${connection.activeConnections}`);
    console.log(`   Total Events: ${connection.totalEvents}`);
    console.log(`   Failure Rate: ${(connection.failureRate * 100).toFixed(1)}%`);

    console.log('\n\nPress Ctrl+C to stop');
  }, 2000);

  // Set up timeout
  setTimeout(() => {
    clearInterval(displayInterval);
    diagnostics.disableAll();
    console.log('\n✅ Monitoring completed');
    process.exit(0);
  }, duration * 1000);

  // Handle Ctrl+C
  process.on('SIGINT', () => {
    clearInterval(displayInterval);
    diagnostics.disableAll();
    console.log('\n\n✅ Monitoring stopped');
    process.exit(0);
  });
}

async function analyzeLogs(args, logger) {
  const logDir = args.find(arg => arg.startsWith('--dir='))?.split('=')[1] || './logs';
  const pattern = args.find(arg => arg.startsWith('--pattern='))?.split('=')[1] || 'error';

  logger.info('Analyzing logs...', { logDir, pattern });

  if (!fs.existsSync(logDir)) {
    console.error(`❌ Log directory not found: ${logDir}`);
    process.exit(1);
  }

  const logFiles = fs.readdirSync(logDir).filter(f => f.endsWith('.log'));

  console.log(`\n📁 Found ${logFiles.length} log files in ${logDir}`);

  const results = {
    totalLines: 0,
    matches: 0,
    files: {},
  };

  const regex = new RegExp(pattern, 'i');

  logFiles.forEach(file => {
    const content = fs.readFileSync(path.join(logDir, file), 'utf8');
    const lines = content.split('\n');
    const matches = lines.filter(line => regex.test(line));

    results.totalLines += lines.length;
    results.matches += matches.length;

    if (matches.length > 0) {
      results.files[file] = {
        matches: matches.length,
        samples: matches.slice(0, 3),
      };
    }
  });

  console.log('\n📊 Analysis Results:');
  console.log(`   Total Lines: ${results.totalLines}`);
  console.log(`   Matches: ${results.matches}`);
  console.log(`   Pattern: ${pattern}`);

  if (results.matches > 0) {
    console.log('\n📄 Files with matches:');
    Object.entries(results.files).forEach(([file, data]) => {
      console.log(`\n   ${file} (${data.matches} matches):`);
      data.samples.forEach(sample => {
        console.log(`      ${sample.substring(0, 100)}...`);
      });
    });
  }
}

function showLoggingConfig(logger) {
  console.log('\n⚙️  Logging Configuration');
  console.log('═══════════════════════════\n');

  const config = loggingConfig.logConfiguration();

  // Configuration is already logged to stderr by logConfiguration()
  // Just add usage instructions

  console.log('\n📝 Environment Variables:');
  console.log('   LOG_LEVEL         - Global log level (TRACE|DEBUG|INFO|WARN|ERROR|FATAL)');
  console.log('   MCP_LOG_LEVEL     - MCP server log level');
  console.log('   TOOLS_LOG_LEVEL   - MCP tools log level');
  console.log('   SWARM_LOG_LEVEL   - Swarm core log level');
  console.log('   AGENT_LOG_LEVEL   - Agent log level');
  console.log('   NEURAL_LOG_LEVEL  - Neural network log level');
  console.log('   LOG_TO_FILE       - Enable file logging (true|false)');
  console.log('   LOG_FORMAT        - Log format (text|json)');
  console.log('   LOG_DIR           - Log directory path');

  console.log('\n💡 Examples:');
  console.log('   LOG_LEVEL=DEBUG npx ruv-swarm mcp start');
  console.log('   MCP_LOG_LEVEL=TRACE TOOLS_LOG_LEVEL=DEBUG npx ruv-swarm mcp start');
  console.log('   LOG_TO_FILE=true LOG_DIR=./mylogs npx ruv-swarm mcp start');
}

function formatReportForConsole(report) {
  const output = [];

  // Connection section
  output.push('🔌 Connection Diagnostics:');
  output.push(`   Active Connections: ${report.connection.connections.activeConnections}`);
  output.push(`   Failure Rate: ${(report.connection.connections.failureRate * 100).toFixed(1)}%`);
  output.push(`   Total Events: ${report.connection.connections.totalEvents}`);

  if (report.connection.patterns.recommendations.length > 0) {
    output.push('\n⚠️  Recommendations:');
    report.connection.patterns.recommendations.forEach(rec => {
      output.push(`   [${rec.severity.toUpperCase()}] ${rec.issue}`);
      output.push(`   → ${rec.suggestion}`);
    });
  }

  // System section
  output.push('\n💻 System Health:');
  output.push(`   Status: ${report.system.status.toUpperCase()}`);
  if (report.system.metrics) {
    Object.entries(report.system.metrics).forEach(([key, value]) => {
      output.push(`   ${key}: ${value}`);
    });
  }

  return output.join('\n');
}

function formatReportAsMarkdown(report) {
  const lines = [
    '# ruv-swarm Diagnostic Report',
    '',
    `Generated: ${report.timestamp}`,
    '',
    '## Connection Diagnostics',
    '',
    `- **Active Connections**: ${report.connection.connections.activeConnections}`,
    `- **Failure Rate**: ${(report.connection.connections.failureRate * 100).toFixed(1)}%`,
    `- **Total Events**: ${report.connection.connections.totalEvents}`,
    '',
  ];

  if (report.connection.patterns.recommendations.length > 0) {
    lines.push('### Recommendations');
    lines.push('');
    report.connection.patterns.recommendations.forEach(rec => {
      lines.push(`- **${rec.severity.toUpperCase()}**: ${rec.issue}`);
      lines.push(`  - ${rec.suggestion}`);
    });
    lines.push('');
  }

  lines.push('## System Health');
  lines.push('');
  lines.push(`- **Status**: ${report.system.status.toUpperCase()}`);

  if (report.system.metrics) {
    lines.push('');
    lines.push('### Metrics');
    lines.push('');
    Object.entries(report.system.metrics).forEach(([key, value]) => {
      lines.push(`- **${key}**: ${value}`);
    });
  }

  return lines.join('\n');
}

function showHelp() {
  console.log(`
🔍 ruv-swarm Diagnostics

Usage: npx ruv-swarm diagnose <command> [options]

Commands:
  test                     Run diagnostic tests
  report [options]         Generate diagnostic report
    --output=<path>        Save report to file
    --format=<json|md>     Output format (default: json)
    
  monitor [options]        Start system monitoring
    --duration=<seconds>   Monitoring duration (default: 60)
    --interval=<ms>        Sample interval (default: 1000)
    
  logs [options]           Analyze log files
    --dir=<path>           Log directory (default: ./logs)
    --pattern=<regex>      Search pattern (default: error)
    
  config                   Show logging configuration
  help                     Show this help message

Examples:
  npx ruv-swarm diagnose test
  npx ruv-swarm diagnose report --output=report.json
  npx ruv-swarm diagnose monitor --duration=120
  npx ruv-swarm diagnose logs --pattern="connection.*failed"
  npx ruv-swarm diagnose config
`);
}

// Export for use in main CLI
export { main as diagnosticsCLI };

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}
