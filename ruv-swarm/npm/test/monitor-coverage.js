#!/usr/bin/env node

import { startLiveMonitoring, trackCoverage, generateProgressReport } from './coverage-dashboard.js';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

console.log('🚀 Starting Coverage Monitoring System...\n');

// Initial baseline
console.log('📊 Establishing baseline coverage...');

async function main() {
  // Get initial coverage
  const initial = await trackCoverage();

  if (initial) {
    console.log('\n📈 BASELINE METRICS:');
    console.log('═══════════════════════════════════════════');
    console.log(`Lines:      ${initial.coverage.lines.toFixed(2)}% (${initial.details.lines})`);
    console.log(`Branches:   ${initial.coverage.branches.toFixed(2)}% (${initial.details.branches})`);
    console.log(`Functions:  ${initial.coverage.functions.toFixed(2)}% (${initial.details.functions})`);
    console.log(`Statements: ${initial.coverage.statements.toFixed(2)}% (${initial.details.statements})`);
    console.log('═══════════════════════════════════════════\n');

    if (initial.uncovered.length > 0) {
      console.log('📍 Files needing coverage (top 5):');
      initial.uncovered.slice(0, 5).forEach(file => {
        console.log(`  - ${file.file}: ${file.lines.toFixed(1)}% lines covered`);
      });
      console.log(`  ... and ${Math.max(0, initial.uncovered.length - 5)} more files\n`);
    }
  }

  // Store initial metrics
  await execAsync(`npx ruv-swarm hook notification --message "Initial coverage: Lines ${initial?.coverage.lines || 0}%, Branches ${initial?.coverage.branches || 0}%, Functions ${initial?.coverage.functions || 0}%, Statements ${initial?.coverage.statements || 0}%" --telemetry true`);

  // Start live monitoring with 15 second intervals
  console.log('🔄 Starting live monitoring (updates every 15 seconds)...\n');
  const monitoringInterval = await startLiveMonitoring(15000);

  // Also check for test file changes
  console.log('👀 Watching for test file changes...\n');

  // Handle graceful shutdown
  process.on('SIGINT', async() => {
    console.log('\n\n⏹️  Stopping monitoring...');
    clearInterval(monitoringInterval);

    const final = await trackCoverage();
    const progress = await generateProgressReport();

    console.log('\n📊 FINAL SESSION METRICS:');
    console.log('═══════════════════════════════════════════');
    console.log(`Lines:      ${final.coverage.lines.toFixed(2)}% (improvement: +${progress.improvement.lines.toFixed(2)}%)`);
    console.log(`Branches:   ${final.coverage.branches.toFixed(2)}% (improvement: +${progress.improvement.branches.toFixed(2)}%)`);
    console.log(`Functions:  ${final.coverage.functions.toFixed(2)}% (improvement: +${progress.improvement.functions.toFixed(2)}%)`);
    console.log(`Statements: ${final.coverage.statements.toFixed(2)}% (improvement: +${progress.improvement.statements.toFixed(2)}%)`);
    console.log('═══════════════════════════════════════════\n');

    // Store final metrics
    await execAsync(`npx ruv-swarm hook notification --message "Session complete: Achieved ${final.coverage.lines.toFixed(1)}% line coverage" --telemetry true`);

    process.exit(0);
  });
}

// Run the monitoring
main().catch(console.error);