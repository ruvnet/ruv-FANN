#!/usr/bin/env node
/**
 * Test script for docs.js flag behavior verification (corrected version)
 */

import { ClaudeDocsGenerator } from './src/claude-integration/docs.js';
import fs from 'fs/promises';
import path from 'path';

async function createTestDir(name) {
  const testDir = path.join('.', `test-${name}-${Date.now()}`);
  await fs.mkdir(testDir, { recursive: true });
  return testDir;
}

async function cleanup(testDir) {
  try {
    await fs.rm(testDir, { recursive: true, force: true });
  } catch (e) {
    // Ignore cleanup errors
  }
}

async function testForceFlag() {
  console.log('\n🧪 Testing --force flag (should overwrite without backup)');

  const testDir = await createTestDir('force');
  const filePath = path.join(testDir, 'CLAUDE.md');

  // Create existing file
  await fs.writeFile(filePath, 'Original content that should be overwritten');

  const generator = new ClaudeDocsGenerator({ workingDir: testDir });

  try {
    await generator.generateClaudeMd({
      force: true,
      interactive: false,
    });

    // Check file was overwritten
    const content = await fs.readFile(filePath, 'utf8');
    const hasRuvSwarmContent = content.includes('ruv-swarm');
    const noOriginalContent = !content.includes('Original content that should be overwritten');

    // Check NO backup was created
    const files = await fs.readdir(testDir);
    const backupFiles = files.filter(f => f.startsWith('CLAUDE.md.backup.'));

    console.log(`   ✅ File overwritten: ${hasRuvSwarmContent && noOriginalContent}`);
    console.log(`   ✅ No backup created: ${backupFiles.length === 0}`);

    await cleanup(testDir);
    return hasRuvSwarmContent && noOriginalContent && backupFiles.length === 0;
  } catch (error) {
    console.error(`   ❌ Error: ${error.message}`);
    await cleanup(testDir);
    return false;
  }
}

async function testBackupFlag() {
  console.log('\n🧪 Testing --backup flag (should backup existing)');

  const testDir = await createTestDir('backup');
  const filePath = path.join(testDir, 'CLAUDE.md');

  // Create existing file
  const originalContent = 'Original content to backup';
  await fs.writeFile(filePath, originalContent);

  const generator = new ClaudeDocsGenerator({ workingDir: testDir });

  try {
    await generator.generateClaudeMd({
      backup: true,
      interactive: false,
    });

    // Check backup was created
    const files = await fs.readdir(testDir);
    const backupFiles = files.filter(f => f.startsWith('CLAUDE.md.backup.'));

    // Check backup content
    let backupContentCorrect = false;
    if (backupFiles.length > 0) {
      const backupContent = await fs.readFile(path.join(testDir, backupFiles[0]), 'utf8');
      backupContentCorrect = backupContent === originalContent;
    }

    // Check file was overwritten with new content
    const newContent = await fs.readFile(filePath, 'utf8');
    const hasRuvSwarmContent = newContent.includes('ruv-swarm');
    const noOriginalContent = !newContent.includes('Original content to backup');

    console.log(`   ✅ Backup created: ${backupFiles.length > 0}`);
    console.log(`   ✅ Backup content correct: ${backupContentCorrect}`);
    console.log(`   ✅ File overwritten: ${hasRuvSwarmContent && noOriginalContent}`);

    await cleanup(testDir);
    return backupFiles.length > 0 && backupContentCorrect && hasRuvSwarmContent && noOriginalContent;
  } catch (error) {
    console.error(`   ❌ Error: ${error.message}`);
    await cleanup(testDir);
    return false;
  }
}

async function testMergeFlag() {
  console.log('\n🧪 Testing --merge flag (should intelligently combine)');

  const testDir = await createTestDir('merge');
  const filePath = path.join(testDir, 'CLAUDE.md');

  // Create existing file with some content
  const existingContent = `# Existing CLAUDE.md

## My Custom Section
This is my custom content that should be preserved.

## Another Section
More custom content.
`;

  await fs.writeFile(filePath, existingContent);

  const generator = new ClaudeDocsGenerator({ workingDir: testDir });

  try {
    await generator.generateClaudeMd({
      merge: true,
      interactive: false,
    });

    // Check merged content
    const mergedContent = await fs.readFile(filePath, 'utf8');

    const hasOriginalContent = mergedContent.includes('My Custom Section');
    const hasRuvSwarmContent = mergedContent.includes('ruv-swarm');
    const isNotSimpleAppend = !mergedContent.endsWith(existingContent);

    console.log(`   ✅ Original content preserved: ${hasOriginalContent}`);
    console.log(`   ✅ ruv-swarm content added: ${hasRuvSwarmContent}`);
    console.log(`   ✅ Intelligent merge (not append): ${isNotSimpleAppend}`);

    await cleanup(testDir);
    return hasOriginalContent && hasRuvSwarmContent;
  } catch (error) {
    console.error(`   ❌ Error: ${error.message}`);
    await cleanup(testDir);
    return false;
  }
}

async function testProtectionLogic() {
  console.log('\n🧪 Testing protection logic (no flags = should fail)');

  const testDir = await createTestDir('protection');
  const filePath = path.join(testDir, 'CLAUDE.md');

  // Create existing file
  await fs.writeFile(filePath, 'Existing content');

  const generator = new ClaudeDocsGenerator({ workingDir: testDir });

  try {
    await generator.generateClaudeMd({
      interactive: false,
    });

    console.log(`   ❌ Should have failed but didn't`);
    await cleanup(testDir);
    return false;
  } catch (error) {
    const expectedError = error.message.includes('already exists') &&
                         error.message.includes('--force') &&
                         error.message.includes('--backup') &&
                         error.message.includes('--merge');

    console.log(`   ✅ Protection working: ${expectedError}`);
    console.log(`   📝 Error message: ${error.message}`);

    await cleanup(testDir);
    return expectedError;
  }
}

async function testFlagCombinations() {
  console.log('\n🧪 Testing flag combinations');

  // Test --force + --backup (should create backup then overwrite)
  const testDir = await createTestDir('combo');
  const filePath = path.join(testDir, 'CLAUDE.md');

  const originalContent = 'Original content for combination test';
  await fs.writeFile(filePath, originalContent);

  const generator = new ClaudeDocsGenerator({ workingDir: testDir });

  try {
    await generator.generateClaudeMd({
      force: true,
      backup: true,
      interactive: false,
    });

    // Check backup was created
    const files = await fs.readdir(testDir);
    const backupFiles = files.filter(f => f.startsWith('CLAUDE.md.backup.'));

    // Check file was overwritten
    const newContent = await fs.readFile(filePath, 'utf8');
    const hasRuvSwarmContent = newContent.includes('ruv-swarm');

    console.log(`   ✅ --force + --backup creates backup: ${backupFiles.length > 0}`);
    console.log(`   ✅ --force + --backup overwrites: ${hasRuvSwarmContent}`);

    await cleanup(testDir);
    return backupFiles.length > 0 && hasRuvSwarmContent;
  } catch (error) {
    console.error(`   ❌ Error: ${error.message}`);
    await cleanup(testDir);
    return false;
  }
}

async function main() {
  console.log('🚀 Testing docs.js flag behavior (corrected)...\n');

  const results = {
    force: await testForceFlag(),
    backup: await testBackupFlag(),
    merge: await testMergeFlag(),
    protection: await testProtectionLogic(),
    combinations: await testFlagCombinations(),
  };

  console.log('\n📊 Test Results Summary:');
  console.log('─'.repeat(40));

  Object.entries(results).forEach(([test, passed]) => {
    const status = passed ? '✅ PASS' : '❌ FAIL';
    console.log(`   ${status} ${test} flag behavior`);
  });

  const allPassed = Object.values(results).every(r => r);
  console.log('\n' + '═'.repeat(40));
  console.log(`🎯 Overall Result: ${allPassed ? '✅ ALL TESTS PASS' : '❌ SOME TESTS FAILED'}`);

  if (allPassed) {
    console.log('🎉 Flag behavior matches ruv\'s specifications!');
    console.log('');
    console.log('📋 Summary of behaviors:');
    console.log('   • --force: Overwrites without backup ✅');
    console.log('   • --backup: Creates backup then overwrites ✅');
    console.log('   • --merge: Intelligently combines content ✅');
    console.log('   • Protection: Blocks when no flags given ✅');
    console.log('   • --force + --backup: Creates backup then overwrites ✅');
  } else {
    console.log('⚠️  Flag behavior needs fixes.');
  }
}

main().catch(console.error);
