/**
 * Documentation generation module for Claude Code integration
 * Generates claude.md and .claude/commands/ documentation
 */

import fs from 'fs/promises';
import path from 'path';
import { getSupportedToolDefinitions } from '../mcp-tools-enhanced.js';

class ClaudeDocsGenerator {
  constructor(options = {}) {
    this.workingDir = options.workingDir || process.cwd();
  }

  /**
     * Generate main claude.md configuration file with protection
     */
  async generateClaudeMd(options = {}) {
    const { force = false, merge = false, backup = false, noBackup = false, interactive = true } = options;

    // Check if CLAUDE.md already exists
    const filePath = path.join(this.workingDir, 'CLAUDE.md');
    const fileExists = await this.fileExists(filePath);

    if (fileExists && !force && !merge && !backup) {
      if (interactive) {
        // Interactive prompt for action
        const action = await this.promptUserAction(filePath);
        if (action === 'cancel') {
          throw new Error('CLAUDE.md generation cancelled by user');
        } else if (action === 'overwrite') {
          await this.createBackup(filePath);
        } else if (action === 'merge') {
          return await this.mergeClaudeMd(filePath);
        }
      } else {
        // Non-interactive mode - fail safely
        throw new Error('CLAUDE.md already exists. Use --force to overwrite, --backup to backup existing, or --merge to combine.');
      }
    } else if (fileExists && force) {
      // Force flag: overwrite with optional backup creation
      if (!noBackup) {
        await this.createBackup(filePath);
        console.log('📄 Backing up existing CLAUDE.md before force overwrite');
      } else {
        console.log('⚠️  Force overwriting existing CLAUDE.md (no backup - disabled by --no-backup)');
      }
    } else if (fileExists && backup && !force && !merge) {
      // Backup flag: create backup then overwrite
      await this.createBackup(filePath);
      console.log('📄 Backing up existing CLAUDE.md before overwriting');
    } else if (fileExists && merge) {
      // Merge with existing content (backup first if backup flag is set)
      if (backup) {
        await this.createBackup(filePath);
      }
      return await this.mergeClaudeMd(filePath, noBackup);
    }
    const content = this.getRuvSwarmContent();

    // Write the new content
    await fs.writeFile(filePath, content);

    // Clean up old backups (keep only last 5)
    await this.cleanupOldBackups(filePath);

    return { file: 'CLAUDE.md', success: true, action: 'created' };
  }

  /**
     * Generate command documentation files in organized subdirectories
     */
  async generateCommandDocs() {
    const commandsDir = path.join(this.workingDir, '.claude', 'commands');
    await fs.mkdir(commandsDir, { recursive: true });
    const files = [];
    for (const tool of getSupportedToolDefinitions()) {
      const filename = `ruv-swarm-${tool.name}.md`;
      const content = `# ${tool.name}\n\n${tool.description}\n\nInput schema:\n\n\`\`\`json\n${JSON.stringify(tool.inputSchema, null, 2)}\n\`\`\`\n\nCreating or assigning metadata does not execute an external model.\n`;
      await fs.writeFile(path.join(commandsDir, filename), content);
      files.push(filename);
    }
    return { files, success: true };
  }

  /** Merge project MCP registration without broadening permissions or hooks. */
  async generateSettingsJson() {
    const filePath = path.join(this.workingDir, '.mcp.json');
    let configuration = {};
    try {
      configuration = JSON.parse(await fs.readFile(filePath, 'utf8'));
    } catch (error) {
      if (error.code !== 'ENOENT') {
        throw error;
      }
    }
    if (!configuration || typeof configuration !== 'object' || Array.isArray(configuration) ||
        (configuration.mcpServers !== undefined && (!configuration.mcpServers ||
        typeof configuration.mcpServers !== 'object' || Array.isArray(configuration.mcpServers)))) {
      throw new TypeError('Existing .mcp.json must contain an object and an object mcpServers map');
    }
    const servers = configuration.mcpServers || {};
    if (!Object.hasOwn(servers, 'ruv-swarm')) {
      configuration.mcpServers = { ...servers, 'ruv-swarm': { command: 'npx', args: ['ruv-swarm', 'mcp', 'start'] } };
      await fs.writeFile(filePath, `${JSON.stringify(configuration, null, 2)}\n`, { mode: 0o600 });
    }
    return { file: '.mcp.json', success: true, permissionsChanged: false };
  }

  async fileExists(filePath) {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  /**
     * Create backup of existing file
     */
  async createBackup(filePath) {
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '');
    const backupPath = `${filePath}.backup.${timestamp}`;

    try {
      await fs.copyFile(filePath, backupPath);
      console.log(`📄 Backup created: ${path.basename(backupPath)}`);
      return backupPath;
    } catch (error) {
      console.error('⚠️  Failed to create backup:', error.message);
      throw error;
    }
  }

  /**
     * Clean up old backup files (keep last 5)
     */
  async cleanupOldBackups(filePath) {
    const dir = path.dirname(filePath);
    const baseName = path.basename(filePath);

    try {
      const files = await fs.readdir(dir);
      const backupFiles = files
        .filter(file => file.startsWith(`${baseName}.backup.`))
        .sort()
        .reverse(); // Most recent first

      // Keep only the 5 most recent backups
      const filesToDelete = backupFiles.slice(5);

      for (const file of filesToDelete) {
        try {
          await fs.unlink(path.join(dir, file));
        } catch {
          // Ignore errors deleting old backups
        }
      }
    } catch {
      // Ignore errors in cleanup
    }
  }

  /**
     * Prompt user for action when CLAUDE.md exists
     */
  async promptUserAction(filePath) {
    // In a CLI environment, we need to use a different approach
    // For now, we'll use process.stdin/stdout directly
    const readline = await import('readline');

    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    return new Promise((resolve) => {
      console.log(`\n📁 CLAUDE.md already exists at: ${filePath}`);
      console.log('Choose an action:');
      console.log('  [o] Overwrite (creates backup)');
      console.log('  [m] Merge with existing content');
      console.log('  [c] Cancel operation');

      rl.question('\nYour choice (o/m/c): ', (answer) => {
        rl.close();

        switch (answer.toLowerCase()) {
        case 'o':
        case 'overwrite':
          resolve('overwrite');
          break;
        case 'm':
        case 'merge':
          resolve('merge');
          break;
        case 'c':
        case 'cancel':
        default:
          resolve('cancel');
        }
      });
    });
  }

  /**
     * Merge ruv-swarm content with existing CLAUDE.md
     */
  async mergeClaudeMd(filePath, noBackup = false) {
    try {
      const existingContent = await fs.readFile(filePath, 'utf8');

      console.log('📝 Merging ruv-swarm configuration with existing CLAUDE.md');

      // Create backup first (unless disabled)
      if (!noBackup) {
        await this.createBackup(filePath);
      } else {
        console.log('📝 Skipping backup creation (disabled by --no-backup)');
      }

      // Generate new ruv-swarm content
      const ruvSwarmContent = this.getRuvSwarmContent();

      // Intelligent merging
      const mergedContent = this.intelligentMerge(existingContent, ruvSwarmContent);

      // Write merged content
      await fs.writeFile(filePath, mergedContent);

      console.log('✅ Successfully merged ruv-swarm configuration with existing CLAUDE.md');

      return { file: 'CLAUDE.md', success: true, action: 'merged' };
    } catch (error) {
      console.error('❌ Failed to merge CLAUDE.md:', error.message);
      throw error;
    }
  }

  /**
     * Get the ruv-swarm specific content (full content from generateClaudeMd)
     */
  getRuvSwarmContent() {
    const tools = getSupportedToolDefinitions().map(tool => `- \`${tool.name}\`: ${tool.description}`).join('\n');
    return `# Claude Code Configuration for ruv-swarm

The project uses ruv-swarm for local metadata, SQLite persistence and checked WASM calculations.
Use the project MCP registration in .mcp.json. Run \`npx ruv-swarm mcp start\` for stdio.
Existing client permission rules and hooks are unchanged.

## Available tools

${tools}

## Actual behavior

Create a swarm with \`swarm_init({topology:"mesh",maxAgents:3})\`, then register an agent with
\`agent_spawn({type:"coder",capabilities:["code"]})\`.
\`task_orchestrate({task:"Review the source",requiredCapabilities:["code"]})\` reserves matching idle agents.
An assigned task is metadata, not completed work. Use the application's actual executor for code, shell or model operations.
After handling an assignment, \`agent_update_status({agentId,status:"idle"})\` releases that agent.

\`neural_infer({layers:[2,1],weights:[2,-1],input:[3,4]})\` computes the actual output [2].
\`forecast_predict({model:"linear",input:[1,3,5]})\` computes the one-step forecast [7].
Benchmarks measure local dense inference timing; they do not measure LLM or SWE-bench performance.

Training, autonomous learning, distributed consensus, automatic code execution and dependency scheduling are unavailable.
State defaults to .ruv-swarm/state.sqlite; RUV_SWARM_DB_PATH selects an explicit database.
Local limits are 64 swarms, 1000 agents, 10000 tasks and bounded monitoring snapshots.

Use Claude's normal permission checks. Permission bypass requires an explicit user-controlled opt-in.
Batch only independent operations; perform dependent initialization and assignment in order.
`;
  }

  intelligentMerge(existingContent, ruvSwarmContent) {
    const existingLines = existingContent.split('\n');
    const newLines = ruvSwarmContent.split('\n');

    // Check if ruv-swarm content already exists
    const ruvSwarmSectionIndex = this.findRuvSwarmSection(existingLines);

    if (ruvSwarmSectionIndex !== -1) {
      // Replace existing ruv-swarm section
      console.log('📝 Updating existing ruv-swarm section in CLAUDE.md');
      const sectionEnd = this.findSectionEnd(existingLines, ruvSwarmSectionIndex);

      // Replace the section
      const beforeSection = existingLines.slice(0, ruvSwarmSectionIndex);
      const afterSection = existingLines.slice(sectionEnd);

      return [...beforeSection, ...newLines, '', ...afterSection].join('\n');
    }
    // Intelligently insert ruv-swarm content
    console.log('📝 Integrating ruv-swarm configuration into existing CLAUDE.md');
    return this.intelligentInsert(existingLines, newLines);

  }

  /**
     * Find existing ruv-swarm section in content
     */
  findRuvSwarmSection(lines) {
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].toLowerCase();
      if (line.includes('ruv-swarm') && (line.startsWith('#') || line.includes('claude code configuration'))) {
        return i;
      }
    }
    return -1;
  }

  /**
     * Intelligently insert new content based on context
     */
  intelligentInsert(existingLines, newLines) {
    // Look for appropriate insertion points
    let insertIndex = -1;

    // 1. After main title but before first major section
    for (let i = 0; i < existingLines.length; i++) {
      if (existingLines[i].startsWith('# ') && i > 0) {
        // Found first major heading after title
        insertIndex = i;
        break;
      }
    }

    // 2. If no major headings, look for end of introductory content
    if (insertIndex === -1) {
      for (let i = 0; i < existingLines.length; i++) {
        if (existingLines[i].startsWith('## ') && i > 5) {
          insertIndex = i;
          break;
        }
      }
    }

    // 3. Fallback: insert after first 10 lines or middle of file
    if (insertIndex === -1) {
      insertIndex = Math.min(10, Math.floor(existingLines.length / 2));
    }

    // Insert the content with proper spacing
    const beforeInsert = existingLines.slice(0, insertIndex);
    const afterInsert = existingLines.slice(insertIndex);

    // Add spacing
    const insertContent = ['', '---', '', ...newLines, '', '---', ''];

    return [...beforeInsert, ...insertContent, ...afterInsert].join('\n');
  }

  /**
     * Find the end of a markdown section
     */
  findSectionEnd(lines, startIndex) {
    // Look for next top-level heading or end of file
    for (let i = startIndex + 1; i < lines.length; i++) {
      if (lines[i].startsWith('# ') && !lines[i].includes('ruv-swarm')) {
        return i;
      }
      // Also check for horizontal rules that might separate sections
      if (lines[i].trim() === '---' && i > startIndex + 10) {
        return i;
      }
    }
    return lines.length;
  }

  /**
     * Generate all documentation files
     */
  async generateAll(options = {}) {
    return {
      claudeMd: await this.generateClaudeMd(options),
      commands: await this.generateCommandDocs(),
      advancedCommands: { files: [], supported: false },
      settings: await this.generateSettingsJson(),
      success: true,
    };
  }

}

export { ClaudeDocsGenerator };