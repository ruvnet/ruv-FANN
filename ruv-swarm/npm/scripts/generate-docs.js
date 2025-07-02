#!/usr/bin/env node

/**
 * Documentation Generation Script for ruv-swarm
 * Automatically generates comprehensive documentation from code and metadata
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const CONFIG = {
  sourceDir: path.resolve(__dirname, '..', 'src'),
  docsDir: path.resolve(__dirname, '..', 'docs'),
  outputDir: path.resolve(__dirname, '..', 'docs', 'generated'),
  packagePath: path.resolve(__dirname, '..', 'package.json'),
  examplesDir: path.resolve(__dirname, '..', 'examples'),
  testDir: path.resolve(__dirname, '..', 'test')
};

// Utility functions
const log = (message) => console.log(`[DocGen] ${message}`);
const error = (message) => console.error(`[DocGen ERROR] ${message}`);

// Read package.json
async function getPackageInfo() {
  try {
    const packageContent = await fs.readFile(CONFIG.packagePath, 'utf8');
    return JSON.parse(packageContent);
  } catch (err) {
    error(`Failed to read package.json: ${err.message}`);
    process.exit(1);
  }
}

// Extract API information from source files
async function extractAPIInfo() {
  const apiInfo = {
    classes: [],
    functions: [],
    interfaces: [],
    types: [],
    constants: []
  };

  try {
    const files = await findSourceFiles(CONFIG.sourceDir);
    
    for (const file of files) {
      const content = await fs.readFile(file, 'utf8');
      const info = parseSourceFile(content, file);
      
      apiInfo.classes.push(...info.classes);
      apiInfo.functions.push(...info.functions);
      apiInfo.interfaces.push(...info.interfaces);
      apiInfo.types.push(...info.types);
      apiInfo.constants.push(...info.constants);
    }
  } catch (err) {
    error(`Failed to extract API info: ${err.message}`);
  }

  return apiInfo;
}

// Find all source files
async function findSourceFiles(dir) {
  const files = [];
  
  async function traverse(currentDir) {
    const entries = await fs.readdir(currentDir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);
      
      if (entry.isDirectory() && entry.name !== 'node_modules') {
        await traverse(fullPath);
      } else if (entry.isFile() && /\.(js|ts|mjs)$/.test(entry.name)) {
        files.push(fullPath);
      }
    }
  }
  
  await traverse(dir);
  return files;
}

// Parse source file for API elements
function parseSourceFile(content, filePath) {
  const info = {
    classes: [],
    functions: [],
    interfaces: [],
    types: [],
    constants: []
  };

  const relativePath = path.relative(CONFIG.sourceDir, filePath);

  // Extract classes
  const classMatches = content.matchAll(/export\s+class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{([^}]+)}/g);
  for (const match of classMatches) {
    const [, name, parent, body] = match;
    const methods = extractMethods(body);
    const properties = extractProperties(body);
    
    info.classes.push({
      name,
      parent,
      methods,
      properties,
      file: relativePath,
      documentation: extractJSDoc(content, match.index)
    });
  }

  // Extract functions
  const functionMatches = content.matchAll(/export\s+(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*{/g);
  for (const match of functionMatches) {
    const [, name, params, returnType] = match;
    
    info.functions.push({
      name,
      parameters: parseParameters(params),
      returnType: returnType?.trim(),
      file: relativePath,
      documentation: extractJSDoc(content, match.index)
    });
  }

  // Extract interfaces (TypeScript)
  const interfaceMatches = content.matchAll(/export\s+interface\s+(\w+)(?:\s+extends\s+([^{]+))?\s*{([^}]+)}/g);
  for (const match of interfaceMatches) {
    const [, name, parent, body] = match;
    const properties = extractInterfaceProperties(body);
    
    info.interfaces.push({
      name,
      parent,
      properties,
      file: relativePath,
      documentation: extractJSDoc(content, match.index)
    });
  }

  // Extract type definitions
  const typeMatches = content.matchAll(/export\s+type\s+(\w+)\s*=\s*([^;]+);/g);
  for (const match of typeMatches) {
    const [, name, definition] = match;
    
    info.types.push({
      name,
      definition: definition.trim(),
      file: relativePath,
      documentation: extractJSDoc(content, match.index)
    });
  }

  // Extract constants
  const constMatches = content.matchAll(/export\s+const\s+(\w+)(?:\s*:\s*([^=]+))?\s*=\s*([^;]+);/g);
  for (const match of constMatches) {
    const [, name, type, value] = match;
    
    info.constants.push({
      name,
      type: type?.trim(),
      value: value.trim(),
      file: relativePath,
      documentation: extractJSDoc(content, match.index)
    });
  }

  return info;
}

// Extract methods from class body
function extractMethods(body) {
  const methods = [];
  const methodMatches = body.matchAll(/(?:async\s+)?(\w+)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*{/g);
  
  for (const match of methodMatches) {
    const [, name, params, returnType] = match;
    
    methods.push({
      name,
      parameters: parseParameters(params),
      returnType: returnType?.trim(),
      isAsync: match[0].includes('async')
    });
  }
  
  return methods;
}

// Extract properties from class body
function extractProperties(body) {
  const properties = [];
  const propertyMatches = body.matchAll(/(?:readonly\s+)?(\w+)(?:\s*:\s*([^;=]+))?(?:\s*=\s*([^;]+))?;/g);
  
  for (const match of propertyMatches) {
    const [, name, type, defaultValue] = match;
    
    properties.push({
      name,
      type: type?.trim(),
      defaultValue: defaultValue?.trim(),
      readonly: match[0].includes('readonly')
    });
  }
  
  return properties;
}

// Extract interface properties
function extractInterfaceProperties(body) {
  const properties = [];
  const propertyMatches = body.matchAll(/(\w+)(\?)?:\s*([^;]+);/g);
  
  for (const match of propertyMatches) {
    const [, name, optional, type] = match;
    
    properties.push({
      name,
      type: type.trim(),
      optional: !!optional,
      description: ''
    });
  }
  
  return properties;
}

// Parse function parameters
function parseParameters(paramsString) {
  if (!paramsString?.trim()) return [];
  
  const params = [];
  const paramMatches = paramsString.matchAll(/(\w+)(\?)?(?:\s*:\s*([^,=]+))?(?:\s*=\s*([^,]+))?/g);
  
  for (const match of paramMatches) {
    const [, name, optional, type, defaultValue] = match;
    
    params.push({
      name,
      type: type?.trim(),
      optional: !!optional || !!defaultValue,
      defaultValue: defaultValue?.trim()
    });
  }
  
  return params;
}

// Extract JSDoc comments
function extractJSDoc(content, position) {
  const beforeContent = content.substring(0, position);
  const lines = beforeContent.split('\n');
  
  let jsdocStart = -1;
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (line.startsWith('*/')) continue;
    if (line.startsWith('/**')) {
      jsdocStart = i;
      break;
    }
    if (!line.startsWith('*') && line !== '') break;
  }
  
  if (jsdocStart === -1) return '';
  
  const jsdocLines = lines.slice(jsdocStart, lines.length);
  return jsdocLines
    .map(line => line.replace(/^\s*\*\s?/, ''))
    .join('\n')
    .trim();
}

// Extract examples from examples directory
async function extractExamples() {
  const examples = [];
  
  try {
    const files = await fs.readdir(CONFIG.examplesDir);
    
    for (const file of files) {
      if (!file.endsWith('.js') && !file.endsWith('.ts')) continue;
      
      const filePath = path.join(CONFIG.examplesDir, file);
      const content = await fs.readFile(filePath, 'utf8');
      
      // Extract description from comments
      const descriptionMatch = content.match(/\/\*\*\s*\n\s*\*\s*([^\n]+)/);
      const description = descriptionMatch ? descriptionMatch[1].trim() : '';
      
      examples.push({
        name: file.replace(/\.(js|ts)$/, ''),
        file: file,
        description,
        content
      });
    }
  } catch (err) {
    log(`No examples directory found or error reading: ${err.message}`);
  }
  
  return examples;
}

// Generate API reference documentation
async function generateAPIReference(apiInfo, packageInfo) {
  const content = `# ruv-swarm API Reference

*Generated automatically from source code*

**Version**: ${packageInfo.version}  
**Generated**: ${new Date().toISOString()}

## Overview

${packageInfo.description}

## Installation

\`\`\`bash
npm install ${packageInfo.name}
\`\`\`

## Table of Contents

- [Classes](#classes)
- [Functions](#functions)
- [Interfaces](#interfaces)
- [Types](#types)
- [Constants](#constants)

---

## Classes

${apiInfo.classes.map(cls => `
### ${cls.name}

${cls.documentation || '*No description available*'}

**File**: \`${cls.file}\`  
${cls.parent ? `**Extends**: \`${cls.parent}\`` : ''}

#### Properties

${cls.properties.length > 0 ? cls.properties.map(prop => `
- **${prop.name}**${prop.readonly ? ' *(readonly)*' : ''}: \`${prop.type || 'any'}\`${prop.defaultValue ? ` = \`${prop.defaultValue}\`` : ''}
`).join('') : '*No public properties*'}

#### Methods

${cls.methods.length > 0 ? cls.methods.map(method => `
##### ${method.name}(${method.parameters.map(p => `${p.name}${p.optional ? '?' : ''}: ${p.type || 'any'}`).join(', ')})${method.returnType ? `: ${method.returnType}` : ''}

${method.isAsync ? '*Async method*' : ''}

**Parameters:**
${method.parameters.length > 0 ? method.parameters.map(p => `
- **${p.name}**${p.optional ? ' *(optional)*' : ''}: \`${p.type || 'any'}\`${p.defaultValue ? ` = \`${p.defaultValue}\`` : ''}
`).join('') : '*No parameters*'}

`).join('') : '*No public methods*'}

---
`).join('')}

## Functions

${apiInfo.functions.map(func => `
### ${func.name}(${func.parameters.map(p => `${p.name}${p.optional ? '?' : ''}: ${p.type || 'any'}`).join(', ')})${func.returnType ? `: ${func.returnType}` : ''}

${func.documentation || '*No description available*'}

**File**: \`${func.file}\`

**Parameters:**
${func.parameters.length > 0 ? func.parameters.map(p => `
- **${p.name}**${p.optional ? ' *(optional)*' : ''}: \`${p.type || 'any'}\`${p.defaultValue ? ` = \`${p.defaultValue}\`` : ''}
`).join('') : '*No parameters*'}

---
`).join('')}

## Interfaces

${apiInfo.interfaces.map(iface => `
### ${iface.name}

${iface.documentation || '*No description available*'}

**File**: \`${iface.file}\`  
${iface.parent ? `**Extends**: \`${iface.parent}\`` : ''}

\`\`\`typescript
interface ${iface.name} {
${iface.properties.map(prop => `  ${prop.name}${prop.optional ? '?' : ''}: ${prop.type};`).join('\n')}
}
\`\`\`

**Properties:**
${iface.properties.map(prop => `
- **${prop.name}**${prop.optional ? ' *(optional)*' : ''}: \`${prop.type}\`
`).join('')}

---
`).join('')}

## Types

${apiInfo.types.map(type => `
### ${type.name}

${type.documentation || '*No description available*'}

**File**: \`${type.file}\`

\`\`\`typescript
type ${type.name} = ${type.definition};
\`\`\`

---
`).join('')}

## Constants

${apiInfo.constants.map(constant => `
### ${constant.name}

${constant.documentation || '*No description available*'}

**File**: \`${constant.file}\`  
**Type**: \`${constant.type || 'auto'}\`  
**Value**: \`${constant.value}\`

---
`).join('')}

## Generated Information

This documentation was automatically generated from the source code on ${new Date().toLocaleDateString()}.

For more information, visit:
- [GitHub Repository](${packageInfo.repository?.url})
- [NPM Package](https://www.npmjs.com/package/${packageInfo.name})
- [Issues](${packageInfo.bugs?.url})
`;

  return content;
}

// Generate examples documentation
async function generateExamplesDoc(examples) {
  const content = `# ruv-swarm Examples

*Code examples demonstrating ruv-swarm usage*

## Table of Contents

${examples.map(example => `- [${example.name}](#${example.name.toLowerCase().replace(/\s+/g, '-')})`).join('\n')}

---

${examples.map(example => `
## ${example.name}

${example.description}

**File**: \`${example.file}\`

\`\`\`javascript
${example.content}
\`\`\`

---
`).join('')}

## Running Examples

All examples can be run using Node.js:

\`\`\`bash
# Clone the repository
git clone https://github.com/ruvnet/ruv-FANN.git
cd ruv-FANN/ruv-swarm/npm

# Install dependencies
npm install

# Run an example
node examples/[example-file].js
\`\`\`

For more examples and use cases, see:
- [Integration Guide](./INTEGRATION_GUIDE.md)
- [API Reference](./API_REFERENCE_COMPLETE.md)
- [GitHub Repository](https://github.com/ruvnet/ruv-FANN)
`;

  return content;
}

// Generate CLI documentation
async function generateCLIDoc(packageInfo) {
  const content = `# ruv-swarm CLI Reference

*Command-line interface for ruv-swarm*

## Installation

\`\`\`bash
# NPX (no installation required)
npx ruv-swarm --help

# Global installation
npm install -g ruv-swarm

# Local installation
npm install ruv-swarm
\`\`\`

## Quick Start

\`\`\`bash
# Initialize a swarm
npx ruv-swarm init mesh 5

# Spawn an agent
npx ruv-swarm spawn researcher "AI Research Assistant"

# Orchestrate a task
npx ruv-swarm orchestrate "Analyze neural architecture trends"

# Check status
npx ruv-swarm status
\`\`\`

## Commands

### Core Commands

#### \`init [topology] [maxAgents]\`

Initialize a new swarm with specified topology and agent limit.

**Arguments:**
- \`topology\` - Swarm topology: \`mesh\`, \`hierarchical\`, \`ring\`, \`star\`
- \`maxAgents\` - Maximum number of agents (1-100)

**Options:**
- \`--claude\` - Set up Claude Code integration
- \`--force\` - Force regeneration of integration files
- \`--forecasting\` - Enable forecasting capabilities

**Examples:**
\`\`\`bash
npx ruv-swarm init mesh 10
npx ruv-swarm init hierarchical 20 --claude
npx ruv-swarm init mesh 5 --claude --force
\`\`\`

#### \`spawn <type> [name]\`

Spawn a new agent in the swarm.

**Arguments:**
- \`type\` - Agent type: \`researcher\`, \`coder\`, \`analyst\`, \`optimizer\`, \`coordinator\`, \`architect\`, \`tester\`
- \`name\` - Optional agent name (1-100 characters)

**Options:**
- \`--no-neural\` - Disable neural network for this agent

**Examples:**
\`\`\`bash
npx ruv-swarm spawn researcher
npx ruv-swarm spawn coder "Senior Developer"
npx ruv-swarm spawn analyst "Data Scientist" --no-neural
\`\`\`

#### \`orchestrate <task>\`

Orchestrate a task across the swarm.

**Arguments:**
- \`task\` - Task description (1-1000 characters)

**Examples:**
\`\`\`bash
npx ruv-swarm orchestrate "Build a REST API with authentication"
npx ruv-swarm orchestrate "Analyze customer behavior patterns"
npx ruv-swarm orchestrate "Optimize database query performance"
\`\`\`

#### \`status [--verbose]\`

Show current swarm status and metrics.

**Options:**
- \`--verbose\`, \`-v\` - Show detailed information

**Examples:**
\`\`\`bash
npx ruv-swarm status
npx ruv-swarm status --verbose
\`\`\`

#### \`monitor [duration]\`

Monitor swarm activity in real-time.

**Arguments:**
- \`duration\` - Monitoring duration in milliseconds (default: 10000)

**Examples:**
\`\`\`bash
npx ruv-swarm monitor
npx ruv-swarm monitor 30000  # Monitor for 30 seconds
\`\`\`

### MCP Commands

#### \`mcp start [options]\`

Start MCP server for Claude Code integration.

**Options:**
- \`--protocol=stdio\` - Use stdio protocol (default)
- \`--host=<host>\` - Bind to specific host (default: localhost)
- \`--port=<port>\` - Use specific port (default: 3000)

**Examples:**
\`\`\`bash
npx ruv-swarm mcp start
npx ruv-swarm mcp start --host 0.0.0.0 --port 3000
\`\`\`

#### \`mcp status\`

Check MCP server status.

#### \`mcp tools\`

List available MCP tools for Claude Code.

#### \`mcp stop\`

Stop MCP server.

### Advanced Commands

#### \`neural <subcommand>\`

Neural network operations.

**Subcommands:**
- \`status\` - Show neural network status
- \`train [options]\` - Train neural models
- \`patterns [model]\` - View learned patterns  
- \`export [options]\` - Export neural weights

**Examples:**
\`\`\`bash
npx ruv-swarm neural status
npx ruv-swarm neural train --model attention --iterations 100
npx ruv-swarm neural patterns --model attention
\`\`\`

#### \`benchmark <subcommand>\`

Performance benchmarking.

**Subcommands:**
- \`run [options]\` - Run performance benchmarks
- \`compare [files]\` - Compare benchmark results

**Examples:**
\`\`\`bash
npx ruv-swarm benchmark run --iterations 10
npx ruv-swarm benchmark run --test swarm-coordination
npx ruv-swarm benchmark compare results-1.json results-2.json
\`\`\`

#### \`performance <subcommand>\`

Performance analysis and optimization.

**Subcommands:**
- \`analyze [options]\` - Analyze performance bottlenecks
- \`optimize [target]\` - Optimize swarm configuration
- \`suggest\` - Get optimization suggestions

**Examples:**
\`\`\`bash
npx ruv-swarm performance analyze --task-id recent
npx ruv-swarm performance optimize --target speed
npx ruv-swarm performance suggest
\`\`\`

#### \`hook <type> [options]\`

Claude Code hooks integration.

**Types:**
- \`pre-task\` - Pre-task hook
- \`post-task\` - Post-task hook
- \`pre-edit\` - Pre-edit hook
- \`post-edit\` - Post-edit hook
- \`git-commit\` - Git commit hook

**Examples:**
\`\`\`bash
npx ruv-swarm hook pre-task --description "Build authentication"
npx ruv-swarm hook post-edit --file app.js --memory-key edit-history
npx ruv-swarm hook git-commit --agent coder-123 --generate-report
\`\`\`

#### \`claude-invoke <prompt>\`

Invoke Claude Code with swarm integration.

**Arguments:**
- \`prompt\` - Prompt for Claude Code

**Examples:**
\`\`\`bash
npx ruv-swarm claude-invoke "Create a development swarm for my project"
npx ruv-swarm claude-invoke "Analyze this codebase and suggest improvements"
\`\`\`

### Utility Commands

#### \`version\`

Show version information.

#### \`help\`

Show help message with all available commands.

## Validation Rules

### Input Validation

- **Topologies**: Must be one of: \`mesh\`, \`hierarchical\`, \`ring\`, \`star\`
- **Max Agents**: Integer between 1 and 100
- **Agent Types**: Must be one of: \`researcher\`, \`coder\`, \`analyst\`, \`optimizer\`, \`coordinator\`, \`architect\`, \`tester\`
- **Agent Names**: 1-100 characters, alphanumeric + spaces/hyphens/underscores/periods
- **Task Descriptions**: 1-1000 characters, non-empty

### Error Handling

All commands include comprehensive error handling and validation:

\`\`\`bash
# Invalid topology
npx ruv-swarm init invalid 5
# Output: ❌ Validation Error: Invalid topology 'invalid'

# Invalid agent count
npx ruv-swarm init mesh 999
# Output: ❌ Validation Error: Invalid maxAgents '999'. Must be between 1 and 100

# Invalid agent type
npx ruv-swarm spawn invalid-type
# Output: ❌ Validation Error: Invalid agent type 'invalid-type'
\`\`\`

## Environment Variables

Configure ruv-swarm behavior with environment variables:

\`\`\`bash
# Core configuration
export RUVA_SWARM_MAX_AGENTS=50
export RUVA_SWARM_TOPOLOGY=mesh
export RUVA_SWARM_PERSISTENCE=sqlite

# Performance tuning
export RUVA_SWARM_WASM_SIMD=true
export RUVA_SWARM_MEMORY_POOL=256MB
export RUVA_SWARM_WORKER_THREADS=4

# Debugging
export RUST_LOG=info
export RUVA_SWARM_DEBUG=true
export DEBUG=ruv-swarm:*
\`\`\`

## Remote Server Usage

ruv-swarm works seamlessly on remote servers using npx:

\`\`\`bash
# Execute on remote server via SSH
ssh user@remote-server 'npx ruv-swarm init mesh 10'

# Start MCP server on remote host
ssh user@remote-server 'npx ruv-swarm mcp start --host 0.0.0.0 &'

# Monitor remote swarm
ssh user@remote-server 'npx ruv-swarm monitor --duration 60000'
\`\`\`

## Exit Codes

- \`0\` - Success
- \`1\` - General error
- \`2\` - Validation error
- \`3\` - Network error
- \`4\` - File system error
- \`5\` - WASM loading error

## Troubleshooting

### Common Issues

**WASM Module Loading**:
\`\`\`bash
npx ruv-swarm features  # Check WASM support
\`\`\`

**Network Issues**:
\`\`\`bash
npx ruv-swarm mcp start --host 0.0.0.0  # Bind to all interfaces
\`\`\`

**Performance Problems**:
\`\`\`bash
npx ruv-swarm benchmark run  # Run performance tests
npx ruv-swarm performance analyze  # Analyze bottlenecks
\`\`\`

### Debug Mode

Enable detailed logging for troubleshooting:

\`\`\`bash
export DEBUG=ruv-swarm:*
export RUVA_SWARM_DEBUG=true
npx ruv-swarm --verbose [command]
\`\`\`

## Support

- **Documentation**: [Full Documentation](./README.md)
- **Issues**: [GitHub Issues](https://github.com/ruvnet/ruv-FANN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ruvnet/ruv-FANN/discussions)

---

*CLI documentation generated for ruv-swarm v${packageInfo.version}*
`;

  return content;
}

// Main function
async function main() {
  log('Starting documentation generation...');
  
  try {
    // Ensure output directory exists
    await fs.mkdir(CONFIG.outputDir, { recursive: true });
    
    // Get package information
    const packageInfo = await getPackageInfo();
    log(`Generating docs for ${packageInfo.name} v${packageInfo.version}`);
    
    // Extract API information
    log('Extracting API information from source files...');
    const apiInfo = await extractAPIInfo();
    
    // Extract examples
    log('Extracting examples...');
    const examples = await extractExamples();
    
    // Generate documentation files
    log('Generating API reference...');
    const apiReference = await generateAPIReference(apiInfo, packageInfo);
    await fs.writeFile(path.join(CONFIG.outputDir, 'API_REFERENCE_GENERATED.md'), apiReference);
    
    log('Generating examples documentation...');
    const examplesDoc = await generateExamplesDoc(examples);
    await fs.writeFile(path.join(CONFIG.outputDir, 'EXAMPLES.md'), examplesDoc);
    
    log('Generating CLI documentation...');
    const cliDoc = await generateCLIDoc(packageInfo);
    await fs.writeFile(path.join(CONFIG.outputDir, 'CLI_REFERENCE.md'), cliDoc);
    
    // Generate summary
    const summary = `# Documentation Generation Summary

Generated on: ${new Date().toISOString()}
Package: ${packageInfo.name} v${packageInfo.version}

## Generated Files

- \`API_REFERENCE_GENERATED.md\` - Auto-generated API reference
- \`EXAMPLES.md\` - Code examples and usage
- \`CLI_REFERENCE.md\` - Command-line interface documentation

## API Analysis

- **Classes**: ${apiInfo.classes.length}
- **Functions**: ${apiInfo.functions.length}
- **Interfaces**: ${apiInfo.interfaces.length}
- **Types**: ${apiInfo.types.length}
- **Constants**: ${apiInfo.constants.length}
- **Examples**: ${examples.length}

## Source Files Analyzed

${(await findSourceFiles(CONFIG.sourceDir)).map(file => `- ${path.relative(CONFIG.sourceDir, file)}`).join('\n')}

## Next Steps

1. Review generated documentation for accuracy
2. Update manual documentation as needed
3. Ensure examples are up to date
4. Run deployment validation

---

*Generated by ruv-swarm documentation generator*
`;
    
    await fs.writeFile(path.join(CONFIG.outputDir, 'GENERATION_SUMMARY.md'), summary);
    
    log('Documentation generation completed successfully!');
    log(`Files generated in: ${CONFIG.outputDir}`);
    log(`- API Reference: ${apiInfo.classes.length} classes, ${apiInfo.functions.length} functions`);
    log(`- Examples: ${examples.length} examples`);
    log(`- CLI Documentation: Complete command reference`);
    
  } catch (err) {
    error(`Documentation generation failed: ${err.message}`);
    process.exit(1);
  }
}

// Run the documentation generator
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { main as generateDocs };