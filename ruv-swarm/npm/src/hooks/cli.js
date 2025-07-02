#!/usr/bin/env node

/**
 * CLI handler for ruv-swarm hooks
 * Usage: npx ruv-swarm hook <type> [options]
 */

import { handleHook } from './index.js';

async function main() {
  const args = process.argv.slice(2);

  // Skip if not a hook command
  if (args[0] !== 'hook') {
    return;
  }

  const [, hookType] = args;
  const options = parseArgs(args.slice(2));

  try {
    const result = await handleHook(hookType, options);

    // Output JSON response for Claude Code to parse
    console.log(JSON.stringify(result, null, 2));

    // Exit with appropriate code
    if (result.continue === false) {
      process.exit(2); // Blocking error
    } else {
      process.exit(0); // Success
    }
  } catch (error) {
    console.error(JSON.stringify({
      continue: true,
      error: error.message,
      stack: process.env.DEBUG ? error.stack : undefined,
    }));
    process.exit(1); // Non-blocking error
  }
}

function parseArgs(args) {
  const options = {};

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg.startsWith('--')) {
      const key = arg.substring(2);

      // Check if next arg is a value or another flag
      if (i + 1 < args.length && !args[i + 1].startsWith('--')) {
        // Next arg is the value
        options[toCamelCase(key)] = args[i + 1];
        i++; // Skip the value in next iteration
      } else {
        // Boolean flag
        options[toCamelCase(key)] = true;
      }
    } else if (!args[i - 1]?.startsWith('--')) {
      // Positional argument
      if (!options._) {
        options._ = [];
      }
      options._.push(arg);
    }
  }

  return options;
}

function toCamelCase(str) {
  return str.replace(/-([a-z])/g, (g) => g[1].toUpperCase());
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { main };