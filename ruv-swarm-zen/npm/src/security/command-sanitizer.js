/**
 * Security module for command sanitization and input validation
 * Addresses critical vulnerabilities in Issue #115
 */

const { spawn } = require('child_process');
const path = require('path');

class CommandSanitizer {
  /**
   * Safe command execution using spawn instead of execSync
   * Prevents command injection by using argument arrays
   */
  static async safeExec(command, args = [], options = {}) {
    return new Promise((resolve, reject) => {
      const child = spawn(command, args, {
        ...options,
        stdio: options.stdio || 'pipe',
      });

      let stdout = '';
      let stderr = '';

      if (child.stdout) {
        child.stdout.on('data', (data) => {
          stdout += data.toString();
        });
      }

      if (child.stderr) {
        child.stderr.on('data', (data) => {
          stderr += data.toString();
        });
      }

      child.on('close', (code) => {
        if (code === 0) {
          resolve({ stdout: stdout.trim(), stderr: stderr.trim(), code });
        } else {
          reject(new Error(`Command failed with code ${code}: ${stderr}`));
        }
      });

      child.on('error', (error) => {
        reject(error);
      });
    });
  }

  /**
   * Validate and sanitize issue numbers
   */
  static validateIssueNumber(issueNumber) {
    const num = parseInt(issueNumber, 10);
    if (isNaN(num) || num <= 0 || num > 999999) {
      throw new Error('Invalid issue number: must be positive integer < 1000000');
    }
    return num;
  }

  /**
   * Validate and sanitize repository owner/name
   */
  static validateRepoIdentifier(identifier) {
    // GitHub username/org name rules: alphanumeric, hyphens, max 39 chars
    const repoRegex = /^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$/;
    if (!repoRegex.test(identifier) || identifier.length > 39) {
      throw new Error('Invalid repository identifier: must be alphanumeric with hyphens, max 39 chars');
    }
    return identifier;
  }

  /**
   * Sanitize swarm ID to prevent path traversal and injection
   */
  static sanitizeSwarmId(swarmId) {
    // Allow only alphanumeric, hyphens, underscores, max 50 chars
    const sanitized = swarmId.replace(/[^a-zA-Z0-9_-]/g, '');
    if (sanitized.length === 0 || sanitized.length > 50) {
      throw new Error('Invalid swarm ID: must be alphanumeric with underscores/hyphens, max 50 chars');
    }
    return sanitized;
  }

  /**
   * Sanitize labels to prevent injection
   */
  static sanitizeLabel(label) {
    // GitHub label rules: no spaces, special chars limited
    const sanitized = label.replace(/[^a-zA-Z0-9._-]/g, '');
    if (sanitized.length === 0 || sanitized.length > 50) {
      throw new Error('Invalid label: must be alphanumeric with dots/underscores/hyphens, max 50 chars');
    }
    return sanitized;
  }

  /**
   * Sanitize comment/message content
   */
  static sanitizeMessage(message) {
    if (typeof message !== 'string') {
      throw new Error('Message must be a string');
    }

    // Remove null bytes and control characters (except newlines/tabs)
    const sanitized = message.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');

    // Limit length to prevent DoS
    if (sanitized.length > 10000) {
      throw new Error('Message too long: maximum 10000 characters');
    }

    return sanitized;
  }

  /**
   * Validate file paths to prevent directory traversal
   */
  static validateFilePath(filePath) {
    const normalized = path.normalize(filePath);

    // Prevent directory traversal
    if (normalized.includes('..') || path.isAbsolute(normalized)) {
      throw new Error('Invalid file path: no directory traversal or absolute paths allowed');
    }

    // Whitelist allowed file extensions
    const allowedExtensions = ['.json', '.md', '.txt', '.yml', '.yaml'];
    const ext = path.extname(normalized).toLowerCase();
    if (!allowedExtensions.includes(ext)) {
      throw new Error(`Invalid file extension: only ${allowedExtensions.join(', ')} allowed`);
    }

    return normalized;
  }

  /**
   * Sanitize branch names
   */
  static sanitizeBranchName(branchName) {
    // Git branch name rules: no spaces, special chars, slashes at start/end
    const sanitized = branchName.replace(/[^a-zA-Z0-9._/-]/g, '');
    if (sanitized.length === 0 || sanitized.length > 100 ||
        sanitized.startsWith('/') || sanitized.endsWith('/') ||
        sanitized.includes('..')) {
      throw new Error('Invalid branch name');
    }
    return sanitized;
  }

  /**
   * Validate environment variables
   */
  static validateEnvironment() {
    const required = ['GITHUB_OWNER', 'GITHUB_REPO'];
    const missing = required.filter(env => !process.env[env]);

    if (missing.length > 0) {
      throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
    }

    // Validate the environment values
    try {
      this.validateRepoIdentifier(process.env.GITHUB_OWNER);
      this.validateRepoIdentifier(process.env.GITHUB_REPO);
    } catch (error) {
      throw new Error(`Invalid environment variable: ${error.message}`);
    }
  }

  /**
   * Create safe GitHub CLI command arguments
   */
  static createGitHubArgs(operation, params) {
    const validOperations = {
      'issue-list': ['issue', 'list'],
      'issue-edit': ['issue', 'edit'],
      'issue-comment': ['issue', 'comment'],
      'pr-create': ['pr', 'create'],
    };

    if (!validOperations[operation]) {
      throw new Error(`Invalid GitHub operation: ${operation}`);
    }

    const baseArgs = validOperations[operation];
    const args = [...baseArgs];

    // Add repository parameter safely
    if (params.repo) {
      const [owner, repo] = params.repo.split('/');
      this.validateRepoIdentifier(owner);
      this.validateRepoIdentifier(repo);
      args.push('--repo', `${owner}/${repo}`);
    }

    // Add other parameters based on operation
    switch (operation) {
      case 'issue-list':
        if (params.state) {
          const validStates = ['open', 'closed', 'all'];
          if (validStates.includes(params.state)) {
            args.push('--state', params.state);
          }
        }
        if (params.label) {
          args.push('--label', this.sanitizeLabel(params.label));
        }
        if (params.limit) {
          const limit = parseInt(params.limit, 10);
          if (limit > 0 && limit <= 1000) {
            args.push('--limit', limit.toString());
          }
        }
        args.push('--json', 'number,title,labels,assignees,state,body');
        break;

      case 'issue-edit':
        if (params.issueNumber) {
          args.push(this.validateIssueNumber(params.issueNumber).toString());
        }
        if (params.addLabel) {
          args.push('--add-label', this.sanitizeLabel(params.addLabel));
        }
        if (params.removeLabel) {
          args.push('--remove-label', this.sanitizeLabel(params.removeLabel));
        }
        break;

      case 'issue-comment':
        if (params.issueNumber) {
          args.push(this.validateIssueNumber(params.issueNumber).toString());
        }
        if (params.body) {
          args.push('--body', this.sanitizeMessage(params.body));
        }
        break;

      case 'pr-create':
        if (params.title) {
          args.push('--title', this.sanitizeMessage(params.title));
        }
        if (params.body) {
          args.push('--body', this.sanitizeMessage(params.body));
        }
        if (params.base) {
          args.push('--base', this.sanitizeBranchName(params.base));
        }
        if (params.head) {
          args.push('--head', this.sanitizeBranchName(params.head));
        }
        break;
    }

    return args;
  }

  /**
   * Create safe git command arguments
   */
  static createGitArgs(operation, params) {
    const validOperations = {
      'checkout': ['checkout'],
      'add': ['add'],
      'commit': ['commit'],
      'push': ['push'],
    };

    if (!validOperations[operation]) {
      throw new Error(`Invalid git operation: ${operation}`);
    }

    const args = [...validOperations[operation]];

    switch (operation) {
      case 'checkout':
        if (params.createBranch) {
          args.push('-b');
        }
        if (params.branch) {
          args.push(this.sanitizeBranchName(params.branch));
        }
        break;

      case 'add':
        if (params.file) {
          args.push(this.validateFilePath(params.file));
        }
        break;

      case 'commit':
        if (params.message) {
          args.push('-m', this.sanitizeMessage(params.message));
        }
        break;

      case 'push':
        if (params.remote) {
          args.push(this.validateRepoIdentifier(params.remote));
        }
        if (params.branch) {
          args.push(this.sanitizeBranchName(params.branch));
        }
        break;
    }

    return args;
  }
}

module.exports = CommandSanitizer;
