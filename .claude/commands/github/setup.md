# GitHub Swarm Coordination Setup

Quick setup guide for GitHub-based swarm coordination in Claude Code.

## Prerequisites

1. GitHub MCP server installed:
   ```bash
   claude mcp add github "npx @modelcontextprotocol/server-github"
   ```

2. GitHub CLI installed (optional but recommended):
   ```bash
   # macOS
   brew install gh
   
   # Linux
   sudo apt install gh
   
   # Then authenticate
   gh auth login
   ```

## Configuration

### 1. Set Your Swarm ID

```bash
# In your shell or .env
export CLAUDE_SWARM_ID="my-unique-swarm-id"

# Or let it auto-generate
export CLAUDE_SWARM_ID="swarm-$(date +%s)"
```

### 2. Configure Repository

```bash
# Set default repo for commands
export GITHUB_OWNER="ruvnet"
export GITHUB_REPO="ruv-FANN"
```

### 3. Add Command Aliases (Optional)

Add to your shell profile:
```bash
# Quick GitHub swarm commands
alias swarm-status="/github status"
alias swarm-tasks="/github tasks"
alias swarm-claim="/github claim"
alias swarm-update="/github update"
```

## First Time Usage

1. **Check system status:**
   ```
   /github status
   ```

2. **Find available work:**
   ```
   /github tasks --priority high
   ```

3. **Claim your first task:**
   ```
   /github claim <issue_number>
   ```

4. **Update progress:**
   ```
   /github update <issue_number> "Started implementation"
   ```

## Label Setup (For Maintainers)

Create these labels in your GitHub repo:

```bash
# Priority labels
gh label create "priority: critical" --color FF0000
gh label create "priority: high" --color FFA500  
gh label create "priority: medium" --color FFFF00
gh label create "priority: low" --color 808080

# Area labels
gh label create "area: core" --color 0052CC
gh label create "area: mcp" --color 5319E7
gh label create "area: neural" --color 206B38
gh label create "area: wasm" --color B60205
gh label create "area: docs" --color 1D76DB

# Coordination labels
gh label create "available" --color 0E8A16
gh label create "swarm-claimed" --color F9D0C4
gh label create "help-wanted" --color 008672
```

## Workflow Integration

### For Individual Contributors

1. Start your session:
   ```
   /github sync
   /github my-tasks
   ```

2. Find and claim work:
   ```
   /github tasks
   /github claim 123
   ```

3. Work and update:
   ```
   # Make changes
   /github update 123 "Progress message"
   
   # When done
   /github complete 123 --pr 456
   ```

### For Team Leads

1. Monitor team:
   ```
   /github swarms
   /github conflicts
   ```

2. Coordinate work:
   ```
   /github coordinate split 789
   /github coordinate sequence 100 101 102
   ```

## Troubleshooting

### Command not found
- Ensure you're in Claude Code
- Check MCP server is running
- Try `/github/status` with full path

### Authentication issues
- Verify GitHub token with `gh auth status`
- Re-add MCP server with fresh token
- Check token permissions (repo, issue access)

### Sync problems
- Run `/github sync --full`
- Check network connectivity
- Verify repository access

## Best Practices

1. **One swarm ID per session** - Don't change mid-work
2. **Update regularly** - Every 30-60 minutes
3. **Release if blocked** - Don't hold tasks
4. **Communicate conflicts** - Use issue comments
5. **Complete or release** - Within 24 hours

## Additional Resources

- Full guide: `/workspaces/ruv-FANN/ruv-swarm/SWARM_COLLABORATION_GUIDE.md`
- Helper script: `/workspaces/ruv-FANN/ruv-swarm/scripts/swarm-helper.sh`
- Issue templates: `/workspaces/ruv-FANN/.github/ISSUE_TEMPLATE/`

## Support

- Create issue with `question` label
- Check existing discussions
- Review closed issues for examples