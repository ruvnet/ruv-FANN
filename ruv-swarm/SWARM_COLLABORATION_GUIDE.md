# 🐝 Swarm Collaboration Guide for ruv-FANN

This guide helps multiple contributors and Claude Code swarms collaborate effectively on the same repository using GitHub's built-in features.

## 📋 Overview

We use GitHub Issues and Projects to coordinate work across multiple swarms without conflicts. Each swarm claims tasks, updates progress, and releases work using labels and comments.

## 🏗️ Project Structure

### 1. GitHub Project Board

Create a project board with these columns:
- **📥 Backlog** - All available tasks
- **🎯 Ready** - Prioritized tasks ready to claim
- **🔄 In Progress** - Tasks being worked on
- **👀 Review** - Completed tasks needing review
- **✅ Done** - Completed and reviewed tasks

### 2. Issue Labels

Create these labels for coordination:

#### Task Type Labels
- `feature` - New functionality
- `bug` - Something needs fixing
- `enhancement` - Improve existing functionality
- `documentation` - Documentation updates
- `test` - Testing improvements
- `refactor` - Code improvements without changing functionality

#### Priority Labels
- `priority: critical` - Urgent, blocks other work
- `priority: high` - Important, should be done soon
- `priority: medium` - Normal priority
- `priority: low` - Nice to have

#### Swarm Coordination Labels
- `available` - Task ready to be claimed
- `swarm-claimed` - Task claimed by a swarm (add swarm ID)
- `swarm-conflict` - Multiple swarms interested in same task
- `help-wanted` - Task needs collaboration

#### Area Labels
- `area: core` - Core swarm functionality
- `area: mcp` - MCP server/tools
- `area: neural` - Neural network features
- `area: wasm` - WebAssembly components
- `area: docs` - Documentation
- `area: tests` - Testing

## 🎯 Workflow for Swarms

### 1. Finding Work

```bash
# List available tasks
gh issue list --label "available" --label "-swarm-claimed"

# List high-priority tasks
gh issue list --label "priority: high" --label "available"

# List tasks in specific area
gh issue list --label "area: neural" --label "available"
```

### 2. Claiming a Task

When a swarm wants to work on an issue:

```bash
# Claim issue #123
gh issue edit 123 --add-label "swarm-claimed"

# Add comment with swarm ID and plan
gh issue comment 123 --body "🐝 Swarm ID: claude-abc123 claiming this task.

Plan:
1. Analyze current implementation
2. Design solution
3. Implement changes
4. Add tests
5. Update documentation

ETA: 2 hours"
```

### 3. Updating Progress

Post regular updates to keep others informed:

```bash
# Progress update
gh issue comment 123 --body "🔄 Progress Update:
- ✅ Analyzed current implementation
- ✅ Designed solution
- 🔄 Implementing changes (60% complete)
- ⏳ Tests pending
- ⏳ Documentation pending"
```

### 4. Completing Work

When finishing a task:

```bash
# Final update
gh issue comment 123 --body "✅ Task Complete:
- All changes implemented in PR #456
- Tests added and passing
- Documentation updated
- Ready for review"

# Remove claimed label
gh issue edit 123 --remove-label "swarm-claimed"

# Move to review
gh issue edit 123 --add-label "needs-review"
```

### 5. Handling Conflicts

If multiple swarms want the same task:

```bash
# Check if already claimed
gh issue view 123

# If conflict, coordinate in comments
gh issue comment 123 --body "🤝 Swarm ID: claude-xyz789 also interested.
Suggestion: I can work on the authentication part while the other swarm handles the UI.
Shall we split this into subtasks?"
```

## 📝 Issue Templates

### Feature Request Template

```markdown
## 🎯 Feature Request

**Description:**
Clear description of the feature

**Benefits:**
- Why this feature is valuable
- Who will benefit

**Implementation Ideas:**
- Potential approach
- Components affected

**Tasks:**
- [ ] Design solution
- [ ] Implement feature
- [ ] Add tests
- [ ] Update documentation

**Labels:** `feature`, `available`, `priority: medium`
```

### Bug Report Template

```markdown
## 🐛 Bug Report

**Description:**
What's broken?

**Steps to Reproduce:**
1. Step one
2. Step two
3. See error

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- OS: 
- Node version:
- ruv-swarm version:

**Labels:** `bug`, `available`, `priority: high`
```

### Task Template

```markdown
## 📋 Task

**Objective:**
What needs to be done

**Context:**
Why this is needed

**Acceptance Criteria:**
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Tests pass

**Related Issues:**
- #xxx
- #yyy

**Labels:** `enhancement`, `available`, `area: core`
```

## 🔄 Best Practices

### For Individual Contributors

1. **Check Before Starting**
   - Always check if an issue is already claimed
   - Look for related issues that might conflict

2. **Communicate Clearly**
   - Post your plan before starting
   - Update progress regularly
   - Ask questions in issue comments

3. **Work in Branches**
   ```bash
   git checkout -b feature/issue-123-description
   ```

4. **Reference Issues**
   - In commits: `Fix authentication bug (#123)`
   - In PRs: `Closes #123`

### For Swarms

1. **Use Unique IDs**
   - Generate consistent swarm IDs
   - Include ID in all comments

2. **Claim Atomically**
   - Claim one task at a time
   - Release if blocked

3. **Update Frequently**
   - Post progress every 30-60 minutes
   - Share blockers immediately

4. **Coordinate on Conflicts**
   - Propose task splitting
   - Offer to collaborate
   - Respect existing claims

## 🎨 GitHub Project Automation

### Auto-move Issues

Add these workflows to `.github/workflows/`:

```yaml
name: Project Automation
on:
  issues:
    types: [opened, labeled, unlabeled]
  pull_request:
    types: [opened, closed]

jobs:
  move-to-ready:
    if: github.event.label.name == 'available'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/add-to-project@v0.5.0
        with:
          project-url: https://github.com/users/ruvnet/projects/XX
          github-token: ${{ secrets.GITHUB_TOKEN }}
          column-name: Ready
```

## 📊 Monitoring Collaboration

### Daily Standup Query

```bash
# Show all active swarm work
gh issue list --label "swarm-claimed" --json number,title,labels,assignees

# Show available high-priority work
gh issue list --label "available" --label "priority: high"

# Show potential conflicts
gh issue list --search "label:swarm-claimed comments:>5"
```

### Weekly Progress

```bash
# Completed this week
gh issue list --state closed --search "closed:>1 week"

# Stale claimed tasks
gh issue list --label "swarm-claimed" --search "updated:<3 days"
```

## 🚀 Getting Started

1. **For New Contributors:**
   ```bash
   # Find your first task
   gh issue list --label "good first issue" --label "available"
   
   # Claim it
   gh issue edit <number> --add-label "swarm-claimed"
   ```

2. **For Swarm Operators:**
   ```bash
   # Set your swarm ID
   export SWARM_ID="swarm-$(date +%s)"
   
   # Find and claim work
   ./scripts/claim-next-task.sh
   ```

3. **For Maintainers:**
   - Review claimed tasks daily
   - Release stale claims (>24h inactive)
   - Mediate conflicts
   - Prioritize backlog

## 📚 Resources

- [GitHub CLI Documentation](https://cli.github.com/manual/)
- [GitHub Projects Guide](https://docs.github.com/en/issues/planning-and-tracking-with-projects)
- [GitHub Issues Guide](https://docs.github.com/en/issues)

---

Remember: Clear communication prevents conflicts. When in doubt, ask in the issue comments!