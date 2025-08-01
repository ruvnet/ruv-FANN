# List Active Patterns

## 🎯 Key Principle
**This tool coordinates Claude Code's actions. It does NOT write code or create content.**

## MCP Tool Usage in Claude Code

**Tool:** `mcp__ruv-swarm-zen__agent_list`

## Parameters
```json
{"filter": "active"}
```

## Description
View all active cognitive patterns and their current focus areas

## Details
Filters:
- **all**: Show all defined patterns
- **active**: Currently engaged patterns
- **idle**: Available but unused patterns
- **busy**: Patterns actively coordinating tasks

## Example Usage

**In Claude Code:**
1. Use the tool: `mcp__ruv-swarm-zen__agent_list`
2. With parameters: `{"filter": "active"}`
3. Claude Code then executes the coordinated plan using its native tools

## Important Reminders
- ✅ This tool provides coordination and structure
- ✅ Claude Code performs all actual implementation
- ❌ The tool does NOT write code
- ❌ The tool does NOT access files directly
- ❌ The tool does NOT execute commands

## See Also
- Main documentation: /claude.md
- Other commands in this category
- Workflow examples in /workflows/
