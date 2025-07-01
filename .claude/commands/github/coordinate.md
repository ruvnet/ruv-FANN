# /github coordinate

Helps coordinate work between multiple swarms by proposing task splits, dependencies, or collaborative approaches.

## Usage

```
/github coordinate <action> [options]
```

## Actions

- `split <issue>` - Propose splitting a large issue
- `merge <issue1> <issue2>` - Suggest combining related issues
- `sequence <issues...>` - Propose order for dependent tasks
- `pair <issue>` - Request collaboration on complex task

## Examples

```
/github coordinate split 45
/github coordinate merge 45 46
/github coordinate sequence 51 52 53
/github coordinate pair 60 --with swarm-beta-456
```

## Implementation

Uses GitHub MCP to:
1. Analyze issue complexity
2. Identify relationships
3. Create coordination proposals
4. Post collaboration comments

## Example Interactions

### Task Splitting
```
> /github coordinate split 45

📊 Analyzing issue #45: "Implement complete authentication system"

Suggested split into 3 tasks:

1. **Backend Authentication API**
   - JWT token generation
   - User validation
   - Session management
   
2. **Frontend Auth Components**  
   - Login/signup forms
   - Auth state management
   - Route protection
   
3. **Testing & Documentation**
   - Integration tests
   - API documentation
   - Usage examples

Posted coordination proposal to issue #45
Other swarms can claim subtasks
```

### Task Sequencing
```
> /github coordinate sequence 51 52 53

📋 Dependency Analysis:

Suggested sequence:
1. #51: Database schema (Required first)
2. #52: API endpoints (Needs schema)  
3. #53: UI components (Needs API)

Dependencies posted to all three issues
Swarms should complete in order
```

### Pairing Request
```
> /github coordinate pair 60 --with swarm-beta-456

🤝 Collaboration Proposal for #60:

Requesting pair work with swarm-beta-456
Task: Complex WASM optimization

Proposed split:
- Your swarm: Performance profiling
- Beta swarm: Implementation
- Both: Testing & validation

Posted collaboration request
Waiting for response from swarm-beta-456
```

## Coordination Types

### 1. Task Splitting
For large issues:
- Analyze complexity
- Identify logical divisions
- Create subtask list
- Enable parallel work

### 2. Task Merging
For related issues:
- Find common work
- Reduce duplication
- Suggest combination
- Simplify tracking

### 3. Sequencing
For dependent work:
- Identify dependencies
- Propose order
- Prevent blocking
- Optimize flow

### 4. Pairing
For complex tasks:
- Request collaboration
- Define roles
- Share expertise
- Faster completion

## Coordination Messages

Auto-generates:
```markdown
🤝 Coordination Proposal

Type: Task Split
Issue: #45

Proposal:
[Detailed split/merge/sequence plan]

Benefits:
- Enables parallel work
- Clearer scope
- Faster completion

How to participate:
1. Review proposal
2. Claim subtasks
3. Coordinate in comments
```

## Best Practices

1. Propose splits for 8+ hour tasks
2. Merge truly duplicate work
3. Sequence to prevent blocking
4. Pair for complex problems
5. Get agreement before proceeding

## Related Commands

- `/github tasks` - Find tasks to coordinate
- `/github conflicts` - Check for conflicts
- `/github swarms` - Find collaborators
- `/github claim` - Claim coordinated tasks