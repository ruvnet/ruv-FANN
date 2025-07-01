#!/bin/bash
# Swarm Helper Script - Simplifies common swarm coordination tasks

set -e

SWARM_ID="${SWARM_ID:-swarm-$(date +%s)}"
REPO="${GITHUB_REPO:-ruvnet/ruv-FANN}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${BLUE}🐝 Swarm Helper - ID: $SWARM_ID${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

function list_available() {
    print_header
    echo -e "${GREEN}📋 Available Tasks:${NC}"
    gh issue list --repo $REPO --label "available" --label "-swarm-claimed" --limit 20
}

function claim_task() {
    local issue_number=$1
    if [ -z "$issue_number" ]; then
        echo -e "${RED}Error: Please provide an issue number${NC}"
        echo "Usage: $0 claim <issue_number>"
        exit 1
    fi
    
    print_header
    echo -e "${YELLOW}🎯 Claiming issue #$issue_number...${NC}"
    
    # Add swarm-claimed label
    gh issue edit $issue_number --repo $REPO --add-label "swarm-claimed"
    
    # Add comment
    gh issue comment $issue_number --repo $REPO --body "🐝 **Swarm ID:** \`$SWARM_ID\` claiming this task.

Starting work now. Will post updates as progress is made.

---
_Automated by swarm-helper.sh_"
    
    echo -e "${GREEN}✅ Successfully claimed issue #$issue_number${NC}"
}

function update_progress() {
    local issue_number=$1
    local message=$2
    
    if [ -z "$issue_number" ] || [ -z "$message" ]; then
        echo -e "${RED}Error: Please provide issue number and message${NC}"
        echo "Usage: $0 update <issue_number> \"<message>\""
        exit 1
    fi
    
    print_header
    echo -e "${YELLOW}📝 Updating issue #$issue_number...${NC}"
    
    gh issue comment $issue_number --repo $REPO --body "🔄 **Progress Update** from \`$SWARM_ID\`:

$message

---
_Automated by swarm-helper.sh_"
    
    echo -e "${GREEN}✅ Progress updated${NC}"
}

function complete_task() {
    local issue_number=$1
    local pr_number=$2
    
    if [ -z "$issue_number" ]; then
        echo -e "${RED}Error: Please provide an issue number${NC}"
        echo "Usage: $0 complete <issue_number> [pr_number]"
        exit 1
    fi
    
    print_header
    echo -e "${YELLOW}✅ Completing issue #$issue_number...${NC}"
    
    # Remove swarm-claimed label
    gh issue edit $issue_number --repo $REPO --remove-label "swarm-claimed"
    
    # Add completion comment
    local comment="✅ **Task Complete** by \`$SWARM_ID\`"
    if [ -n "$pr_number" ]; then
        comment="$comment

Changes implemented in PR #$pr_number"
    fi
    comment="$comment

---
_Automated by swarm-helper.sh_"
    
    gh issue comment $issue_number --repo $REPO --body "$comment"
    
    echo -e "${GREEN}✅ Task marked as complete${NC}"
}

function release_task() {
    local issue_number=$1
    local reason=$2
    
    if [ -z "$issue_number" ]; then
        echo -e "${RED}Error: Please provide an issue number${NC}"
        echo "Usage: $0 release <issue_number> [reason]"
        exit 1
    fi
    
    print_header
    echo -e "${YELLOW}🔓 Releasing issue #$issue_number...${NC}"
    
    # Remove swarm-claimed label
    gh issue edit $issue_number --repo $REPO --remove-label "swarm-claimed"
    
    # Add release comment
    local comment="🔓 **Task Released** by \`$SWARM_ID\`"
    if [ -n "$reason" ]; then
        comment="$comment

Reason: $reason"
    fi
    comment="$comment

This task is now available for other contributors.

---
_Automated by swarm-helper.sh_"
    
    gh issue comment $issue_number --repo $REPO --body "$comment"
    
    echo -e "${GREEN}✅ Task released${NC}"
}

function status() {
    print_header
    echo -e "${GREEN}📊 Swarm Coordination Status:${NC}"
    echo
    
    echo -e "${BLUE}Available Tasks:${NC}"
    gh issue list --repo $REPO --label "available" --label "-swarm-claimed" --limit 5
    echo
    
    echo -e "${BLUE}Currently Claimed:${NC}"
    gh issue list --repo $REPO --label "swarm-claimed" --limit 10
    echo
    
    echo -e "${BLUE}My Active Tasks:${NC}"
    gh issue list --repo $REPO --search "commenter:@me swarm-claimed" --limit 5
}

function help() {
    print_header
    echo "Usage: $0 <command> [arguments]"
    echo
    echo "Commands:"
    echo "  list              - List available tasks"
    echo "  claim <number>    - Claim an issue"
    echo "  update <number> <message> - Post progress update"
    echo "  complete <number> [pr]    - Mark task as complete"
    echo "  release <number> [reason] - Release a claimed task"
    echo "  status            - Show coordination status"
    echo "  help              - Show this help message"
    echo
    echo "Environment Variables:"
    echo "  SWARM_ID     - Your swarm identifier (default: auto-generated)"
    echo "  GITHUB_REPO  - Repository to work on (default: ruvnet/ruv-FANN)"
}

# Main command dispatcher
case "$1" in
    list)
        list_available
        ;;
    claim)
        claim_task "$2"
        ;;
    update)
        update_progress "$2" "$3"
        ;;
    complete)
        complete_task "$2" "$3"
        ;;
    release)
        release_task "$2" "$3"
        ;;
    status)
        status
        ;;
    help|--help|-h)
        help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        help
        exit 1
        ;;
esac