#!/bin/bash

# NPX Command Test Suite for ruv-swarm v1.0.6
# Tests all commands listed in issue #45

set -e

echo "================================================"
echo "ruv-swarm v1.0.6 NPX Integration Test Suite"
echo "================================================"
echo "Date: $(date)"
echo "Node Version: $(node --version)"
echo "NPM Version: $(npm --version)"
echo ""

# Test results file
RESULTS_FILE="/app/test-results/npx-test-results.json"
mkdir -p $(dirname $RESULTS_FILE)

# Initialize results
echo '{
  "testSuite": "npx-integration",
  "version": "1.0.6",
  "timestamp": "'$(date -Iseconds)'",
  "nodeVersion": "'$(node --version)'",
  "tests": []
}' > $RESULTS_FILE

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to test a command
test_npx_command() {
    local command="$1"
    local description="$2"
    local expected_output="$3"
    
    echo "Testing: $description"
    echo "Command: npx ruv-swarm $command"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Create temp file for output
    local output_file=$(mktemp)
    local start_time=$(date +%s%N)
    
    # Run command and capture output
    if npx ruv-swarm $command > $output_file 2>&1; then
        local end_time=$(date +%s%N)
        local duration=$((($end_time - $start_time) / 1000000))
        
        # Check if expected output exists
        if [ -n "$expected_output" ]; then
            if grep -q "$expected_output" $output_file; then
                echo "✅ PASSED - Found expected output"
                PASSED_TESTS=$((PASSED_TESTS + 1))
                local status="passed"
            else
                echo "❌ FAILED - Expected output not found"
                echo "Expected: $expected_output"
                echo "Got: $(cat $output_file | head -n 5)"
                FAILED_TESTS=$((FAILED_TESTS + 1))
                local status="failed"
            fi
        else
            echo "✅ PASSED - Command executed successfully"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            local status="passed"
        fi
        
        # Add to results
        jq --arg cmd "$command" \
           --arg desc "$description" \
           --arg status "$status" \
           --arg duration "$duration" \
           '.tests += [{
               "command": $cmd,
               "description": $desc,
               "status": $status,
               "duration": $duration
           }]' $RESULTS_FILE > ${RESULTS_FILE}.tmp && mv ${RESULTS_FILE}.tmp $RESULTS_FILE
    else
        echo "❌ FAILED - Command exited with error"
        cat $output_file | head -n 10
        FAILED_TESTS=$((FAILED_TESTS + 1))
        
        # Add to results
        jq --arg cmd "$command" \
           --arg desc "$description" \
           --arg error "$(cat $output_file | head -n 10)" \
           '.tests += [{
               "command": $cmd,
               "description": $desc,
               "status": "failed",
               "error": $error
           }]' $RESULTS_FILE > ${RESULTS_FILE}.tmp && mv ${RESULTS_FILE}.tmp $RESULTS_FILE
    fi
    
    rm -f $output_file
    echo ""
}

echo "1. Testing MCP Server Start"
echo "=========================="
test_npx_command "mcp start --test-mode" "MCP Server Start (Test Mode)" "MCP server ready"

echo "2. Testing Swarm Initialization"
echo "==============================="
test_npx_command "init mesh 4" "Initialize mesh topology with 4 agents" "Swarm initialized"
test_npx_command "init hierarchical 8" "Initialize hierarchical topology with 8 agents" "Swarm initialized"
test_npx_command "init ring 6" "Initialize ring topology with 6 agents" "Swarm initialized"
test_npx_command "init star 5" "Initialize star topology with 5 agents" "Swarm initialized"

echo "3. Testing Agent Spawning"
echo "========================"
test_npx_command "spawn researcher \"Data Analyst\"" "Spawn researcher agent" "Agent spawned"
test_npx_command "spawn coder \"Backend Dev\"" "Spawn coder agent" "Agent spawned"
test_npx_command "spawn analyst \"System Architect\"" "Spawn analyst agent" "Agent spawned"
test_npx_command "spawn tester \"QA Engineer\"" "Spawn tester agent" "Agent spawned"
test_npx_command "spawn coordinator \"Project Lead\"" "Spawn coordinator agent" "Agent spawned"

echo "4. Testing Task Orchestration"
echo "============================"
test_npx_command "orchestrate \"Test task execution\"" "Orchestrate simple task" "Task orchestrated"

echo "5. Testing Status Commands"
echo "========================="
test_npx_command "status" "Basic status check" "Swarm Status"
test_npx_command "status --verbose" "Verbose status check" "Topology:"

echo "6. Testing Monitoring"
echo "===================="
test_npx_command "monitor 5" "Monitor swarm for 5 seconds" "Monitoring"

echo "7. Testing Neural Commands"
echo "========================="
test_npx_command "neural status" "Neural network status" "Neural"
test_npx_command "neural train" "Neural training" "Training"
test_npx_command "neural patterns" "Neural patterns" "Patterns"

echo "8. Testing Benchmark Commands"
echo "============================"
test_npx_command "benchmark quick" "Quick benchmark" "Benchmark"
test_npx_command "benchmark memory" "Memory benchmark" "Memory"

echo "9. Testing Performance Commands"
echo "=============================="
test_npx_command "performance analyze" "Performance analysis" "Performance"
test_npx_command "performance optimize" "Performance optimization" "Optimization"

echo "10. Testing Hook Commands"
echo "========================"
test_npx_command "hook pre-task --description \"Test hook\"" "Pre-task hook" "continue"
test_npx_command "hook post-task --task-id \"test-123\"" "Post-task hook" "continue"
test_npx_command "hook notification --message \"Test notification\"" "Notification hook" "continue"

echo "11. Testing Claude Integration"
echo "============================="
test_npx_command "claude-invoke \"Test prompt\"" "Claude invoke command" ""

echo "12. Testing Help and Version"
echo "==========================="
test_npx_command "--help" "Help command" "Usage:"
test_npx_command "--version" "Version command" "1.0.6"

# Update final results
jq --arg total "$TOTAL_TESTS" \
   --arg passed "$PASSED_TESTS" \
   --arg failed "$FAILED_TESTS" \
   '.summary = {
       "total": ($total | tonumber),
       "passed": ($passed | tonumber),
       "failed": ($failed | tonumber),
       "passRate": (($passed | tonumber) / ($total | tonumber) * 100)
   }' $RESULTS_FILE > ${RESULTS_FILE}.tmp && mv ${RESULTS_FILE}.tmp $RESULTS_FILE

echo ""
echo "================================================"
echo "Test Summary"
echo "================================================"
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo "Pass Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
echo ""
echo "Results saved to: $RESULTS_FILE"

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Some tests failed!"
    exit 1
fi