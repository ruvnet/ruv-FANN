#!/usr/bin/env python3
"""
Simple test script to validate stdio MCP server functionality
"""
import subprocess
import json
import sys

def test_mcp_server():
    # Start the server process
    server_path = "../../target/debug/ruv-swarm-mcp-stdio"
    process = subprocess.Popen(
        [server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Test 1: Initialize
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    print("Sending initialize request...")
    process.stdin.write(json.dumps(init_request) + "\n")
    process.stdin.flush()
    
    # Read response
    try:
        response_line = process.stdout.readline()
        print(f"Initialize response: {response_line.strip()}")
        
        if response_line:
            response = json.loads(response_line)
            if "result" in response:
                print("✅ Initialize successful!")
                
                # Send initialized notification (required by MCP protocol)
                initialized_notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                }
                
                print("Sending initialized notification...")
                process.stdin.write(json.dumps(initialized_notification) + "\n")
                process.stdin.flush()
                
            else:
                print("❌ Initialize failed:", response)
                
        # Test 2: List tools
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        print("\nSending tools/list request...")
        process.stdin.write(json.dumps(list_tools_request) + "\n")
        process.stdin.flush()
        
        # Read tools response
        tools_response_line = process.stdout.readline()
        print(f"Tools response: {tools_response_line.strip()}")
        
        if tools_response_line:
            tools_response = json.loads(tools_response_line)
            if "result" in tools_response and "tools" in tools_response["result"]:
                tools = tools_response["result"]["tools"]
                print(f"✅ Found {len(tools)} tools:")
                for tool in tools:
                    print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
            else:
                print("❌ Tools list failed:", tools_response)
                
        # Test 3: Call increment tool
        increment_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "increment",
                "arguments": {}
            }
        }
        
        print("\nSending increment tool call...")
        process.stdin.write(json.dumps(increment_request) + "\n")
        process.stdin.flush()
        
        # Read increment response
        increment_response_line = process.stdout.readline()
        print(f"Increment response: {increment_response_line.strip()}")
        
        if increment_response_line:
            increment_response = json.loads(increment_response_line)
            if "result" in increment_response:
                print("✅ Increment tool call successful!")
            else:
                print("❌ Increment tool call failed:", increment_response)
                
    except Exception as e:
        print(f"❌ Error during test: {e}")
        
    finally:
        # Cleanup
        process.terminate()
        process.wait()
        
        # Print stderr for debugging
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"\nServer stderr:\n{stderr_output}")

if __name__ == "__main__":
    test_mcp_server()