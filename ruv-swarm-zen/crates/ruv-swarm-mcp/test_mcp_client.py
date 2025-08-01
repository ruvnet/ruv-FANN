#!/usr/bin/env python3
"""Test MCP client for ruv-swarm-mcp stdio server."""

import json
import subprocess
import sys
import asyncio
import time

async def test_mcp_stdio():
    """Test the MCP stdio server."""
    
    # Start the MCP server
    process = subprocess.Popen(
        ["./target/debug/ruv-swarm-mcp-stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    def send_request(request):
        """Send a JSON-RPC request."""
        json_request = json.dumps(request) + "\n"
        print(f"Sending: {json_request.strip()}")
        process.stdin.write(json_request)
        process.stdin.flush()
        
        # Read response
        response = process.stdout.readline()
        print(f"Received: {response.strip()}")
        return json.loads(response) if response.strip() else None
    
    try:
        # 1. Initialize request
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
        
        init_response = send_request(init_request)
        print(f"Initialize response: {init_response}")
        
        # 2. List tools request
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        tools_response = send_request(tools_request)
        print(f"Tools response: {tools_response}")
        
        # 3. Test spawn_agent tool
        spawn_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "spawn_agent",
                "arguments": {
                    "agent_type": "researcher",
                    "name": "test-agent",
                    "capabilities": ["research", "analysis"]
                }
            }
        }
        
        spawn_response = send_request(spawn_request)
        print(f"Spawn agent response: {spawn_response}")
        
        # 4. Test query_swarm tool
        query_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "query_swarm",
                "arguments": {
                    "query": "status"
                }
            }
        }
        
        query_response = send_request(query_request)
        print(f"Query swarm response: {query_response}")
        
        # 5. Test monitor_swarm tool
        monitor_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "monitor_swarm",
                "arguments": {}
            }
        }
        
        monitor_response = send_request(monitor_request)
        print(f"Monitor swarm response: {monitor_response}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        # Clean up
        process.terminate()
        process.wait()

if __name__ == "__main__":
    asyncio.run(test_mcp_stdio())