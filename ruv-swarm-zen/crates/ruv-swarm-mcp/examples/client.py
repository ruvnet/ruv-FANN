#!/usr/bin/env python3

"""
Example MCP client for RUV-Swarm in Python

This demonstrates how to interact with the MCP server using WebSocket
"""

import json
import asyncio
import websockets
from datetime import datetime
from typing import Any, Dict, Optional


class MCPClient:
    def __init__(self, url: str = "ws://localhost:3000/mcp"):
        self.url = url
        self.ws = None
        self.request_id = 0
        self.pending_requests: Dict[int, asyncio.Future] = {}
        
    async def connect(self):
        """Connect to the MCP server"""
        self.ws = await websockets.connect(self.url)
        print(f"Connected to MCP server at {self.url}")
        
        # Start message handler
        asyncio.create_task(self._handle_messages())
        
    async def _handle_messages(self):
        """Handle incoming messages from the server"""
        async for message in self.ws:
            try:
                response = json.loads(message)
                print(f"Received: {json.dumps(response, indent=2)}")
                
                if "id" in response and response["id"] in self.pending_requests:
                    future = self.pending_requests.pop(response["id"])
                    
                    if "error" in response:
                        future.set_exception(Exception(response["error"]["message"]))
                    else:
                        future.set_result(response.get("result"))
                        
            except Exception as e:
                print(f"Error handling message: {e}")
                
    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a request to the MCP server"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.request_id
        }
        
        future = asyncio.Future()
        self.pending_requests[self.request_id] = future
        
        await self.ws.send(json.dumps(request))
        print(f"Sent: {json.dumps(request, indent=2)}")
        
        return await future
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP session"""
        return await self.send_request("initialize")
        
    async def list_tools(self) -> Dict[str, Any]:
        """List available tools"""
        return await self.send_request("tools/list")
        
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool"""
        return await self.send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
        
    async def close(self):
        """Close the connection"""
        if self.ws:
            await self.ws.close()
            print("Disconnected from MCP server")


async def main():
    """Example usage of the MCP client"""
    client = MCPClient()
    
    try:
        # Connect to server
        await client.connect()
        
        # Initialize session
        print("\n=== Initializing Session ===")
        init_result = await client.initialize()
        print(f"Session ID: {init_result.get('sessionId')}")
        
        # List available tools
        print("\n=== Listing Tools ===")
        tools_result = await client.list_tools()
        print(f"Available tools: {len(tools_result['tools'])}")
        for tool in tools_result['tools'][:3]:  # Show first 3 tools
            print(f"  - {tool['name']}: {tool['description']}")
        
        # Spawn multiple agents
        print("\n=== Spawning Agents ===")
        agent_types = ["researcher", "coder", "analyst"]
        agents = []
        
        for agent_type in agent_types:
            result = await client.call_tool("ruv-swarm.spawn", {
                "agent_type": agent_type,
                "name": f"{agent_type.capitalize()} Agent",
                "capabilities": {
                    "languages": ["python", "rust"],
                    "frameworks": ["pytorch", "tensorflow"],
                    "tools": ["git", "docker"],
                    "specializations": ["ML", "distributed systems"],
                    "max_concurrent_tasks": 3
                }
            })
            agents.append(result)
            print(f"Spawned {agent_type}: {result['agent_id']}")
            
        # Create tasks
        print("\n=== Creating Tasks ===")
        tasks = [
            ("research", "Research neural architecture search methods", "high"),
            ("development", "Implement distributed training framework", "medium"),
            ("analysis", "Analyze performance bottlenecks", "high")
        ]
        
        for task_type, description, priority in tasks:
            result = await client.call_tool("ruv-swarm.task.create", {
                "task_type": task_type,
                "description": description,
                "priority": priority
            })
            print(f"Created task: {result['task_id']} - {description}")
            
        # Orchestrate a complex task
        print("\n=== Orchestrating Complex Task ===")
        orchestrate_result = await client.call_tool("ruv-swarm.orchestrate", {
            "objective": "Build a distributed machine learning pipeline",
            "strategy": "development",
            "mode": "hierarchical",
            "max_agents": 5,
            "parallel": True
        })
        print(f"Orchestration started: {orchestrate_result['task_id']}")
        
        # Store research findings
        print("\n=== Storing Research Findings ===")
        findings = {
            "topic": "Distributed ML Training",
            "methods": [
                "Data parallelism",
                "Model parallelism",
                "Pipeline parallelism"
            ],
            "frameworks": {
                "PyTorch": "torch.distributed",
                "TensorFlow": "tf.distribute",
                "JAX": "jax.pmap"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        store_result = await client.call_tool("ruv-swarm.memory.store", {
            "key": "ml_training_research",
            "value": findings,
            "ttl_secs": 3600  # 1 hour TTL
        })
        print(f"Stored findings with key: {store_result['key']}")
        
        # Retrieve findings
        print("\n=== Retrieving Findings ===")
        get_result = await client.call_tool("ruv-swarm.memory.get", {
            "key": "ml_training_research"
        })
        if get_result["found"]:
            print(f"Retrieved findings: {json.dumps(get_result['value'], indent=2)}")
            
        # Query swarm state
        print("\n=== Querying Swarm State ===")
        state_result = await client.call_tool("ruv-swarm.query", {
            "include_metrics": True
        })
        print(f"Active agents: {state_result['total_agents']}")
        print(f"Active tasks: {state_result['active_tasks']}")
        print(f"Completed tasks: {state_result['completed_tasks']}")
        
        if "metrics" in state_result:
            metrics = state_result["metrics"]
            print(f"Success rate: {metrics['success_rate'] * 100:.1f}%")
            print(f"Agent utilization: {metrics['agent_utilization'] * 100:.1f}%")
            
        # Get optimization recommendations
        print("\n=== Getting Optimization Recommendations ===")
        optimize_result = await client.call_tool("ruv-swarm.optimize", {
            "target_metric": "throughput",
            "auto_apply": False
        })
        
        if optimize_result["recommendations"]:
            print("Recommendations:")
            for rec in optimize_result["recommendations"]:
                print(f"  - {rec['description']}")
                print(f"    Impact: {rec['impact']}, Priority: {rec['priority']}")
                
        # List all agents
        print("\n=== Listing All Agents ===")
        agents_result = await client.call_tool("ruv-swarm.agent.list", {
            "include_inactive": False,
            "sort_by": "created_at"
        })
        print(f"Total agents: {agents_result['count']}")
        for agent in agents_result["agents"][:5]:  # Show first 5
            print(f"  - {agent['name']} ({agent['agent_type']}): {agent['status']}")
            
        # Monitor events
        print("\n=== Starting Event Monitor ===")
        monitor_result = await client.call_tool("ruv-swarm.monitor", {
            "event_types": ["agent_spawned", "task_completed", "optimization_applied"],
            "duration_secs": 10
        })
        print(f"Monitoring for {monitor_result['duration_secs']} seconds...")
        
        # Wait for monitoring to complete
        await asyncio.sleep(5)
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())