#!/usr/bin/env python3
"""Bounded JSON-RPC proof against the actual compiled stdio server (stdlib only)."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import uuid

MAX_LINE = 1024 * 1024
OUTPUT_BUDGET = 512 * 1024
EXPECTED_TOOLS = {
    "spawn", "orchestrate", "query", "monitor", "optimize", "memory_store",
    "memory_get", "task_create", "agent_list", "agent_metrics",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def binary_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def exercise(binary):
    checks, stdout_bytes, stderr_bytes = [], 0, 0
    with tempfile.TemporaryDirectory(prefix="ruv-swarm-stdio-") as temporary:
        environment = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "LANG") if key in os.environ}
        environment["RUV_SWARM_DB_PATH"] = str(Path(temporary) / "protocol.db")
        process = await asyncio.create_subprocess_exec(
            str(binary), cwd=temporary, env=environment,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=MAX_LINE + 1,
            start_new_session=os.name == "posix",
        )

        async def drain_stderr():
            nonlocal stderr_bytes
            while True:
                chunk = await process.stderr.read(4096)
                if not chunk:
                    break
                stderr_bytes += len(chunk)
                require(stderr_bytes <= 65536, "Server stderr exceeded 64 KiB")

        stderr = asyncio.create_task(drain_stderr())

        async def send(message):
            data = json.dumps(message, allow_nan=False, ensure_ascii=False).encode() + b"\n"
            process.stdin.write(data)
            await asyncio.wait_for(process.stdin.drain(), 5)

        async def request(identifier, method, params=None):
            nonlocal stdout_bytes
            body = {"jsonrpc": "2.0", "id": identifier, "method": method}
            if params is not None:
                body["params"] = params
            await send(body)
            line = await asyncio.wait_for(process.stdout.readline(), 5)
            stdout_bytes += len(line)
            require(line and stdout_bytes <= OUTPUT_BUDGET, "Missing response or stdout budget exceeded")
            response = json.loads(line)
            require(response.get("jsonrpc") == "2.0" and response.get("id") == identifier,
                    "Unexpected protocol response or stdout contamination")
            if stderr.done():
                stderr.result()
            return response

        async def call(identifier, name, arguments):
            response = await request(identifier, "tools/call", {"name": name, "arguments": arguments})
            require("error" not in response, f"{name} failed: {response.get('error')}")
            result = response["result"]
            require(result.get("isError", False) is False, f"{name} returned a tool error")
            return "\n".join(item["text"] for item in result["content"] if item.get("type") == "text")

        try:
            response = await request(1, "initialize", {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "ruv-swarm-stdio-regression", "version": "1.0"},
            })
            result = response.get("result", {})
            require(result.get("protocolVersion") == "2024-11-05", "Legacy protocol negotiation changed")
            require("tools" in result.get("capabilities", {}), "Tools capability missing")
            require(result.get("serverInfo", {}).get("name") == "ruv-swarm-mcp", "Wrong server implementation")
            checks.append("initialize_and_legacy_negotiation")
            await send({"jsonrpc": "2.0", "method": "notifications/initialized"})

            response = await request(2, "tools/list", {})
            tools = response.get("result", {}).get("tools", [])
            require({tool["name"] for tool in tools} == EXPECTED_TOOLS, "Actual tool catalogue changed")
            require(all(tool.get("inputSchema", {}).get("type") == "object" for tool in tools), "Invalid tool schemas")
            checks.append("real_tool_catalogue_and_schemas")

            stored = {"answer": 42, "label": "café"}
            await call(3, "memory_store", {"key": "roundtrip", "value": stored})
            text = await call(4, "memory_get", {"key": "roundtrip"})
            require(text.startswith("Retrieved: ") and json.loads(text[len("Retrieved: "):]) == stored,
                    "Real memory round trip failed")
            checks.append("real_memory_round_trip")

            text = await call(5, "spawn", {"agent_type": "researcher", "name": "stdio-regression"})
            identifier = str(uuid.UUID(text.split("ID: ", 1)[1]))
            require(identifier in await call(6, "agent_list", {}), "Persisted agent missing from list")
            require("1 agents" in await call(7, "query", {}), "Swarm state did not reflect the actual agent")
            checks.append("real_agent_spawn_and_persistence")

            response = await request(8, "tools/call", {"name": "memory_get", "arguments": {"key": 123}})
            require("error" not in response and response.get("result", {}).get("isError") is True,
                    "Invalid arguments must produce the SDK's in-band tool error")
            checks.append("invalid_arguments_return_tool_error")

            oversized = {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {
                "name": "memory_store", "arguments": {"key": "oversized-was-dispatched", "value": "x" * MAX_LINE}}}
            await send(oversized)
            response = await request(10, "ping")
            require("result" in response and "error" not in response, "Connection did not recover after oversized line")
            require("No data found" in await call(11, "memory_get", {"key": "oversized-was-dispatched"}),
                    "Oversized request reached the actual tool handler")
            checks.append("oversized_line_not_dispatched_and_stream_recovers")

            process.stdin.close()
            await asyncio.wait_for(process.wait(), 5)
            await asyncio.wait_for(stderr, 5)
            require(process.returncode == 0, f"Server failed during EOF shutdown: {process.returncode}")
            checks.append("graceful_eof_shutdown")
            return {"status": "passed", "binary": str(binary), "binarySha256": binary_hash(binary),
                    "sdk": "rmcp 3.9.0", "protocolVersion": result["protocolVersion"], "checks": checks,
                    "stdioLineLimit": MAX_LINE, "stdoutBytes": stdout_bytes, "stderrBytes": stderr_bytes,
                    "scope": "Actual compiled stdio server, private temporary SQLite database; no HTTP client or provider calls."}
        finally:
            if process.returncode is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 2)
                except asyncio.TimeoutError:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    await asyncio.wait_for(process.wait(), 2)
            if not stderr.done():
                stderr.cancel()
            await asyncio.gather(stderr, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="Actual compiled ruv-swarm-mcp-stdio executable")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    require(binary.is_file(), "Binary must be an ordinary executable file")
    async def bounded():
        return await asyncio.wait_for(exercise(binary), 30)
    print(json.dumps(asyncio.run(bounded()), allow_nan=False))


if __name__ == "__main__":
    main()
