#!/bin/bash
# MCP Server wrapper for ruv-swarm
# Ensures clean stdio communication for Claude Code

cd /workspaces/ruv-FANN/ruv-swarm/npm
exec node bin/ruv-swarm-enhanced.js mcp start --protocol=stdio