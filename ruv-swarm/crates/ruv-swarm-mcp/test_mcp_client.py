#!/usr/bin/env python3
"""Compatibility entry point for the bounded real-server stdio protocol proof.

Pass --binary /absolute/path/to/ruv-swarm-mcp-stdio. Protocol failures exit nonzero.
"""
from test_stdio import main


if __name__ == "__main__":
    main()
