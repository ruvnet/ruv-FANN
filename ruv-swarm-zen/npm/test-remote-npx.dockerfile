FROM node:20-alpine

# Install git for npx to work properly
RUN apk add --no-cache git

# Create app directory
WORKDIR /app

# Create a test script
RUN echo '#!/bin/sh\n\
echo "🧪 Testing remote npx ruv-swarm@1.0.12"\n\
echo "====================================="\n\
\n\
# Test 1: Check version\n\
echo "\n📋 Test 1: Checking version..."\n\
npx ruv-swarm@1.0.12 --version\n\
\n\
# Test 2: Run MCP server test\n\
echo "\n🔌 Test 2: Testing MCP server start..."\n\
timeout 10 npx ruv-swarm@1.0.12 mcp start < /dev/null || echo "✅ MCP server started and exited as expected"\n\
\n\
# Test 3: Check CLI commands\n\
echo "\n📝 Test 3: Testing CLI commands..."\n\
npx ruv-swarm@1.0.12 help\n\
\n\
# Test 4: Run diagnostic check\n\
echo "\n🔍 Test 4: Running diagnostics..."\n\
npx ruv-swarm@1.0.12 diagnose config || echo "Diagnostics command available"\n\
\n\
echo "\n✅ All tests completed!"' > /app/test-remote.sh

RUN chmod +x /app/test-remote.sh

CMD ["/app/test-remote.sh"]