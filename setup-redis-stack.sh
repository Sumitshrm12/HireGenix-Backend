#!/bin/bash

# Redis Stack Setup Script for macOS
# This script installs Redis Stack with RediSearch module for vector similarity search

echo "================================================"
echo "🚀 Redis Stack Setup for HireGenix Backend"
echo "================================================"
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew is not installed!"
    echo "Please install Homebrew first: https://brew.sh"
    echo "Run: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "✅ Homebrew found"
echo ""

# Install Redis Stack
echo "📦 Installing Redis Stack (includes RediSearch module)..."
echo ""

# Check if Redis Stack is already installed
if brew list redis-stack &> /dev/null; then
    echo "ℹ️  Redis Stack is already installed"
    echo "Checking for updates..."
    brew upgrade redis-stack || true
else
    echo "Installing Redis Stack..."
    brew tap redis-stack/redis-stack
    brew install redis-stack
fi

echo ""
echo "✅ Redis Stack installed successfully!"
echo ""

# Start Redis Stack service
echo "🔧 Configuring Redis Stack service..."
echo ""

# Stop any existing Redis service to avoid conflicts
if brew services list | grep -q "redis.*started"; then
    echo "⚠️  Regular Redis service is running. Stopping it..."
    brew services stop redis 2>/dev/null || true
fi

# Start Redis Stack
echo "Starting Redis Stack service..."
brew services start redis-stack

# Wait for Redis to start
echo "Waiting for Redis Stack to start..."
sleep 3

# Check if Redis is running
if redis-cli ping &> /dev/null; then
    echo "✅ Redis Stack is running!"
else
    echo "❌ Redis Stack failed to start. Trying manual start..."
    redis-stack-server --daemonize yes
    sleep 2
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis Stack started manually!"
    else
        echo "❌ Could not start Redis Stack. Please check logs."
        exit 1
    fi
fi

echo ""

# Verify RediSearch module is loaded
echo "🔍 Verifying RediSearch module..."
if redis-cli MODULE LIST | grep -q "search"; then
    echo "✅ RediSearch module is loaded and ready!"
else
    echo "⚠️  RediSearch module not detected. Redis Stack may need restart."
    echo "Try: brew services restart redis-stack"
fi

echo ""

# Display Redis configuration
echo "📋 Redis Stack Configuration:"
echo "   Host: localhost"
echo "   Port: 6379"
echo "   Modules: RediSearch, RedisJSON, RedisGraph, RedisTimeSeries, RedisBloom"
echo ""

# Test vector search capability
echo "🧪 Testing vector search capability..."
redis-cli FT._LIST > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Vector search (FT.CREATE) commands are available!"
else
    echo "⚠️  Vector search commands not available. Module may need initialization."
fi

echo ""
echo "================================================"
echo "✅ Redis Stack Setup Complete!"
echo "================================================"
echo ""
echo "📝 Next Steps:"
echo "1. Ensure your .env file has:"
echo "   REDIS_HOST=localhost"
echo "   REDIS_PORT=6379"
echo "   REDIS_PASSWORD="
echo ""
echo "2. Activate your Python venv:"
echo "   cd Backend"
echo "   source .venv/bin/activate  # or: source venv/bin/activate"
echo ""
echo "3. Install/upgrade Python Redis client:"
echo "   pip install -r requirements.txt"
echo ""
echo "4. Start your FastAPI backend:"
echo "   uvicorn main:app --reload --port 8000"
echo ""
echo "🔧 Useful Commands:"
echo "   - Check status: brew services list | grep redis"
echo "   - Stop service: brew services stop redis-stack"
echo "   - Restart: brew services restart redis-stack"
echo "   - Redis CLI: redis-cli"
echo "   - View logs: brew services info redis-stack"
echo ""
echo "================================================"