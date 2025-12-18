#!/bin/bash

echo "🔧 Fixing Python Backend Dependencies..."
echo ""

# Activate virtual environment
source ./venv/bin/activate

echo "📦 Installing missing tf-keras package..."
pip install tf-keras

echo ""
echo "✅ Dependencies fixed!"
echo ""
echo "🚀 You can now run the backend with:"
echo "   cd Backend && ./venv/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"