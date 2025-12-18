#!/bin/bash

echo "🔧 Fixing Backend Dependencies..."
echo "================================="

# Navigate to Backend directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create it first:"
    echo "   python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Uninstall problematic packages
echo "🗑️  Uninstalling problematic packages..."
pip uninstall -y protobuf deepface tf-keras tensorflow

# Install protobuf with correct version
echo "📥 Installing protobuf 3.20.x (compatible with DeepFace)..."
pip install "protobuf>=3.20.0,<4.0.0"

# Install tensorflow and tf-keras
echo "📥 Installing TensorFlow and tf-keras..."
pip install tensorflow>=2.15.0
pip install tf-keras>=2.15.0

# Install DeepFace
echo "📥 Installing DeepFace..."
pip install deepface>=0.0.79

# Reinstall other dependencies
echo "📥 Installing remaining dependencies..."
pip install -r requirements.txt

# Verify installations
echo ""
echo "✅ Verifying installations..."
python -c "import protobuf; print(f'✓ protobuf version: {protobuf.__version__}')" 2>/dev/null || echo "⚠️  protobuf check failed"
python -c "import tensorflow; print(f'✓ tensorflow version: {tensorflow.__version__}')" 2>/dev/null || echo "⚠️  tensorflow check failed"
python -c "import deepface; print('✓ deepface installed successfully')" 2>/dev/null || echo "⚠️  deepface check failed"
python -c "from langchain_openai import AzureChatOpenAI; print('✓ langchain_openai installed successfully')" 2>/dev/null || echo "⚠️  langchain_openai check failed"

echo ""
echo "✅ Dependencies fixed successfully!"
echo ""
echo "📝 Next steps:"
echo "   1. Copy .env.example to .env if not already done"
echo "   2. Update TEXT_EMBEDDING_ENDPOINT in your .env file"
echo "   3. Run: python main.py or uvicorn main:socket_app --host 0.0.0.0 --port 8000 --reload"
echo ""