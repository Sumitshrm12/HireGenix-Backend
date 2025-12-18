#!/bin/bash

# HireGenix Agentic AI Installation Script
# This script installs all required dependencies for the agentic AI system

set -e  # Exit on error

echo "🚀 Installing HireGenix Agentic AI Dependencies..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create one first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    exit 1
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Could not activate virtual environment"
    exit 1
fi

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo ""
echo "📦 Installing core dependencies..."
pip install fastapi>=0.109.0 uvicorn[standard]>=0.27.0

# Install agentic AI stack
echo ""
echo "🤖 Installing Agentic AI stack..."
pip install langchain>=0.1.0 langchain-openai>=0.0.5 langchain-community>=0.0.20
pip install langgraph>=0.0.25 dspy-ai>=2.4.0 langsmith>=0.0.77

# Install vector databases
echo ""
echo "🗄️  Installing vector databases..."
pip install chromadb>=0.4.22 faiss-cpu>=1.7.4
pip install qdrant-client>=1.7.0 pinecone-client>=3.0.0

# Install AI/ML libraries
echo ""
echo "🧠 Installing AI/ML libraries..."
pip install openai>=1.12.0 anthropic>=0.18.0
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers>=4.37.0 sentence-transformers>=2.3.0
pip install scikit-learn>=1.4.0 ultralytics>=8.1.0

# Install document processing
echo ""
echo "📄 Installing document processing libraries..."
pip install PyPDF2>=3.0.1 python-docx>=1.1.0
pip install pytesseract>=0.3.10 easyocr>=1.7.0
pip install pdf2image>=1.16.3 pdfplumber>=0.10.3

# Install computer vision
echo ""
echo "👁️  Installing computer vision libraries..."
pip install mediapipe>=0.10.9 opencv-python-headless>=4.9.0
pip install pillow>=10.2.0

# Install web scraping
echo ""
echo "🌐 Installing web scraping libraries..."
pip install crawl4ai[all]>=0.2.0 playwright>=1.41.0
pip install tavily-python>=0.3.0 beautifulsoup4>=4.12.3
pip install lxml>=5.1.0 requests>=2.31.0

# Install database libraries
echo ""
echo "💾 Installing database libraries..."
pip install psycopg2-binary>=2.9.9 sqlalchemy>=2.0.25
pip install asyncpg>=0.29.0 redis>=5.0.1

# Install remaining dependencies
echo ""
echo "📦 Installing remaining dependencies..."
pip install -r requirements.txt

# Install playwright browsers
echo ""
echo "🌐 Installing Playwright browsers..."
playwright install

# Initialize ChromaDB
echo ""
echo "🗄️  Initializing ChromaDB..."
python -c "import chromadb; client = chromadb.Client(); print('✅ ChromaDB initialized successfully')"

echo ""
echo "=================================================="
echo "✅ Installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Set up environment variables in .env file"
echo "2. Run: python -m pytest tests/ to verify installation"
echo "3. Start server: uvicorn main:app --reload"
echo ""
echo "For more information, see: AGENTIC_AI_TRANSFORMATION.md"