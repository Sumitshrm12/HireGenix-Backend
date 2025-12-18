#!/usr/bin/env python
"""
Production-grade startup script for HireGenix Backend
Handles warning suppression before any library imports
"""

# CRITICAL: Must be the very first lines before ANY imports
import warnings
import os

# Suppress urllib3 LibreSSL warning (macOS Python 3.9 compatibility)
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Now safe to import urllib3 and suppress its specific warning
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

# Import and run the main application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
