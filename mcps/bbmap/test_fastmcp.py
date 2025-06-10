#!/usr/bin/env python3
"""Test FastMCP import"""
try:
    import fastmcp
    print("✅ FastMCP imported successfully")
    if hasattr(fastmcp, '__version__'):
        print(f"   Version: {fastmcp.__version__}")
except ImportError as e:
    print(f"❌ FastMCP import failed: {e}")
