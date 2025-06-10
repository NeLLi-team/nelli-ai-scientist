#!/usr/bin/env python3
"""Test BBMap tools import"""
import sys
from pathlib import Path

# Add BBMap src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from bbmap_tools import BBMapToolkit
    toolkit = BBMapToolkit()
    print("✅ BBMapToolkit imported successfully")
    print(f"   Shifter image: {toolkit.shifter_image}")
    print(f"   Base command: {' '.join(toolkit.base_command)}")
except ImportError as e:
    print(f"❌ BBMapToolkit import failed: {e}")
except Exception as e:
    print(f"❌ BBMapToolkit error: {e}")
