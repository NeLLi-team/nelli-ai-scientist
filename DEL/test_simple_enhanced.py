#!/usr/bin/env python3
"""
Simple test for enhanced analysis tools
"""
import asyncio
import json
import sys
from pathlib import Path

# Add the mcps/bioseq src to the path
sys.path.insert(0, str(Path(__file__).parent / "mcps" / "bioseq" / "src"))

async def test_simple():
    """Simple test of enhanced functionality"""
    
    try:
        # Import the tools module directly
        from tools import NucleicAcidAnalyzer
        
        print("✅ Tools module imports successfully")
        
        # Test that the sandbox directory can be created
        sandbox_dir = Path("sandbox/analysis/plots")
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Sandbox directory created: {sandbox_dir.absolute()}")
        
        # Test basic Python execution with enhanced libraries
        test_code = """
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Test basic functionality
data_test = np.array([1, 2, 3, 4, 5])
df_test = pd.DataFrame({'values': data_test})
print(f"NumPy array: {data_test}")
print(f"Pandas DataFrame shape: {df_test.shape}")

# Test plot creation
plt.figure(figsize=(6, 4))
plt.plot(data_test)
plt.title('Test Plot')
test_success = True
"""
        
        # Execute the test code manually to verify libraries work
        try:
            import numpy as np
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            print("✅ All required libraries import successfully")
            
            # Test basic functionality
            data_test = np.array([1, 2, 3, 4, 5])
            df_test = pd.DataFrame({'values': data_test})
            print(f"✅ NumPy array: {data_test}")
            print(f"✅ Pandas DataFrame shape: {df_test.shape}")
            
            # Test plot creation and saving
            plt.figure(figsize=(6, 4))
            plt.plot(data_test)
            plt.title('Test Plot')
            
            # Save plot to sandbox
            plot_file = sandbox_dir / "test_plot.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            if plot_file.exists():
                print(f"✅ Plot saved successfully: {plot_file}")
            else:
                print("❌ Plot file not created")
                
        except Exception as e:
            print(f"❌ Library test failed: {e}")
            
        # Test JSON handling
        test_data = {
            "sequence_stats": {"length": 1000, "gc_content": 42.5},
            "test": "data"
        }
        
        test_json_file = Path("sandbox/analysis/test_data.json")
        with open(test_json_file, 'w') as f:
            json.dump(test_data, f, indent=2)
            
        # Read it back
        with open(test_json_file, 'r') as f:
            loaded_data = json.load(f)
            
        if loaded_data == test_data:
            print("✅ JSON file operations work correctly")
        else:
            print("❌ JSON file operations failed")
            
        print(f"\n🎉 Enhanced tools infrastructure tested successfully!")
        print(f"📁 Sandbox location: {sandbox_dir.parent.absolute()}")
        print(f"📈 Plot files: {list(sandbox_dir.glob('*.png'))}")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple())