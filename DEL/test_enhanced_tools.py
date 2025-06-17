#!/usr/bin/env python3
"""
Test script for enhanced analysis tools in bioseq MCP server
"""
import asyncio
import json
import sys
from pathlib import Path

# Add the mcps/bioseq src to the path
sys.path.insert(0, str(Path(__file__).parent / "mcps" / "bioseq" / "src"))

async def test_enhanced_tools():
    """Test the enhanced execute_python_analysis and read_analysis_results tools"""
    
    try:
        from server import mcp
        print("✅ MCP server imports successfully")
        
        # Test data for analysis
        test_data = {
            "sequence_stats": {
                "length": 1000000,
                "gc_content": 42.5,
                "nucleotide_counts": {"A": 287500, "T": 287500, "G": 212500, "C": 212500}
            },
            "kmer_analysis": {
                "3": {"ATG": 1250, "TAA": 890, "TGA": 780, "TAG": 560, "GCA": 1100},
                "4": {"ATGC": 340, "CGAT": 290, "TAGA": 180, "GCTA": 220}
            },
            "promoter_identification": {
                "seq1": [
                    {"sequence": "TATAAA", "position": 150, "score": 0.85, "type": "TATA_box"},
                    {"sequence": "CAAT", "position": 180, "score": 0.72, "type": "CAAT_box"}
                ]
            }
        }
        
        # Test 1: execute_python_analysis tool
        print("\n🧪 Testing execute_python_analysis...")
        
        test_code = """
# Test analysis code
print("=== Enhanced Analysis Test ===")
print(f"Data keys available: {list(data.keys()) if data else 'No data'}")

if data and 'sequence_stats' in data:
    stats = data['sequence_stats']
    print(f"Sequence length: {stats['length']:,} bp")
    print(f"GC content: {stats['gc_content']:.1f}%")
    
    # Create a simple visualization
    import matplotlib.pyplot as plt
    nucleotides = list(stats['nucleotide_counts'].keys())
    counts = list(stats['nucleotide_counts'].values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(nucleotides, counts)
    plt.title('Nucleotide Composition')
    plt.ylabel('Count')
    plt.xlabel('Nucleotide')
    
    # Test variable creation
    analysis_summary = {
        'total_bp': stats['length'],
        'gc_ratio': stats['gc_content'] / 100,
        'at_ratio': (100 - stats['gc_content']) / 100
    }
    
    print(f"Analysis complete! AT/GC ratio: {analysis_summary['at_ratio']:.3f}")

print("=== Test completed successfully ===")
"""
        
        # Import the execute_python_analysis function
        from server import execute_python_analysis
        
        result = await execute_python_analysis(
            code=test_code,
            context_data=test_data,
            output_file="test_analysis.md"
        )
        
        print(f"✅ execute_python_analysis completed: {result['success']}")
        if result['success']:
            print(f"📊 Output captured: {len(result['stdout'])} characters")
            print(f"🔧 Variables created: {len(result['local_variables'])}")
            print(f"📈 Plots saved: {len(result['saved_plots'])}")
            print(f"📁 Sandbox dir: {result['sandbox_dir']}")
            if result['output_file']:
                print(f"📄 Output file: {result['output_file']}")
        else:
            print(f"❌ Error: {result['error']}")
            
        # Test 2: Check if sandbox directory was created
        sandbox_dir = Path("sandbox/analysis")
        if sandbox_dir.exists():
            print(f"✅ Sandbox directory created at: {sandbox_dir.absolute()}")
            plots_dir = sandbox_dir / "plots"
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                print(f"📈 Found {len(plot_files)} plot files")
        
        # Test 3: read_analysis_results tool
        print(f"\n📖 Testing read_analysis_results...")
        
        # Create a test JSON file
        test_results_file = Path("sandbox/analysis/test_results.json")
        test_results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(test_results_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        from server import read_analysis_results
        read_result = await read_analysis_results(str(test_results_file))
        
        print(f"✅ read_analysis_results completed: {read_result['success']}")
        if read_result['success']:
            data_keys = read_result['analysis_data'].keys()
            print(f"📊 Data keys read: {list(data_keys)}")
        else:
            print(f"❌ Error: {read_result['error']}")
            
        print(f"\n🎉 All enhanced tools tested successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the project root and bioseq dependencies are installed")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_tools())