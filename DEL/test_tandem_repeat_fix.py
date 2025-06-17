#!/usr/bin/env python3
"""
Test script to verify tandem repeat analysis is working correctly
"""

import asyncio
import sys
from pathlib import Path

# Add bioseq source to path
sys.path.append('/home/fschulz/dev/nelli-ai-scientist/mcps/bioseq/src')

from tools import NucleicAcidAnalyzer

async def test_tandem_repeat_analysis():
    """Test tandem repeat detection with a small example"""
    
    analyzer = NucleicAcidAnalyzer()
    
    # Create test sequences with known tandem repeats
    test_sequences = [
        {
            "id": "test_seq_1",
            "sequence": "ATGATGATGATGATGATG" * 3  # ATG repeat
        },
        {
            "id": "test_seq_2", 
            "sequence": "GCGCGCGCGCGCGCGC" * 2  # GC repeat
        },
        {
            "id": "test_seq_3",
            "sequence": "TAGTTAGTTAGTTAGTTAG"  # TAGT repeat
        }
    ]
    
    print("🧬 Testing Tandem Repeat Detection")
    print("=" * 60)
    
    # Test with sequences directly
    result = await analyzer.repeat_detection(sequences=test_sequences, min_repeat_length=2, max_repeat_length=10)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Detection completed successfully!")
        print(f"\nTotal sequences analyzed: {result['total_sequences']}")
        print(f"Total length: {result['total_length']} bp")
        print(f"\nTandem repeats found: {len(result['tandem_repeats'])}")
        
        # Show most frequent repeats
        if result['tandem_repeats']:
            # Count repeat units
            repeat_counts = {}
            for repeat in result['tandem_repeats']:
                unit = repeat['repeat_unit']
                if unit not in repeat_counts:
                    repeat_counts[unit] = 0
                repeat_counts[unit] += 1
            
            # Sort by frequency
            sorted_repeats = sorted(repeat_counts.items(), key=lambda x: x[1], reverse=True)
            
            print("\n📊 Most Frequently Found Tandem Repeats:")
            print("-" * 40)
            for unit, count in sorted_repeats[:5]:
                print(f"  {unit}: {count} occurrences")
    
    # Test with file path (if example file exists)
    example_file = Path("/home/fschulz/dev/nelli-ai-scientist/example/AC3300027503___Ga0255182_1000024.fna")
    if example_file.exists():
        print(f"\n\n🧬 Testing with real file: {example_file.name}")
        print("=" * 60)
        
        result = await analyzer.repeat_detection(sequences=str(example_file), min_repeat_length=10, max_repeat_length=50)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Detection completed!")
            print(f"\nFile info:")
            print(f"  - Total sequences: {result['total_sequences']}")
            print(f"  - Total length: {result['total_length']:,} bp")
            print(f"  - File sampled: {result.get('file_sampled', False)}")
            
            if result['tandem_repeats']:
                # Analyze repeat units
                repeat_units = {}
                for repeat in result['tandem_repeats']:
                    unit = repeat['repeat_unit']
                    if unit not in repeat_units:
                        repeat_units[unit] = {
                            'count': 0,
                            'total_copies': 0,
                            'positions': []
                        }
                    repeat_units[unit]['count'] += 1
                    repeat_units[unit]['total_copies'] += repeat['copy_number']
                    repeat_units[unit]['positions'].append(repeat['start'])
                
                # Sort by frequency
                sorted_units = sorted(repeat_units.items(), 
                                    key=lambda x: x[1]['count'], 
                                    reverse=True)
                
                print(f"\n📊 Top 10 Most Frequently Found Tandem Repeats:")
                print("-" * 60)
                print(f"{'Repeat Unit':<15} {'Count':<10} {'Avg Copies':<12} {'Example Position'}")
                print("-" * 60)
                
                for unit, data in sorted_units[:10]:
                    avg_copies = data['total_copies'] / data['count']
                    print(f"{unit:<15} {data['count']:<10} {avg_copies:<12.1f} {data['positions'][0]}")

if __name__ == "__main__":
    asyncio.run(test_tandem_repeat_analysis())