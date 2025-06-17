#!/usr/bin/env python3
"""
Test script for the new bioinformatics methods
Tests the giant virus promoter search, GC skew analysis, and CpG island detection
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from biotools import BioToolkit


async def test_new_methods():
    """Test the new bioinformatics analysis methods"""
    
    # Create toolkit instance
    toolkit = BioToolkit()
    
    # Test sequences (simple examples)
    test_sequences = [
        {
            "id": "test_seq_1",
            "sequence": "ATCGATCGATCGATCGTATATAAAATTGAATCGATCGATCGAAAAAATTGGATCGATCGTATATAAATCGATCGATCG"
        },
        {
            "id": "test_seq_2", 
            "sequence": "CGATCGATCGATCGCGCGCGCGCGCGATCGATCGAAAATTGAATCGATCGATATATATATATATATATCGATCGATCG"
        }
    ]
    
    print("🧪 Testing New Bioinformatics Methods")
    print("=" * 50)
    
    # Test 1: Giant virus promoter search
    print("\n1. Testing Giant Virus Promoter Search...")
    try:
        promoter_result = await toolkit.giant_virus_promoter_search(test_sequences)
        print(f"   ✅ Found {promoter_result['summary']['total_motifs_found']} giant virus motifs")
        print(f"   📊 Motif types: {list(promoter_result['summary']['motif_type_distribution'].keys())}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Assembly stats
    print("\n2. Testing Assembly Statistics...")
    try:
        assembly_result = await toolkit.assembly_stats(test_sequences)
        print(f"   ✅ Assembly stats calculated")
        print(f"   📊 Total length: {assembly_result['total_length']} bp")
        print(f"   📊 GC content: {assembly_result['gc_content']}%")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: K-mer analysis
    print("\n3. Testing K-mer Analysis...")
    try:
        kmer_result = await toolkit.kmer_analysis(test_sequences, k_values=[3, 4])
        print(f"   ✅ K-mer analysis completed")
        print(f"   📊 K-mer types analyzed: {list(kmer_result.keys())}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: For longer sequences, test GC skew and CpG islands
    long_sequence = [
        {
            "id": "long_test_seq",
            "sequence": "A" * 5000 + "T" * 5000 + "G" * 5000 + "C" * 5000  # 20kb test sequence
        }
    ]
    
    print("\n4. Testing GC Skew Analysis (long sequence)...")
    try:
        gc_skew_result = await toolkit.gc_skew_analysis(long_sequence, window_size=1000, step_size=500)
        print(f"   ✅ GC skew analysis completed")
        if gc_skew_result['summary']:
            print(f"   📊 Mean GC skew: {gc_skew_result['summary']['overall_mean_skew']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n5. Testing CpG Island Detection...")
    try:
        cpg_result = await toolkit.cpg_island_detection(long_sequence)
        print(f"   ✅ CpG island detection completed")
        print(f"   📊 Islands found: {cpg_result['summary']['total_cpg_islands']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_new_methods())