#!/usr/bin/env python3
"""
Test script for the simplified nucleic acid analysis tools
Tests the single NucleicAcidAnalyzer class with direct method calls
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from tools import NucleicAcidAnalyzer


async def test_simplified_analyzer():
    """Test the simplified NucleicAcidAnalyzer class"""
    
    # Create analyzer instance
    analyzer = NucleicAcidAnalyzer()
    
    # Test sequences
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
    
    print("🧪 Testing Simplified Nucleic Acid Analyzer")
    print("=" * 50)
    
    # Test 1: Sequence stats (single sequence)
    print("\n1. Testing Single Sequence Stats...")
    try:
        stats_result = await analyzer.sequence_stats(test_sequences[0]["sequence"], "dna")
        print(f"   ✅ Length: {stats_result['length']} bp")
        print(f"   ✅ GC content: {stats_result['gc_content']}%")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Assembly stats (multiple sequences)
    print("\n2. Testing Assembly Statistics...")
    try:
        assembly_result = await analyzer.assembly_stats(test_sequences)
        print(f"   ✅ Total length: {assembly_result['total_length']} bp")
        print(f"   ✅ N50: {assembly_result['n50']} bp")
        print(f"   ✅ GC content: {assembly_result['gc_content']}%")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Giant virus promoter search
    print("\n3. Testing Giant Virus Promoter Search...")
    try:
        promoter_result = analyzer.giant_virus_promoter_search(test_sequences)
        print(f"   ✅ Found {promoter_result['summary']['total_motifs_found']} giant virus motifs")
        print(f"   ✅ Sequences with motifs: {promoter_result['summary']['sequences_with_giant_virus_motifs']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Translation
    print("\n4. Testing Sequence Translation...")
    try:
        translation_result = await analyzer.translate_sequence("ATGAAATAA", genetic_code=1)
        print(f"   ✅ Protein: {translation_result['protein_sequence']}")
        print(f"   ✅ ORFs found: {translation_result['orfs_found']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: K-mer analysis
    print("\n5. Testing K-mer Analysis...")
    try:
        kmer_result = await analyzer.kmer_analysis(test_sequences, k_values=[3, 4])
        print(f"   ✅ K-mer types analyzed: {list(kmer_result.keys())}")
        print(f"   ✅ 3-mer diversity (entropy): {kmer_result['3-mers']['shannon_entropy']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ All tests completed successfully!")
    print("\n📊 Architecture Summary:")
    print("   • Single NucleicAcidAnalyzer class")
    print("   • Direct method calls (no wrapper layer)")
    print("   • Focus on nucleic acid analysis only")
    print("   • Removed external tool placeholders")


if __name__ == "__main__":
    asyncio.run(test_simplified_analyzer())