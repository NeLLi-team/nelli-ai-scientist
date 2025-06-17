#!/usr/bin/env python3
"""
Test script to calculate tandem repeat percentage from real data
"""

import json
from pathlib import Path

def calculate_repeat_percentage(repeat_data, full_data=None):
    """Calculate tandem repeat percentage"""
    if not repeat_data or 'tandem_repeats' not in repeat_data:
        return "No tandem repeat data available"
    
    tandem_repeats = repeat_data['tandem_repeats']
    if not tandem_repeats:
        return "No tandem repeats found"
    
    # Calculate genome percentage
    total_genome_length = repeat_data.get('total_length', 0)
    
    # Try to get total length from various sources
    if total_genome_length == 0 and full_data:
        # Check assembly stats
        if 'assembly_stats' in full_data:
            total_genome_length = full_data['assembly_stats'].get('total_length', 0)
    
    if total_genome_length == 0:
        # Try alternative ways to get total length
        if 'per_sequence_stats' in repeat_data:
            total_genome_length = sum(stat.get('length', 0) for stat in repeat_data['per_sequence_stats'])
    
    # Also check if it's already in summary
    if total_genome_length == 0 and 'summary' in repeat_data:
        total_genome_length = repeat_data['summary'].get('total_length', 0)
    
    total_repeat_bases = sum(repeat.get('total_length', 0) for repeat in tandem_repeats)
    
    if total_genome_length > 0:
        repeat_percentage = (total_repeat_bases / total_genome_length) * 100
        
        print(f"📊 Tandem Repeat Genome Coverage")
        print(f"Total genome analyzed: {total_genome_length:,} bp")
        print(f"Total bases in tandem repeats: {total_repeat_bases:,} bp")
        print(f"Percentage of genome consisting of tandem repeats: {repeat_percentage:.2f}%")
        
        # Add some context
        if repeat_percentage < 1:
            print("This is a relatively low percentage, suggesting the genome has few repetitive regions.")
        elif repeat_percentage < 5:
            print("This is a moderate percentage, typical for many bacterial genomes.")
        elif repeat_percentage < 15:
            print("This is a high percentage, indicating significant repetitive content.")
        else:
            print("This is a very high percentage, suggesting extensive repetitive regions.")
            
        return repeat_percentage
    else:
        print("Unable to calculate percentage - total genome length not available")
        return None

# Test with real data
analysis_file = Path("/home/fschulz/dev/nelli-ai-scientist/reports/AC3300027503___Ga0255182_1000024_analysis.json")

if analysis_file.exists():
    latest_file = analysis_file
    
    print(f"Loading analysis from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Find repeat data
    repeat_data = None
    if 'repeat_analysis' in data:
        repeat_data = data['repeat_analysis']
        print(f"Found repeat_analysis with keys: {list(repeat_data.keys())}")
    elif 'repeats' in data:
        repeat_data = data['repeats']
    else:
        # Search nested structure
        for key, value in data.items():
            if isinstance(value, dict) and 'tandem_repeats' in value:
                repeat_data = value
                break
    
    if repeat_data:
        print("\n" + "="*60)
        percentage = calculate_repeat_percentage(repeat_data, data)
        print("="*60)
        
        # Also show some details about the repeats
        if 'tandem_repeats' in repeat_data:
            repeats = repeat_data['tandem_repeats']
            print(f"\nFound {len(repeats)} tandem repeat instances")
            
            # Show longest repeat
            if repeats:
                longest = max(repeats, key=lambda r: r.get('total_length', 0))
                print(f"Longest repeat: {longest.get('repeat_unit', 'Unknown')} "
                      f"({longest.get('copy_number', 0)} copies, "
                      f"{longest.get('total_length', 0)} bp total)")
    else:
        print("No repeat data found in the analysis file")
        # Show what keys are available
        print(f"Available keys: {list(data.keys())}")
else:
    print("No analysis files found")