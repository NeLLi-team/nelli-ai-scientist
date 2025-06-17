#!/usr/bin/env python3
"""
Demo script showing how to properly analyze and display tandem repeat results
This is what the agent should be doing when asked for tandem repeat details
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

def analyze_tandem_repeats(repeat_data):
    """
    Analyze tandem repeat data to find the most frequently occurring patterns
    """
    if not repeat_data or 'tandem_repeats' not in repeat_data:
        return "No tandem repeat data available"
    
    tandem_repeats = repeat_data['tandem_repeats']
    
    if not tandem_repeats:
        return "No tandem repeats found in the sequences"
    
    # Collect statistics
    repeat_unit_stats = defaultdict(lambda: {
        'count': 0,
        'total_copies': 0,
        'total_length': 0,
        'positions': [],
        'copy_numbers': []
    })
    
    # Analyze each repeat
    for repeat in tandem_repeats:
        unit = repeat['repeat_unit']
        stats = repeat_unit_stats[unit]
        
        stats['count'] += 1
        stats['total_copies'] += repeat['copy_number']
        stats['total_length'] += repeat['total_length']
        stats['positions'].append(repeat['start'])
        stats['copy_numbers'].append(repeat['copy_number'])
    
    # Sort by frequency
    sorted_units = sorted(repeat_unit_stats.items(), 
                         key=lambda x: x[1]['count'], 
                         reverse=True)
    
    # Format results
    results = []
    results.append("## 📊 Most Frequently Found Tandem Repeats\n")
    results.append(f"**Total unique repeat units:** {len(repeat_unit_stats)}")
    results.append(f"**Total tandem repeat instances:** {len(tandem_repeats)}\n")
    
    results.append("### Top 10 Most Frequent Tandem Repeats:\n")
    results.append("| Repeat Unit | Occurrences | Avg Copy Number | Total Bases | Category |")
    results.append("|-------------|-------------|-----------------|-------------|----------|")
    
    for i, (unit, stats) in enumerate(sorted_units[:10]):
        avg_copies = stats['total_copies'] / stats['count']
        
        # Categorize by length
        unit_len = len(unit)
        if unit_len <= 2:
            category = "Dinucleotide"
        elif unit_len <= 3:
            category = "Trinucleotide"
        elif unit_len <= 6:
            category = "Microsatellite"
        else:
            category = "Minisatellite"
        
        results.append(f"| {unit} | {stats['count']} | {avg_copies:.1f} | {stats['total_length']} | {category} |")
    
    # Additional analysis
    results.append("\n### Repeat Unit Composition Analysis:\n")
    
    # AT-rich vs GC-rich
    at_rich = []
    gc_rich = []
    palindromic = []
    
    for unit, stats in repeat_unit_stats.items():
        at_content = (unit.count('A') + unit.count('T')) / len(unit)
        
        if at_content >= 0.7:
            at_rich.append((unit, stats['count']))
        elif at_content <= 0.3:
            gc_rich.append((unit, stats['count']))
        
        # Check if palindromic
        if unit == unit[::-1]:
            palindromic.append((unit, stats['count']))
    
    if at_rich:
        results.append(f"**AT-rich repeats ({len(at_rich)} types):** " + 
                      ", ".join([f"{u} ({c}x)" for u, c in sorted(at_rich, key=lambda x: x[1], reverse=True)[:5]]))
    
    if gc_rich:
        results.append(f"**GC-rich repeats ({len(gc_rich)} types):** " + 
                      ", ".join([f"{u} ({c}x)" for u, c in sorted(gc_rich, key=lambda x: x[1], reverse=True)[:5]]))
    
    if palindromic:
        results.append(f"**Palindromic repeats ({len(palindromic)} types):** " + 
                      ", ".join([f"{u} ({c}x)" for u, c in sorted(palindromic, key=lambda x: x[1], reverse=True)[:5]]))
    
    # Length distribution
    results.append("\n### Repeat Unit Length Distribution:\n")
    length_dist = Counter(len(unit) for unit in repeat_unit_stats.keys())
    
    for length in sorted(length_dist.keys()):
        results.append(f"- **{length} bp units:** {length_dist[length]} different sequences")
    
    return "\n".join(results)


# Example usage
if __name__ == "__main__":
    # Load example data if available
    analysis_files = list(Path("/home/fschulz/dev/nelli-ai-scientist/reports").glob("*_analysis.json"))
    
    if analysis_files:
        # Use the most recent analysis file
        latest_file = max(analysis_files, key=lambda f: f.stat().st_mtime)
        
        print(f"Loading analysis from: {latest_file}")
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Find repeat data
        if 'repeat_analysis' in data:
            repeat_data = data['repeat_analysis']
        elif 'repeats' in data:
            repeat_data = data['repeats']
        else:
            # Search nested structure
            for key, value in data.items():
                if isinstance(value, dict) and 'tandem_repeats' in value:
                    repeat_data = value
                    break
            else:
                repeat_data = None
        
        if repeat_data:
            analysis = analyze_tandem_repeats(repeat_data)
            print("\n" + analysis)
        else:
            print("No repeat data found in the analysis file")
    else:
        # Create mock data for demonstration
        mock_data = {
            'tandem_repeats': [
                {'repeat_unit': 'AT', 'copy_number': 15, 'start': 1000, 'total_length': 30},
                {'repeat_unit': 'AT', 'copy_number': 10, 'start': 2000, 'total_length': 20},
                {'repeat_unit': 'GC', 'copy_number': 8, 'start': 3000, 'total_length': 16},
                {'repeat_unit': 'ATG', 'copy_number': 5, 'start': 4000, 'total_length': 15},
                {'repeat_unit': 'TAGA', 'copy_number': 7, 'start': 5000, 'total_length': 28},
            ]
        }
        
        print("\nDemo with mock data:")
        analysis = analyze_tandem_repeats(mock_data)
        print(analysis)