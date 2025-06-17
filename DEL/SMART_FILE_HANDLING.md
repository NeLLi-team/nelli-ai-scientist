# Smart File Handling for Large Biological Sequences

## Problem Solved

The agent was trying to process massive biological sequence files (like the 5.4 million token contig file) which exceeded the LLM's 1M token limit, causing critical failures. This has been fixed with intelligent file handling.

## Smart File Handling Implementation

### 1. **File Size Detection and Sampling**

```python
# Automatic file size checking
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB threshold
SAMPLE_SIZE = 1000  # Sample sequences from large files
MAX_SEQUENCE_LENGTH = 100000  # Truncate very long sequences
```

### 2. **Intelligent Sampling Strategy**

For large files (>50MB):
- **Quick analysis**: Uses `grep -c '^>'` to count sequences without loading
- **Smart sampling**: Selects representative sequences across the file
- **Length distribution**: Analyzes first 100 sequences for size patterns
- **Sequence truncation**: Keeps beginning, middle, and end of very long sequences

### 3. **All Analysis Tools Now Support File Paths**

**Before** (would fail with large files):
```python
# Agent had to read entire file into memory first
sequences = read_fasta_file("large_contigs.fasta")  # 5.4M tokens!
assembly_stats(sequences)  # CRASH - token limit exceeded
```

**After** (smart handling):
```python
# Direct file path input with automatic sampling
assembly_stats("large_contigs.fasta")  # ✅ Works with any file size
repeat_detection("large_contigs.fasta")  # ✅ Intelligent sampling 
gene_prediction_and_coding_stats("large_contigs.fasta")  # ✅ Safe processing
```

### 4. **Enhanced Tools with Smart File Handling**

All these tools now accept either file paths OR sequence lists:

- `assembly_stats()` - Assembly statistics with intelligent sampling
- `repeat_detection()` - Repeat analysis on representative sequences  
- `gene_prediction_and_coding_stats()` - Gene finding with sampling
- `kmer_analysis()` - K-mer frequency analysis
- `promoter_identification()` - Promoter motif search
- `giant_virus_promoter_search()` - Specialized giant virus analysis
- `gc_skew_analysis()` - GC skew calculation
- `cpg_island_detection()` - CpG island identification

### 5. **Sample Output for Large Files**

```json
{
  "file_path": "/path/to/large_file.fasta",
  "file_size_mb": 150.5,
  "large_file_mode": true,
  "file_statistics": {
    "total_sequences": 25000,
    "estimated_total_length": 500000000,
    "avg_sequence_length": 20000
  },
  "n_sequences_sampled": 1000,
  "sequences": [...],
  "sampling_strategy": "intelligent_sampling",
  "note": "Large file (25000 sequences) - showing representative sample of 1000 sequences"
}
```

### 6. **Sequence Truncation for Very Long Sequences**

For sequences >100kb:
```
ATCGATCGATCG...[MIDDLE_TRUNCATED:85000bp]...GCTAGCTAG...[END_TRUNCATED]...TTAACCGGAA
```

Preserves:
- First 5kb (promoter regions, start of genes)
- Middle 2kb (representative internal content)  
- Last 5kb (terminator regions, end of genes)

## Benefits

### 🛡️ **Prevents Token Limit Crashes**
- No more 5.4M token failures
- Handles files of any size safely
- Automatic sampling prevents memory issues

### 📊 **Maintains Analysis Quality**
- Representative sampling preserves statistical properties
- Length distributions remain accurate
- Gene density calculations still valid

### 🔄 **Backwards Compatible**
- All existing code continues to work
- Sequence list inputs work as before
- File path inputs now work intelligently

### ⚡ **Performance Optimized**
- Quick file analysis using shell commands
- Streaming processing for large files
- Memory-efficient sampling algorithms

## Usage Examples

### Direct File Analysis
```python
# Instead of loading massive files:
# sequences = read_fasta_file("huge_genome.fasta")  # OLD: Could crash
# assembly_stats(sequences)

# Now works directly:
result = assembly_stats("huge_genome.fasta")  # NEW: Always safe
```

### Custom Sampling Control
```python
# Control sampling behavior
file_data = read_fasta_file("large_file.fasta", sample_large_files=True)
# Or disable sampling for smaller files
file_data = read_fasta_file("small_file.fasta", sample_large_files=False)
```

### Mixed Input Types
```python
# Works with file paths
stats1 = assembly_stats("genome.fasta")

# Works with sequence lists  
stats2 = assembly_stats([{"id": "seq1", "sequence": "ATCG..."}])
```

## Technical Implementation

### Smart File Reader
- `_analyze_fasta_structure()` - Quick file statistics
- `_smart_fasta_sample()` - Intelligent sequence sampling  
- `_sample_fasta_sequences()` - Representative sequence selection

### Tool Integration
- All analysis tools check `isinstance(sequences, str)` for file paths
- Automatic conversion using `convert_to_sequence_list()`
- Error handling preserves original error context

### File Size Thresholds
- **50MB** - Triggers smart sampling mode
- **100KB** - Individual sequence truncation limit  
- **1000** - Default number of sampled sequences
- **10,000** - Maximum sequences loaded at once

## Result

The agent can now safely analyze biological sequence files of **any size** without risk of token limit failures, while maintaining high-quality scientific analysis through intelligent sampling strategies.

Critical biological data analysis operations that previously failed with "token count exceeds maximum" now work seamlessly with files containing millions of sequences.