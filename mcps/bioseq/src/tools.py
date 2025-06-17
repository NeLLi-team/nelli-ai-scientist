"""
Nucleic acid sequence analysis tools
Focused on DNA/RNA analysis including assembly stats, promoter detection, 
GC skew analysis, CpG islands, and specialized giant virus analysis
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from collections import Counter, defaultdict
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import re
from itertools import product

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
import pyrodigal

logger = logging.getLogger(__name__)


class NucleicAcidAnalyzer:
    """Comprehensive nucleic acid sequence analysis toolkit for DNA and RNA"""
    
    def __init__(self, n_threads: int = 8):
        self.n_threads = n_threads
        self.valid_nucleotides = set('ATCGNU')  # Including U for RNA
        self.ambiguous_nucleotides = set('RYSWKMBDHVN')
        
        # Smart file handling limits
        self.MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        self.SAMPLE_SIZE = 1000  # Number of sequences to sample for large files
        self.MAX_SEQUENCE_LENGTH = 100000  # Max individual sequence length to process
        self.MAX_TOTAL_SEQUENCES = 10000  # Max total sequences to load at once
        
    def validate_nucleic_acid(self, sequence: str) -> Dict[str, Any]:
        """Validate if input is a valid nucleic acid sequence"""
        sequence = sequence.upper()
        seq_set = set(sequence)
        
        # Check for valid nucleotides
        valid_chars = self.valid_nucleotides | self.ambiguous_nucleotides
        invalid_chars = seq_set - valid_chars
        
        if invalid_chars:
            return {
                "valid": False,
                "sequence_type": None,
                "invalid_characters": list(invalid_chars),
                "error": f"Invalid characters found: {', '.join(invalid_chars)}"
            }
        
        # Determine if DNA or RNA
        has_t = 'T' in seq_set
        has_u = 'U' in seq_set
        
        if has_t and has_u:
            return {
                "valid": False,
                "sequence_type": None,
                "error": "Sequence contains both T and U"
            }
        
        sequence_type = "rna" if has_u else "dna"
        
        # Calculate composition
        composition = {
            "standard_nucleotides": sum(1 for c in sequence if c in self.valid_nucleotides),
            "ambiguous_nucleotides": sum(1 for c in sequence if c in self.ambiguous_nucleotides),
            "percentage_ambiguous": (sum(1 for c in sequence if c in self.ambiguous_nucleotides) / len(sequence) * 100) if sequence else 0
        }
        
        return {
            "valid": True,
            "sequence_type": sequence_type,
            "length": len(sequence),
            "composition": composition
        }

    async def sequence_stats(self, sequence: str, sequence_type: str = "dna") -> Dict[str, Any]:
        """Calculate comprehensive sequence statistics"""
        # Validate nucleic acid
        validation = self.validate_nucleic_acid(sequence)
        if not validation["valid"]:
            return {"error": validation["error"]}
        
        # Basic stats
        seq_upper = sequence.upper()
        gc_content = (seq_upper.count('G') + seq_upper.count('C')) / len(sequence) * 100 if sequence else 0
        
        # Count nucleotides
        nucleotide_counts = Counter(seq_upper)
        
        return {
            "length": len(sequence),
            "type": sequence_type,
            "gc_content": round(gc_content, 2),
            "nucleotide_counts": dict(nucleotide_counts),
            "ambiguous_bases": sum(1 for n in seq_upper if n in self.ambiguous_nucleotides)
        }

    async def translate_sequence(self, sequence: str, genetic_code: int = 1) -> Dict[str, Any]:
        """Translate DNA/RNA sequence to protein"""
        try:
            # Validate nucleic acid
            validation = self.validate_nucleic_acid(sequence)
            if not validation["valid"]:
                return {"error": validation["error"]}
            
            # Convert RNA to DNA if needed
            seq_upper = sequence.upper()
            if 'U' in seq_upper:
                seq_upper = seq_upper.replace('U', 'T')
            
            # Create Bio.Seq object
            bio_seq = Seq(seq_upper)
            
            # Translate
            protein = str(bio_seq.translate(table=genetic_code))
            
            # Find all ORFs
            orfs = []
            for frame in range(3):
                frame_seq = bio_seq[frame:]
                frame_trans = str(frame_seq.translate(table=genetic_code))
                
                # Find ORFs (start codon to stop codon)
                start = 0
                while True:
                    try:
                        start = frame_trans.index('M', start)
                        stop = frame_trans.index('*', start)
                        if stop - start >= 10:  # Minimum 10 amino acids
                            orfs.append({
                                "frame": frame + 1,
                                "start": frame + start * 3,
                                "end": frame + (stop + 1) * 3,
                                "length": (stop - start + 1) * 3,
                                "protein": frame_trans[start:stop+1]
                            })
                        start = stop + 1
                    except ValueError:
                        break
            
            return {
                "protein_sequence": protein,
                "length": len(protein),
                "orfs_found": len(orfs),
                "orfs": orfs[:10],  # First 10 ORFs
                "genetic_code": genetic_code
            }
            
        except Exception as e:
            return {"error": f"Translation failed: {str(e)}"}

    async def read_fasta_file(self, file_path: str, sample_large_files: bool = True) -> Dict[str, Any]:
        """Smart FASTA file reader that handles large files intelligently"""
        import os
        
        try:
            # Check file size first
            file_size = os.path.getsize(file_path)
            
            if file_size > self.MAX_FILE_SIZE and sample_large_files:
                # File is too large - use smart sampling
                return await self._smart_fasta_sample(file_path, file_size)
            else:
                # File is small enough - read normally
                return await self._read_fasta_normal(file_path, file_size)
                
        except Exception as e:
            return {"error": f"Failed to read FASTA file: {str(e)}"}
    
    async def _read_fasta_normal(self, file_path: str, file_size: int) -> Dict[str, Any]:
        """Read FASTA file normally (for smaller files)"""
        sequences = []
        sequence_count = 0
        
        for record in SeqIO.parse(file_path, "fasta"):
            sequence_count += 1
            
            # Limit number of sequences to prevent memory issues
            if sequence_count > self.MAX_TOTAL_SEQUENCES:
                break
                
            # Truncate very long sequences
            seq_str = str(record.seq)
            if len(seq_str) > self.MAX_SEQUENCE_LENGTH:
                seq_str = seq_str[:self.MAX_SEQUENCE_LENGTH] + "...[TRUNCATED]"
                
            sequences.append({
                "id": record.id,
                "sequence": seq_str,
                "description": record.description,
                "length": len(record.seq)
            })
        
        return {
            "file_path": file_path,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024*1024), 2),
            "n_sequences": len(sequences),
            "total_sequences_in_file": sequence_count,
            "sequences": sequences,
            "truncated": sequence_count > self.MAX_TOTAL_SEQUENCES
        }
    
    async def _smart_fasta_sample(self, file_path: str, file_size: int) -> Dict[str, Any]:
        """Smart sampling for large FASTA files"""
        # First, get file statistics without loading everything
        file_stats = await self._analyze_fasta_structure(file_path)
        
        # Sample sequences intelligently
        sampled_sequences = await self._sample_fasta_sequences(file_path, file_stats)
        
        return {
            "file_path": file_path,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024*1024), 2),
            "large_file_mode": True,
            "file_statistics": file_stats,
            "n_sequences_sampled": len(sampled_sequences),
            "sequences": sampled_sequences,
            "sampling_strategy": "intelligent_sampling",
            "note": f"Large file ({file_stats['total_sequences']} sequences) - showing representative sample of {len(sampled_sequences)} sequences"
        }
    
    async def _analyze_fasta_structure(self, file_path: str) -> Dict[str, Any]:
        """Quickly analyze FASTA file structure without loading all sequences"""
        sequence_count = 0
        total_length = 0
        length_distribution = []
        id_samples = []
        
        # Use head/tail approach for quick analysis
        import subprocess
        
        try:
            # Count total sequences quickly
            result = subprocess.run(['grep', '-c', '^>', file_path], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                sequence_count = int(result.stdout.strip())
        except:
            # Fallback to manual counting (limited)
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if line.startswith('>'):
                        sequence_count += 1
                    if i > 100000:  # Limit scanning to first 100k lines
                        break
        
        # Sample sequences for length analysis
        sample_count = 0
        current_seq_length = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq_length > 0:
                        length_distribution.append(current_seq_length)
                        total_length += current_seq_length
                        current_seq_length = 0
                        
                    # Sample ID
                    if sample_count < 10:
                        id_samples.append(line[1:])  # Remove '>'
                        sample_count += 1
                else:
                    current_seq_length += len(line)
                    
                # Limit analysis to prevent long scanning
                if sample_count >= 100:
                    break
        
        return {
            "total_sequences": sequence_count,
            "estimated_total_length": total_length,
            "sample_length_distribution": length_distribution,
            "sample_ids": id_samples,
            "avg_sequence_length": total_length / len(length_distribution) if length_distribution else 0
        }
    
    async def _sample_fasta_sequences(self, file_path: str, file_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sample sequences intelligently from large FASTA file"""
        total_sequences = file_stats["total_sequences"]
        sample_size = min(self.SAMPLE_SIZE, total_sequences)
        
        # Calculate sampling interval
        if total_sequences <= sample_size:
            sampling_interval = 1
        else:
            sampling_interval = total_sequences // sample_size
            
        sequences = []
        current_index = 0
        next_sample_index = 0
        
        for record in SeqIO.parse(file_path, "fasta"):
            if current_index == next_sample_index:
                # Include this sequence in sample
                seq_str = str(record.seq)
                
                # Truncate very long sequences but keep some representative content
                if len(seq_str) > self.MAX_SEQUENCE_LENGTH:
                    # Keep beginning, some middle, and end
                    start = seq_str[:5000]
                    middle_pos = len(seq_str) // 2
                    middle = seq_str[middle_pos-1000:middle_pos+1000]
                    end = seq_str[-5000:]
                    seq_str = f"{start}...[MIDDLE_TRUNCATED:{len(seq_str)-12000}bp]...{middle}...[END_TRUNCATED]...{end}"
                
                sequences.append({
                    "id": record.id,
                    "sequence": seq_str,
                    "description": record.description,
                    "length": len(record.seq),
                    "original_index": current_index,
                    "truncated": len(record.seq) > self.MAX_SEQUENCE_LENGTH
                })
                
                next_sample_index = current_index + sampling_interval
                
                # Stop if we have enough samples
                if len(sequences) >= sample_size:
                    break
                    
            current_index += 1
            
            # Safety break for extremely large files
            if current_index > total_sequences * 2:
                break
        
        return sequences

    async def convert_to_sequence_list(self, input_data) -> List[Dict[str, str]]:
        """Convert file path or existing sequence list to proper sequence format"""
        if isinstance(input_data, str):
            # It's a file path
            file_result = await self.read_fasta_file(input_data, sample_large_files=True)
            if "error" in file_result:
                raise ValueError(f"Failed to read file: {file_result['error']}")
            return file_result.get("sequences", [])
        elif isinstance(input_data, list):
            # Already a sequence list
            return input_data
        else:
            raise ValueError(f"Input must be file path (str) or sequence list, got {type(input_data)}")

    async def write_json_report(self, data: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Write analysis results to a JSON report file"""
        try:
            import json
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Get file size
            import os
            file_size = os.path.getsize(output_path)
            
            return {
                "success": True,
                "output_path": output_path,
                "file_size": file_size
            }
        except Exception as e:
            return {"error": f"Failed to write JSON report: {str(e)}"}
    
    async def assembly_stats(self, sequences: List[Dict[str, str]]) -> Dict[str, Any]:
        """Calculate comprehensive assembly statistics including N50, L50, etc."""
        # Handle file path input by converting to sequence list
        if isinstance(sequences, str):
            file_result = await self.read_fasta_file(sequences, sample_large_files=True)
            if "error" in file_result:
                return file_result
            sequences = file_result.get("sequences", [])
        
        if not sequences:
            return {"error": "No sequences provided"}
        
        # Extract sequence lengths
        seq_lengths = []
        total_gc = 0
        total_n = 0
        total_length = 0
        
        for seq_dict in sequences:
            seq = seq_dict.get("sequence", "")
            seq_upper = seq.upper()
            length = len(seq)
            seq_lengths.append(length)
            total_length += length
            
            # Count GC and N
            gc_count = seq_upper.count('G') + seq_upper.count('C')
            n_count = seq_upper.count('N')
            total_gc += gc_count
            total_n += n_count
        
        # Sort lengths for N50/L50 calculation
        seq_lengths.sort(reverse=True)
        
        # Calculate N50 and L50
        cumsum = 0
        n50 = 0
        l50 = 0
        half_total = total_length / 2
        
        for i, length in enumerate(seq_lengths):
            cumsum += length
            if cumsum >= half_total:
                n50 = length
                l50 = i + 1
                break
        
        # Calculate N90 and L90
        cumsum = 0
        n90 = 0
        l90 = 0
        ninety_percent = total_length * 0.9
        
        for i, length in enumerate(seq_lengths):
            cumsum += length
            if cumsum >= ninety_percent:
                n90 = length
                l90 = i + 1
                break
        
        # GC content
        gc_content = (total_gc / total_length * 100) if total_length > 0 else 0
        n_content = (total_n / total_length * 100) if total_length > 0 else 0
        
        # Size distribution
        size_ranges = {
            "<1kb": sum(1 for l in seq_lengths if l < 1000),
            "1-10kb": sum(1 for l in seq_lengths if 1000 <= l < 10000),
            "10-100kb": sum(1 for l in seq_lengths if 10000 <= l < 100000),
            "100kb-1Mb": sum(1 for l in seq_lengths if 100000 <= l < 1000000),
            ">1Mb": sum(1 for l in seq_lengths if l >= 1000000)
        }
        
        return {
            "total_length": total_length,
            "n_sequences": len(sequences),
            "gc_content": round(gc_content, 2),
            "n_content": round(n_content, 2),
            "n50": n50,
            "l50": l50,
            "n90": n90,
            "l90": l90,
            "longest_sequence": seq_lengths[0] if seq_lengths else 0,
            "shortest_sequence": seq_lengths[-1] if seq_lengths else 0,
            "mean_length": round(total_length / len(sequences), 2) if sequences else 0,
            "median_length": seq_lengths[len(seq_lengths)//2] if seq_lengths else 0,
            "size_distribution": size_ranges
        }
    
    async def repeat_detection(self, sequences: List[Dict[str, str]], 
                             min_repeat_length: int = 10,
                             max_repeat_length: int = 100) -> Dict[str, Any]:
        """Detect various types of repeats in sequences"""
        # Handle file path input by converting to sequence list
        if isinstance(sequences, str):
            file_result = await self.read_fasta_file(sequences, sample_large_files=True)
            if "error" in file_result:
                return file_result
            sequences = file_result.get("sequences", [])
        
        # Handle case where sequences might be a list of strings (sequence data)
        if sequences and isinstance(sequences[0], str):
            # Convert to expected format
            sequences = [{"id": f"seq_{i}", "sequence": seq} for i, seq in enumerate(sequences)]
        
        # Ensure sequences is in the correct format
        if not sequences:
            return {"error": "No sequences provided"}
        
        total_length = sum(len(s["sequence"]) for s in sequences)
        repeat_stats = {
            "total_sequences": len(sequences),
            "total_length": total_length,
            "tandem_repeats": [],
            "inverted_repeats": [],
            "direct_repeats": [],
            "per_sequence_stats": []
        }
        
        total_repeat_bases = 0
        
        for seq_dict in sequences:
            seq_id = seq_dict.get("id", "unknown")
            sequence = seq_dict["sequence"].upper()
            seq_length = len(sequence)
            seq_repeat_bases = 0
            
            # Tandem repeats detection
            tandem_repeats = self._find_tandem_repeats(sequence, min_repeat_length, max_repeat_length)
            for repeat in tandem_repeats:
                repeat["sequence_id"] = seq_id
                repeat_stats["tandem_repeats"].append(repeat)
                seq_repeat_bases += repeat["total_length"]
            
            # Simple sequence repeats (SSRs)
            ssrs = self._find_ssrs(sequence)
            
            # Calculate repeat density for this sequence
            repeat_percentage = (seq_repeat_bases / seq_length * 100) if seq_length > 0 else 0
            
            repeat_stats["per_sequence_stats"].append({
                "sequence_id": seq_id,
                "sequence_length": seq_length,
                "repeat_bases": seq_repeat_bases,
                "repeat_percentage": round(repeat_percentage, 2),
                "n_tandem_repeats": len(tandem_repeats),
                "n_ssrs": len(ssrs),
                "ssr_types": Counter([ssr["type"] for ssr in ssrs])
            })
            
            total_repeat_bases += seq_repeat_bases
        
        # Overall statistics
        repeat_stats["summary"] = {
            "total_genome_length": total_length,
            "total_repeat_bases": total_repeat_bases,
            "overall_repeat_percentage": round((total_repeat_bases / total_length * 100) if total_length > 0 else 0, 2),
            "n_tandem_repeats": len(repeat_stats["tandem_repeats"]),
            "repeat_size_distribution": self._get_repeat_size_distribution(repeat_stats["tandem_repeats"])
        }
        
        return repeat_stats
    
    def _find_tandem_repeats(self, sequence: str, min_length: int, max_length: int) -> List[Dict[str, Any]]:
        """Find tandem repeats in a sequence"""
        repeats = []
        seq_len = len(sequence)
        
        for repeat_len in range(min_length, min(max_length + 1, seq_len // 2)):
            i = 0
            while i < seq_len - repeat_len:
                pattern = sequence[i:i + repeat_len]
                if 'N' in pattern:  # Skip regions with Ns
                    i += 1
                    continue
                
                # Check how many times this pattern repeats
                count = 1
                j = i + repeat_len
                while j + repeat_len <= seq_len and sequence[j:j + repeat_len] == pattern:
                    count += 1
                    j += repeat_len
                
                if count >= 2:  # At least 2 copies
                    repeats.append({
                        "start": i,
                        "end": i + count * repeat_len,
                        "repeat_unit": pattern,
                        "unit_length": repeat_len,
                        "copy_number": count,
                        "total_length": count * repeat_len
                    })
                    i = j  # Skip past this repeat
                else:
                    i += 1
        
        return repeats
    
    def _find_ssrs(self, sequence: str) -> List[Dict[str, Any]]:
        """Find simple sequence repeats (microsatellites)"""
        ssrs = []
        
        # Define SSR patterns (1-6 bp)
        for unit_length in range(1, 7):
            # Skip if sequence is too short
            if len(sequence) < unit_length * 2:
                continue
                
            # Use regex to find SSRs
            if unit_length == 1:  # Mononucleotide
                pattern = r'([ATCG])\1{4,}'  # At least 5 copies
                min_copies = 5
            elif unit_length == 2:  # Dinucleotide
                pattern = r'([ATCG]{2})\1{3,}'  # At least 4 copies
                min_copies = 4
            else:  # Longer motifs
                pattern = rf'([ATCG]{{{unit_length}}})\1{{2,}}'  # At least 3 copies
                min_copies = 3
            
            for match in re.finditer(pattern, sequence):
                unit = match.group(1)
                full_match = match.group(0)
                copy_number = len(full_match) // len(unit)
                
                if copy_number >= min_copies:
                    ssrs.append({
                        "type": f"{unit_length}bp",
                        "unit": unit,
                        "start": match.start(),
                        "end": match.end(),
                        "copy_number": copy_number,
                        "length": len(full_match)
                    })
        
        return ssrs
    
    def _get_repeat_size_distribution(self, repeats: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of repeat sizes"""
        size_dist = {
            "10-20bp": 0,
            "20-50bp": 0,
            "50-100bp": 0,
            "100-500bp": 0,
            ">500bp": 0
        }
        
        for repeat in repeats:
            length = repeat["total_length"]
            if length <= 20:
                size_dist["10-20bp"] += 1
            elif length <= 50:
                size_dist["20-50bp"] += 1
            elif length <= 100:
                size_dist["50-100bp"] += 1
            elif length <= 500:
                size_dist["100-500bp"] += 1
            else:
                size_dist[">500bp"] += 1
        
        return size_dist

    async def gene_prediction_and_coding_stats(self, sequences: List[Dict[str, str]], 
                                             genetic_code: int = 11,
                                             meta_mode: bool = True) -> Dict[str, Any]:
        """Predict genes and calculate coding density using Pyrodigal"""
        # Handle file path input by converting to sequence list
        if isinstance(sequences, str):
            file_result = await self.read_fasta_file(sequences, sample_large_files=True)
            if "error" in file_result:
                return file_result
            sequences = file_result.get("sequences", [])
        
        try:
            # Initialize gene finder
            if meta_mode:
                gene_finder = pyrodigal.GeneFinder(meta=True)
            else:
                # Train on sequences for single genome mode
                training_seq = "".join(s["sequence"] for s in sequences[:10])  # Use first 10 sequences
                gene_finder = pyrodigal.GeneFinder()
                gene_finder.train(training_seq, translation_table=genetic_code)
            
            all_genes = []
            codon_counts = Counter()
            total_coding_length = 0
            total_genome_length = 0
            per_sequence_stats = []
            
            for seq_dict in sequences:
                seq_id = seq_dict.get("id", "unknown")
                sequence = seq_dict["sequence"]
                seq_length = len(sequence)
                total_genome_length += seq_length
                
                # Find genes
                if meta_mode:
                    genes = gene_finder.find_genes(sequence)
                else:
                    genes = gene_finder.find_genes(sequence)
                
                coding_length = 0
                seq_codon_counts = Counter()
                
                for i, gene in enumerate(genes):
                    gene_info = {
                        "sequence_id": seq_id,
                        "gene_id": f"{seq_id}_gene_{i+1}",
                        "start": gene.begin,
                        "end": gene.end,
                        "strand": gene.strand,
                        "length": abs(gene.end - gene.begin),
                        "partial": gene.partial_begin or gene.partial_end,
                        "rbs_motif": gene.rbs_motif if hasattr(gene, 'rbs_motif') else None,
                        "start_codon": gene.start_type if hasattr(gene, 'start_type') else None
                    }
                    
                    # Get gene sequence and count codons
                    if gene.strand == 1:
                        gene_seq = sequence[gene.begin-1:gene.end]
                    else:
                        gene_seq = str(Seq(sequence[gene.begin-1:gene.end]).reverse_complement())
                    
                    # Count codons
                    for j in range(0, len(gene_seq) - 2, 3):
                        codon = gene_seq[j:j+3]
                        if len(codon) == 3 and 'N' not in codon:
                            codon_counts[codon] += 1
                            seq_codon_counts[codon] += 1
                    
                    coding_length += gene_info["length"]
                    all_genes.append(gene_info)
                
                total_coding_length += coding_length
                coding_density = (coding_length / seq_length * 100) if seq_length > 0 else 0
                
                per_sequence_stats.append({
                    "sequence_id": seq_id,
                    "sequence_length": seq_length,
                    "n_genes": len(genes),
                    "coding_length": coding_length,
                    "coding_density": round(coding_density, 2),
                    "mean_gene_length": round(coding_length / len(genes), 2) if genes else 0,
                    "genes_per_kb": round(len(genes) / (seq_length / 1000), 2) if seq_length > 0 else 0
                })
            
            # Calculate codon usage bias
            total_codons = sum(codon_counts.values())
            codon_usage = {}
            
            # Group by amino acid
            codon_to_aa = {
                'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
                'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
                'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
                'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
                'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
                'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
                'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
                'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
                'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
                'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
                'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
                'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
                'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
                'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
                'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
                'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
            }
            
            aa_codon_counts = defaultdict(lambda: defaultdict(int))
            for codon, count in codon_counts.items():
                if codon in codon_to_aa:
                    aa = codon_to_aa[codon]
                    aa_codon_counts[aa][codon] = count
            
            # Calculate RSCU (Relative Synonymous Codon Usage)
            rscu_values = {}
            for aa, codons in aa_codon_counts.items():
                if aa == '*':  # Skip stop codons
                    continue
                total_aa_count = sum(codons.values())
                n_synonymous = len(codons)
                for codon, count in codons.items():
                    expected = total_aa_count / n_synonymous
                    rscu = (count / expected) if expected > 0 else 0
                    rscu_values[codon] = round(rscu, 3)
            
            # Overall statistics
            overall_coding_density = (total_coding_length / total_genome_length * 100) if total_genome_length > 0 else 0
            
            return {
                "summary": {
                    "total_genes": len(all_genes),
                    "total_genome_length": total_genome_length,
                    "total_coding_length": total_coding_length,
                    "overall_coding_density": round(overall_coding_density, 2),
                    "mean_gene_length": round(total_coding_length / len(all_genes), 2) if all_genes else 0,
                    "genetic_code": genetic_code,
                    "meta_mode": meta_mode
                },
                "per_sequence_stats": per_sequence_stats,
                "codon_usage": {
                    "total_codons": total_codons,
                    "codon_counts": dict(codon_counts.most_common(20)),  # Top 20 codons
                    "rscu_values": rscu_values,
                    "gc3_content": self._calculate_gc3(codon_counts)
                },
                "gene_length_distribution": self._get_gene_length_distribution(all_genes)
            }
            
        except Exception as e:
            logger.error(f"Gene prediction failed: {str(e)}")
            return {"error": f"Gene prediction failed: {str(e)}"}
    
    def _calculate_gc3(self, codon_counts: Counter) -> float:
        """Calculate GC content at third codon position"""
        gc3_count = 0
        total_count = 0
        
        for codon, count in codon_counts.items():
            if len(codon) == 3:
                third_pos = codon[2]
                if third_pos in 'GC':
                    gc3_count += count
                if third_pos in 'ATGC':
                    total_count += count
        
        return round((gc3_count / total_count * 100) if total_count > 0 else 0, 2)
    
    def _get_gene_length_distribution(self, genes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of gene lengths"""
        length_dist = {
            "<300bp": 0,
            "300-600bp": 0,
            "600-900bp": 0,
            "900-1500bp": 0,
            "1500-3000bp": 0,
            ">3000bp": 0
        }
        
        for gene in genes:
            length = gene["length"]
            if length < 300:
                length_dist["<300bp"] += 1
            elif length < 600:
                length_dist["300-600bp"] += 1
            elif length < 900:
                length_dist["600-900bp"] += 1
            elif length < 1500:
                length_dist["900-1500bp"] += 1
            elif length < 3000:
                length_dist["1500-3000bp"] += 1
            else:
                length_dist[">3000bp"] += 1
        
        return length_dist
    
    async def kmer_analysis(self, sequences: List[Dict[str, str]], 
                          k_values: List[int] = [3, 4, 5, 6],
                          per_sequence: bool = False) -> Dict[str, Any]:
        """Analyze k-mer frequencies"""
        # Handle file path input by converting to sequence list
        if isinstance(sequences, str):
            file_result = await self.read_fasta_file(sequences, sample_large_files=True)
            if "error" in file_result:
                return file_result
            sequences = file_result.get("sequences", [])
        
        results = {}
        
        for k in k_values:
            if per_sequence:
                # Analyze each sequence separately
                per_seq_kmers = []
                for seq_dict in sequences:
                    seq_id = seq_dict.get("id", "unknown")
                    kmers = self._count_kmers(seq_dict["sequence"], k)
                    total_kmers = sum(kmers.values())
                    
                    # Get top 10 k-mers
                    top_kmers = []
                    for kmer, count in kmers.most_common(10):
                        frequency = (count / total_kmers * 100) if total_kmers > 0 else 0
                        top_kmers.append({
                            "kmer": kmer,
                            "count": count,
                            "frequency": round(frequency, 3)
                        })
                    
                    per_seq_kmers.append({
                        "sequence_id": seq_id,
                        "total_kmers": total_kmers,
                        "unique_kmers": len(kmers),
                        "top_10_kmers": top_kmers
                    })
                
                results[f"{k}-mers"] = {"per_sequence": per_seq_kmers}
            else:
                # Combine all sequences
                all_kmers = Counter()
                for seq_dict in sequences:
                    seq_kmers = self._count_kmers(seq_dict["sequence"], k)
                    all_kmers.update(seq_kmers)
                
                total_kmers = sum(all_kmers.values())
                
                # Get top 10 k-mers
                top_kmers = []
                for kmer, count in all_kmers.most_common(10):
                    frequency = (count / total_kmers * 100) if total_kmers > 0 else 0
                    top_kmers.append({
                        "kmer": kmer,
                        "count": count,
                        "frequency": round(frequency, 3),
                        "expected_frequency": round(100 / (4**k), 3)  # Expected for random sequence
                    })
                
                # Calculate diversity metrics
                shannon_entropy = self._calculate_shannon_entropy(all_kmers)
                
                results[f"{k}-mers"] = {
                    "total_kmers": total_kmers,
                    "unique_kmers": len(all_kmers),
                    "shannon_entropy": round(shannon_entropy, 3),
                    "top_10_kmers": top_kmers
                }
        
        return results
    
    def _count_kmers(self, sequence: str, k: int) -> Counter:
        """Count k-mers in a sequence"""
        sequence = sequence.upper()
        kmers = Counter()
        
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if 'N' not in kmer:  # Skip k-mers with ambiguous bases
                kmers[kmer] += 1
        
        return kmers
    
    def _calculate_shannon_entropy(self, kmer_counts: Counter) -> float:
        """Calculate Shannon entropy for k-mer distribution"""
        total = sum(kmer_counts.values())
        if total == 0:
            return 0
        
        entropy = 0
        for count in kmer_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    async def promoter_identification(self, sequences: List[Dict[str, str]], 
                                    upstream_length: int = 100) -> Dict[str, Any]:
        """Identify potential promoter regions using motif patterns"""
        # Handle file path input by converting to sequence list
        if isinstance(sequences, str):
            file_result = await self.read_fasta_file(sequences, sample_large_files=True)
            if "error" in file_result:
                return file_result
            sequences = file_result.get("sequences", [])
        
        # Now continue with promoter analysis
        # Common prokaryotic promoter motifs
        promoter_motifs = {
            "pribnow_box": ["TATAAT", "TATTAT", "TATAGT", "TATAAT"],
            "-35_box": ["TTGACA", "TTGACT", "TTGACC", "TTGTCA"],
            "shine_dalgarno": ["AGGAGG", "AGGAGA", "GGAGGT", "GAGGAG"]
        }
        
        # Eukaryotic motifs
        euk_motifs = {
            "tata_box": ["TATAAA", "TATAWAW"],  # W = A or T
            "caat_box": ["GGCCAATCT", "CCAAT"],
            "gc_box": ["GGGCGG", "GGCGGG"]
        }
        
        promoter_predictions = []
        motif_counts = defaultdict(int)
        
        for seq_dict in sequences:
            seq_id = seq_dict.get("id", "unknown")
            sequence = seq_dict["sequence"].upper()
            
            # Look for promoter motifs
            seq_promoters = []
            
            # Search for prokaryotic motifs
            for motif_type, motifs in promoter_motifs.items():
                for motif in motifs:
                    # Allow for 1 mismatch
                    for match in self._fuzzy_search(sequence, motif, max_mismatches=1):
                        seq_promoters.append({
                            "type": motif_type,
                            "motif": motif,
                            "position": match,
                            "sequence_context": sequence[max(0, match-10):match+len(motif)+10]
                        })
                        motif_counts[motif_type] += 1
            
            # Store per-sequence results
            if seq_promoters:
                promoter_predictions.append({
                    "sequence_id": seq_id,
                    "n_promoter_motifs": len(seq_promoters),
                    "motifs_found": seq_promoters[:10]  # Limit to first 10
                })
        
        # Calculate promoter density
        total_length = sum(len(s["sequence"]) for s in sequences)
        
        return {
            "summary": {
                "sequences_with_promoters": len(promoter_predictions),
                "total_sequences": len(sequences),
                "total_motifs_found": sum(motif_counts.values()),
                "motif_type_counts": dict(motif_counts),
                "promoter_density_per_kb": round(sum(motif_counts.values()) / (total_length / 1000), 2) if total_length > 0 else 0
            },
            "per_sequence_predictions": promoter_predictions
        }
    
    def _fuzzy_search(self, sequence: str, pattern: str, max_mismatches: int = 1) -> List[int]:
        """Find pattern in sequence allowing for mismatches"""
        positions = []
        pattern_len = len(pattern)
        
        for i in range(len(sequence) - pattern_len + 1):
            substring = sequence[i:i + pattern_len]
            mismatches = sum(1 for a, b in zip(substring, pattern) if a != b)
            if mismatches <= max_mismatches:
                positions.append(i)
        
        return positions

    async def giant_virus_promoter_search(self, sequences: List[Dict[str, str]], 
                                   upstream_length: int = 150) -> Dict[str, Any]:
        """Search for giant virus-specific promoter motifs based on NCLDV research"""
        # Handle file path input by converting to sequence list
        if isinstance(sequences, str):
            file_result = await self.read_fasta_file(sequences, sample_large_files=True)
            if "error" in file_result:
                return file_result
            sequences = file_result.get("sequences", [])
        
        # Continue with giant virus promoter search
        
        # Giant virus promoter motifs from the literature
        giant_virus_motifs = {
            # Mimivirus motifs
            "mimivirus_early": {
                "motifs": ["AAAATTGA", "AAAATTGG", "GAAATTGA"],
                "description": "Early gene promoter in mimiviruses (45% of genes)"
            },
            "mimivirus_late": {
                # Two 10-nt segments separated by 4-nt spacer
                "pattern": r"[AT]{10}[ATCG]{4}[AT]{10}",
                "description": "Late gene promoter in mimiviruses"
            },
            
            # MEGA-box (proposed ancestral motif)
            "mega_box": {
                "motifs": ["TATATAAAATTGA", "TATATAAAATTGG"],
                "description": "Proposed ancestral NCLDV promoter motif"
            },
            
            # Asfarviridae family motifs (faustovirus, kaumoebavirus, asfarvirus)
            "asfar_tattt": {
                "motifs": ["TATTT", "TTTTT", "ATTTT"],
                "description": "A/T-rich promoter motif in Asfarviridae"
            },
            "asfar_tatata": {
                "motifs": ["TATATA", "ATATAT", "TATATT"],
                "description": "TATATA box in Asfarviridae"
            },
            
            # Marseillevirus motifs
            "marseille_motif": {
                "motifs": ["AAATATTT", "AAATATTT", "TAATATTT"],
                "description": "Abundant motif in Marseillevirus intergenic regions"
            },
            
            # CroV (Cafeteria roenbergensis virus) late promoter
            "crov_late": {
                "pattern": r"[AT]{3,6}TCTA[AT]{3,6}",
                "description": "CroV late gene promoter with TCTA core"
            },
            
            # Phycodnavirus-like early motifs
            "phyco_early": {
                "motifs": ["AAAAATTGA", "AAAATTGAA", "AAAATTGAT"],
                "description": "Early promoter in phycodnaviruses and related giant viruses"
            }
        }
        
        all_motif_hits = []
        motif_summary = defaultdict(int)
        sequences_with_motifs = 0
        
        for seq_dict in sequences:
            seq_id = seq_dict.get("id", "unknown")
            sequence = seq_dict["sequence"].upper()
            seq_motifs = []
            
            # Search for simple motifs
            for motif_type, motif_info in giant_virus_motifs.items():
                if "motifs" in motif_info:
                    for motif in motif_info["motifs"]:
                        # Search allowing 1 mismatch
                        positions = self._fuzzy_search(sequence, motif, max_mismatches=1)
                        for pos in positions:
                            # Check if in upstream region (assuming genes throughout sequence)
                            context_start = max(0, pos - 20)
                            context_end = min(len(sequence), pos + len(motif) + 20)
                            
                            seq_motifs.append({
                                "type": motif_type,
                                "motif": motif,
                                "position": pos,
                                "upstream_distance": "variable",  # Would need gene positions
                                "sequence_context": sequence[context_start:context_end],
                                "description": motif_info["description"]
                            })
                            motif_summary[motif_type] += 1
                
                # Search for regex patterns
                elif "pattern" in motif_info:
                    import re
                    pattern = motif_info["pattern"]
                    for match in re.finditer(pattern, sequence):
                        seq_motifs.append({
                            "type": motif_type,
                            "motif": match.group(),
                            "position": match.start(),
                            "upstream_distance": "variable",
                            "sequence_context": sequence[max(0, match.start()-20):match.end()+20],
                            "description": motif_info["description"]
                        })
                        motif_summary[motif_type] += 1
            
            if seq_motifs:
                sequences_with_motifs += 1
                all_motif_hits.append({
                    "sequence_id": seq_id,
                    "n_motifs": len(seq_motifs),
                    "motif_details": seq_motifs[:20]  # Limit output
                })
        
        # Calculate enrichment for A/T-rich regions
        at_rich_regions = self._find_at_rich_regions(sequences)
        
        return {
            "summary": {
                "sequences_analyzed": len(sequences),
                "sequences_with_giant_virus_motifs": sequences_with_motifs,
                "total_motifs_found": sum(motif_summary.values()),
                "motif_type_distribution": dict(motif_summary),
                "motif_density_per_kb": round(sum(motif_summary.values()) / (sum(len(s["sequence"]) for s in sequences) / 1000), 2)
            },
            "at_rich_analysis": at_rich_regions,
            "detailed_results": all_motif_hits[:100]  # Limit to first 100 sequences
        }
    
    def _find_at_rich_regions(self, sequences: List[Dict[str, str]], 
                             window: int = 100, 
                             min_at_content: float = 70) -> Dict[str, Any]:
        """Find A/T-rich regions that are characteristic of giant virus promoters"""
        at_rich_count = 0
        total_windows = 0
        
        for seq_dict in sequences:
            sequence = seq_dict["sequence"].upper()
            for i in range(0, len(sequence) - window + 1, window // 2):  # 50% overlap
                window_seq = sequence[i:i + window]
                at_count = window_seq.count('A') + window_seq.count('T')
                at_content = (at_count / window * 100) if window > 0 else 0
                
                total_windows += 1
                if at_content >= min_at_content:
                    at_rich_count += 1
        
        return {
            "at_rich_windows": at_rich_count,
            "total_windows": total_windows,
            "percentage_at_rich": round((at_rich_count / total_windows * 100) if total_windows > 0 else 0, 2),
            "criteria": f">={min_at_content}% AT content in {window}bp windows"
        }
    
    async def gc_skew_analysis(self, sequences: List[Dict[str, str]], 
                              window_size: int = 10000, 
                              step_size: int = 5000) -> Dict[str, Any]:
        """
        Calculate GC skew to identify replication origins and strand bias
        GC skew = (G - C) / (G + C)
        """
        # Handle file path input by converting to sequence list
        if isinstance(sequences, str):
            file_result = await self.read_fasta_file(sequences, sample_large_files=True)
            if "error" in file_result:
                return file_result
            sequences = file_result.get("sequences", [])
        
        results = {
            "per_sequence_analysis": [],
            "summary": {}
        }
        
        for seq_dict in sequences:
            seq_id = seq_dict.get("id", "unknown")
            sequence = seq_dict["sequence"].upper()
            seq_length = len(sequence)
            
            # Skip if sequence is too short
            if seq_length < window_size:
                results["per_sequence_analysis"].append({
                    "sequence_id": seq_id,
                    "error": f"Sequence too short ({seq_length}bp) for window size ({window_size}bp)"
                })
                continue
            
            gc_skew_values = []
            cumulative_skew = []
            positions = []
            cumulative = 0
            
            # Calculate GC skew using sliding window
            for i in range(0, seq_length - window_size + 1, step_size):
                window = sequence[i:i + window_size]
                g_count = window.count('G')
                c_count = window.count('C')
                
                # Calculate GC skew
                if g_count + c_count > 0:
                    gc_skew = (g_count - c_count) / (g_count + c_count)
                else:
                    gc_skew = 0
                
                gc_skew_values.append(gc_skew)
                cumulative += gc_skew
                cumulative_skew.append(cumulative)
                positions.append(i + window_size // 2)  # Middle of window
            
            # Find potential replication origins (where cumulative skew changes sign)
            origin_candidates = []
            for i in range(1, len(cumulative_skew)):
                if (cumulative_skew[i-1] < 0 and cumulative_skew[i] > 0) or \
                   (cumulative_skew[i-1] > 0 and cumulative_skew[i] < 0):
                    origin_candidates.append({
                        "position": positions[i],
                        "skew_change": cumulative_skew[i] - cumulative_skew[i-1]
                    })
            
            # Find regions of maximum and minimum skew
            if gc_skew_values:
                max_skew_idx = np.argmax(gc_skew_values)
                min_skew_idx = np.argmin(gc_skew_values)
                
                seq_result = {
                    "sequence_id": seq_id,
                    "sequence_length": seq_length,
                    "window_size": window_size,
                    "step_size": step_size,
                    "n_windows": len(gc_skew_values),
                    "mean_gc_skew": round(np.mean(gc_skew_values), 4),
                    "std_gc_skew": round(np.std(gc_skew_values), 4),
                    "max_skew": {
                        "value": round(gc_skew_values[max_skew_idx], 4),
                        "position": positions[max_skew_idx]
                    },
                    "min_skew": {
                        "value": round(gc_skew_values[min_skew_idx], 4),
                        "position": positions[min_skew_idx]
                    },
                    "potential_origins": origin_candidates[:5],  # Top 5 candidates
                    "n_origin_candidates": len(origin_candidates)
                }
            else:
                seq_result = {
                    "sequence_id": seq_id,
                    "error": "Unable to calculate GC skew"
                }
            
            results["per_sequence_analysis"].append(seq_result)
        
        # Overall summary
        all_mean_skews = [r["mean_gc_skew"] for r in results["per_sequence_analysis"] 
                          if "mean_gc_skew" in r]
        
        if all_mean_skews:
            results["summary"] = {
                "n_sequences_analyzed": len(sequences),
                "overall_mean_skew": round(np.mean(all_mean_skews), 4),
                "overall_std_skew": round(np.std(all_mean_skews), 4),
                "sequences_with_origin_candidates": sum(1 for r in results["per_sequence_analysis"] 
                                                       if "n_origin_candidates" in r and r["n_origin_candidates"] > 0)
            }
        
        return results
    
    async def cpg_island_detection(self, sequences: List[Dict[str, str]], 
                                  min_length: int = 200,
                                  gc_threshold: float = 50.0,
                                  oe_ratio_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Detect CpG islands in sequences
        Criteria:
        - Length >= 200 bp
        - GC content >= 50%
        - Observed/Expected CpG ratio >= 0.6
        """
        # Handle file path input by converting to sequence list
        if isinstance(sequences, str):
            file_result = await self.read_fasta_file(sequences, sample_large_files=True)
            if "error" in file_result:
                return file_result
            sequences = file_result.get("sequences", [])
        
        results = {
            "per_sequence_analysis": [],
            "summary": {}
        }
        
        total_islands = 0
        total_island_length = 0
        
        for seq_dict in sequences:
            seq_id = seq_dict.get("id", "unknown")
            sequence = seq_dict["sequence"].upper()
            seq_length = len(sequence)
            
            cpg_islands = []
            
            # Scan sequence with sliding window
            for window_size in [200, 300, 500, 1000]:  # Multiple window sizes
                for i in range(0, seq_length - window_size + 1, 100):  # Step by 100bp
                    window = sequence[i:i + window_size]
                    
                    # Calculate GC content
                    gc_count = window.count('G') + window.count('C')
                    gc_content = (gc_count / window_size * 100) if window_size > 0 else 0
                    
                    if gc_content < gc_threshold:
                        continue
                    
                    # Calculate observed CpG dinucleotides
                    cpg_observed = window.count('CG')
                    
                    # Calculate expected CpG
                    c_count = window.count('C')
                    g_count = window.count('G')
                    cpg_expected = (c_count * g_count) / window_size if window_size > 0 else 0
                    
                    # Calculate O/E ratio
                    oe_ratio = cpg_observed / cpg_expected if cpg_expected > 0 else 0
                    
                    if oe_ratio >= oe_ratio_threshold:
                        # Check if this island overlaps with existing ones
                        is_new = True
                        for island in cpg_islands:
                            if (i < island["end"] and i + window_size > island["start"]):
                                # Extend existing island
                                island["start"] = min(island["start"], i)
                                island["end"] = max(island["end"], i + window_size)
                                island["length"] = island["end"] - island["start"]
                                is_new = False
                                break
                        
                        if is_new:
                            cpg_islands.append({
                                "start": i,
                                "end": i + window_size,
                                "length": window_size,
                                "gc_content": round(gc_content, 2),
                                "oe_ratio": round(oe_ratio, 3),
                                "cpg_count": cpg_observed
                            })
            
            # Merge overlapping islands and recalculate statistics
            merged_islands = []
            cpg_islands.sort(key=lambda x: x["start"])
            
            for island in cpg_islands:
                if not merged_islands or island["start"] > merged_islands[-1]["end"]:
                    # Recalculate statistics for the island
                    island_seq = sequence[island["start"]:island["end"]]
                    gc_count = island_seq.count('G') + island_seq.count('C')
                    island["gc_content"] = round((gc_count / len(island_seq) * 100) if island_seq else 0, 2)
                    island["cpg_count"] = island_seq.count('CG')
                    
                    # Recalculate O/E ratio
                    c_count = island_seq.count('C')
                    g_count = island_seq.count('G')
                    cpg_expected = (c_count * g_count) / len(island_seq) if island_seq else 0
                    island["oe_ratio"] = round(island["cpg_count"] / cpg_expected if cpg_expected > 0 else 0, 3)
                    
                    merged_islands.append(island)
                else:
                    # Merge with previous island
                    prev = merged_islands[-1]
                    prev["end"] = max(prev["end"], island["end"])
                    prev["length"] = prev["end"] - prev["start"]
                    
                    # Recalculate statistics
                    island_seq = sequence[prev["start"]:prev["end"]]
                    gc_count = island_seq.count('G') + island_seq.count('C')
                    prev["gc_content"] = round((gc_count / len(island_seq) * 100) if island_seq else 0, 2)
                    prev["cpg_count"] = island_seq.count('CG')
                    
                    c_count = island_seq.count('C')
                    g_count = island_seq.count('G')
                    cpg_expected = (c_count * g_count) / len(island_seq) if island_seq else 0
                    prev["oe_ratio"] = round(prev["cpg_count"] / cpg_expected if cpg_expected > 0 else 0, 3)
            
            # Calculate coverage
            island_coverage = sum(island["length"] for island in merged_islands)
            coverage_percentage = (island_coverage / seq_length * 100) if seq_length > 0 else 0
            
            total_islands += len(merged_islands)
            total_island_length += island_coverage
            
            results["per_sequence_analysis"].append({
                "sequence_id": seq_id,
                "sequence_length": seq_length,
                "n_cpg_islands": len(merged_islands),
                "total_island_length": island_coverage,
                "coverage_percentage": round(coverage_percentage, 2),
                "islands": merged_islands[:10],  # First 10 islands
                "mean_island_length": round(island_coverage / len(merged_islands), 2) if merged_islands else 0,
                "mean_gc_in_islands": round(np.mean([i["gc_content"] for i in merged_islands]), 2) if merged_islands else 0,
                "mean_oe_ratio": round(np.mean([i["oe_ratio"] for i in merged_islands]), 3) if merged_islands else 0
            })
        
        # Summary statistics
        total_genome_length = sum(len(s["sequence"]) for s in sequences)
        results["summary"] = {
            "total_sequences": len(sequences),
            "sequences_with_cpg_islands": sum(1 for r in results["per_sequence_analysis"] if r["n_cpg_islands"] > 0),
            "total_cpg_islands": total_islands,
            "total_island_length": total_island_length,
            "genome_coverage": round((total_island_length / total_genome_length * 100) if total_genome_length > 0 else 0, 2),
            "criteria": {
                "min_length": min_length,
                "gc_threshold": gc_threshold,
                "oe_ratio_threshold": oe_ratio_threshold
            }
        }
        
        return results

    async def analyze_fasta_file(self, file_path: str, sequence_type: str = "dna") -> Dict[str, Any]:
        """Comprehensive analysis of sequences in a FASTA file"""
        try:
            # Read sequences
            sequences = []
            for record in SeqIO.parse(file_path, "fasta"):
                sequences.append({
                    "id": record.id,
                    "sequence": str(record.seq),
                    "description": record.description
                })
            
            if not sequences:
                return {"error": "No sequences found in file"}
            
            # Validate first sequence
            validation = self.validate_nucleic_acid(sequences[0]["sequence"])
            if not validation["valid"]:
                return {"error": f"Invalid nucleic acid sequence: {validation['error']}"}
            
            sequence_type = validation["sequence_type"]
            
            # Run all analyses
            results = {
                "file_info": {
                    "file_path": file_path,
                    "sequence_type": sequence_type,
                    "n_sequences": len(sequences)
                }
            }
            
            # Assembly statistics
            logger.info("Calculating assembly statistics...")
            results["assembly_stats"] = await self.assembly_stats(sequences)
            
            # K-mer analysis
            logger.info("Performing k-mer analysis...")
            results["kmer_analysis"] = await self.kmer_analysis(sequences, k_values=[4, 5, 6])
            
            # Repeat detection
            logger.info("Detecting repeats...")
            results["repeat_analysis"] = await self.repeat_detection(sequences)
            
            # Gene prediction and coding stats (only for DNA)
            if sequence_type == "dna":
                logger.info("Predicting genes and calculating coding statistics...")
                results["gene_analysis"] = await self.gene_prediction_and_coding_stats(sequences)
                
                logger.info("Identifying promoters...")
                results["promoter_analysis"] = await self.promoter_identification(sequences)
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}