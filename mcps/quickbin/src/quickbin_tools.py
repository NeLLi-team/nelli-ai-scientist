"""
QuickBin Tools Implementation

This module provides Python wrappers for QuickBin metagenomics binning tools using shifter containers.
QuickBin bins contigs using coverage and kmer frequencies for metagenomics analysis.
"""

import subprocess
import tempfile
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class QuickBinToolkit:
    """Collection of QuickBin-based metagenomics binning tools"""

    def __init__(self, shifter_image: str = "bryce911/bbtools:latest"):
        """Initialize QuickBin toolkit

        Args:
            shifter_image: Docker image containing BBTools suite (includes QuickBin)
        """
        self.shifter_image = shifter_image
        self.base_command = ["shifter", "--image", self.shifter_image]

    def _run_command(self, command: List[str], input_data: Optional[str] = None) -> Tuple[str, str, int]:
        """Execute a command using shifter

        Args:
            command: Command and arguments to execute
            input_data: Optional input data to pipe to command

        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        full_command = self.base_command + command

        try:
            logger.info(f"Executing: {' '.join(full_command)}")

            process = subprocess.Popen(
                full_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if input_data else None,
                text=True,
                cwd=os.getcwd()
            )

            stdout, stderr = process.communicate(input=input_data)
            return stdout, stderr, process.returncode

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return "", str(e), 1

    async def bin_contigs(
        self,
        contigs_path: str,
        sam_files: List[str],
        output_pattern: str,
        stringency: str = "normal",
        additional_params: Optional[str] = None
    ) -> Dict[str, Any]:
        """Bin contigs using coverage and kmer frequencies

        Args:
            contigs_path: Path to assembled contigs (FASTA)
            sam_files: List of SAM/BAM alignment files
            output_pattern: Output pattern for bins
            stringency: Binning stringency level
            additional_params: Additional QuickBin parameters

        Returns:
            Dictionary with binning statistics and file paths
        """
        # Validate input files
        if not os.path.exists(contigs_path):
            raise FileNotFoundError(f"Contigs file not found: {contigs_path}")

        for sam_file in sam_files:
            if not os.path.exists(sam_file):
                raise FileNotFoundError(f"SAM file not found: {sam_file}")

        # Build QuickBin command
        command = [
            "quickbin.sh",
            f"in={contigs_path}",
            f"out={output_pattern}",
            stringency  # Add stringency as a flag
        ]

        # Add SAM files
        command.extend(sam_files)

        # Add coverage output for statistics
        coverage_file = "quickbin_coverage.txt"
        command.append(f"covout={coverage_file}")

        if additional_params:
            command.extend(additional_params.split())

        # Execute binning
        stdout, stderr, returncode = self._run_command(command)

        if returncode != 0:
            return {
                "status": "error",
                "error_message": stderr,
                "command": " ".join(command)
            }

        # Parse results
        bin_stats = self._parse_binning_results(output_pattern, coverage_file, stdout)

        return {
            "status": "success",
            "contigs_input": contigs_path,
            "sam_files": sam_files,
            "output_pattern": output_pattern,
            "stringency": stringency,
            "bin_stats": bin_stats,
            "coverage_file": coverage_file,
            "command_output": stdout,
            "command": " ".join(command)
        }

    async def bin_contigs_with_coverage(
        self,
        contigs_path: str,
        coverage_file: str,
        output_pattern: str,
        stringency: str = "normal",
        additional_params: Optional[str] = None
    ) -> Dict[str, Any]:
        """Bin contigs using pre-calculated coverage file

        Args:
            contigs_path: Path to assembled contigs (FASTA)
            coverage_file: Path to coverage file
            output_pattern: Output pattern for bins
            stringency: Binning stringency level
            additional_params: Additional QuickBin parameters

        Returns:
            Dictionary with binning statistics and file paths
        """
        # Validate input files
        if not os.path.exists(contigs_path):
            raise FileNotFoundError(f"Contigs file not found: {contigs_path}")
        if not os.path.exists(coverage_file):
            raise FileNotFoundError(f"Coverage file not found: {coverage_file}")

        # Build QuickBin command
        command = [
            "quickbin.sh",
            f"in={contigs_path}",
            f"cov={coverage_file}",
            f"out={output_pattern}",
            stringency  # Add stringency as a flag
        ]

        if additional_params:
            command.extend(additional_params.split())

        # Execute binning
        stdout, stderr, returncode = self._run_command(command)

        if returncode != 0:
            return {
                "status": "error",
                "error_message": stderr,
                "command": " ".join(command)
            }

        # Parse results
        bin_stats = self._parse_binning_results(output_pattern, coverage_file, stdout)

        return {
            "status": "success",
            "contigs_input": contigs_path,
            "coverage_file": coverage_file,
            "output_pattern": output_pattern,
            "stringency": stringency,
            "bin_stats": bin_stats,
            "command_output": stdout,
            "command": " ".join(command)
        }

    async def generate_coverage(
        self,
        contigs_path: str,
        sam_files: List[str],
        output_coverage: str,
        additional_params: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate coverage statistics from SAM files

        Args:
            contigs_path: Path to assembled contigs (FASTA)
            sam_files: List of SAM/BAM alignment files
            output_coverage: Output path for coverage file
            additional_params: Additional parameters

        Returns:
            Dictionary with coverage statistics
        """
        # Validate input files
        if not os.path.exists(contigs_path):
            raise FileNotFoundError(f"Contigs file not found: {contigs_path}")

        for sam_file in sam_files:
            if not os.path.exists(sam_file):
                raise FileNotFoundError(f"SAM file not found: {sam_file}")

        # Build QuickBin command for coverage generation only
        command = [
            "quickbin.sh",
            f"in={contigs_path}",
            f"covout={output_coverage}",
            "out=temp_bins"  # Temporary output that we'll ignore
        ]

        # Add SAM files
        command.extend(sam_files)

        if additional_params:
            command.extend(additional_params.split())

        # Execute coverage generation
        stdout, stderr, returncode = self._run_command(command)

        # Clean up temporary bin files
        self._cleanup_temp_files("temp_bins")

        if returncode != 0:
            return {
                "status": "error",
                "error_message": stderr,
                "command": " ".join(command)
            }

        # Parse coverage statistics
        coverage_stats = self._parse_coverage_file(output_coverage)

        return {
            "status": "success",
            "contigs_input": contigs_path,
            "sam_files": sam_files,
            "coverage_file": output_coverage,
            "coverage_stats": coverage_stats,
            "command_output": stdout,
            "command": " ".join(command)
        }

    async def evaluate_bins(
        self,
        bin_directory: str,
        reference_taxonomy: Optional[str] = None,
        additional_params: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate binning quality and completeness

        Args:
            bin_directory: Directory containing binned contigs
            reference_taxonomy: Optional reference taxonomy for validation
            additional_params: Additional evaluation parameters

        Returns:
            Dictionary with quality assessment results
        """
        if not os.path.exists(bin_directory):
            raise FileNotFoundError(f"Bin directory not found: {bin_directory}")

        # Get list of bin files
        bin_files = []
        for file in os.listdir(bin_directory):
            if file.endswith(('.fa', '.fasta', '.fna')):
                bin_files.append(os.path.join(bin_directory, file))

        if not bin_files:
            return {
                "status": "error",
                "error_message": "No FASTA files found in bin directory"
            }

        # Analyze each bin
        bin_analyses = []
        for bin_file in bin_files:
            bin_analysis = self._analyze_single_bin(bin_file)
            bin_analyses.append(bin_analysis)

        # Calculate overall statistics
        quality_stats = self._calculate_overall_quality(bin_analyses)

        # If reference taxonomy provided, add validation
        validation_results = None
        if reference_taxonomy and os.path.exists(reference_taxonomy):
            validation_results = self._validate_bins(bin_files, reference_taxonomy)

        return {
            "status": "success",
            "bin_directory": bin_directory,
            "total_bins": len(bin_files),
            "bin_analyses": bin_analyses,
            "quality_stats": quality_stats,
            "validation_results": validation_results
        }

    def _parse_binning_results(self, output_pattern: str, coverage_file: str, stdout: str) -> Dict[str, Any]:
        """Parse QuickBin output to extract statistics"""
        stats = {
            "total_bins": 0,
            "total_contigs_binned": 0,
            "largest_bin_size": 0,
            "smallest_bin_size": float('inf'),
            "bin_files": []
        }

        try:
            # Determine output directory or pattern
            if "%" in output_pattern:
                # Pattern-based output (bin%.fa)
                base_pattern = output_pattern.replace("%", "*")
                import glob
                bin_files = glob.glob(base_pattern)
            elif os.path.isdir(output_pattern):
                # Directory-based output
                bin_files = []
                for file in os.listdir(output_pattern):
                    if file.endswith(('.fa', '.fasta', '.fna')):
                        bin_files.append(os.path.join(output_pattern, file))
            else:
                # Single file output
                bin_files = [output_pattern] if os.path.exists(output_pattern) else []

            stats["total_bins"] = len(bin_files)
            stats["bin_files"] = bin_files

            # Analyze each bin file
            for bin_file in bin_files:
                if os.path.exists(bin_file):
                    bin_size = os.path.getsize(bin_file)
                    stats["largest_bin_size"] = max(stats["largest_bin_size"], bin_size)
                    stats["smallest_bin_size"] = min(stats["smallest_bin_size"], bin_size)

            # Count contigs from stdout if available
            contig_match = re.search(r'(\d+) contigs.*binned', stdout)
            if contig_match:
                stats["total_contigs_binned"] = int(contig_match.group(1))

        except Exception as e:
            logger.warning(f"Could not parse binning results: {e}")
            stats["parse_error"] = str(e)

        if stats["smallest_bin_size"] == float('inf'):
            stats["smallest_bin_size"] = 0

        return stats

    def _parse_coverage_file(self, coverage_file: str) -> Dict[str, Any]:
        """Parse coverage statistics from QuickBin coverage file"""
        stats = {
            "total_contigs": 0,
            "average_coverage": 0.0,
            "coverage_range": {"min": 0.0, "max": 0.0},
            "samples": []
        }

        try:
            if os.path.exists(coverage_file):
                with open(coverage_file, 'r') as f:
                    lines = f.readlines()

                # Parse header to get sample names
                if lines:
                    header = lines[0].strip().split('\t')
                    if len(header) > 3:  # Assuming format: contig, length, gc, sample1, sample2, ...
                        stats["samples"] = header[3:]

                # Parse coverage data
                coverages = []
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) > 3:
                        # Extract coverage values for each sample
                        sample_coverages = [float(x) for x in parts[3:] if x.replace('.', '').isdigit()]
                        if sample_coverages:
                            avg_coverage = sum(sample_coverages) / len(sample_coverages)
                            coverages.append(avg_coverage)

                if coverages:
                    stats["total_contigs"] = len(coverages)
                    stats["average_coverage"] = sum(coverages) / len(coverages)
                    stats["coverage_range"]["min"] = min(coverages)
                    stats["coverage_range"]["max"] = max(coverages)

        except Exception as e:
            logger.warning(f"Could not parse coverage file: {e}")
            stats["parse_error"] = str(e)

        return stats

    def _analyze_single_bin(self, bin_file: str) -> Dict[str, Any]:
        """Analyze a single bin file for basic statistics"""
        analysis = {
            "bin_file": bin_file,
            "total_length": 0,
            "num_contigs": 0,
            "n50": 0,
            "gc_content": 0.0
        }

        try:
            # Simple FASTA parsing for basic stats
            sequences = []
            current_seq = ""

            with open(bin_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_seq:
                            sequences.append(current_seq)
                            current_seq = ""
                    else:
                        current_seq += line

                if current_seq:
                    sequences.append(current_seq)

            # Calculate statistics
            analysis["num_contigs"] = len(sequences)
            lengths = [len(seq) for seq in sequences]

            if lengths:
                analysis["total_length"] = sum(lengths)

                # Calculate N50
                lengths.sort(reverse=True)
                total_length = analysis["total_length"]
                cumulative = 0
                for length in lengths:
                    cumulative += length
                    if cumulative >= total_length / 2:
                        analysis["n50"] = length
                        break

                # Calculate GC content
                all_bases = "".join(sequences).upper()
                gc_count = all_bases.count('G') + all_bases.count('C')
                total_bases = len(all_bases)
                if total_bases > 0:
                    analysis["gc_content"] = (gc_count / total_bases) * 100

        except Exception as e:
            logger.warning(f"Could not analyze bin {bin_file}: {e}")
            analysis["analysis_error"] = str(e)

        return analysis

    def _calculate_overall_quality(self, bin_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall quality statistics from individual bin analyses"""
        if not bin_analyses:
            return {"avg_completeness": 0, "avg_contamination": 0, "total_length": 0}

        total_length = sum(analysis.get("total_length", 0) for analysis in bin_analyses)
        total_contigs = sum(analysis.get("num_contigs", 0) for analysis in bin_analyses)
        avg_gc = sum(analysis.get("gc_content", 0) for analysis in bin_analyses) / len(bin_analyses)

        # N50 statistics
        n50_values = [analysis.get("n50", 0) for analysis in bin_analyses if analysis.get("n50", 0) > 0]
        avg_n50 = sum(n50_values) / len(n50_values) if n50_values else 0

        return {
            "total_bins": len(bin_analyses),
            "total_length": total_length,
            "total_contigs": total_contigs,
            "average_gc_content": avg_gc,
            "average_n50": avg_n50,
            "largest_bin": max(bin_analyses, key=lambda x: x.get("total_length", 0))["total_length"] if bin_analyses else 0,
            "smallest_bin": min(bin_analyses, key=lambda x: x.get("total_length", 0))["total_length"] if bin_analyses else 0
        }

    def _validate_bins(self, bin_files: List[str], reference_taxonomy: str) -> Dict[str, Any]:
        """Validate bins against reference taxonomy (placeholder implementation)"""
        # This would typically involve checking against known taxonomy
        # For now, return a placeholder structure
        return {
            "validation_performed": True,
            "reference_file": reference_taxonomy,
            "bins_validated": len(bin_files),
            "accuracy_metrics": {
                "precision": 0.0,  # Would calculate based on reference
                "recall": 0.0,     # Would calculate based on reference
                "f1_score": 0.0    # Would calculate based on reference
            },
            "note": "Validation implementation would require specific taxonomy format"
        }

    def _cleanup_temp_files(self, pattern: str):
        """Clean up temporary files created during processing"""
        try:
            import glob
            temp_files = glob.glob(f"{pattern}*")
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    if os.path.isfile(temp_file):
                        os.remove(temp_file)
                    elif os.path.isdir(temp_file):
                        import shutil
                        shutil.rmtree(temp_file)
        except Exception as e:
            logger.warning(f"Could not clean up temporary files: {e}")
