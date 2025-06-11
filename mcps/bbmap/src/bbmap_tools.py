"""
BBMap Tools Implementation

This module provides Python wrappers for BBMap bioinformatics tools using shifter containers.
BBMap is a suite of tools for read mapping, quality assessment, and sequence analysis.
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


class BBMapToolkit:
    """Collection of BBMap-based bioinformatics tools"""

    def __init__(self, shifter_image: str = "bryce911/bbtools:39.27"):
        """Initialize BBMap toolkit

        Args:
            shifter_image: Docker image containing BBTools suite
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

    async def map_reads(
        self,
        reference_path: str,
        reads_path: str,
        output_sam: str,
        additional_params: Optional[str] = None
    ) -> Dict[str, Any]:
        """Map reads to reference genome using BBMap

        Args:
            reference_path: Path to reference genome (FASTA)
            reads_path: Path to reads file (FASTQ)
            output_sam: Output SAM file path
            additional_params: Additional BBMap parameters

        Returns:
            Dictionary with mapping statistics and file paths
        """
        # Validate input files
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference file not found: {reference_path}")
        if not os.path.exists(reads_path):
            raise FileNotFoundError(f"Reads file not found: {reads_path}")

        # Build BBMap command
        command = [
            "bbmap.sh",
            f"ref={reference_path}",
            f"in={reads_path}",
            f"out={output_sam}",
            "overwrite=true"
        ]

        if additional_params:
            command.extend(additional_params.split())

        # Execute mapping
        stdout, stderr, returncode = self._run_command(command)

        if returncode != 0:
            raise RuntimeError(f"BBMap failed: {stderr}")

        # Parse mapping statistics from stdout/stderr
        stats = self._parse_stdout_stats(stdout, stderr)

        result = {
            "status": "success",
            "output_sam": output_sam,
            "mapping_stats": stats,
            "command_used": " ".join(self.base_command + command),
            "stdout": stdout,
            "stderr": stderr if stderr else None
        }

        return result

    async def quality_stats(
        self,
        fastq_path: str,
        output_prefix: str = "quality_stats"
    ) -> Dict[str, Any]:
        """Generate quality statistics for FASTQ files

        Args:
            fastq_path: Path to FASTQ file
            output_prefix: Prefix for output files

        Returns:
            Dictionary with quality statistics
        """
        if not os.path.exists(fastq_path):
            raise FileNotFoundError(f"FASTQ file not found: {fastq_path}")

        stats_file = f"{output_prefix}.txt"
        hist_file = f"{output_prefix}_hist.txt"

        command = [
            "readlength.sh",
            f"in={fastq_path}",
            f"out={stats_file}",
            f"hist={hist_file}"
        ]

        stdout, stderr, returncode = self._run_command(command)

        if returncode != 0:
            raise RuntimeError(f"Quality stats failed: {stderr}")

        # Parse quality statistics
        stats = self._parse_quality_stats(stats_file, hist_file)

        result = {
            "status": "success",
            "fastq_file": fastq_path,
            "quality_stats": stats,
            "output_files": {
                "stats": stats_file,
                "histogram": hist_file
            },
            "command_used": " ".join(self.base_command + command)
        }

        return result

    async def coverage_analysis(
        self,
        sam_path: str,
        reference_path: str,
        output_prefix: str = "coverage"
    ) -> Dict[str, Any]:
        """Analyze coverage from SAM/BAM file

        Args:
            sam_path: Path to SAM/BAM file
            reference_path: Path to reference genome
            output_prefix: Prefix for output files

        Returns:
            Dictionary with coverage statistics
        """
        if not os.path.exists(sam_path):
            raise FileNotFoundError(f"SAM file not found: {sam_path}")
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference file not found: {reference_path}")

        coverage_file = f"{output_prefix}_coverage.txt"
        stats_file = f"{output_prefix}_stats.txt"

        command = [
            "pileup.sh",
            f"in={sam_path}",
            f"ref={reference_path}",
            f"out={coverage_file}",
            f"stats={stats_file}",
            "hist=coverage_histogram.txt"
        ]

        stdout, stderr, returncode = self._run_command(command)

        if returncode != 0:
            raise RuntimeError(f"Coverage analysis failed: {stderr}")

        # Parse coverage statistics
        stats = self._parse_coverage_stats(stats_file)

        result = {
            "status": "success",
            "sam_file": sam_path,
            "coverage_stats": stats,
            "output_files": {
                "coverage": coverage_file,
                "stats": stats_file,
                "histogram": "coverage_histogram.txt"
            },
            "command_used": " ".join(self.base_command + command)
        }

        return result

    async def filter_reads(
        self,
        input_fastq: str,
        output_fastq: str,
        min_length: int = 50,
        min_quality: float = 20.0,
        additional_params: Optional[str] = None
    ) -> Dict[str, Any]:
        """Filter reads based on quality and length

        Args:
            input_fastq: Input FASTQ file
            output_fastq: Output filtered FASTQ file
            min_length: Minimum read length
            min_quality: Minimum average quality score
            additional_params: Additional filtering parameters

        Returns:
            Dictionary with filtering statistics
        """
        if not os.path.exists(input_fastq):
            raise FileNotFoundError(f"Input FASTQ not found: {input_fastq}")

        command = [
            "bbduk.sh",
            f"in={input_fastq}",
            f"out={output_fastq}",
            f"minlen={min_length}",
            f"maq={min_quality}",
            "stats=filter_stats.txt"
        ]

        if additional_params:
            command.extend(additional_params.split())

        stdout, stderr, returncode = self._run_command(command)

        if returncode != 0:
            raise RuntimeError(f"Read filtering failed: {stderr}")

        # Parse filtering statistics
        stats = self._parse_filter_stats("filter_stats.txt")

        result = {
            "status": "success",
            "input_file": input_fastq,
            "output_file": output_fastq,
            "filter_params": {
                "min_length": min_length,
                "min_quality": min_quality
            },
            "filter_stats": stats,
            "command_used": " ".join(self.base_command + command)
        }

        return result

    def _parse_mapping_stats(self, stats_file: str) -> Dict[str, Any]:
        """Parse BBMap mapping statistics"""
        stats = {}

        try:
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    content = f.read()

                # Extract key statistics using regex patterns
                patterns = {
                    'reads_used': r'Reads Used:\s*(\d+)',
                    'mapped_reads': r'Mapped:\s*(\d+)',
                    'mapping_rate': r'Mapped:\s*\d+\s*\((\d+\.?\d*)%\)',
                    'average_identity': r'Average identity:\s*(\d+\.?\d*)%',
                    'average_coverage': r'Average coverage:\s*(\d+\.?\d*)'
                }

                for key, pattern in patterns.items():
                    match = re.search(pattern, content)
                    if match:
                        try:
                            stats[key] = float(match.group(1))
                        except ValueError:
                            stats[key] = match.group(1)

        except Exception as e:
            logger.warning(f"Could not parse mapping stats: {e}")
            stats = {"error": "Could not parse statistics file"}

        return stats

    def _parse_quality_stats(self, stats_file: str, hist_file: str) -> Dict[str, Any]:
        """Parse quality statistics from readlength.sh output"""
        stats = {}

        try:
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    content = f.read()

                # Parse basic statistics
                patterns = {
                    'total_reads': r'Reads:\s*(\d+)',
                    'total_bases': r'Bases:\s*(\d+)',
                    'average_length': r'Average:\s*(\d+\.?\d*)',
                    'median_length': r'Median:\s*(\d+)',
                    'mode_length': r'Mode:\s*(\d+)'
                }

                for key, pattern in patterns.items():
                    match = re.search(pattern, content)
                    if match:
                        try:
                            stats[key] = float(match.group(1))
                        except ValueError:
                            stats[key] = match.group(1)

        except Exception as e:
            logger.warning(f"Could not parse quality stats: {e}")
            stats = {"error": "Could not parse statistics file"}

        return stats

    def _parse_coverage_stats(self, stats_file: str) -> Dict[str, Any]:
        """Parse coverage statistics from pileup.sh output"""
        stats = {}

        try:
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    content = f.read()

                # Parse coverage statistics
                patterns = {
                    'average_coverage': r'Average coverage:\s*(\d+\.?\d*)',
                    'coverage_std': r'Coverage std dev:\s*(\d+\.?\d*)',
                    'percent_covered': r'Percent covered:\s*(\d+\.?\d*)%',
                    'plus_reads': r'Plus reads:\s*(\d+)',
                    'minus_reads': r'Minus reads:\s*(\d+)'
                }

                for key, pattern in patterns.items():
                    match = re.search(pattern, content)
                    if match:
                        try:
                            stats[key] = float(match.group(1))
                        except ValueError:
                            stats[key] = match.group(1)

        except Exception as e:
            logger.warning(f"Could not parse coverage stats: {e}")
            stats = {"error": "Could not parse statistics file"}

        return stats

    def _parse_filter_stats(self, stats_file: str) -> Dict[str, Any]:
        """Parse filtering statistics from bbduk.sh output"""
        stats = {}

        try:
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    content = f.read()

                # Parse filtering statistics
                patterns = {
                    'input_reads': r'Input:\s*(\d+) reads',
                    'output_reads': r'Result:\s*(\d+) reads',
                    'filtered_reads': r'Discarded:\s*(\d+) reads',
                    'filtering_rate': r'Discarded:\s*\d+ reads \((\d+\.?\d*)%\)'
                }

                for key, pattern in patterns.items():
                    match = re.search(pattern, content)
                    if match:
                        try:
                            stats[key] = float(match.group(1))
                        except ValueError:
                            stats[key] = match.group(1)

        except Exception as e:
            logger.warning(f"Could not parse filter stats: {e}")
            stats = {"error": "Could not parse statistics file"}

        return stats

    def _parse_stdout_stats(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse BBMap mapping statistics from stdout/stderr output"""
        stats = {}

        # Combine stdout and stderr for parsing
        content = stdout + "\n" + stderr

        try:
            # Extract key statistics using regex patterns
            patterns = {
                'reads_used': r'Reads Used:\s*(\d+)',
                'mapped_reads': r'Mapped:\s*(\d+)',
                'mapping_rate': r'Mapped:\s*\d+\s*\((\d+\.?\d*)%\)',
                'average_identity': r'Average identity:\s*(\d+\.?\d*)%',
                'average_coverage': r'Average coverage:\s*(\d+\.?\d*)',
                'reads_processed': r'Reads:\s*(\d+)',
                'bases_processed': r'Bases:\s*(\d+)'
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    try:
                        stats[key] = float(match.group(1))
                    except ValueError:
                        stats[key] = match.group(1)

            # If no specific stats found, provide basic info
            if not stats:
                stats = {
                    "status": "completed",
                    "output_available": len(stdout) > 0,
                    "stdout_length": len(stdout),
                    "stderr_length": len(stderr)
                }

        except Exception as e:
            logger.warning(f"Could not parse stdout stats: {e}")
            stats = {"error": "Could not parse statistics from output"}

        return stats