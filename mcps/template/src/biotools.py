"""
BioPython-based tools implementation
"""

from typing import Dict, Any, List
import logging
from io import StringIO

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import gc_fraction, molecular_weight
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import numpy as np

logger = logging.getLogger(__name__)


class BioToolkit:
    """Collection of BioPython-based tools"""

    async def sequence_stats(self, sequence: str, sequence_type: str) -> Dict[str, Any]:
        """Calculate comprehensive sequence statistics"""
        try:
            seq = Seq(sequence.upper())

            stats = {"length": len(seq), "type": sequence_type}

            if sequence_type in ["dna", "rna"]:
                # Nucleotide statistics
                stats.update(
                    {
                        "gc_content": gc_fraction(seq) * 100,
                        "at_content": (1 - gc_fraction(seq)) * 100,
                        "composition": {
                            "A": seq.count("A"),
                            "T": seq.count("T") if sequence_type == "dna" else 0,
                            "U": seq.count("U") if sequence_type == "rna" else 0,
                            "G": seq.count("G"),
                            "C": seq.count("C"),
                            "N": seq.count("N"),
                        },
                    }
                )

                # Find ORFs for DNA
                if sequence_type == "dna":
                    try:
                        orfs = self._find_orfs(seq)
                        stats["orfs"] = {
                            "count": len(orfs),
                            "longest": max([orf["length"] for orf in orfs]) if orfs else 0,
                            "details": orfs[:5],  # First 5 ORFs
                        }
                    except Exception as e:
                        logger.warning(f"ORF finding failed: {str(e)}")
                        stats["orfs"] = {
                            "count": 0,
                            "longest": 0,
                            "details": [],
                            "error": f"ORF analysis failed: {str(e)}"
                        }

            else:  # protein
                # Protein statistics
                stats.update(
                    {
                        "molecular_weight": molecular_weight(seq, seq_type="protein"),
                        "composition": self._amino_acid_composition(seq),
                    }
                )

                # Basic properties
                stats["properties"] = {
                    "hydrophobic": sum(seq.count(aa) for aa in "AILMFWYV"),
                    "hydrophilic": sum(seq.count(aa) for aa in "RNDQEHKS"),
                    "aromatic": sum(seq.count(aa) for aa in "FWY"),
                    "positive": sum(seq.count(aa) for aa in "RKH"),
                    "negative": sum(seq.count(aa) for aa in "DE"),
                }

            return stats
        except Exception as e:
            logger.error(f"Sequence statistics calculation failed: {str(e)}")
            return {
                "error": f"Failed to calculate sequence statistics: {str(e)}",
                "length": len(sequence) if sequence else 0,
                "type": sequence_type
            }

    def _find_orfs(self, seq: Seq, min_length: int = 100) -> List[Dict[str, Any]]:
        """Find open reading frames"""
        orfs = []

        for strand, nuc in [(+1, seq), (-1, seq.reverse_complement())]:
            for frame in range(3):
                try:
                    # Clean sequence - replace ambiguous nucleotides with N
                    clean_seq = str(nuc[frame:])
                    # Replace any non-standard nucleotides with N
                    clean_seq = ''.join(c if c in 'ATCGN' else 'N' for c in clean_seq)
                    clean_seq_obj = Seq(clean_seq)
                    
                    # Translate with ambiguous_dna_by_name=True to handle N's properly
                    trans = clean_seq_obj.translate(to_stop=False)
                    trans_str = str(trans)
                    aa_start = 0

                    while aa_start < len(trans_str):
                        # Find next Met
                        aa_start = trans_str.find("M", aa_start)
                        if aa_start == -1:
                            break

                        # Find next stop
                        aa_end = trans_str.find("*", aa_start)
                        if aa_end == -1:
                            aa_end = len(trans_str)

                        # Check length
                        if (aa_end - aa_start) * 3 >= min_length:
                            orfs.append(
                                {
                                    "start": frame + aa_start * 3,
                                    "end": frame + aa_end * 3,
                                    "strand": strand,
                                    "frame": frame,
                                    "length": (aa_end - aa_start) * 3,
                                    "protein": trans_str[aa_start:aa_end],
                                }
                            )

                        aa_start = aa_end + 1
                except Exception as e:
                    logger.warning(f"Translation failed for frame {frame}, strand {strand}: {str(e)}")
                    continue

        return sorted(orfs, key=lambda x: x["length"], reverse=True)

    def _amino_acid_composition(self, seq: Seq) -> Dict[str, int]:
        """Calculate amino acid composition"""
        aa_list = "ACDEFGHIKLMNPQRSTVWY"
        return {aa: seq.count(aa) for aa in aa_list}

    async def blast_local(
        self, sequence: str, database: str, program: str, e_value: float = 0.001
    ) -> Dict[str, Any]:
        """Run local BLAST search (simulated)"""

        # In a real implementation, this would run actual BLAST
        # For now, return simulated results

        return {
            "query": sequence[:50] + "...",
            "database": database,
            "program": program,
            "parameters": {"e_value": e_value},
            "hits": [
                {
                    "accession": "NM_001126",
                    "description": "Simulated BLAST hit",
                    "e_value": 1e-50,
                    "bit_score": 234.5,
                    "identity": 95.5,
                    "coverage": 98.0,
                    "alignment": {
                        "query_start": 1,
                        "query_end": len(sequence),
                        "subject_start": 100,
                        "subject_end": 100 + len(sequence),
                    },
                }
            ],
            "statistics": {
                "db_sequences": 1000000,
                "db_letters": 500000000,
                "effective_space": 450000000,
            },
        }

    async def multiple_alignment(
        self, sequences: List[Dict[str, str]], algorithm: str = "clustalw"
    ) -> Dict[str, Any]:
        """Perform multiple sequence alignment"""

        # Create SeqRecord objects
        records = []
        for seq_dict in sequences:
            record = SeqRecord(
                Seq(seq_dict["sequence"]), id=seq_dict["id"], description=""
            )
            records.append(record)

        # In real implementation, would call actual alignment tools
        # For now, return the sequences as-is (unaligned)

        # Simulate alignment by padding sequences
        max_len = max(len(r.seq) for r in records)
        aligned_seqs = []

        for record in records:
            seq_str = str(record.seq)
            # Simple right-padding simulation
            aligned_seq = seq_str + "-" * (max_len - len(seq_str))
            aligned_seqs.append({"id": record.id, "sequence": aligned_seq})

        # Calculate simple identity matrix
        n = len(aligned_seqs)
        identity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                seq1 = aligned_seqs[i]["sequence"]
                seq2 = aligned_seqs[j]["sequence"]
                matches = sum(
                    a == b for a, b in zip(seq1, seq2) if a != "-" and b != "-"
                )
                length = sum(1 for a, b in zip(seq1, seq2) if a != "-" or b != "-")
                identity_matrix[i][j] = (matches / length * 100) if length > 0 else 0

        return {
            "algorithm": algorithm,
            "aligned_sequences": aligned_seqs,
            "alignment_length": max_len,
            "identity_matrix": identity_matrix.tolist(),
            "consensus": self._calculate_consensus(aligned_seqs),
        }

    def _calculate_consensus(self, aligned_seqs: List[Dict[str, str]]) -> str:
        """Calculate consensus sequence"""
        if not aligned_seqs:
            return ""

        alignment_length = len(aligned_seqs[0]["sequence"])
        consensus = []

        for i in range(alignment_length):
            column = [seq["sequence"][i] for seq in aligned_seqs]
            # Most common non-gap character
            non_gap = [c for c in column if c != "-"]
            if non_gap:
                consensus.append(max(set(non_gap), key=non_gap.count))
            else:
                consensus.append("-")

        return "".join(consensus)

    async def phylogenetic_tree(
        self, alignment: str, method: str = "nj"
    ) -> Dict[str, Any]:
        """Build phylogenetic tree from alignment"""

        # Parse alignment
        handle = StringIO(alignment)
        try:
            aln = AlignIO.read(handle, "fasta")
        except Exception:
            # If not in FASTA format, try to parse manually
            sequences = []
            for line in alignment.strip().split("\n"):
                if line.startswith(">"):
                    if sequences and len(sequences[-1]) == 2:
                        sequences[-1].append("")
                    sequences.append([line[1:].strip()])
                elif sequences:
                    if len(sequences[-1]) == 1:
                        sequences[-1].append(line.strip())
                    else:
                        sequences[-1][1] += line.strip()

            # Create alignment
            records = [
                SeqRecord(Seq(seq[1]), id=seq[0]) for seq in sequences if len(seq) == 2
            ]
            aln = MultipleSeqAlignment(records)

        # Calculate distance matrix
        calculator = DistanceCalculator("identity")
        dm = calculator.get_distance(aln)

        # Build tree
        constructor = DistanceTreeConstructor()

        if method == "nj":
            tree = constructor.nj(dm)
        elif method == "upgma":
            tree = constructor.upgma(dm)
        else:
            # For ML, would use different approach
            tree = constructor.nj(dm)  # Fallback to NJ

        # Convert tree to Newick format
        output = StringIO()
        Phylo.write(tree, output, "newick")
        newick = output.getvalue().strip()

        return {
            "method": method,
            "newick": newick,
            "taxa_count": len(aln),
            "alignment_length": aln.get_alignment_length(),
            "tree_stats": {
                "terminals": tree.count_terminals(),
                "total_branch_length": tree.total_branch_length(),
            },
        }

    async def translate_sequence(
        self, sequence: str, genetic_code: int = 1
    ) -> Dict[str, Any]:
        """Translate DNA/RNA sequence to protein"""

        seq = Seq(sequence.upper().replace("U", "T"))  # Convert RNA to DNA

        # Translate in all three frames
        translations = {}

        for frame in range(3):
            frame_seq = seq[frame:]
            # Trim to multiple of 3
            trim_length = len(frame_seq) % 3
            if trim_length:
                frame_seq = frame_seq[:-trim_length]

            protein = frame_seq.translate(table=genetic_code, to_stop=True)
            translations[f"frame_{frame+1}"] = {
                "sequence": str(protein),
                "length": len(protein),
                "stops": str(frame_seq.translate(table=genetic_code)).count("*"),
            }

        # Also translate reverse complement
        rev_comp = seq.reverse_complement()
        for frame in range(3):
            frame_seq = rev_comp[frame:]
            trim_length = len(frame_seq) % 3
            if trim_length:
                frame_seq = frame_seq[:-trim_length]

            protein = frame_seq.translate(table=genetic_code, to_stop=True)
            translations[f"frame_-{frame+1}"] = {
                "sequence": str(protein),
                "length": len(protein),
                "stops": str(frame_seq.translate(table=genetic_code)).count("*"),
            }

        # Find longest ORF
        longest_orf = max(translations.items(), key=lambda x: x[1]["length"])

        return {
            "input_length": len(sequence),
            "genetic_code": genetic_code,
            "translations": translations,
            "longest_orf": {
                "frame": longest_orf[0],
                "protein": longest_orf[1]["sequence"],
                "length": longest_orf[1]["length"],
            },
        }

    async def read_fasta_file(self, file_path: str) -> Dict[str, Any]:
        """Read sequences from a FASTA file"""
        import os
        from Bio import SeqIO

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": f"File not found: {file_path}"}

        try:
            sequences = []
            for record in SeqIO.parse(file_path, "fasta"):
                sequences.append(
                    {
                        "id": record.id,
                        "description": record.description,
                        "sequence": str(record.seq),
                        "length": len(record.seq),
                    }
                )

            logger.info(f"Successfully read {len(sequences)} sequences from {file_path}")
            return {
                "file_path": file_path,
                "num_sequences": len(sequences),
                "sequences": sequences,
            }
        except Exception as e:
            logger.error(f"Failed to read FASTA file {file_path}: {str(e)}")
            return {"error": f"Failed to read FASTA file: {str(e)}"}

    async def write_json_report(
        self, data: Dict[str, Any], output_path: str
    ) -> Dict[str, Any]:
        """Write analysis results to a JSON report file"""
        import json
        import os
        from datetime import datetime

        try:
            # Add metadata to the report
            report = {
                "metadata": {
                    "generated_by": "NeLLi AI Scientist MCP Server",
                    "server_type": "biotools",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                },
                "analysis_results": data,
            }

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

            return {
                "status": "success",
                "output_path": output_path,
                "file_size": os.path.getsize(output_path),
            }
        except Exception as e:
            return {"error": f"Failed to write JSON report: {str(e)}"}

    async def analyze_fasta_file(
        self, file_path: str, sequence_type: str = "dna"
    ) -> Dict[str, Any]:
        """Comprehensive analysis of sequences in a FASTA file"""
        # First read the file
        fasta_data = await self.read_fasta_file(file_path)

        if "error" in fasta_data:
            return fasta_data

        # Analyze each sequence
        analyses = []
        for seq_record in fasta_data["sequences"]:
            seq_analysis = await self.sequence_stats(
                seq_record["sequence"], sequence_type
            )
            seq_analysis["sequence_id"] = seq_record["id"]
            seq_analysis["description"] = seq_record["description"]
            analyses.append(seq_analysis)

        # Calculate summary statistics
        total_length = sum(len(seq["sequence"]) for seq in fasta_data["sequences"])
        avg_length = (
            total_length / len(fasta_data["sequences"])
            if fasta_data["sequences"]
            else 0
        )

        if sequence_type in ["dna", "rna"]:
            avg_gc = (
                sum(analysis.get("gc_content", 0) for analysis in analyses)
                / len(analyses)
                if analyses
                else 0
            )
            summary_stats = {
                "total_sequences": fasta_data["num_sequences"],
                "total_length": total_length,
                "average_length": avg_length,
                "average_gc_content": avg_gc,
                "longest_sequence": max(
                    (analysis["length"] for analysis in analyses), default=0
                ),
                "shortest_sequence": min(
                    (analysis["length"] for analysis in analyses), default=0
                ),
            }
        else:  # protein
            avg_mw = (
                sum(analysis.get("molecular_weight", 0) for analysis in analyses)
                / len(analyses)
                if analyses
                else 0
            )
            summary_stats = {
                "total_sequences": fasta_data["num_sequences"],
                "total_length": total_length,
                "average_length": avg_length,
                "average_molecular_weight": avg_mw,
                "longest_sequence": max(
                    (analysis["length"] for analysis in analyses), default=0
                ),
                "shortest_sequence": min(
                    (analysis["length"] for analysis in analyses), default=0
                ),
            }

        return {
            "file_info": {"file_path": file_path, "sequence_type": sequence_type},
            "summary_statistics": summary_stats,
            "individual_analyses": analyses,
        }
