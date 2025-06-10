import subprocess
import re
from fastmcp import FastMCP

import logging
logger = logging.getLogger(__name__)

mcp = FastMCP("rbs_prediction")
# toolkit = BioToolkit()

class SequenceRBS:
    def __init__(self, sequence_id):
        self.sequence_id = sequence_id
        self.cds_predictions = []
        self.rbs_predictions = []

    def add_cds(self, start, end, strand, frame, score, length, product):
        self.cds_predictions.append({
            'start': start,
            'end': end,
            'strand': strand,
            'frame': frame,
            'score': score,
            'length': length,
            'product': product
        })

    def add_rbs(self, position, sequence):
        self.rbs_predictions.append({
            'position': position,
            'sequence': sequence
        })

    def to_dict(self):
        return {
            'sequence_id': self.sequence_id,
            'cds_predictions': self.cds_predictions,
            'rbs_predictions': self.rbs_predictions
        }

@mcp.tool
async def predict_cds_and_rbs(nucleotide_sequence, sequence_id, tag_meta=1):
    # Write the nucleotide sequence to a temporary file
    with open('temp_sequence.fasta', 'w') as f:
        f.write(f">{sequence_id}\n{nucleotide_sequence}\n")

    # Run Prodigal to predict CDS
    seq_len = len(nucleotide_sequence)
    if seq_len < 20000 or tag_meta==1:
        result = subprocess.run(['prodigal', '-p', 'meta', '-i', 'temp_sequence.fasta', '-f', 'gff', '-o', 'prodigal_output.gff'], capture_output=True, text=True)
    else:
        result = subprocess.run(['prodigal', '-i', 'temp_sequence.fasta', '-f', 'gff', '-o', 'prodigal_output.gff'], capture_output=True, text=True)

    # Check if Prodigal ran successfully
    if result.returncode != 0:
        raise Exception(f"Prodigal failed with error: {result.stderr}")

    # Parse the GFF output to extract CDS and RBS information
    cds_predictions = []
    rbs_predictions = []

    with open('prodigal_output.gff', 'r') as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.strip().split('\t')
                if parts[2] == 'CDS':
                    start = int(parts[3])
                    end = int(parts[4])
                    strand = parts[6]
                    attributes = dict(re.findall(r'(\S+)=(\S+)', parts[8]))
                    frame = attributes.get('codon_start', '?')
                    score = attributes.get('score', '?')
                    length = end - start + 1
                    product = attributes.get('product', '?')
                    cds_predictions.append((start, end, strand, frame, score, length, product))
                elif parts[2] == 'rRNA':
                    # Assuming RBS is annotated as 'rRNA' for simplicity
                    start = int(parts[3])
                    end = int(parts[4])
                    strand = parts[6]
                    rbs_sequence = nucleotide_sequence[start-1:end]
                    rbs_predictions.append((start, rbs_sequence))

    # Create a SequenceRBS object and add predictions
    sequence_rbs = SequenceRBS(sequence_id)
    for cds in cds_predictions:
        sequence_rbs.add_cds(*cds)
    for rbs in rbs_predictions:
        sequence_rbs.add_rbs(*rbs)

    return sequence_rbs

if __name__ == "__main__":
    mcp.run()