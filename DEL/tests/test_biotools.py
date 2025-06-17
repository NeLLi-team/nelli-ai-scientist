"""
Tests for BioPython tools
"""

import pytest
from src.biotools import BioToolkit


@pytest.fixture
def toolkit():
    """Create BioToolkit instance"""
    return BioToolkit()


@pytest.mark.asyncio
async def test_sequence_stats_dna(toolkit):
    """Test DNA sequence statistics"""
    result = await toolkit.sequence_stats(
        sequence="ATCGATCGATCGATCG", sequence_type="dna"
    )

    assert result["length"] == 16
    assert result["gc_content"] == 50.0
    assert result["composition"]["A"] == 4
    assert result["composition"]["T"] == 4


@pytest.mark.asyncio
async def test_sequence_stats_protein(toolkit):
    """Test protein sequence statistics"""
    result = await toolkit.sequence_stats(
        sequence="MVLSPADKTNVKAAW", sequence_type="protein"
    )

    assert result["length"] == 15
    assert "molecular_weight" in result
    assert result["properties"]["hydrophobic"] > 0


@pytest.mark.asyncio
async def test_translate_sequence(toolkit):
    """Test sequence translation"""
    result = await toolkit.translate_sequence(
        sequence="ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
    )

    assert "translations" in result
    assert "frame_1" in result["translations"]
    assert len(result["translations"]) == 6  # 3 forward + 3 reverse


@pytest.mark.asyncio
async def test_multiple_alignment(toolkit):
    """Test multiple sequence alignment"""
    sequences = [
        {"id": "seq1", "sequence": "ATCGATCG"},
        {"id": "seq2", "sequence": "ATCGATGG"},
        {"id": "seq3", "sequence": "ATCGATTG"},
    ]

    result = await toolkit.multiple_alignment(sequences)

    assert len(result["aligned_sequences"]) == 3
    assert "consensus" in result
    assert "identity_matrix" in result
