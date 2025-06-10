"""
Tests for BBMap tools
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.bbmap_tools import BBMapToolkit


@pytest.fixture
def toolkit():
    """Create BBMapToolkit instance"""
    return BBMapToolkit()


@pytest.fixture
def sample_fasta():
    """Create a sample FASTA file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(">test_contig\n")
        f.write("ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG\n")
        f.write("ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG\n")
        return f.name


@pytest.fixture
def sample_fastq():
    """Create a sample FASTQ file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fastq', delete=False) as f:
        f.write("@read1\n")
        f.write("ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG\n")
        f.write("+\n")
        f.write("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n")
        f.write("@read2\n")
        f.write("GCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA\n")
        f.write("+\n")
        f.write("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n")
        return f.name


@pytest.fixture
def cleanup_files():
    """Cleanup test files after tests"""
    files_to_cleanup = []
    yield files_to_cleanup
    for file_path in files_to_cleanup:
        if os.path.exists(file_path):
            os.unlink(file_path)


def test_bbmap_toolkit_initialization():
    """Test BBMapToolkit initialization"""
    toolkit = BBMapToolkit()
    assert toolkit.shifter_image == "bryce911/bbtools:latest"
    assert toolkit.base_command == ["shifter", "--image", "bryce911/bbtools:latest"]


def test_bbmap_toolkit_custom_image():
    """Test BBMapToolkit with custom image"""
    custom_image = "custom/bbtools:v1.0"
    toolkit = BBMapToolkit(shifter_image=custom_image)
    assert toolkit.shifter_image == custom_image
    assert toolkit.base_command == ["shifter", "--image", custom_image]


@patch('subprocess.Popen')
def test_run_command_success(mock_popen):
    """Test successful command execution"""
    # Mock successful command execution
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("output", "")
    mock_process.returncode = 0
    mock_popen.return_value = mock_process

    toolkit = BBMapToolkit()
    stdout, stderr, returncode = toolkit._run_command(["test", "command"])

    assert stdout == "output"
    assert stderr == ""
    assert returncode == 0


@patch('subprocess.Popen')
def test_run_command_failure(mock_popen):
    """Test failed command execution"""
    # Mock failed command execution
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("", "error message")
    mock_process.returncode = 1
    mock_popen.return_value = mock_process

    toolkit = BBMapToolkit()
    stdout, stderr, returncode = toolkit._run_command(["test", "command"])

    assert stdout == ""
    assert stderr == "error message"
    assert returncode == 1


@pytest.mark.asyncio
async def test_map_reads_file_not_found():
    """Test map_reads with non-existent files"""
    toolkit = BBMapToolkit()

    with pytest.raises(FileNotFoundError, match="Reference file not found"):
        await toolkit.map_reads(
            reference_path="/nonexistent/reference.fasta",
            reads_path="/nonexistent/reads.fastq",
            output_sam="output.sam"
        )


@pytest.mark.asyncio
@patch.object(BBMapToolkit, '_run_command')
@patch.object(BBMapToolkit, '_parse_mapping_stats')
async def test_map_reads_success(mock_parse_stats, mock_run_command, sample_fasta, sample_fastq, cleanup_files):
    """Test successful read mapping"""
    # Setup mocks
    mock_run_command.return_value = ("mapping output", "", 0)
    mock_parse_stats.return_value = {
        "reads_used": 1000,
        "mapped_reads": 950,
        "mapping_rate": 95.0
    }

    toolkit = BBMapToolkit()
    output_sam = "test_output.sam"
    cleanup_files.append(output_sam)

    result = await toolkit.map_reads(
        reference_path=sample_fasta,
        reads_path=sample_fastq,
        output_sam=output_sam
    )

    assert result["status"] == "success"
    assert result["output_sam"] == output_sam
    assert result["mapping_stats"]["mapping_rate"] == 95.0
    assert "command_used" in result


@pytest.mark.asyncio
@patch.object(BBMapToolkit, '_run_command')
async def test_map_reads_bbmap_failure(mock_run_command, sample_fasta, sample_fastq):
    """Test map_reads when BBMap command fails"""
    # Mock BBMap command failure
    mock_run_command.return_value = ("", "BBMap execution failed", 1)

    toolkit = BBMapToolkit()

    with pytest.raises(RuntimeError, match="BBMap failed"):
        await toolkit.map_reads(
            reference_path=sample_fasta,
            reads_path=sample_fastq,
            output_sam="output.sam"
        )


@pytest.mark.asyncio
async def test_quality_stats_file_not_found():
    """Test quality_stats with non-existent FASTQ file"""
    toolkit = BBMapToolkit()

    with pytest.raises(FileNotFoundError, match="FASTQ file not found"):
        await toolkit.quality_stats(fastq_path="/nonexistent/reads.fastq")


@pytest.mark.asyncio
@patch.object(BBMapToolkit, '_run_command')
@patch.object(BBMapToolkit, '_parse_quality_stats')
async def test_quality_stats_success(mock_parse_stats, mock_run_command, sample_fastq):
    """Test successful quality statistics generation"""
    # Setup mocks
    mock_run_command.return_value = ("quality stats output", "", 0)
    mock_parse_stats.return_value = {
        "total_reads": 1000,
        "average_length": 150.5,
        "total_bases": 150500
    }

    toolkit = BBMapToolkit()

    result = await toolkit.quality_stats(
        fastq_path=sample_fastq,
        output_prefix="test_quality"
    )

    assert result["status"] == "success"
    assert result["fastq_file"] == sample_fastq
    assert result["quality_stats"]["total_reads"] == 1000
    assert "output_files" in result


@pytest.mark.asyncio
@patch.object(BBMapToolkit, '_run_command')
@patch.object(BBMapToolkit, '_parse_coverage_stats')
async def test_coverage_analysis_success(mock_parse_stats, mock_run_command, sample_fasta):
    """Test successful coverage analysis"""
    # Setup mocks
    mock_run_command.return_value = ("coverage analysis output", "", 0)
    mock_parse_stats.return_value = {
        "average_coverage": 25.5,
        "percent_covered": 95.2
    }

    toolkit = BBMapToolkit()

    # Create a dummy SAM file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sam', delete=False) as sam_file:
        sam_file.write("@SQ\tSN:test_contig\tLN:100\n")
        sam_file.write("read1\t0\ttest_contig\t1\t60\t50M\t*\t0\t0\tATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG\t*\n")
        sam_path = sam_file.name

    try:
        result = await toolkit.coverage_analysis(
            sam_path=sam_path,
            reference_path=sample_fasta,
            output_prefix="test_coverage"
        )

        assert result["status"] == "success"
        assert result["sam_file"] == sam_path
        assert result["coverage_stats"]["average_coverage"] == 25.5
        assert "output_files" in result
    finally:
        os.unlink(sam_path)


@pytest.mark.asyncio
@patch.object(BBMapToolkit, '_run_command')
@patch.object(BBMapToolkit, '_parse_filter_stats')
async def test_filter_reads_success(mock_parse_stats, mock_run_command, sample_fastq, cleanup_files):
    """Test successful read filtering"""
    # Setup mocks
    mock_run_command.return_value = ("filtering output", "", 0)
    mock_parse_stats.return_value = {
        "input_reads": 1000,
        "output_reads": 950,
        "filtered_reads": 50
    }

    toolkit = BBMapToolkit()
    output_fastq = "filtered_reads.fastq"
    cleanup_files.append(output_fastq)

    result = await toolkit.filter_reads(
        input_fastq=sample_fastq,
        output_fastq=output_fastq,
        min_length=30,
        min_quality=15.0
    )

    assert result["status"] == "success"
    assert result["input_file"] == sample_fastq
    assert result["output_file"] == output_fastq
    assert result["filter_params"]["min_length"] == 30
    assert result["filter_stats"]["output_reads"] == 950


def test_parse_mapping_stats():
    """Test parsing of mapping statistics"""
    toolkit = BBMapToolkit()

    # Create a mock stats file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("Reads Used: 1000\n")
        f.write("Mapped: 950 (95.00%)\n")
        f.write("Average identity: 98.5%\n")
        f.write("Average coverage: 25.3\n")
        stats_file = f.name

    try:
        stats = toolkit._parse_mapping_stats(stats_file)
        assert stats["reads_used"] == 1000.0
        assert stats["mapped_reads"] == 950.0
        assert stats["mapping_rate"] == 95.00
        assert stats["average_identity"] == 98.5
    finally:
        os.unlink(stats_file)


def test_parse_mapping_stats_missing_file():
    """Test parsing mapping stats with missing file"""
    toolkit = BBMapToolkit()
    stats = toolkit._parse_mapping_stats("/nonexistent/file.txt")
    assert "error" in stats


def test_parse_quality_stats():
    """Test parsing of quality statistics"""
    toolkit = BBMapToolkit()

    # Create a mock stats file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("Reads: 1000\n")
        f.write("Bases: 150000\n")
        f.write("Average: 150.0\n")
        f.write("Median: 148\n")
        f.write("Mode: 150\n")
        stats_file = f.name

    try:
        stats = toolkit._parse_quality_stats(stats_file, "hist_file.txt")
        assert stats["total_reads"] == 1000.0
        assert stats["total_bases"] == 150000.0
        assert stats["average_length"] == 150.0
    finally:
        os.unlink(stats_file)


if __name__ == "__main__":
    pytest.main([__file__])
