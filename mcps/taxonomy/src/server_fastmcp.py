"""
Taxonomy MCP Server using FastMCP
"""

from fastmcp import FastMCP
import json
import logging
import sys
from pathlib import Path
import taxopy

from pydantic import BaseModel, ConfigDict

taxdump_dir = Path(__file__).parent.parent / "ictv-taxdump"


# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Reduce FastMCP logging noise during startup
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

# Create FastMCP server
mcp = FastMCP("Taxonomy Tools 🧬")


@mcp.tool
def get_taxonomy_info(name: str) -> list[dict[str, str]]:
    """
    Get taxonomy information from a viral name.
    The taxdump is obtained from running the create_ictv_taxonomy.sh script
    """
    matched_ranks = []
    taxdb_url = "https://github.com/shenwei356/ictv-taxdump/releases/latest/download/ictv-taxdump.tar.gz"

    taxdb = taxopy.TaxDb(
        taxdump_url=taxdb_url,
    )

    taxids = taxopy.taxid_from_name(name, taxdb, fuzzy=True)
    for taxid in taxids:
        taxon = taxopy.Taxon(taxid, taxdb)
        matched_ranks.append(taxon.rank_name_dictionary)

    return matched_ranks


if __name__ == "__main__":
    # Run with stdio transport by default
    mcp.run()
