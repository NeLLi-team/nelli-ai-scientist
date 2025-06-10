"""
Simple and Reliable Filesystem MCP Server
Uses Python's built-in pathlib and os for robust file operations
"""

from fastmcp import FastMCP
import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Reduce FastMCP logging noise
logging.getLogger('fastmcp').setLevel(logging.WARNING)
logging.getLogger('mcp').setLevel(logging.WARNING)

# Create FastMCP server
mcp = FastMCP("Simple Filesystem 📁")

# Security: only allow operations in safe directories
REPO_BASE = str(Path(__file__).parent.parent.parent.parent.absolute())
ALLOWED_DIRS = ["/tmp", REPO_BASE]

def _check_path_security(path: str) -> bool:
    """Check if path is in allowed directories"""
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(allowed) for allowed in ALLOWED_DIRS)

@mcp.tool
async def tree_view(path: str = ".", max_depth: int = 3, show_hidden: bool = False, 
                    file_extensions: str = "") -> dict:
    """
    Display directory structure in a tree format with file details
    
    Args:
        path: Directory path to explore (default: current directory)
        max_depth: Maximum depth to traverse (default: 3)
        show_hidden: Include hidden files/directories (default: False)  
        file_extensions: Only show files with these extensions (e.g., "py,txt,fna,fasta")
    """
    try:
        if not _check_path_security(path):
            return {"error": f"Access denied. Path must be in: {ALLOWED_DIRS}"}
        
        start_path = Path(path).resolve()
        if not start_path.exists():
            return {"error": f"Path does not exist: {path}"}
        
        if not start_path.is_dir():
            return {"error": f"Path is not a directory: {path}"}
        
        # Parse file extensions filter
        allowed_extensions = None
        if file_extensions and file_extensions.strip():
            allowed_extensions = [ext.strip().lower() for ext in file_extensions.split(',')]
            allowed_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in allowed_extensions]
        
        tree_structure = []
        file_count = 0
        dir_count = 0
        total_size = 0
        
        def build_tree(current_path: Path, prefix: str = "", depth: int = 0):
            nonlocal file_count, dir_count, total_size
            
            if depth > max_depth:
                return
            
            try:
                # Get all items in directory
                items = list(current_path.iterdir())
                
                # Filter hidden files if needed
                if not show_hidden:
                    items = [item for item in items if not item.name.startswith('.')]
                
                # Sort: directories first, then files
                dirs = [item for item in items if item.is_dir()]
                files = [item for item in items if item.is_file()]
                dirs.sort(key=lambda x: x.name.lower())
                files.sort(key=lambda x: x.name.lower())
                
                all_items = dirs + files
                
                for i, item in enumerate(all_items):
                    is_last = i == len(all_items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    
                    if item.is_dir():
                        dir_count += 1
                        tree_structure.append({
                            "line": f"{prefix}{current_prefix}{item.name}/",
                            "type": "directory",
                            "name": item.name,
                            "path": str(item),
                            "depth": depth
                        })
                        # Recurse into subdirectory
                        build_tree(item, next_prefix, depth + 1)
                    
                    elif item.is_file():
                        # Check file extension filter
                        if allowed_extensions:
                            file_ext = item.suffix.lower()
                            if file_ext not in allowed_extensions:
                                continue
                        
                        file_count += 1
                        size = item.stat().st_size
                        total_size += size
                        
                        # Format file size
                        size_str = format_file_size(size)
                        
                        tree_structure.append({
                            "line": f"{prefix}{current_prefix}{item.name} ({size_str})",
                            "type": "file", 
                            "name": item.name,
                            "path": str(item),
                            "size": size,
                            "size_formatted": size_str,
                            "extension": item.suffix.lower(),
                            "depth": depth
                        })
            
            except PermissionError:
                tree_structure.append({
                    "line": f"{prefix}├── [Permission Denied]",
                    "type": "error",
                    "depth": depth
                })
        
        # Start building tree
        tree_structure.append({
            "line": f"{start_path.name}/",
            "type": "root",
            "name": start_path.name,
            "path": str(start_path),
            "depth": 0
        })
        
        build_tree(start_path)
        
        # Create tree display string
        tree_display = "\n".join([item["line"] for item in tree_structure])
        
        return {
            "path": str(start_path),
            "tree_display": tree_display,
            "tree_structure": tree_structure,
            "summary": {
                "directories": dir_count,
                "files": file_count,
                "total_size": total_size,
                "total_size_formatted": format_file_size(total_size),
                "max_depth": max_depth,
                "file_extensions_filter": file_extensions
            },
            "success": True
        }
        
    except Exception as e:
        return {"error": f"Failed to generate tree view: {str(e)}"}

@mcp.tool  
async def find_files(path: str = ".", pattern: str = "*", extensions: str = "", 
                     max_depth: int = 5) -> dict:
    """
    Find files by pattern or extension recursively
    
    Args:
        path: Starting directory (default: current directory)
        pattern: Filename pattern using wildcards (e.g., "*contigs*", "*.py")
        extensions: File extensions to search for (e.g., "fna,fasta,fastq,py,txt")
        max_depth: Maximum search depth (default: 5)
    """
    try:
        import fnmatch
        
        if not _check_path_security(path):
            return {"error": f"Access denied. Path must be in: {ALLOWED_DIRS}"}
        
        start_path = Path(path).resolve()
        if not start_path.exists():
            return {"error": f"Path does not exist: {path}"}
        
        # Parse extensions
        allowed_extensions = None
        if extensions and extensions.strip():
            allowed_extensions = [ext.strip().lower() for ext in extensions.split(',')]
            allowed_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in allowed_extensions]
        
        found_files = []
        
        def search_files(current_path: Path, depth: int = 0):
            if depth > max_depth:
                return
            
            try:
                for item in current_path.iterdir():
                    if item.is_file():
                        # Check pattern match
                        matches_pattern = fnmatch.fnmatch(item.name, pattern)
                        
                        # Check extension match  
                        matches_extension = True
                        if allowed_extensions:
                            matches_extension = item.suffix.lower() in allowed_extensions
                        
                        if matches_pattern and matches_extension:
                            size = item.stat().st_size
                            found_files.append({
                                "name": item.name,
                                "path": str(item),
                                "relative_path": str(item.relative_to(start_path)),
                                "size": size,
                                "size_formatted": format_file_size(size),
                                "extension": item.suffix.lower(),
                                "parent_dir": str(item.parent)
                            })
                    
                    elif item.is_dir() and not item.name.startswith('.'):
                        search_files(item, depth + 1)
                        
            except PermissionError:
                pass  # Skip directories we can't access
        
        search_files(start_path)
        
        # Sort by name
        found_files.sort(key=lambda x: x['name'])
        
        return {
            "search_path": str(start_path),
            "pattern": pattern,
            "extensions": extensions,
            "max_depth": max_depth,
            "found_files": found_files,
            "total_files": len(found_files),
            "total_size": sum(f["size"] for f in found_files),
            "total_size_formatted": format_file_size(sum(f["size"] for f in found_files)),
            "success": True
        }
        
    except Exception as e:
        return {"error": f"Failed to search files: {str(e)}"}

@mcp.tool
async def read_file(path: str, max_lines: int = 0) -> dict:
    """
    Read file contents safely
    
    Args:
        path: Path to the file to read
        max_lines: Maximum number of lines to read (default: read all)
    """
    try:
        if not _check_path_security(path):
            return {"error": f"Access denied. Path must be in: {ALLOWED_DIRS}"}
        
        file_path = Path(path)
        if not file_path.exists():
            return {"error": f"File not found: {path}"}
        
        if not file_path.is_file():
            return {"error": f"Path is not a file: {path}"}
        
        size = file_path.stat().st_size
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if max_lines and max_lines > 0:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line.rstrip('\n\r'))
                    content = '\n'.join(lines)
                    truncated = True
                else:
                    content = f.read()
                    truncated = False
        except UnicodeDecodeError:
            # Try binary read for non-text files
            with open(file_path, 'rb') as f:
                raw_content = f.read(1000) if max_lines else f.read()  
                content = f"[Binary file - showing first {len(raw_content)} bytes]\n{raw_content}"
                truncated = len(raw_content) < size
        
        return {
            "path": str(file_path),
            "content": content,
            "size": size,
            "size_formatted": format_file_size(size),
            "lines": content.count('\n') + 1 if content else 0,
            "truncated": truncated,
            "max_lines": max_lines,
            "success": True
        }
        
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}

@mcp.tool
async def write_file(path: str, content: str) -> dict:
    """
    Write content to a file
    
    Args:
        path: Path to the file to write
        content: Content to write
    """
    try:
        if not _check_path_security(path):
            return {"error": f"Access denied. Path must be in: {ALLOWED_DIRS}"}
        
        file_path = Path(path)
        
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        size = file_path.stat().st_size
        
        return {
            "path": str(file_path),
            "size": size,
            "size_formatted": format_file_size(size),
            "lines": content.count('\n') + 1,
            "success": True
        }
        
    except Exception as e:
        return {"error": f"Failed to write file: {str(e)}"}

@mcp.tool
async def find_file_by_name(filename: str, search_path: str = ".", max_depth: int = 5) -> dict:
    """
    Find a file by name in the directory tree
    
    Args:
        filename: Name of the file to find (e.g., "contigs100k.fna")
        search_path: Directory to search in (default: current directory)
        max_depth: Maximum search depth (default: 5)
    """
    try:
        if not _check_path_security(search_path):
            return {"error": f"Access denied. Path must be in: {ALLOWED_DIRS}"}
        
        start_path = Path(search_path).resolve()
        if not start_path.exists():
            return {"error": f"Search path does not exist: {search_path}"}
        
        found_files = []
        
        def search_for_file(current_path: Path, depth: int = 0):
            if depth > max_depth:
                return
            
            try:
                for item in current_path.iterdir():
                    if item.is_file() and item.name == filename:
                        size = item.stat().st_size
                        found_files.append({
                            "name": item.name,
                            "path": str(item),
                            "relative_path": str(item.relative_to(start_path)),
                            "size": size,
                            "size_formatted": format_file_size(size),
                            "parent_dir": str(item.parent)
                        })
                    elif item.is_dir() and not item.name.startswith('.'):
                        search_for_file(item, depth + 1)
                        
            except PermissionError:
                pass  # Skip directories we can't access
        
        search_for_file(start_path)
        
        return {
            "filename": filename,
            "search_path": str(start_path),
            "max_depth": max_depth,
            "found_files": found_files,
            "total_found": len(found_files),
            "success": True
        }
        
    except Exception as e:
        return {"error": f"Failed to find file: {str(e)}"}

@mcp.tool
async def file_info(path: str) -> dict:
    """
    Get detailed information about a file or directory
    
    Args:
        path: Path to examine
    """
    try:
        if not _check_path_security(path):
            return {"error": f"Access denied. Path must be in: {ALLOWED_DIRS}"}
        
        item_path = Path(path)
        if not item_path.exists():
            return {"error": f"Path does not exist: {path}"}
        
        stat = item_path.stat()
        
        result = {
            "path": str(item_path),
            "name": item_path.name,
            "exists": True,
            "type": "directory" if item_path.is_dir() else "file",
            "size": stat.st_size,
            "size_formatted": format_file_size(stat.st_size),
            "modified": stat.st_mtime,
            "success": True
        }
        
        if item_path.is_file():
            result.update({
                "extension": item_path.suffix.lower(),
                "stem": item_path.stem
            })
            
            # Try to detect if it's a text file
            try:
                with open(item_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    result["is_text"] = True
                    result["first_line"] = first_line.strip()
            except UnicodeDecodeError:
                result["is_text"] = False
                result["first_line"] = "[Binary file]"
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to get file info: {str(e)}"}

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"

# Add resources for help and examples
@mcp.resource("simple-filesystem://help")
async def help_resource() -> str:
    """Get help information for simple filesystem tools"""
    return json.dumps({
        "tools": {
            "tree_view": "Display directory structure in tree format with file sizes",
            "find_files": "Find files by pattern or extension recursively", 
            "read_file": "Read file contents with optional line limit",
            "write_file": "Write content to a file",
            "file_info": "Get detailed information about a file or directory"
        },
        "examples": {
            "tree_view": "tree_view(path='data', max_depth=2, file_extensions='fna,fasta')",
            "find_files": "find_files(path='data', extensions='fna,fasta,fastq')",
            "read_file": "read_file(path='data/file.fna', max_lines=10)",
            "file_info": "file_info(path='data/nelli_hackathon/contigs.fna')"
        },
        "allowed_directories": ALLOWED_DIRS
    }, indent=2)

if __name__ == "__main__":
    # Run the server
    mcp.run()