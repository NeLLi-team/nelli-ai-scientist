# moducomp_mcp.py
from mcp.server.fastmcp import FastMCP
import subprocess
import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create an MCP server
mcp = FastMCP("moducomp-server")

# Configuration - Set your moducomp project path here
MODUCOMP_PROJECT_PATH = "./moducomp"  # Update this path!

# You can also set it via environment variable
if 'MODUCOMP_PROJECT_PATH' in os.environ:
    MODUCOMP_PROJECT_PATH = os.environ['MODUCOMP_PROJECT_PATH']

def get_moducomp_path() -> Path:
    """Get the path to the moducomp project directory"""
    return Path(MODUCOMP_PROJECT_PATH).expanduser().resolve()

def check_pixi_environment(project_path: Path = None):
    """Check if pixi is available and moducomp environment is set up"""
    if project_path is None:
        project_path = get_moducomp_path()
    
    issues = []
    
    # Check if pixi is installed
    if not shutil.which('pixi'):
        issues.append("pixi is not installed")
    
    # Check if moducomp directory exists
    if not project_path.exists():
        issues.append(f"moducomp directory not found at: {project_path}")
        return issues
    
    # Check if we're in a pixi project (pixi.toml exists)
    pixi_toml = project_path / 'pixi.toml'
    if not pixi_toml.exists():
        issues.append(f"pixi.toml not found in: {project_path}")
    
    # Check if pixi environment is installed
    try:
        result = subprocess.run(['pixi', 'info'], 
                              capture_output=True, text=True, cwd=project_path)
        if result.returncode != 0:
            issues.append("pixi environment not properly set up")
    except FileNotFoundError:
        issues.append("pixi command not found")
    
    return issues

def run_in_pixi_env(command: List[str], project_path: Path = None, **kwargs):
    """Run a command in the pixi environment"""
    if project_path is None:
        project_path = get_moducomp_path()
    
    pixi_command = ['pixi', 'run'] + command
    kwargs['cwd'] = str(project_path)  # Set working directory
    return subprocess.run(pixi_command, **kwargs)

@mcp.tool()
def set_moducomp_path(path: str) -> str:
    """
    Set the path to the moducomp project directory
    
    Args:
        path: Path to the moducomp project directory
    """
    global MODUCOMP_PROJECT_PATH
    
    project_path = Path(path).expanduser().resolve()
    
    if not project_path.exists():
        return f"Error: Directory does not exist: {project_path}"
    
    pixi_toml = project_path / 'pixi.toml'
    if not pixi_toml.exists():
        return f"Error: pixi.toml not found in {project_path}. This doesn't appear to be a moducomp project."
    
    moducomp_py = project_path / 'moducomp.py'
    if not moducomp_py.exists():
        return f"Warning: moducomp.py not found in {project_path}. Is this the correct moducomp directory?"
    
    MODUCOMP_PROJECT_PATH = str(project_path)
    return f"Successfully set moducomp project path to: {project_path}"

@mcp.tool()
def get_current_moducomp_path() -> str:
    """Get the currently configured moducomp project path"""
    project_path = get_moducomp_path()
    exists = project_path.exists()
    is_valid = (project_path / 'pixi.toml').exists() and (project_path / 'moducomp.py').exists()
    
    return f"""Current moducomp project path: {project_path}
Directory exists: {exists}
Valid moducomp project: {is_valid}"""

@mcp.tool()
def check_environment() -> str:
    """Check the current pixi environment and dependencies"""
    
    project_path = get_moducomp_path()
    issues = check_pixi_environment(project_path)
    
    # Get pixi info
    pixi_info = {}
    try:
        result = subprocess.run(['pixi', 'info'], 
                              capture_output=True, text=True, cwd=project_path)
        if result.returncode == 0:
            pixi_info['pixi_info'] = result.stdout
    except:
        pixi_info['pixi_info'] = "Could not get pixi info"
    
    # Check if in moducomp directory
    is_moducomp = (project_path / 'pixi.toml').exists() and (project_path / 'moducomp.py').exists()
    
    info = {
        "mcp_server_directory": os.getcwd(),
        "moducomp_project_path": str(project_path),
        "moducomp_project_exists": project_path.exists(),
        "is_valid_moducomp_project": is_moducomp,
        "python_executable": sys.executable,
        "eggnog_data_dir": os.environ.get('EGGNOG_DATA_DIR', 'Not set'),
        "pixi_available": shutil.which('pixi') is not None,
        "issues": issues,
        **pixi_info
    }
    
    return json.dumps(info, indent=2)

@mcp.tool()
def setup_moducomp_project(project_path: str = None) -> str:
    """
    Clone and set up the moducomp project with pixi
    
    Args:
        project_path: Path where to clone the project (optional, uses configured path if not provided)
    """
    try:
        if not shutil.which('pixi'):
            return "Error: pixi is not installed. Please install pixi first: curl -fsSL https://pixi.sh/install.sh | sh"
        
        # If project_path is provided, clone there
        if project_path:
            project_path = Path(project_path).expanduser().resolve()
            logger.info(f"Cloning moducomp to {project_path}")
            
            # Check if directory already exists
            if project_path.exists():
                if (project_path / 'pixi.toml').exists():
                    logger.info(f"Directory {project_path} already exists and appears to be a moducomp project")
                else:
                    return f"Error: Directory {project_path} exists but doesn't appear to be a moducomp project"
            else:
                # Create parent directory if needed
                project_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Clone the repository
                result = subprocess.run([
                    'git', 'clone', 
                    'https://github.com/juanvillada/moducomp.git',
                    str(project_path)
                ], capture_output=True, text=True, check=True)
                
                logger.info(f"Successfully cloned moducomp to {project_path}")
            
            # Update the global path
            global MODUCOMP_PROJECT_PATH
            MODUCOMP_PROJECT_PATH = str(project_path)
        else:
            project_path = get_moducomp_path()
            
            # If using default path and it doesn't exist, clone it
            if not project_path.exists():
                logger.info(f"Cloning moducomp to default location: {project_path}")
                project_path.parent.mkdir(parents=True, exist_ok=True)
                
                result = subprocess.run([
                    'git', 'clone', 
                    'https://github.com/juanvillada/moducomp.git',
                    str(project_path)
                ], capture_output=True, text=True, check=True)
        
        # Check if we're in the right directory
        pixi_toml = project_path / 'pixi.toml'
        if not pixi_toml.exists():
            return f"Error: pixi.toml not found in {project_path}. Make sure this is the moducomp project directory."
        
        # Check if dependencies are already installed
        pixi_dir = project_path / '.pixi'
        if pixi_dir.exists():
            logger.info("Pixi dependencies appear to be already installed")
            return f"Moducomp project is already set up at {project_path}. Dependencies are installed."
        
        # Install dependencies using pixi
        logger.info("Installing dependencies with pixi...")
        logger.info("This may take a few minutes on first run...")
        
        result = subprocess.run(['pixi', 'install'], 
                              capture_output=True, text=True, check=True, cwd=project_path)
        
        return f"""Successfully set up moducomp project at {project_path}!

Installation output:
{result.stdout}

Next steps:
1. Use setup_eggnog_database to download the eggNOG database
2. Use run_moducomp_analysis to analyze your genomic data
"""
        
    except subprocess.CalledProcessError as e:
        return f"Error setting up project: {e.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def setup_eggnog_database(force_download: bool = False) -> str:
    """
    Download and setup the eggNOG database using pixi environment
    Database will be stored in: moducomp/eggnog-db
    
    Args:
        force_download: Force re-download even if database exists
    """
    try:
        project_path = get_moducomp_path()
        
        # Hardcode the database directory to be inside the moducomp project
        data_dir = project_path / "eggnog-db"
        
        # Check pixi environment
        issues = check_pixi_environment(project_path)
        if issues:
            return f"Pixi environment issues: {', '.join(issues)}"
        
        # Set environment variable
        os.environ['EGGNOG_DATA_DIR'] = str(data_dir)
        logger.info(f"Setting up eggNOG database in: {data_dir}")
        
        # Create directory if it doesn't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if database already exists (unless force_download)
        if not force_download:
            existing_files = list(data_dir.glob("*.dmnd"))
            if existing_files:
                return f"Database already exists in {data_dir}. Use force_download=True to re-download."
        
        # Run the download script in pixi environment
        logger.info("Starting eggNOG database download in pixi environment...")
        logger.info("This may take a while and requires significant disk space...")
        
        # Try different ways to call the download script
        download_commands = [
            ['download_eggnog_data.py'],  # As executable
            ['python', 'download_eggnog_data.py'],  # As python script
            ['python', '-m', 'eggnogdb.download_eggnog_data'],  # As module
        ]
        
        last_error = None
        for cmd in download_commands:
            try:
                logger.info(f"Trying command: {' '.join(cmd)}")
                
                result = run_in_pixi_env(
                    cmd,
                    project_path=project_path,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=7200,  # 2 hour timeout
                    env={**os.environ, 'EGGNOG_DATA_DIR': str(data_dir)}
                )
                
                return f"Successfully downloaded eggNOG database to {data_dir}\nCommand used: {' '.join(cmd)}\nOutput:\n{result.stdout}"
                
            except subprocess.CalledProcessError as e:
                last_error = f"Command '{' '.join(cmd)}' failed: {e.stderr}"
                logger.warning(last_error)
                continue
            except subprocess.TimeoutExpired:
                return f"Error: Download process timed out after 2 hours using command: {' '.join(cmd)}"
        
        # If all commands failed
        return f"All download attempts failed. Last error: {last_error}"
    
    except Exception as e:
        return f"Error: {str(e)}"
    

@mcp.tool()
def check_eggnog_database_status() -> str:
    """
    Check the status of the eggNOG database in the hardcoded location
    """
    try:
        project_path = get_moducomp_path()
        data_dir = project_path / "eggnog-db"
        
        # Check if directory exists
        if not data_dir.exists():
            return f"❌ Database directory does not exist: {data_dir}\nRun setup_eggnog_database() to create it."
        
        # Check for database files
        database_files = {
            "diamond": list(data_dir.glob("*.dmnd")),
            "annotations": list(data_dir.glob("*.annotations*")),
            "orthologs": list(data_dir.glob("*.orthologs*")),
            "other": list(data_dir.glob("*"))
        }
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
        total_size_gb = total_size / (1024**3)
        
        # Count files
        total_files = len([f for f in data_dir.rglob("*") if f.is_file()])
        
        status = f"""eggNOG Database Status:
Location: {data_dir}
Directory exists: ✅
Total files: {total_files}
Total size: {total_size_gb:.2f} GB

File breakdown:
- Diamond files (.dmnd): {len(database_files['diamond'])}
- Annotation files: {len(database_files['annotations'])}
- Ortholog files: {len(database_files['orthologs'])}
- Other files: {len(database_files['other']) - len(database_files['diamond']) - len(database_files['annotations']) - len(database_files['orthologs'])}
"""

        # Check environment variable
        env_var = os.environ.get('EGGNOG_DATA_DIR', 'Not set')
        status += f"\nEGGNOG_DATA_DIR environment variable: {env_var}"
        
        if env_var != str(data_dir):
            status += f"\n⚠️  Environment variable doesn't match database location"
        
        # List some key files
        if database_files['diamond']:
            status += f"\n\nDiamond database files found:"
            for f in database_files['diamond'][:5]:  # Show first 5
                status += f"\n  - {f.name}"
        
        return status
        
    except Exception as e:
        return f"Error checking database status: {str(e)}"
    

@mcp.tool()
def check_download_script_availability() -> str:
    """
    Check if download_eggnog_data.py is available in the pixi environment
    """
    try:
        project_path = get_moducomp_path()
        
        # Check pixi environment first
        issues = check_pixi_environment(project_path)
        if issues:
            return f"Pixi environment issues: {', '.join(issues)}"
        
        # Test different ways to access the download script
        test_commands = [
            ['download_eggnog_data.py', '--help'],
            ['python', 'download_eggnog_data.py', '--help'],
            ['python', '-m', 'eggnogdb.download_eggnog_data', '--help'],
            ['which', 'download_eggnog_data.py'],
            ['find', '.', '-name', 'download_eggnog_data.py']
        ]
        
        results = []
        for cmd in test_commands:
            try:
                logger.info(f"Testing command: {' '.join(cmd)}")
                
                result = run_in_pixi_env(
                    cmd,
                    project_path=project_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                status = "✅" if result.returncode == 0 else "❌"
                results.append(f"{status} {' '.join(cmd)}: {result.stdout.strip()[:200]}")
                
            except subprocess.CalledProcessError as e:
                results.append(f"❌ {' '.join(cmd)}: {e.stderr.strip()[:200]}")
            except Exception as e:
                results.append(f"❌ {' '.join(cmd)}: {str(e)[:200]}")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error testing download script: {str(e)}"

@mcp.tool()
def get_pixi_shell_instructions() -> str:
    """
    Get instructions for manually entering pixi shell and running commands
    """
    project_path = get_moducomp_path()
    
    return f"""To manually test the eggNOG download in pixi shell:

1. Open a terminal and navigate to the moducomp directory:
   cd {project_path}

2. Enter the pixi shell:
   pixi shell

3. Set the environment variable:
   export EGGNOG_DATA_DIR="/path/to/your/database/directory"

4. Try running the download script:
   download_eggnog_data.py
   
   Or if that doesn't work, try:
   python download_eggnog_data.py
   
   Or:
   python -m eggnogdb.download_eggnog_data

5. Check if the script exists:
   which download_eggnog_data.py
   find . -name "*download*eggnog*"

6. List available pixi commands:
   pixi list

Current configured project path: {project_path}
"""


@mcp.tool()
def run_moducomp_analysis(
    input_faa_files: List[str],
    output_dir: str,
    genome_names: Optional[List[str]] = None,
    max_combinations: int = 2
) -> str:
    """
    Run moducomp analysis using pixi environment
    Uses the hardcoded eggNOG database location: moducomp/eggnog-db
    
    Args:
        input_faa_files: List of paths to FAA (protein sequence) files
        output_dir: Directory to store analysis results
        genome_names: Optional list of genome names
        max_combinations: Maximum number of genomes to combine
    """
    try:
        project_path = get_moducomp_path()
        
        # Use hardcoded database path
        eggnog_data_dir = project_path / "eggnog-db"
        
        # Check pixi environment
        issues = check_pixi_environment(project_path)
        if issues:
            return f"Pixi environment issues: {', '.join(issues)}"
        
        # Check if database exists
        if not eggnog_data_dir.exists():
            return f"eggNOG database not found at {eggnog_data_dir}. Run setup_eggnog_database() first."
        
        # Set up environment
        env_vars = os.environ.copy()
        env_vars['EGGNOG_DATA_DIR'] = str(eggnog_data_dir)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert relative paths to absolute paths
        abs_input_files = [str(Path(f).resolve()) for f in input_faa_files]
        abs_output_dir = str(Path(output_dir).resolve())
        
        # Prepare command arguments for moducomp
        cmd = ['python', 'moducomp.py']  # Adjust based on actual script
        cmd.extend(['--input'] + abs_input_files)
        cmd.extend(['--output', abs_output_dir])
        cmd.extend(['--max-combinations', str(max_combinations)])
        
        if genome_names:
            cmd.extend(['--genome-names'] + genome_names)
        
        # Run moducomp in pixi environment
        logger.info("Running moducomp analysis in pixi environment...")
        logger.info(f"Using eggNOG database: {eggnog_data_dir}")
        
        result = run_in_pixi_env(
            cmd,
            project_path=project_path,
            capture_output=True,
            text=True,
            check=True,
            env=env_vars
        )
        
        return f"Analysis completed successfully!\nOutput directory: {abs_output_dir}\nUsed database: {eggnog_data_dir}\n{result.stdout}"
    
    except subprocess.CalledProcessError as e:
        return f"Error running moducomp: {e.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"
    
    

@mcp.tool()
def install_pixi_dependencies() -> str:
    """
    Install pixi dependencies for the moducomp project
    Equivalent to: cd moducomp && pixi install
    """
    try:
        project_path = get_moducomp_path()
        
        # Check if pixi is available
        if not shutil.which('pixi'):
            return "Error: pixi is not installed. Please install pixi first: curl -fsSL https://pixi.sh/install.sh | sh"
        
        # Check if moducomp directory exists
        if not project_path.exists():
            return f"Error: moducomp directory not found at: {project_path}. Run setup_moducomp_project first."
        
        # Check if pixi.toml exists
        pixi_toml = project_path / 'pixi.toml'
        if not pixi_toml.exists():
            return f"Error: pixi.toml not found in {project_path}. This doesn't appear to be a moducomp project."
        
        # Install dependencies using pixi
        logger.info(f"Installing pixi dependencies in: {project_path}")
        logger.info("This may take a few minutes...")
        
        result = subprocess.run(
            ['pixi', 'install'], 
            capture_output=True, 
            text=True, 
            check=True, 
            cwd=str(project_path)
        )
        
        return f"Successfully installed pixi dependencies!\nOutput:\n{result.stdout}"
        
    except subprocess.CalledProcessError as e:
        return f"Error installing pixi dependencies: {e.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def check_pixi_installation_status() -> str:
    """
    Check if pixi dependencies are properly installed
    """
    try:
        project_path = get_moducomp_path()
        
        if not project_path.exists():
            return f"❌ moducomp directory not found at: {project_path}"
        
        # Check if .pixi directory exists (created after pixi install)
        pixi_dir = project_path / '.pixi'
        pixi_installed = pixi_dir.exists()
        
        # Check pixi info
        pixi_info_available = False
        pixi_info_output = ""
        try:
            result = subprocess.run(
                ['pixi', 'info'], 
                capture_output=True, 
                text=True, 
                cwd=str(project_path)
            )
            if result.returncode == 0:
                pixi_info_available = True
                pixi_info_output = result.stdout
        except:
            pass
        
        # Check if we can list environments
        env_list_available = False
        try:
            result = subprocess.run(
                ['pixi', 'list'], 
                capture_output=True, 
                text=True, 
                cwd=str(project_path)
            )
            if result.returncode == 0:
                env_list_available = True
        except:
            pass
        
        status = f"""Pixi Installation Status for {project_path}:
{'✅' if pixi_installed else '❌'} .pixi directory exists: {pixi_installed}
{'✅' if pixi_info_available else '❌'} pixi info available: {pixi_info_available}
{'✅' if env_list_available else '❌'} pixi environment accessible: {env_list_available}

"""
        
        if pixi_info_available:
            status += f"Pixi Info:\n{pixi_info_output}"
        
        if not pixi_installed:
            status += "\n💡 Run install_pixi_dependencies to install dependencies"
        
        return status
        
    except Exception as e:
        return f"Error checking pixi installation: {str(e)}"
    

@mcp.tool()
def test_pixi_environment() -> str:
    """Test if the pixi environment is working correctly"""
    try:
        project_path = get_moducomp_path()
        
        # Test pixi info
        result = run_in_pixi_env(['python', '--version'], 
                               project_path=project_path,
                               capture_output=True, text=True)
        python_version = result.stdout.strip()
        
        # Test if eggnog-mapper is available
        result = run_in_pixi_env(['emapper.py', '--version'], 
                               project_path=project_path,
                               capture_output=True, text=True)
        emapper_version = result.stdout.strip() if result.returncode == 0 else "Not available"
        
        # Test if moducomp script exists
        moducomp_exists = (project_path / 'moducomp.py').exists()
        
        return f"""Pixi Environment Test Results:
Project Path: {project_path}
✅ Python version: {python_version}
{'✅' if result.returncode == 0 else '❌'} eggNOG-mapper: {emapper_version}
{'✅' if moducomp_exists else '❌'} moducomp.py exists: {moducomp_exists}
"""
        
    except Exception as e:
        return f"Error testing pixi environment: {str(e)}"

if __name__ == "__main__":
    logger.info("Starting moducomp MCP server with pixi support...")
    logger.info(f"Configured moducomp project path: {MODUCOMP_PROJECT_PATH}")
    mcp.run()