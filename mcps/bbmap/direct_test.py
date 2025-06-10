#!/usr/bin/env python3
"""
Direct BBMap Test - Step by step with full error handling
"""

import subprocess
import os
import sys
import time

def main():
    print("🧬 Direct BBMap Test")
    print("=" * 40)

    # Data files
    contigs = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/contigs.fa"
    reads = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/reads.fq.gz"
    output = "direct_test.sam"

    print(f"📁 Input files:")
    print(f"   Contigs: {contigs}")
    print(f"   Reads: {reads}")
    print(f"   Output: {output}")

    # Check files exist
    if not os.path.exists(contigs):
        print(f"❌ Contigs file not found!")
        return 1
    if not os.path.exists(reads):
        print(f"❌ Reads file not found!")
        return 1

    print(f"✅ Input files verified")

    # Test shifter access first
    print(f"\n🔧 Testing Shifter container access...")
    try:
        result = subprocess.run([
            "shifter", "--image", "bryce911/bbtools:latest",
            "bbmap.sh", "--help"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print(f"✅ Shifter container accessible")
            print(f"   BBMap version info found in help")
        else:
            print(f"❌ Shifter container test failed")
            print(f"   Return code: {result.returncode}")
            print(f"   Error: {result.stderr}")
            return 1

    except subprocess.TimeoutExpired:
        print(f"❌ Shifter container test timed out")
        return 1
    except Exception as e:
        print(f"❌ Error testing shifter: {e}")
        return 1

    # Build BBMap command
    cmd = [
        "shifter", "--image", "bryce911/bbtools:latest",
        "bbmap.sh",
        f"ref={contigs}",
        f"in={reads}",
        f"out={output}",
        "minid=0.85",
        "maxindel=100",
        "fast=t",
        "threads=auto",
        "overwrite=t",
        "-Xmx8g"  # Limit memory to 8GB
    ]

    print(f"\n🚀 Starting BBMap alignment...")
    print(f"Command: {' '.join(cmd)}")

    # Run BBMap with timeout
    start_time = time.time()
    try:
        # Run with shorter timeout for testing (5 minutes)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        elapsed = time.time() - start_time
        print(f"\n⏱️  Completed in {elapsed/60:.1f} minutes")
        print(f"📊 Return code: {result.returncode}")

        if result.returncode == 0:
            print(f"🎉 BBMap completed successfully!")

            # Check output file
            if os.path.exists(output):
                size = os.path.getsize(output) / (1024**2)
                print(f"✅ SAM file created: {size:.1f} MB")

                # Show first few lines
                print(f"\n📄 SAM file preview:")
                with open(output, 'r') as f:
                    for i, line in enumerate(f):
                        if i < 3:
                            print(f"   {line.strip()[:80]}...")
                        else:
                            break

                return 0
            else:
                print(f"❌ SAM file not created despite successful return code")

        else:
            print(f"❌ BBMap failed with return code: {result.returncode}")

        # Show output/error
        if result.stdout:
            print(f"\n📤 STDOUT:")
            print(result.stdout[-1000:])  # Last 1000 chars

        if result.stderr:
            print(f"\n📤 STDERR:")
            print(result.stderr[-1000:])  # Last 1000 chars

        return result.returncode

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"\n⏰ BBMap timed out after {elapsed/60:.1f} minutes")
        print(f"   This is normal for large datasets - try with more time")
        return 1

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Error after {elapsed/60:.1f} minutes: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
