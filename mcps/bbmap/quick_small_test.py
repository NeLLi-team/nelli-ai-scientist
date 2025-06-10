#!/usr/bin/env python3
"""
Quick BBMap Test - Small data subset first
"""

import subprocess
import os
import sys
import time

def main():
    print("🧬 Quick BBMap Test - Small Dataset")
    print("=" * 45)

    # Original data files
    contigs_orig = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/contigs.fa"
    reads_orig = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/reads.fq.gz"

    # Create small test files
    contigs_small = "test_contigs_small.fa"
    reads_small = "test_reads_small.fq"
    output = "test_small.sam"

    print(f"📁 Creating small test files...")

    # Create small contigs file (first 1000 lines)
    try:
        with open(contigs_orig, 'r') as f_in, open(contigs_small, 'w') as f_out:
            for i, line in enumerate(f_in):
                if i >= 1000:  # First 1000 lines
                    break
                f_out.write(line)
        print(f"✅ Small contigs file created: {os.path.getsize(contigs_small)/1024:.1f} KB")
    except Exception as e:
        print(f"❌ Error creating small contigs: {e}")
        return 1

    # Create small reads file (first 10000 lines from gunzipped)
    try:
        result = subprocess.run([
            "gunzip", "-c", reads_orig
        ], stdout=subprocess.PIPE, text=True)

        if result.returncode == 0:
            lines = result.stdout.split('\n')
            with open(reads_small, 'w') as f_out:
                for i, line in enumerate(lines):
                    if i >= 10000:  # First 10k lines = ~2500 reads
                        break
                    f_out.write(line + '\n')
            print(f"✅ Small reads file created: {os.path.getsize(reads_small)/1024:.1f} KB")
        else:
            print(f"❌ Error extracting reads")
            return 1
    except Exception as e:
        print(f"❌ Error creating small reads: {e}")
        return 1

    # Test BBMap with small files
    print(f"\n🚀 Testing BBMap with small files...")

    cmd = [
        "shifter", "--image", "bryce911/bbtools:latest",
        "bbmap.sh",
        f"ref={contigs_small}",
        f"in={reads_small}",
        f"out={output}",
        "overwrite=t",
        "-Xmx2g"  # Only 2GB memory
    ]

    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # 2 min timeout

        elapsed = time.time() - start_time
        print(f"\n⏱️  Completed in {elapsed:.1f} seconds")
        print(f"📊 Return code: {result.returncode}")

        if result.returncode == 0:
            print(f"🎉 BBMap small test successful!")

            if os.path.exists(output):
                size = os.path.getsize(output) / 1024
                print(f"✅ SAM file created: {size:.1f} KB")

                # Count alignments
                with open(output, 'r') as f:
                    header_lines = 0
                    alignment_lines = 0
                    for line in f:
                        if line.startswith('@'):
                            header_lines += 1
                        elif line.strip():
                            alignment_lines += 1

                print(f"📊 SAM Statistics:")
                print(f"   Header lines: {header_lines}")
                print(f"   Alignment lines: {alignment_lines}")

                print(f"\n✅ BBMap is working! Ready for full dataset")
                return 0
            else:
                print(f"❌ SAM file not created")
        else:
            print(f"❌ BBMap failed: {result.returncode}")

        # Show output
        if result.stdout:
            print(f"\nSTDOUT (last 500 chars):")
            print(result.stdout[-500:])
        if result.stderr:
            print(f"\nSTDERR (last 500 chars):")
            print(result.stderr[-500:])

        return result.returncode

    except subprocess.TimeoutExpired:
        print(f"\n⏰ Timed out after 2 minutes")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    finally:
        # Cleanup small files
        for f in [contigs_small, reads_small]:
            if os.path.exists(f):
                os.remove(f)
                print(f"🗑️  Cleaned up {f}")

if __name__ == "__main__":
    sys.exit(main())
