#!/usr/bin/env python3
"""
BBMap Progress Monitor

This script monitors the progress of your BBMap run and provides updates.
"""

import os
import time
import subprocess

def monitor_bbmap_progress():
    """Monitor BBMap execution progress"""

    print("🔍 BBMap Progress Monitor")
    print("=" * 40)

    # Files to monitor
    results_dir = "microbiome_bbmap_results"
    sam_file = os.path.join(results_dir, "microbiome_alignment.sam")
    stats_file = "mapping_stats.txt"

    start_time = time.time()

    while True:
        current_time = time.time()
        runtime = current_time - start_time

        print(f"\n⏰ Runtime: {runtime:.0f} seconds ({runtime/60:.1f} minutes)")

        # Check for output files
        if os.path.exists(sam_file):
            sam_size = os.path.getsize(sam_file) / (1024**2)  # MB
            print(f"📄 SAM file: {sam_size:.1f} MB (growing...)")
        else:
            print("📄 SAM file: Not yet created")

        if os.path.exists(stats_file):
            stats_size = os.path.getsize(stats_file) / 1024  # KB
            print(f"📊 Stats file: {stats_size:.1f} KB")
        else:
            print("📊 Stats file: Not yet created")

        # Check if BBMap process is still running
        result = subprocess.run(
            ["pgrep", "-f", "bbmap.sh"],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print("🔄 BBMap process: RUNNING")
            print("   (This is normal - large files take time)")
        else:
            print("✅ BBMap process: COMPLETED")
            break

        # Wait before next check
        time.sleep(30)  # Check every 30 seconds

    print(f"\n🎉 BBMap execution completed!")

    # Final file check
    if os.path.exists(sam_file):
        final_size = os.path.getsize(sam_file) / (1024**2)  # MB
        print(f"✅ Final SAM file: {final_size:.1f} MB")

        # Quick SAM file validation
        with open(sam_file, 'r') as f:
            line_count = 0
            header_lines = 0
            alignment_lines = 0

            for line in f:
                line_count += 1
                if line.startswith('@'):
                    header_lines += 1
                else:
                    alignment_lines += 1

                # Don't count all lines for very large files
                if line_count > 100000:
                    break

            print(f"📈 SAM file preview:")
            print(f"   Header lines: {header_lines}")
            print(f"   Alignment lines: {alignment_lines}+")
            print(f"   Total lines checked: {line_count}")

if __name__ == "__main__":
    try:
        monitor_bbmap_progress()
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")
