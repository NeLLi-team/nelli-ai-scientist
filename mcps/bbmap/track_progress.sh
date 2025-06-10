#!/bin/bash
"""
BBMap Progress Tracker

Track the progress of your BBMap run in real-time.
"""

echo "🔍 BBMap Progress Tracker"
echo "=========================="
echo "Monitoring your microbiome data processing..."
echo ""

EXPECTED_OUTPUT="quick_test_alignment.sam"
STATS_FILE="mapping_stats.txt"
START_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date +%s)
    RUNTIME=$((CURRENT_TIME - START_TIME))
    RUNTIME_MIN=$((RUNTIME / 60))

    echo "⏰ Runtime: ${RUNTIME}s (${RUNTIME_MIN}m)"

    # Check if BBMap process is running
    if pgrep -f "bbmap.sh" > /dev/null; then
        echo "🔄 BBMap process: RUNNING"

        # Check output file size if it exists
        if [ -f "$EXPECTED_OUTPUT" ]; then
            SIZE_MB=$(du -m "$EXPECTED_OUTPUT" 2>/dev/null | cut -f1)
            echo "📄 SAM file: ${SIZE_MB}MB (growing...)"
        else
            echo "📄 SAM file: Not yet created"
        fi

        # Check stats file
        if [ -f "$STATS_FILE" ]; then
            SIZE_KB=$(du -k "$STATS_FILE" 2>/dev/null | cut -f1)
            echo "📊 Stats file: ${SIZE_KB}KB"
        else
            echo "📊 Stats file: Not yet created"
        fi

    else
        echo "✅ BBMap process: COMPLETED"
        break
    fi

    echo "---"
    sleep 30  # Check every 30 seconds
done

echo ""
echo "🎉 BBMap processing completed!"

# Final file check
if [ -f "$EXPECTED_OUTPUT" ]; then
    FINAL_SIZE=$(du -m "$EXPECTED_OUTPUT" | cut -f1)
    LINE_COUNT=$(wc -l < "$EXPECTED_OUTPUT")
    echo "✅ Final SAM file: ${FINAL_SIZE}MB with ${LINE_COUNT} lines"
else
    echo "❌ SAM file not found"
fi
