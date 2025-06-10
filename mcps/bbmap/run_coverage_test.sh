#!/bin/bash
# Direct Coverage Analysis Test

echo "🧬 BBMap Coverage Analysis Test" > coverage_test.log
echo "Timestamp: $(date)" >> coverage_test.log

# Input files
SAM_FILE="direct_test.sam"
REF_FILE="/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/contigs.fa"
COVERAGE_OUT="coverage_analysis_coverage.txt"
STATS_OUT="coverage_analysis_stats.txt"
HIST_OUT="coverage_histogram.txt"

echo "📁 Input files:" >> coverage_test.log
echo "  SAM: $SAM_FILE" >> coverage_test.log
echo "  Reference: $REF_FILE" >> coverage_test.log

# Check files exist
if [ ! -f "$SAM_FILE" ]; then
    echo "❌ SAM file not found!" >> coverage_test.log
    exit 1
fi

if [ ! -f "$REF_FILE" ]; then
    echo "❌ Reference file not found!" >> coverage_test.log
    exit 1
fi

SAM_SIZE=$(du -h "$SAM_FILE" | cut -f1)
REF_SIZE=$(du -h "$REF_FILE" | cut -f1)
echo "✅ Files found - SAM: $SAM_SIZE, Reference: $REF_SIZE" >> coverage_test.log

# Run coverage analysis
echo "🔍 Starting coverage analysis at $(date)..." >> coverage_test.log

shifter --image bryce911/bbtools:latest pileup.sh \
    in="$SAM_FILE" \
    ref="$REF_FILE" \
    out="$COVERAGE_OUT" \
    stats="$STATS_OUT" \
    hist="$HIST_OUT" \
    -Xmx8g \
    > coverage_output.log 2>&1

EXIT_CODE=$?
echo "Coverage analysis completed at $(date) with exit code: $EXIT_CODE" >> coverage_test.log

if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 Coverage analysis successful!" >> coverage_test.log

    # Check output files
    for file in "$COVERAGE_OUT" "$STATS_OUT" "$HIST_OUT"; do
        if [ -f "$file" ]; then
            SIZE=$(du -h "$file" | cut -f1)
            LINES=$(wc -l < "$file")
            echo "✅ $file: $SIZE ($LINES lines)" >> coverage_test.log
        else
            echo "❌ $file not found" >> coverage_test.log
        fi
    done

    # Show sample of stats file
    if [ -f "$STATS_OUT" ]; then
        echo "📊 Sample statistics:" >> coverage_test.log
        head -10 "$STATS_OUT" >> coverage_test.log
    fi

else
    echo "❌ Coverage analysis failed with exit code: $EXIT_CODE" >> coverage_test.log
fi

echo "Coverage analysis output (last 1000 chars):" >> coverage_test.log
tail -c 1000 coverage_output.log >> coverage_test.log

echo "Test completed at $(date)" >> coverage_test.log
