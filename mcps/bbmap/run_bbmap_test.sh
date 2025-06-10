#!/bin/bash
# BBMap Test Script - Output to files

echo "🧬 BBMap Test Script Starting..." > bbmap_test.log
echo "Timestamp: $(date)" >> bbmap_test.log

# Data files
CONTIGS="/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/contigs.fa"
READS="/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/reads.fq.gz"
OUTPUT="bbmap_alignment.sam"

echo "Input files:" >> bbmap_test.log
echo "  Contigs: $CONTIGS" >> bbmap_test.log
echo "  Reads: $READS" >> bbmap_test.log
echo "  Output: $OUTPUT" >> bbmap_test.log

# Check files exist
if [ ! -f "$CONTIGS" ]; then
    echo "❌ Contigs file not found!" >> bbmap_test.log
    exit 1
fi

if [ ! -f "$READS" ]; then
    echo "❌ Reads file not found!" >> bbmap_test.log
    exit 1
fi

echo "✅ Input files verified" >> bbmap_test.log

# Get file sizes
CONTIGS_SIZE=$(du -m "$CONTIGS" | cut -f1)
READS_SIZE=$(du -m "$READS" | cut -f1)
echo "File sizes: Contigs=${CONTIGS_SIZE}MB, Reads=${READS_SIZE}MB" >> bbmap_test.log

# Test shifter first
echo "Testing shifter container..." >> bbmap_test.log
timeout 30 shifter --image bryce911/bbtools:latest bbmap.sh --help > shifter_test.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Shifter container accessible" >> bbmap_test.log
else
    echo "❌ Shifter container test failed" >> bbmap_test.log
    cat shifter_test.log >> bbmap_test.log
    exit 1
fi

# Run BBMap alignment
echo "Starting BBMap alignment at $(date)..." >> bbmap_test.log

shifter --image bryce911/bbtools:latest bbmap.sh \
    ref="$CONTIGS" \
    in="$READS" \
    out="$OUTPUT" \
    minid=0.85 \
    maxindel=100 \
    fast=t \
    threads=auto \
    overwrite=t \
    -Xmx8g \
    > bbmap_output.log 2>&1

EXIT_CODE=$?
echo "BBMap completed at $(date) with exit code: $EXIT_CODE" >> bbmap_test.log

if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 BBMap completed successfully!" >> bbmap_test.log

    if [ -f "$OUTPUT" ]; then
        OUTPUT_SIZE=$(du -m "$OUTPUT" | cut -f1)
        echo "✅ SAM file created: ${OUTPUT_SIZE}MB" >> bbmap_test.log

        # Count lines in SAM file
        TOTAL_LINES=$(wc -l < "$OUTPUT")
        HEADER_LINES=$(grep -c "^@" "$OUTPUT")
        ALIGNMENT_LINES=$((TOTAL_LINES - HEADER_LINES))

        echo "📊 SAM Statistics:" >> bbmap_test.log
        echo "  Total lines: $TOTAL_LINES" >> bbmap_test.log
        echo "  Header lines: $HEADER_LINES" >> bbmap_test.log
        echo "  Alignment lines: $ALIGNMENT_LINES" >> bbmap_test.log

        # Show first few lines
        echo "📄 SAM file preview:" >> bbmap_test.log
        head -5 "$OUTPUT" >> bbmap_test.log

    else
        echo "❌ SAM file not created" >> bbmap_test.log
    fi
else
    echo "❌ BBMap failed with exit code: $EXIT_CODE" >> bbmap_test.log
fi

echo "BBMap output (last 1000 characters):" >> bbmap_test.log
tail -c 1000 bbmap_output.log >> bbmap_test.log

echo "Test completed at $(date)" >> bbmap_test.log
