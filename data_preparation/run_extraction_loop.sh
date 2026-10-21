#!/bin/bash
# Runs extract_oasis_structures.py repeatedly every 10 minutes while the
# FastSurfer batch job (PID check via run_prediction.py) is still running,
# then runs it one final time after FastSurfer finishes.

LOGFILE=/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/run_extraction_loop.log
echo "Starting extraction loop at $(date)" >> "$LOGFILE"

while pgrep -f "FastSurferCNN/run_prediction.py" > /dev/null; do
    echo "[$(date)] Running incremental extraction pass..." >> "$LOGFILE"
    conda run -n fastsurfer python3 /media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/extract_oasis_structures.py >> "$LOGFILE" 2>&1
    sleep 600
done

echo "[$(date)] FastSurfer batch finished. Running final extraction pass..." >> "$LOGFILE"
conda run -n fastsurfer python3 /media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/extract_oasis_structures.py >> "$LOGFILE" 2>&1
echo "[$(date)] Extraction loop finished. Running final split merge..." >> "$LOGFILE"
conda run -n fastsurfer python3 /media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/merge_oasis_splits.py >> "$LOGFILE" 2>&1
echo "[$(date)] ALL DONE." >> "$LOGFILE"
