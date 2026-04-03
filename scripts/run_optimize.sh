#!/bin/bash
cd /app/poker

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/optimize_${TIMESTAMP}.log"
mkdir -p logs

echo "=== Poker AI Optimization Loop ==="
echo "Started: $(date)"
echo "Log: ${LOG_FILE}"
echo ""

claude -p "Read the file /app/poker/scripts/optimize_loop.md and follow all instructions in it. Start now." \
    --allowedTools 'Bash(command:*)' 'Read' 'Write' 'Edit' 'Grep' 'Glob' \
    --output-format stream-json \
    --verbose \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "Finished: $(date)"
