#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating data directories..."
mkdir -p models data/spots data/training

echo "Running tests..."
python3 -m pytest tests/ -q

echo ""
echo "Next steps:"
echo "  make train      # Deep CFR training"
echo "  make selfplay   # self-play statistics for a checkpoint"
echo "  make webui      # job runner and spot explorer"
