#!/bin/bash

echo "======================================================================"
echo "POKER TRAINING ENGINE SETUP"
echo "======================================================================"

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Creating necessary directories..."
mkdir -p models
mkdir -p data/logs
mkdir -p data/spots
mkdir -p data/training

echo ""
echo "Running setup check..."
python scripts/check_setup.py

echo ""
echo "======================================================================"
echo "SETUP COMPLETE"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Quick test: python scripts/quick_test.py"
echo "  2. Run tests: pytest tests/ -v"
echo "  3. GUI demo: python scripts/play_gui.py"
echo "  4. CFR demo: python scripts/demo_cfr.py"
echo ""
