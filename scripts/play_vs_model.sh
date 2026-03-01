#!/bin/bash

echo "=========================================="
echo "Play Poker vs Trained AI"
echo "=========================================="
echo ""

if [ "$1" == "--list" ] || [ "$1" == "-l" ]; then
    python3 scripts/play_gui_vs_model.py --list
    exit 0
fi

if [ -z "$1" ]; then
    echo "Available models:"
    python3 scripts/play_gui_vs_model.py --list
    echo ""
    echo "Usage:"
    echo "  ./scripts/play_vs_model.sh <model_file>"
    echo ""
    echo "Examples:"
    echo "  ./scripts/play_vs_model.sh models/final_model.pt"
    echo "  ./scripts/play_vs_model.sh models/checkpoint_iter_500.pt"
    echo ""
    exit 1
fi

python3 scripts/play_gui_vs_model.py --model "$1"
