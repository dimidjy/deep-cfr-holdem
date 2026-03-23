#!/bin/bash
set -e

echo "=========================================="
echo "  Training supervised models: 5p, 4p, 3p, 2p"
echo "=========================================="

# ── Phase 1: Build spot DB + training data for 5p and 4p in parallel ──────
echo ""
echo "========== PHASE 1: 5p + 4p data (parallel) =========="

(
  echo "[5p] Step 1: Building spot DB..."
  python3 scripts/build_spot_db.py \
    --num-players 5 --workers 11 --output data/spot_db_5p.npz

  echo "[5p] Step 2: Building training data..."
  python3 scripts/build_training_data.py \
    --good-players data/good_players.txt \
    --bad-players data/bad_players.txt \
    --spot-db data/spot_db_5p.npz \
    --fold-ratio 0.15 \
    --min-lose-pct 60 \
    --spot-k 100 \
    --max-made-hand 0.25 \
    --workers 11 \
    --num-players 5 \
    --output data/training_data_5p.npz

  echo "[5p] Steps 1-2 DONE"
) > logs/train_5p.log 2>&1 &
PID_5P=$!

(
  echo "[4p] Step 1: Building spot DB..."
  python3 scripts/build_spot_db.py \
    --num-players 4 --workers 11 --output data/spot_db_4p.npz

  echo "[4p] Step 2: Building training data..."
  python3 scripts/build_training_data.py \
    --good-players data/good_players.txt \
    --bad-players data/bad_players.txt \
    --spot-db data/spot_db_4p.npz \
    --fold-ratio 0.15 \
    --min-lose-pct 60 \
    --spot-k 100 \
    --max-made-hand 0.25 \
    --workers 11 \
    --num-players 4 \
    --output data/training_data_4p.npz

  echo "[4p] Steps 1-2 DONE"
) > logs/train_4p.log 2>&1 &
PID_4P=$!

echo "  5p PID=$PID_5P, 4p PID=$PID_4P"
echo "  Logs: logs/train_5p.log, logs/train_4p.log"
echo "  Waiting..."
wait $PID_5P $PID_4P
echo "========== PHASE 1 COMPLETE =========="

# ── Phase 2: Build spot DB + training data for 3p and 2p in parallel ──────
echo ""
echo "========== PHASE 2: 3p + 2p data (parallel) =========="

(
  echo "[3p] Step 1: Building spot DB..."
  python3 scripts/build_spot_db.py \
    --num-players 3 --workers 11 --output data/spot_db_3p.npz &&
  python3 scripts/build_training_data.py \
    --good-players data/good_players.txt \
    --bad-players data/bad_players.txt \
    --spot-db data/spot_db_3p.npz \
    --fold-ratio 0.15 \
    --min-lose-pct 60 \
    --spot-k 100 \
    --max-made-hand 0.25 \
    --workers 11 \
    --num-players 3 \
    --output data/training_data_3p.npz

  echo "[3p] Steps 1-2 DONE"
) > logs/train_3p.log 2>&1 &
PID_3P=$!

(
  echo "[2p] Step 1: Building spot DB..."
  python3 scripts/build_spot_db.py \
    --num-players 2 --workers 11 --output data/spot_db_2p.npz &&
  python3 scripts/build_training_data.py \
    --good-players data/good_players.txt \
    --bad-players data/bad_players.txt \
    --spot-db data/spot_db_2p.npz \
    --fold-ratio 0.15 \
    --min-lose-pct 60 \
    --spot-k 100 \
    --max-made-hand 0.25 \
    --workers 11 \
    --num-players 2 \
    --output data/training_data_2p.npz

  echo "[2p] Steps 1-2 DONE"
) > logs/train_2p.log 2>&1 &
PID_2P=$!

echo "  3p PID=$PID_3P, 2p PID=$PID_2P"
echo "  Logs: logs/train_3p.log, logs/train_2p.log"
echo "  Waiting..."
wait $PID_3P $PID_2P
echo "========== PHASE 2 COMPLETE =========="

# ── Phase 3: Train models one by one (GPU) ───────────────────────────────
echo ""
echo "========== PHASE 3: Training models (sequential, GPU) =========="

echo "[5p] Training..."
python3 scripts/train_supervised.py \
  --training-data data/training_data_5p.npz \
  --device cuda \
  --hidden-layers 512,512,256 \
  --epochs 100 \
  --lr 0.0003 \
  --output models/supervised_5p.pt

echo "[4p] Training..."
python3 scripts/train_supervised.py \
  --training-data data/training_data_4p.npz \
  --device cuda \
  --hidden-layers 512,512,256 \
  --epochs 100 \
  --lr 0.0003 \
  --output models/supervised_4p.pt

echo "[3p] Training..."
python3 scripts/train_supervised.py \
  --training-data data/training_data_3p.npz \
  --device cuda \
  --hidden-layers 512,512,256 \
  --epochs 100 \
  --lr 0.0003 \
  --output models/supervised_3p.pt

echo "[2p] Training..."
python3 scripts/train_supervised.py \
  --training-data data/training_data_2p.npz \
  --device cuda \
  --hidden-layers 512,512,256 \
  --epochs 100 \
  --lr 0.0003 \
  --output models/supervised_2p.pt

echo ""
echo "=========================================="
echo "  ALL DONE"
echo "=========================================="
echo "Models saved:"
echo "  models/supervised_5p.pt"
echo "  models/supervised_4p.pt"
echo "  models/supervised_3p.pt"
echo "  models/supervised_2p.pt"
