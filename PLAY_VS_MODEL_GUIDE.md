# Play Against Trained Models - Guide

## 🎮 Quick Start

### Method 1: Using Python Script (Recommended)

```bash
# List available models
python3 scripts/play_gui_vs_model.py --list

# Play against a specific model
python3 scripts/play_gui_vs_model.py --model models/final_model.pt

# Play 6-max against a model
python3 scripts/play_gui_vs_model.py --model models/final_model.pt --players 6
```

### Method 2: Using Shell Script

```bash
# List available models
./scripts/play_vs_model.sh --list

# Play against a model
./scripts/play_vs_model.sh models/final_model.pt
```

---

## 📋 Available Models

After training, you'll have these model files in `models/`:

| Model File | Description |
|------------|-------------|
| `final_model.pt` | Final model after all training iterations |
| `checkpoint_iter_100.pt` | Model after 100 iterations |
| `checkpoint_iter_200.pt` | Model after 200 iterations |
| `checkpoint_iter_500.pt` | Model after 500 iterations |
| `checkpoint_iter_1000.pt` | Model after 1000 iterations |

**Tip**: Later checkpoints are generally stronger, but early checkpoints can be interesting to see AI learning progression!

---

## 🎯 Usage Examples

### Play Against Final Model
```bash
python3 scripts/play_gui_vs_model.py --model models/final_model.pt
```

### Play Against Mid-Training Model
```bash
python3 scripts/play_gui_vs_model.py --model models/checkpoint_iter_500.pt
```

### Compare Different Models
Play multiple sessions against different checkpoints to see improvement:

```bash
# Play vs early model (weaker)
python3 scripts/play_gui_vs_model.py --model models/checkpoint_iter_100.pt

# Play vs final model (stronger)
python3 scripts/play_gui_vs_model.py --model models/final_model.pt
```

---

## 🎮 Controls

Once the GUI opens:

| Control | Action |
|---------|--------|
| **Mouse Click** | Select action buttons |
| **R** | Start new hand |
| **SPACE** | Random action (for testing) |
| **Q** | Quit |

---

## 📊 What You'll See

### Game Info Panel (Top Right)
- Current hand number
- Your win/loss record
- Pot size
- Your chips
- AI chips

### Action History (Top Left)
- Last 8 actions taken
- Color-coded:
  - 🟡 Yellow: Hand markers
  - 🟢 Green: Your actions
  - 🔴 Red: AI actions

### Game Status (Top Center)
- Current game stage (Pre-Flop, Flop, Turn, River)
- Whose turn it is
- Game results (win/loss/tie)

---

## 🤖 AI Behavior

### Random AI (Default)
```bash
python3 scripts/play_gui.py
```
- Takes random legal actions
- Good for testing game mechanics
- Not strategic

### Trained AI (With Model)
```bash
python3 scripts/play_gui_vs_model.py --model models/final_model.pt
```
- Uses learned strategy from training
- Makes strategic decisions based on:
  - Hand strength
  - Position
  - Pot odds
  - Opponent patterns (learned during training)
- Strength depends on training quality and iterations

---

## 🔧 Troubleshooting

### "Model file not found"

**Solution**: Check if the model exists
```bash
ls -lh models/
```

If no models exist, you need to train first:
```bash
python3 scripts/train.py
```

### "Failed to load model"

**Possible causes**:
1. **Wrong number of players**: Model trained for 2 players, trying to load for 6
   ```bash
   # If model was trained with 6 players:
   python3 scripts/play_gui_vs_model.py --model models/final_model.pt --players 6
   ```

2. **Corrupted model file**: Re-train or use a different checkpoint

3. **Incompatible model format**: Model from older version

### GUI doesn't open

**Check**:
```bash
# Test pygame
python3 -c "import pygame; print('Pygame OK')"

# Test basic GUI
python3 scripts/play_gui.py
```

---

## 📈 Evaluating AI Strength

### Play Multiple Hands
- Play at least 50-100 hands to get a good sample
- Track your win rate
- Note patterns in AI play

### Compare Checkpoints
Play against different training checkpoints:

```bash
# Early training (weaker)
python3 scripts/play_gui_vs_model.py --model models/checkpoint_iter_100.pt

# Mid training
python3 scripts/play_gui_vs_model.py --model models/checkpoint_iter_500.pt

# Final (strongest)
python3 scripts/play_gui_vs_model.py --model models/final_model.pt
```

### Signs of Good Training

**Strong AI shows**:
- Folding weak hands
- Raising with strong hands
- Appropriate bet sizing
- Position awareness
- Adapting to pot odds

**Weak AI shows**:
- Random-looking play
- Calling too much
- Not raising strong hands
- Ignoring position

---

## 🎓 Advanced Usage

### Test Multiple Models in Sequence

Create a test script:

```bash
#!/bin/bash
for model in models/checkpoint_iter_*.pt; do
    echo "Testing $model"
    python3 scripts/play_gui_vs_model.py --model "$model"
done
```

### Record Results

Keep notes on performance:
```
Checkpoint 100: Won 45/100 hands (45% win rate)
Checkpoint 500: Won 38/100 hands (38% win rate)
Final model:    Won 35/100 hands (35% win rate)
```

Lower win rate against later models = AI is getting stronger!

---

## 📝 Model Information

### What's Saved in a Model?

Each `.pt` file contains:
- **Advantage networks** (2 networks, one per player)
- **Strategy networks** (2 networks, one per player)
- **Optimizer states**
- **Training iteration number**

### Model Size

Typical model sizes:
- **Small networks** (128, 128): ~5-10 MB
- **Medium networks** (256, 256, 128): ~20-30 MB
- **Large networks** (512, 512, 256): ~50-80 MB

---

## 🚀 Quick Reference

### List Models
```bash
python3 scripts/play_gui_vs_model.py --list
```

### Play vs Model
```bash
python3 scripts/play_gui_vs_model.py --model models/final_model.pt
```

### Play vs Random (No Model)
```bash
python3 scripts/play_gui.py
```

### Train a Model First
```bash
python3 scripts/train.py
```

---

## 💡 Tips

1. **Start with random AI** to learn the interface
2. **Train for at least 500 iterations** before expecting strong play
3. **Compare checkpoints** to see learning progression
4. **Play many hands** (50+) for meaningful evaluation
5. **Note AI mistakes** - they indicate areas needing more training
6. **Try different positions** - AI should play differently from different positions

---

## 🎉 Enjoy!

You now have a complete system to:
- ✅ Train poker AI models
- ✅ Load and play against them
- ✅ Evaluate their strength
- ✅ Compare different training stages

**Have fun testing your AI!** 🃏🤖
