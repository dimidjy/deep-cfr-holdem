# AI Behavior and Training Notes

## 🤖 Understanding AI Behavior

### Why Does the AI Fold When It Can Check?

This is **normal for early-stage training**. Here's why:

#### 1. **Learning Process**
- The AI starts with random behavior
- It gradually learns through self-play
- Early iterations (< 500) may show suboptimal plays
- The AI needs to explore all actions to learn their values

#### 2. **Action Space in Poker**
In PettingZoo's Texas Hold'em, actions are:
- **Action 0**: Fold
- **Action 1**: Check/Call
- **Action 2**: Raise Half Pot
- **Action 3**: Raise Full Pot
- **Action 4**: All-In

**Both Fold and Check/Call are always legal actions**, even when checking is free.

#### 3. **Why This Happens**
- Early in training, the AI hasn't learned that folding when you can check for free is always bad
- The strategy network outputs probabilities for all legal actions
- If the model hasn't converged, it might assign non-zero probability to folding

---

## 📈 Training Progression

### Expected Behavior by Iteration

| Iterations | Expected AI Behavior |
|------------|---------------------|
| 0-100 | Random-like play, frequent mistakes |
| 100-300 | Learning basic concepts, still makes obvious errors |
| 300-500 | Understands position, starting to avoid terrible plays |
| 500-1000 | Decent play, fewer obvious mistakes |
| 1000-2000 | Good strategic play, understands pot odds |
| 2000+ | Strong play, near-optimal in many situations |

### Signs of Good Training

✅ **Good AI behavior**:
- Rarely folds when checking is free (after 500+ iterations)
- Raises with strong hands
- Folds weak hands to raises
- Adjusts bet sizing based on pot
- Plays tighter from early position

❌ **Signs of insufficient training**:
- Folding when checking is free
- Calling with any two cards
- Random bet sizing
- Ignoring position
- Not responding to opponent aggression

---

## 🔧 How to Fix/Improve

### Option 1: Train Longer
```bash
# Increase iterations in config
{
  "training": {
    "num_iterations": 2000  // Instead of 1000
  }
}
```

### Option 2: Use Later Checkpoints
```bash
# Use a later checkpoint
python3 scripts/play_gui_vs_model.py --model models/checkpoint_iter_800.pt
```

### Option 3: Increase Traversals
```bash
# More experience per iteration
{
  "training": {
    "traversals_per_iteration": 2000  // Instead of 1000
  }
}
```

### Option 4: Add Reward Shaping (Advanced)

You could modify the training to penalize obviously bad plays, but this risks biasing the learning process.

---

## 🎯 Recommended Training Settings

### For Quick Testing (1-2 hours)
```json
{
  "training": {
    "num_iterations": 500,
    "traversals_per_iteration": 1000
  }
}
```
**Result**: Basic competent play

### For Good AI (4-8 hours)
```json
{
  "training": {
    "num_iterations": 1500,
    "traversals_per_iteration": 1500
  }
}
```
**Result**: Solid strategic play

### For Strong AI (8-16 hours)
```json
{
  "training": {
    "num_iterations": 3000,
    "traversals_per_iteration": 2000
  }
}
```
**Result**: Near-optimal play

---

## 📊 Evaluating AI Quality

### Method 1: Play Against It
- Play 100+ hands
- Track win rate
- Note obvious mistakes

### Method 2: Compare Checkpoints
```bash
# Early (should be weak)
python3 scripts/play_gui_vs_model.py --model models/checkpoint_iter_100.pt

# Late (should be strong)
python3 scripts/play_gui_vs_model.py --model models/checkpoint_iter_1000.pt
```

### Method 3: Self-Play Evaluation
Run the evaluation in the training script:
```python
eval_results = trainer.evaluate(num_games=1000)
```

Look for:
- **Converging rewards**: Players should have similar average rewards
- **Decreasing variance**: More consistent play over time

---

## 🐛 Common Issues

### Issue: AI Always Folds
**Cause**: Very early training or corrupted model
**Fix**: Train for at least 300 iterations

### Issue: AI Always Calls
**Cause**: Model learned calling is safe
**Fix**: Train longer, increase exploration

### Issue: AI Makes Random Bets
**Cause**: Insufficient training data
**Fix**: Increase `traversals_per_iteration`

### Issue: AI Doesn't Adapt
**Cause**: Model capacity too small or not enough training
**Fix**: 
- Increase network size in config
- Train for more iterations

---

## 🎓 Technical Explanation

### Why CFR Takes Time to Converge

**Counterfactual Regret Minimization (CFR)** works by:
1. Playing many games
2. Computing regret for not taking each action
3. Adjusting strategy to minimize regret
4. Repeating until convergence

**Key insight**: The AI needs to try folding when checking is free many times, realize it's always bad, and build up enough regret to stop doing it.

### The Math
- **Regret for folding when can check**: Always negative (you lose the pot)
- **Regret for checking**: Zero or positive (you stay in the hand)
- **After N iterations**: Regret for folding accumulates, probability decreases

**Convergence time**: O(1/√N) where N is number of iterations

This means:
- 100 iterations: Still learning basics
- 1000 iterations: Good approximation
- 10000 iterations: Near-optimal

---

## 💡 Best Practices

### For Development
1. Use small iterations (100-200) for quick tests
2. Check behavior at each checkpoint
3. Increase iterations when behavior looks reasonable

### For Production
1. Train for 2000+ iterations
2. Evaluate against random and previous checkpoints
3. Use the best-performing checkpoint, not necessarily the final one

### For Research
1. Track exploitability over time
2. Compare against known optimal strategies
3. Test against human players

---

## 📚 Further Reading

- **CFR Algorithm**: [Original Paper](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf)
- **Deep CFR**: [DeepStack Paper](https://www.science.org/doi/10.1126/science.aam6960)
- **PettingZoo**: [Documentation](https://pettingzoo.farama.org/)

---

## ✅ Summary

**The AI folding when it can check is normal during early training.**

**Solutions**:
1. ✅ Train for 500+ iterations
2. ✅ Use later checkpoints
3. ✅ Increase traversals per iteration
4. ✅ Be patient - CFR takes time to converge

**Remember**: A weak early-stage AI is a sign of proper learning, not a bug! 🎓
