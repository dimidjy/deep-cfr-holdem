# Quick Start: Parallel Training

## 🚀 30-Second Setup

### 1. Test Your System
```bash
python3 scripts/test_parallel.py
```

### 2. Run Training

**CPU Only (8+ cores)**:
```bash
python3 scripts/train_parallel.py
```

**GPU Server**:
```bash
python3 scripts/train_async.py
```

---

## ⚡ Performance Gains

| Your Hardware | Use This | Expected Speedup |
|---------------|----------|------------------|
| 4-8 CPU cores | `train_parallel.py` | 3-4x faster |
| 8-16 CPU cores | `train_parallel.py` | 4-6x faster |
| CPU + GPU | `train_async.py` | 8-15x faster |

---

## ⚙️ Quick Config

Edit `configs/training_config_parallel.json`:

```json
{
  "training": {
    "device": "cuda",        // Change to "cpu" if no GPU
    "num_workers": 16,       // Set to: (your CPU cores - 2)
    "batch_size": 256        // 128 for CPU, 256-512 for GPU
  }
}
```

---

## 📊 What to Expect

### Training Time (1000 iterations)

- **Before**: 5-11 hours
- **After (Parallel)**: 1.5-3 hours
- **After (Async GPU)**: 30-90 minutes

### During Training

You'll see:
```
Parallel trainer initialized with 16 workers
Using 16 worker processes

Iteration 100/1000 completed - Buffer: 15234
Iteration 200/1000 completed - Buffer: 30567
...
```

---

## 🔧 Troubleshooting

**GPU not working?**
```bash
# Check GPU
nvidia-smi

# Use CPU instead
# Edit config: "device": "cpu"
```

**Out of memory?**
```json
// Reduce these in config:
{"batch_size": 128, "num_workers": 8}
```

**Too slow?**
```json
// Increase these in config:
{"batch_size": 512, "num_workers": 14}
```

---

## 📚 More Info

- Full guide: `PARALLEL_TRAINING_GUIDE.md`
- Technical details: `PERFORMANCE_OPTIMIZATION_SUMMARY.md`
- Validation: `python3 scripts/validate_train.py`

---

## ✅ That's It!

You're ready for **high-performance training**! 🎉
