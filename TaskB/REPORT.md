# Task B Report — LLM Family Authorship Attribution

## Task Description

Task B is an 11-class authorship attribution problem. Given a code snippet, the model must
identify which LLM family (or a human) produced it. This is significantly harder than binary
detection (Task A) because the model must distinguish between stylistically similar generators
that all produce syntactically valid, functionally correct code.

**Classes:**
| Label | Family |
|---|---|
| 0 | Human |
| 1 | deepseek-ai |
| 2 | qwen |
| 3 | 01-ai |
| 4 | bigcode |
| 5 | gemma |
| 6 | phi |
| 7 | meta-llama |
| 8 | ibm-granite |
| 9 | mistral |
| 10 | openai |

**Evaluation metric:** Macro F1 (equal weight per class regardless of frequency)

---

## Dataset

### Raw Data
- **Total training samples:** 500,000
- **Distribution:** Heavily skewed — Human accounts for 88.4% (442,096 samples); most AI
  families have only 2,000–11,000 samples each

| Label | Family | Raw Count | % |
|---|---|---|---|
| 0 | Human | 442,096 | 88.4% |
| 1 | deepseek-ai | 4,162 | 0.8% |
| 2 | qwen | 8,993 | 1.8% |
| 3 | 01-ai | 3,029 | 0.6% |
| 4 | bigcode | 2,227 | 0.4% |
| 5 | gemma | 1,968 | 0.4% |
| 6 | phi | 5,783 | 1.2% |
| 7 | meta-llama | 8,197 | 1.6% |
| 8 | ibm-granite | 8,127 | 1.6% |
| 9 | mistral | 4,608 | 0.9% |
| 10 | openai | 10,810 | 2.2% |

### Balanced Training Data
To address the extreme imbalance, data was balanced using a **per-(language, label) cap** of 550
samples per group, mirroring the approach used in Task A.

- **Balanced train samples:** 57,614 (from 500,000 raw — 11.5% used)
- **Balanced val samples:** 11,340
- **Programming languages:** Python, Java, JavaScript, C#, C++, Go, C, PHP (8 languages)

**Known data limitation:** PHP samples are missing for labels 3, 4, 6, and 8. These generator
families never produced PHP code in the dataset. This structurally limits per-class F1 for
those generators.

---

## Model Architecture

The same hybrid architecture used in Task A is applied here, extended to 11-class output.

```
backbone_embedding [3072]  ──► LayerNorm ──► tanh(Linear(3072→3072)) ──► [3072]
handcrafted_features [102] ──► MLP (102→256→128) ──────────────────────► [128]
                                                                              │
                                                               concat [3200] ──► Fusion MLP ──► [11]
```

**Backbone:** StarCoder2-3B (frozen, 8-bit quantized)
- Pre-computed mean-pooled embeddings: `[N, 3072]` cached as `.pt` files
- No gradient flows through backbone during training

**Handcrafted features (102-dim):**
- 12 stylometric: avg line length, indent style, blank line ratio, comment density, etc.
- 57 pattern: regex counts for keywords, operators, naming conventions
- 33 AST: node type distributions (functions, loops, conditionals, etc.)

**Fusion MLP:**
- `Linear(3200 → 512) → LayerNorm → GELU → Dropout(0.2)`
- `Linear(512 → 256) → LayerNorm → GELU → Dropout(0.1)`
- `Linear(256 → 11)`

**Loss:** CrossEntropyLoss with inverse-frequency class weights (auto-computed from training
label distribution). Mean-normalised so the average weight = 1.

**Trainable parameters:** ~8M (head only; backbone frozen)

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-6 |
| Weight decay | 0.01 |
| Batch size | 64 |
| Epochs | 50 |
| Warmup ratio | 0.1 |
| Max grad norm | 0.5 |
| Scheduler | Linear decay with warmup |
| Mixed precision | bfloat16 |
| Max token length | 512 |

---

## Results

### Training Curve (selected epochs)

| Epoch | Train Loss | Val Loss | Val Macro F1 |
|---|---|---|---|
| 1 | 2.3222 | 2.1045 | 0.2202 |
| 5 | 1.7399 | 1.6774 | 0.3577 |
| 10 | 1.4631 | 1.5036 | 0.4140 |
| 15 | 1.3080 | 1.4431 | 0.4515 |
| 20 | 1.1922 | 1.3937 | 0.4644 |
| 25 | 1.1010 | 1.3716 | 0.4716 |
| 30 | 1.0252 | 1.3629 | 0.4896 |
| 40 | 0.9227 | 1.3540 | 0.4978 |
| **49** | **0.9315** | **1.3621** | **0.5007** |
| 50 | 0.8714 | 1.3536 | 0.5011 |

**Best checkpoint:** Epoch 49 — Val Macro F1 = **0.5014**

### Test Set Score

| Metric | Value |
|---|---|
| Competition Baseline | 0.2286 |
| Test Macro F1 (Run 1, best) | **0.2900** |
| Improvement over baseline | +0.0614 (+26.9% relative) |
| Val Macro F1 (best) | 0.5014 |
| Val–Test gap | 0.2114 |

### Architecture Experiment

A modified architecture was tested (removed tanh pooler, increased dropout to 0.3/0.2,
added label smoothing 0.1):

| Run | Architecture | Val F1 | Test F1 |
|---|---|---|---|
| Run 1 | Original | 0.5014 | **0.2900** |
| Run 2 | Modified | 0.4376 | 0.2568 |

> Run 2 architecture: removed tanh pooler, dropout increased to 0.3/0.2, label smoothing 0.1.
> Reverted to Run 1 architecture after performance degraded.

The modifications reduced performance. The original architecture was restored.

---

## Analysis

### Convergence
The model converged around **epoch 25–27**. Val loss became flat (1.37–1.36) while train loss
continued declining — a textbook overfitting plateau. The additional 25 epochs gained only
+0.01 F1. A 25-epoch cutoff would have been equally effective.

### Val–Test Distribution Shift
The 0.21-point gap between val F1 (0.50) and test F1 (0.29) is the primary challenge.
Our balanced validation set has equal representation across all 11 classes, but the test set
likely reflects a natural, skewed distribution (dominated by human and a few major generators).

The model learned to predict all 11 classes with roughly equal frequency, which is penalized
when the test set expects a skewed distribution.

### Structural Limitations
1. **Missing PHP labels:** Labels 3, 4, 6, 8 have zero PHP samples in training. The model
   cannot learn those generator signatures for PHP, hurting per-class recall.
2. **Small-count generators:** bigcode (2,227 raw) and gemma (1,968 raw) have the fewest
   samples after capping. Their class-level F1 is expected to be the lowest.
3. **Semantic similarity:** Different LLM families produce stylistically similar code. The
   102-dim handcrafted features may not capture sufficient discriminative signal between,
   e.g., deepseek vs mistral vs qwen.

### Prediction Distribution (Test Submission)
| Label | Predicted Count | % |
|---|---|---|
| 0 (Human) | 198,165 | 39.6% |
| 5 (gemma) | 53,440 | 10.7% |
| 10 (openai) | 81,007 | 16.2% |
| Others | 167,388 | 33.5% |

The model over-predicts human (39.6%) and openai (16.2%), suggesting these are the easiest
classes to identify from code style.
