# Task C Report — Hybrid Code Detection

## Task Description

Task C is a 4-class classification problem focused on detecting not just whether code is
AI-generated, but *how* it was produced. This is the most nuanced of the three subtasks,
requiring the model to distinguish between genuinely human-written code, fully machine-generated
code, code that is a mixture of human and LLM contributions, and adversarially generated code
designed to mimic human style.

**Classes:**
| Label | Category | Description |
|---|---|---|
| 0 | Human-written | Code written entirely by a human |
| 1 | Machine-generated | Code produced entirely by an LLM |
| 2 | Hybrid | Code partially written or completed by an LLM |
| 3 | Adversarial | Code generated via adversarial prompts or RLHF to mimic human style |

**Evaluation metric:** Macro F1

**Baseline score:** 0.48120 (competition organizer)

---

## Dataset

### Raw Data
- **Training samples:** 900,000
- **Validation samples:** 200,000
- **Test samples:** 500,000

| Label | Category | Raw Train Count | % |
|---|---|---|---|
| 0 | Human-written | 485,483 | 53.9% |
| 1 | Machine-generated | 210,471 | 23.4% |
| 2 | Hybrid | 85,520 | 9.5% |
| 3 | Adversarial | 118,526 | 13.2% |

The dataset is naturally skewed — human-written code dominates at 53.9%, and Hybrid code
is the rarest class at 9.5%. This imbalance is reflected in the difficulty of each class:
Hybrid is inherently the hardest to detect as it blends human and machine characteristics.

### Balancing Strategy
Data was balanced using a **per-(language, label) cap** across 8 programming languages,
matching the methodology used in Task A. Two configurations were evaluated:

| Configuration | Cap per (lang, label) | Train Samples | % of Raw Used |
|---|---|---|---|
| Run 1 (50k) | 1,550 | 49,243 | 5.5% |
| Run 2 / Run 3 (150k) | 5,000 | 150,980 | 16.8% |

**Programming languages (8):** Python, Java, JavaScript, C#, C++, Go, C, PHP

### Training Label Distribution (150k run)
| Label | Category | Count | % |
|---|---|---|---|
| 0 | Human-written | 40,000 | 26.5% |
| 1 | Machine-generated | 37,581 | 24.9% |
| 2 | Hybrid | 37,206 | 24.6% |
| 3 | Adversarial | 36,193 | 24.0% |

Near-equal balance achieved across all four classes.

**Known data limitations (natural caps, not balancing artifacts):**
- C × Hybrid: 3,116 samples (below cap of 5,000)
- C++ × Hybrid: 4,659
- Go × Hybrid: 4,431
- PHP × Machine-generated: 2,581
- PHP × Adversarial: 1,193 (most constrained group)

These reflect genuine scarcity of certain language-category combinations in the source data.

---

## Model Architecture

```
backbone_embedding [3072]  ──► LayerNorm ──► tanh(Linear(3072→3072)) ──► [3072]
handcrafted_features [102] ──► MLP (102→256→128) ──────────────────────► [128]
                                                                              │
                                                               concat [3200] ──► Fusion MLP ──► [4]
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
- `Linear(256 → 4)`

**Loss:** CrossEntropyLoss with inverse-frequency class weights

**Trainable parameters:** ~8M (head only; backbone frozen)

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-6 |
| Weight decay | 0.01 |
| Batch size | 64 |
| Warmup ratio | 0.1 |
| Max grad norm | 0.5 |
| Scheduler | Linear decay with warmup |
| Mixed precision | bfloat16 |
| Max token length | 512 |

---

## Results

### Test Set Scores (Competition)

| Run | Train Samples | Epochs | Test Macro F1 |
|---|---|---|---|
| Baseline | — | — | 0.48120 |
| Run 1 | 49,243 (50k balanced) | 15 | 0.51845 |
| Run 2 | 150,980 (150k balanced) | 15 | 0.54877 |
| **Run 3** | **150,980 (150k balanced)** | **50** | **0.55585** |

**Best result: 0.55585** — improvement of **+0.07465** over baseline (+15.5% relative)

### Observations Across Runs

**Effect of dataset size (Run 1 → Run 2, same 15 epochs):**
- +0.03032 F1 (+5.9% relative)
- Tripling training data (49k → 151k) brought consistent improvement
- The model saw more diverse hybrid and adversarial examples, which are the rarest classes

**Effect of training duration (Run 2 → Run 3, same 150k data):**
- +0.00708 F1 (+1.3% relative)
- Extending from 15 to 50 epochs gave diminishing returns
- The architecture converges within the first 15–20 epochs; additional epochs contribute marginally

**Scaling trend:**
```
Baseline  →  50k/15ep  →  150k/15ep  →  150k/50ep
  0.481       0.518        0.549         0.556
         +0.037        +0.031        +0.007
```
Dataset size has a substantially larger effect than training duration.

### Prediction Distribution (Test Submission — Run 3)
| Label | Category | Predicted Count | % |
|---|---|---|---|
| 0 | Human-written | 252,997 | 50.6% |
| 1 | Machine-generated | 101,611 | 20.3% |
| 2 | Hybrid | 92,608 | 18.5% |
| 3 | Adversarial | 52,784 | 10.6% |

The model predicts human-written at 50.6%, broadly consistent with the raw training distribution
(53.9%), suggesting reasonable calibration with less distribution shift than seen in Task B.

---

## Analysis

### Why Task C is Harder Than It Appears
Despite being 4-class like a coarser version of Task B, Task C presents unique challenges:

1. **Hybrid code** (label 2) is by nature ambiguous — it contains both human and machine
   patterns simultaneously, making it the hardest class to separate from both human-written
   and fully machine-generated code.
2. **Adversarial code** (label 3) is explicitly designed to mimic human style via RLHF or
   adversarial prompting, directly attacking the stylometric and pattern features our model
   relies on.
3. The **9.5% natural frequency of Hybrid** in raw data means even after balancing, the model
   has seen fewer examples of this class than any other.

### Why 0.556 is the Likely Ceiling for This Approach
- **Adversarial samples defeat handcrafted features:** The 102-dim stylometric/AST features
  were designed to detect generic AI patterns. Adversarial code that was tuned to avoid
  exactly these patterns will produce misleading feature values.
- **Hybrid code is ambiguous at the embedding level:** Mean-pooled StarCoder2 embeddings
  represent a holistic code summary. A file that is 50% human + 50% LLM may produce an
  embedding indistinguishable from either class.
- **PHP × Adversarial bottleneck:** Only 1,193 training samples exist for this combination,
  structurally limiting recall on adversarial PHP code.

### Effect of Data Quantity vs Training Duration
The largest single improvement came from **tripling training data (50k → 150k)**, not from
training longer. This confirms the bottleneck for Task C is data quantity, not model capacity
or training duration. Increasing the cap further (e.g., 10,000 per group → ~320k samples)
would likely push F1 higher, bounded by the PHP × Adversarial natural limit.
