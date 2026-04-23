# DiffuTran — Architecture Flows, Variants & Unexplored Ideas

Two models live in this repo. Both accept a training CSV plus an optional
constraint and output synthetic tabular rows. Diagrams follow the same
format as `../../docs/models/`.

---

## DiffuTron (Diffusion Transformer)

### Full Pipeline Flow

```
INPUT: train.csv + constraint("col=val")
         │
         ▼
┌─────────────────────────────────────────────────┐
│  STEP 1 · DATA PREPROCESSING                    │
│  src/common/data_utils.py                       │
│  Infer col types from CSV dtypes                │
│  Numerical  → StandardScaler  (per-column)      │
│  Categorical→ LabelEncoder    (integer codes)   │
│  Categoricals one-hot-expanded in flat vector   │
│  condition_col → separate LabelEncoder          │
│  Output: float32 array [N, Σ feature_dims]      │
│  Artifacts: preprocessor.json                   │
│             condition_encoder.json              │
│                                                 │
│  Single method — no variants                    │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴──────────────────────────────────────┐
         │  UNTRIED IDEAS at Step 1:                     │
         │  • QuantileTransformer instead of Standard    │
         │    (already used in DS-Flow, same codebase)   │
         │  • Learned per-category embeddings            │
         │  • Missing-value indicator columns (NaN→cat)  │
         │  • Ordinal encoding for ordered categories    │
         └───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  STEP 2 · MODEL TRAINING                        │
│  DiffusionModel  src/diffutron/model.py         │
│  Token-per-feature Transformer:                 │
│    Per-feature input projections  (Linear)      │
│    6 layers  d_model=256  8 heads  ff=1024      │
│    Pre-norm, GELU, batch_first                  │
│  CFG: class_embedding(num_classes+1, d_model)   │
│    null class = index num_classes               │
│  GaussianDiffusion  src/diffutron/diffusion.py  │
│    Schedule: linear or cosine  T=1,000 steps    │
│    Objective: ε-prediction (noise prediction)   │
│  Optimizer: AdamW  LR=1e-4                      │
│  Artifacts: checkpoint_epoch_final.pt           │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴──────────────────────────────────────┐
         │  UNTRIED IDEAS at Step 2:                     │
         │  • x0-prediction instead of ε-prediction      │
         │  • Variance learning (IDDPM style)            │
         │  • Self-conditioning                          │
         │  • More epochs (200) + early stopping         │
         │  • FlashAttention-2 + RoPE (used in DS-Flow)  │
         │  • Cross-attention conditioning (vs concat)   │
         └───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  STEP 3 · CONSTRAINT → MASK                     │
│  Parse "col=val" → encoded float target value   │
│  constraint_mask [n_features] binary:           │
│    0 = constrained  (fixed)                     │
│    1 = free         (generated)                 │
│  constraint_x0 [batch, n_features] float        │
│                                                 │
│  All runs share this step identically           │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴──────────────────────────────────────┐
         │  UNTRIED IDEAS at Step 3:                     │
         │  • Range constraints: col ∈ [lo, hi]          │
         │  • Multi-column AND constraints               │
         │  • Soft mask: partial conditioning weight     │
         └───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  STEP 4 · REPAINT REVERSE DIFFUSION  t=T..0     │
│                                                 │
│  Start: x_T ~ N(0, I)                           │
│                                                 │
│  DDPM mode  (η=1.0 — default):                  │
│    1,000 stochastic DDPM steps                  │
│                                                 │
│  DDIM mode  (η=0.0):                            │
│    N strided deterministic steps  (e.g. 50)     │
│    controlled via --num_inference_steps         │
│                                                 │
│  CFG at each step:                              │
│    ε = ε_uncond + w·(ε_cond − ε_uncond)         │
│                                                 │
│  REPAINT constraint re-injection each step:     │
│    z[mask=0] ← q_sample(x0_target, t_prev)      │
│    only mask=1 features evolve freely           │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴──────────────────────────────────────┐
         │  UNTRIED IDEAS at Step 4:                     │
         │  • Jump-back REPAINT (re-noise + re-diffuse j │
         │    times — original REPAINT paper method)     │
         │  • Gradient guidance on free dims toward      │
         │    constraint-consistent distribution         │
         │  • Cross-attention conditioning instead of    │
         │    REPAINT masking                            │
         │  • Fewer re-injections (every K steps, not 1) │
         └───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  STEP 5 · POST-PROCESSING                       │
│  Inverse StandardScaler → numerical values      │
│  LabelEncoder.inverse_transform → cat strings   │
│  Output: synthetic DataFrame  (n_samples rows)  │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴──────────────────────────────────────┐
         │  UNTRIED IDEAS at Step 5:                     │
         │  • Reject-and-resample filter                 │
         │  • Re-rank by plausibility score from pool    │
         │  • Post-hoc repair for constraint failures    │
         └───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  STEP 6 · CSR CHECK                             │
│  Verify % of rows satisfying constraint         │
└─────────────────────────────────────────────────┘
```

### Variant Summary Table

| Aspect | Detail |
|---|---|
| **Architecture** | Token-per-feature Transformer, d_model=256, 6 layers, 8 heads |
| **Diffusion** | Gaussian DDPM, T=1000, ε-prediction |
| **Noise schedule** | Linear (default) or cosine |
| **Constraint handling** | REPAINT masking — re-inject noisy target at each denoising step |
| **Inference** | DDPM 1000 steps or DDIM N steps (η=0.0 deterministic) |
| **CFG** | class_embedding + null token at index `num_classes` |
| **Key difference vs DS-Flow** | Flat vector + DDPM vs token sequence + OT-CFM |

### Key source files

| Component | File |
|---|---|
| Preprocessing | `src/common/data_utils.py` |
| Model | `src/diffutron/model.py` |
| Diffusion engine | `src/diffutron/diffusion.py` |
| Embeddings | `src/diffutron/embeddings.py` |
| Training entry | `scripts/train.py` |
| Generation entry | `scripts/generate_test.py` |

---

## DS-Flow (Dual-Stream Flow Model)

### Full Pipeline Flow

```
INPUT: train.csv + constraint("col=val")
         │
         ▼
┌─────────────────────────────────────────────────┐
│  STEP 1 · DATA TOKENIZATION                     │
│  src/flow_model/data/tokenizer.py               │
│  Numerical → QuantileTransformer per col        │
│    output_distribution='normal'  (Gaussian)     │
│    jitter if < 10% unique vals                  │
│    NaN → binary indicator category column       │
│  Categorical → LabelEncoder per col             │
│    hash fallback if cardinality > 100           │
│    '<MISSING>' sentinel for NaN                 │
│  Learned layers (init during fit):              │
│    num_projections: Linear(1, d_model)          │
│    cat_embeddings: Embedding(card, d_model)     │
│    num_decoders:   Linear(d_model, 1)           │
│    cat_decoders:   Linear(d_model, card)        │
│  Output: x_num [N, N_num]  x_cat [N, N_cat]    │
│  Temporal: x_num [B, T, N_num]                  │
│  Artifacts: tokenizer.pt                        │
│                                                 │
│  Single method — no variants                    │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴──────────────────────────────────────┐
         │  UNTRIED IDEAS at Step 1:                     │
         │  • Different n_quantiles (256, 512 vs 1000)   │
         │  • Byte-pair encoding for high-cardinality    │
         │    categorical columns                        │
         │  • Bypass quantile: raw normalised floats     │
         └───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  STEP 2 · MODEL TRAINING                        │
│  DualStreamTransformer                          │
│    src/flow_model/models/dual_stream.py         │
│    Global token (learnable row summary)         │
│    Local tokens (per-column, d_model)           │
│    Per block: Global→Local cross-attn           │
│               CFG conditioning cross-attn       │
│               Local→Global cross-attn           │
│               Local self-attn with 1D-RoPE      │
│               FFN for each stream               │
│    FlashAttention-2 (F.scaled_dot_product)      │
│    TimestepEmbedder: sinusoidal + 2×Linear+SiLU │
│  OTCFM engine                                   │
│    src/flow_model/models/flow_matching.py       │
│    Numerical: OT-CFM, x_t=t·x1+(1−t)·x0        │
│      target velocity u_t = x1 − x0             │
│      loss: MSE(v_θ, u_t)                        │
│    Categorical: discrete reverse DDPM           │
│      keep original with prob t                  │
│      loss: CrossEntropy(cat_decoder(v_θ), y)    │
│    Reconstruction loss  λ=0.1  (anti-collapse)  │
│  CFG: class_embedder(num_classes, d_model)      │
│    uncond = pass y=None (no null token)         │
│  Optimizer: Adam  LR=1e-4  WD=1e-5             │
│  Artifacts: model_final.pt + tokenizer.pt       │
│  Temporal: is_temporal=True  sequence_len=T     │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴──────────────────────────────────────┐
         │  UNTRIED IDEAS at Step 2:                     │
         │  • CFG training: add null-token like DiffuTron│
         │    (currently uncond = missing y, not learned)│
         │  • Rectified flow (straight-line OT paths)    │
         │    → single-step sampling in limit            │
         │  • Per-column flow heads                      │
         │  • More layers (8) for larger datasets        │
         │  • Temporal support in sampler (training only)│
         └───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  STEP 3 · CONSTRAINT PARSING                    │
│  Parse "col=val" → typed constraint dict        │
│  cat_constraints: {col → class_idx}             │
│    encoded via LabelEncoder or hash fallback    │
│  num_constraints: {col → float}                 │
│    encoded via QuantileTransformer              │
│    projected to embedding via num_projections   │
│  Passed natively to OTCFM.sample()              │
│                                                 │
│  ✓ Native support for both cat + num columns    │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴──────────────────────────────────────┐
         │  UNTRIED IDEAS at Step 3:                     │
         │  • Range constraints: col ∈ [lo, hi] clipped  │
         │    trajectory in ODE integration              │
         │  • Compound multi-column AND constraints      │
         │  • Soft constraints: constraint strength γ    │
         └───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  STEP 4 · ODE + DISCRETE DDPM SAMPLING  t=0..1  │
│                                                 │
│  Numerical: ODE integration                     │
│    Start: x_0 ~ N(0, I)  [n, N_num, d_model]   │
│                                                 │
│  Euler solver  (--solver euler):                │
│    x_{t+dt} = x_t + v_θ(x_t, t) · dt           │
│    default 20 steps                             │
│                                                 │
│  DPM++ solver  (--solver dpm++  — default):     │
│    2nd-order midpoint per step                  │
│    x_mid = x_t + v_t · dt/2                     │
│    x_{t+dt} = x_t + v_mid · dt                  │
│    recommended 10 steps  (50× faster than DDPM) │
│                                                 │
│  Categorical: discrete reverse DDPM each step   │
│    logits = cat_decoder(v_θ[N_num:])            │
│    x1 = multinomial(softmax(logits))            │
│    keep x1 with prob t_next, else resample      │
│                                                 │
│  Adaptive CFG: w(t) = w_max · (1 − t)           │
│    vt = v_uncond + w(t)·(v_cond − v_uncond)     │
│                                                 │
│  Pinned columns re-applied after each step      │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴──────────────────────────────────────┐
         │  UNTRIED IDEAS at Step 4:                     │
         │  • SDE mode: add noise per Euler step         │
         │    (Langevin dynamics — diversity vs quality) │
         │  • Langevin correction step post-integration  │
         │  • Adaptive step count: more steps for hard   │
         │    / low-support constraints                  │
         │  • Constant CFG weight (ablate adaptive decay)│
         └───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  STEP 5 · POST-PROCESSING                       │
│  tokenizer.inverse_transform(x_tokens)          │
│  Numerical: num_decoder(token) →                │
│    clip [-5, 5] → QuantileTransformer.inverse   │
│  Categorical: cat_decoder(v_θ) → argmax →       │
│    LabelEncoder.inverse_transform               │
│  NaN restoration from missing indicator cols    │
│  Integer cols rounded + recast to original dtype│
│  Output: synthetic DataFrame  (n_samples rows)  │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴──────────────────────────────────────┐
         │  UNTRIED IDEAS at Step 5:                     │
         │  • Reject-and-resample (low priority — native │
         │    OT-CFM already gives high CSR)             │
         │  • Re-rank by plausibility score from pool    │
         └───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  STEP 6 · CSR CHECK                             │
│  Verify % of rows satisfying constraint         │
└─────────────────────────────────────────────────┘
```

### Variant Summary Table

| Aspect | Detail |
|---|---|
| **Architecture** | DualStreamTransformer — global row token + local column tokens |
| **Attention** | FlashAttention-2 (`F.scaled_dot_product_attention`) with 1D-RoPE |
| **Flow objective** | OT-CFM for numericals + discrete reverse DDPM for categoricals |
| **Reconstruction loss** | λ=0.1 anti-collapse auxiliary loss on token decoders |
| **Inference — Euler** | 1st-order, default 20 steps |
| **Inference — DPM++** | 2nd-order midpoint, default 10 steps (recommended) |
| **CFG** | Adaptive w(t)=w_max·(1−t); uncond via y=None (no null token) |
| **Constraint handling** | Native OT-CFM pin — no masking, no guidance tricks |
| **Temporal** | Training supports `[B, T, N, D]`; sampler is non-temporal only |
| **Key difference vs DiffuTron** | OT-CFM + RoPE + dual-stream vs DDPM + standard transformer |

### Key source files

| Component | File |
|---|---|
| Tokenizer | `src/flow_model/data/tokenizer.py` |
| DataLoader | `src/flow_model/data/loader.py` |
| Backbone | `src/flow_model/models/dual_stream.py` |
| Flow engine | `src/flow_model/models/flow_matching.py` |
| Config | `src/flow_model/config.py` |
| Training entry | `src/flow_model/train.py` |
| Sampling entry | `src/flow_model/sampler.py` |

---

## Unexplored Ideas — Priority Order

### Highest leverage (both models)
- **CFG training for DS-Flow**: DiffuTron already trains with a null token at
  index `num_classes`. DS-Flow currently passes `y=None` for unconditional,
  which skips conditioning cross-attention entirely — no learned null embedding.
  Aligning DS-Flow to the DiffuTron CFG pattern (explicit null token) would make
  guidance more principled and likely improve CSR.

- **Range constraints**: both models encode numerical constraints as a single
  pinned value. Extending `num_constraints` to `{col: (lo, hi)}` and clipping
  the ODE trajectory to the interval at each step is a low-code change with
  direct benchmark benefit.

### DiffuTron-specific
- **Cross-attention conditioning**: replace REPAINT masking with cross-attention
  over (constraint_col, constraint_val) token pairs. More information-efficient;
  free columns attend to constraints at every layer. DiffuTron's transformer
  backbone already has the infrastructure.
- **Jump-back REPAINT**: re-add noise for `j` steps and re-diffuse. Better joint
  coherence on multi-column constraints at the cost of `j×` more compute.
- **QuantileTransformer preprocessing**: DiffuTron still uses StandardScaler.
  Swapping in QuantileTransformer (already validated in DS-Flow) should reduce
  outlier sensitivity.

### DS-Flow-specific
- **Temporal sampler**: training already supports `[B, T, N, D]` via
  `is_temporal=True`. The sampler (`OTCFM.sample`) is non-temporal. Extending
  it to reshape tokens into time steps is the missing half of the temporal
  feature.
- **SDE mode**: add small Gaussian noise at each Euler step (Langevin dynamics).
  Trades determinism for diversity — may help on datasets where free columns
  collapse to the mode.
- **Rectified flow**: re-train with straight-line OT paths. Enables single-step
  sampling in the limit.
