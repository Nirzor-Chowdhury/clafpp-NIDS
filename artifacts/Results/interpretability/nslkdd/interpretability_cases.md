# Symbolic Interpretability — Case Studies


## True Positives (correctly flagged attacks)

### Test sample #17393 (Probe)
  - Neural meta score: **0.9964**
  - Symbolic (raw): 0.5689 (lambda=0.0)
  - Final score: **0.9964** vs threshold 0.01 → predicted **attack**
  - Rules fired:
    - `service_sweep_dispersion`: 1.0

### Test sample #9800 (DoS)
  - Neural meta score: **0.9489**
  - Symbolic (raw): 0.5603 (lambda=0.0)
  - Final score: **0.9489** vs threshold 0.01 → predicted **attack**
  - Rules fired:
    - `syn_flood_consensus`: 1.0

### Test sample #14723 (DoS)
  - Neural meta score: **0.9998**
  - Symbolic (raw): 0.9 (lambda=0.0)
  - Final score: **0.9998** vs threshold 0.01 → predicted **attack**
  - Rules fired:
    - `syn_flood_consensus`: 1.0
    - `service_sweep_dispersion`: 0.7732

### Test sample #2003 (DoS)
  - Neural meta score: **0.9997**
  - Symbolic (raw): 0.4597 (lambda=0.0)
  - Final score: **0.9997** vs threshold 0.01 → predicted **attack**
  - Rules fired:
    - `service_sweep_dispersion`: 0.7515

### Test sample #9678 (DoS)
  - Neural meta score: **0.9995**
  - Symbolic (raw): 0.4078 (lambda=0.0)
  - Final score: **0.9995** vs threshold 0.01 → predicted **attack**
  - Rules fired:
    - `service_sweep_dispersion`: 0.6334


## False Negatives (missed attacks)

### Test sample #12166 (DoS)
  - Neural meta score: **0.0023**
  - Symbolic (raw): 0.1318 (lambda=0.0)
  - Final score: **0.0023** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #5181 (R2L)
  - Neural meta score: **0.0009**
  - Symbolic (raw): 0.1482 (lambda=0.0)
  - Final score: **0.0009** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #10019 (R2L)
  - Neural meta score: **0.0014**
  - Symbolic (raw): 0.1508 (lambda=0.0)
  - Final score: **0.0014** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #9840 (DoS)
  - Neural meta score: **0.002**
  - Symbolic (raw): 0.1301 (lambda=0.0)
  - Final score: **0.002** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #18550 (R2L)
  - Neural meta score: **0.0014**
  - Symbolic (raw): 0.1494 (lambda=0.0)
  - Final score: **0.0014** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.


## False Positives (misflagged normal traffic)

### Test sample #11025 (Normal)
  - Neural meta score: **0.1279**
  - Symbolic (raw): 0.1527 (lambda=0.0)
  - Final score: **0.1279** vs threshold 0.01 → predicted **attack**
  - No rules fired above threshold.

### Test sample #19008 (Normal)
  - Neural meta score: **0.9971**
  - Symbolic (raw): 0.5651 (lambda=0.0)
  - Final score: **0.9971** vs threshold 0.01 → predicted **attack**
  - Rules fired:
    - `service_sweep_dispersion`: 0.9914

### Test sample #9807 (Normal)
  - Neural meta score: **0.9991**
  - Symbolic (raw): 0.1296 (lambda=0.0)
  - Final score: **0.9991** vs threshold 0.01 → predicted **attack**
  - No rules fired above threshold.

### Test sample #8188 (Normal)
  - Neural meta score: **0.9976**
  - Symbolic (raw): 0.4385 (lambda=0.0)
  - Final score: **0.9976** vs threshold 0.01 → predicted **attack**
  - Rules fired:
    - `service_sweep_dispersion`: 0.6956

### Test sample #3863 (Normal)
  - Neural meta score: **0.1316**
  - Symbolic (raw): 0.1521 (lambda=0.0)
  - Final score: **0.1316** vs threshold 0.01 → predicted **attack**
  - No rules fired above threshold.


## True Negatives (correctly ignored normal traffic)

### Test sample #21983 (Normal)
  - Neural meta score: **0.0006**
  - Symbolic (raw): 0.1522 (lambda=0.0)
  - Final score: **0.0006** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #2080 (Normal)
  - Neural meta score: **0.0007**
  - Symbolic (raw): 0.1506 (lambda=0.0)
  - Final score: **0.0007** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #11855 (Normal)
  - Neural meta score: **0.0004**
  - Symbolic (raw): 0.1558 (lambda=0.0)
  - Final score: **0.0004** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #17222 (Normal)
  - Neural meta score: **0.005**
  - Symbolic (raw): 0.1507 (lambda=0.0)
  - Final score: **0.005** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #16643 (Normal)
  - Neural meta score: **0.0009**
  - Symbolic (raw): 0.2712 (lambda=0.0)
  - Final score: **0.0009** vs threshold 0.01 → predicted **normal**
  - Rules fired:
    - `low_entropy_repetition`: 0.6376
