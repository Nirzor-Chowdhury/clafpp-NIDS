# Symbolic Interpretability — Case Studies


## True Positives (correctly flagged attacks)

### Test sample #17364 (DoS)
  - Neural meta score: **0.9985**
  - Symbolic (raw): 0.3434 (lambda=0.0)
  - Final score: **0.9985** vs threshold 0.01 → predicted **attack**
  - No rules fired above threshold.

### Test sample #9801 (DoS)
  - Neural meta score: **0.9997**
  - Symbolic (raw): 0.4782 (lambda=0.0)
  - Final score: **0.9997** vs threshold 0.01 → predicted **attack**
  - Rules fired:
    - `service_sweep_dispersion`: 0.7936

### Test sample #14723 (DoS)
  - Neural meta score: **0.9997**
  - Symbolic (raw): 0.9 (lambda=0.0)
  - Final score: **0.9997** vs threshold 0.01 → predicted **attack**
  - Rules fired:
    - `syn_flood_consensus`: 1.0
    - `service_sweep_dispersion`: 0.7732

### Test sample #1986 (DoS)
  - Neural meta score: **0.9996**
  - Symbolic (raw): 0.8466 (lambda=0.0)
  - Final score: **0.9996** vs threshold 0.01 → predicted **attack**
  - Rules fired:
    - `syn_flood_consensus`: 1.0
    - `service_sweep_dispersion`: 0.6517

### Test sample #9676 (DoS)
  - Neural meta score: **0.9997**
  - Symbolic (raw): 0.35 (lambda=0.0)
  - Final score: **0.9997** vs threshold 0.01 → predicted **attack**
  - Rules fired:
    - `low_entropy_repetition`: 0.9922


## False Negatives (missed attacks)

### Test sample #12197 (DoS)
  - Neural meta score: **0.002**
  - Symbolic (raw): 0.1302 (lambda=0.0)
  - Final score: **0.002** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #5239 (R2L)
  - Neural meta score: **0.0018**
  - Symbolic (raw): 0.1296 (lambda=0.0)
  - Final score: **0.0018** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #10031 (R2L)
  - Neural meta score: **0.0009**
  - Symbolic (raw): 0.1305 (lambda=0.0)
  - Final score: **0.0009** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #9840 (DoS)
  - Neural meta score: **0.0021**
  - Symbolic (raw): 0.1301 (lambda=0.0)
  - Final score: **0.0021** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #18626 (R2L)
  - Neural meta score: **0.001**
  - Symbolic (raw): 0.1305 (lambda=0.0)
  - Final score: **0.001** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.


## False Positives (misflagged normal traffic)

### Test sample #10932 (Normal)
  - Neural meta score: **0.0119**
  - Symbolic (raw): 0.1296 (lambda=0.0)
  - Final score: **0.0119** vs threshold 0.01 → predicted **attack**
  - No rules fired above threshold.

### Test sample #19073 (Normal)
  - Neural meta score: **0.9987**
  - Symbolic (raw): 0.5729 (lambda=0.0)
  - Final score: **0.9987** vs threshold 0.01 → predicted **attack**
  - Rules fired:
    - `service_sweep_dispersion`: 0.9998

### Test sample #9949 (Normal)
  - Neural meta score: **0.0261**
  - Symbolic (raw): 0.1508 (lambda=0.0)
  - Final score: **0.0261** vs threshold 0.01 → predicted **attack**
  - No rules fired above threshold.

### Test sample #8188 (Normal)
  - Neural meta score: **0.9987**
  - Symbolic (raw): 0.4385 (lambda=0.0)
  - Final score: **0.9987** vs threshold 0.01 → predicted **attack**
  - Rules fired:
    - `service_sweep_dispersion`: 0.6956

### Test sample #4030 (Normal)
  - Neural meta score: **0.0396**
  - Symbolic (raw): 0.1782 (lambda=0.0)
  - Final score: **0.0396** vs threshold 0.01 → predicted **attack**
  - No rules fired above threshold.


## True Negatives (correctly ignored normal traffic)

### Test sample #21987 (Normal)
  - Neural meta score: **0.0004**
  - Symbolic (raw): 0.1746 (lambda=0.0)
  - Final score: **0.0004** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #2072 (Normal)
  - Neural meta score: **0.0003**
  - Symbolic (raw): 0.1525 (lambda=0.0)
  - Final score: **0.0003** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #11855 (Normal)
  - Neural meta score: **0.0004**
  - Symbolic (raw): 0.1558 (lambda=0.0)
  - Final score: **0.0004** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #17222 (Normal)
  - Neural meta score: **0.0009**
  - Symbolic (raw): 0.1507 (lambda=0.0)
  - Final score: **0.0009** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.

### Test sample #16641 (Normal)
  - Neural meta score: **0.0081**
  - Symbolic (raw): 0.1523 (lambda=0.0)
  - Final score: **0.0081** vs threshold 0.01 → predicted **normal**
  - No rules fired above threshold.
