# Symbolic Interpretability — Case Studies


## True Positives (correctly flagged attacks)

### Test sample #19287 (Uploading)
  - Neural meta score: **0.9971**
  - Symbolic (raw): 0.3475 (lambda=0.0)
  - Final score: **0.9971** vs threshold 0.73 → predicted **attack**
  - No rules fired above threshold.

### Test sample #11156 (DDoS_ICMP)
  - Neural meta score: **0.9959**
  - Symbolic (raw): 0.5827 (lambda=0.0)
  - Final score: **0.9959** vs threshold 0.73 → predicted **attack**
  - No rules fired above threshold.

### Test sample #16376 (DDoS_HTTP)
  - Neural meta score: **0.998**
  - Symbolic (raw): 0.3475 (lambda=0.0)
  - Final score: **0.998** vs threshold 0.73 → predicted **attack**
  - No rules fired above threshold.

### Test sample #2403 (DDoS_ICMP)
  - Neural meta score: **0.9961**
  - Symbolic (raw): 0.575 (lambda=0.0)
  - Final score: **0.9961** vs threshold 0.73 → predicted **attack**
  - No rules fired above threshold.

### Test sample #10955 (Uploading)
  - Neural meta score: **0.9991**
  - Symbolic (raw): 0.3475 (lambda=0.0)
  - Final score: **0.9991** vs threshold 0.73 → predicted **attack**
  - No rules fired above threshold.


## False Negatives (missed attacks)

### Test sample #11135 (Port_Scanning)
  - Neural meta score: **0.3864**
  - Symbolic (raw): 0.6884 (lambda=0.0)
  - Final score: **0.3864** vs threshold 0.73 → predicted **normal**
  - Rules fired:
    - `recon_dns_arp`: 1.0

### Test sample #4751 (Port_Scanning)
  - Neural meta score: **0.4362**
  - Symbolic (raw): 0.6884 (lambda=0.0)
  - Final score: **0.4362** vs threshold 0.73 → predicted **normal**
  - Rules fired:
    - `recon_dns_arp`: 1.0

### Test sample #9221 (Port_Scanning)
  - Neural meta score: **0.1764**
  - Symbolic (raw): 0.6884 (lambda=0.0)
  - Final score: **0.1764** vs threshold 0.73 → predicted **normal**
  - Rules fired:
    - `recon_dns_arp`: 1.0

### Test sample #8630 (Port_Scanning)
  - Neural meta score: **0.4057**
  - Symbolic (raw): 0.6884 (lambda=0.0)
  - Final score: **0.4057** vs threshold 0.73 → predicted **normal**
  - Rules fired:
    - `recon_dns_arp`: 1.0

### Test sample #17397 (Ransomware)
  - Neural meta score: **0.5203**
  - Symbolic (raw): 0.477 (lambda=0.0)
  - Final score: **0.5203** vs threshold 0.73 → predicted **normal**
  - No rules fired above threshold.


## False Positives (misflagged normal traffic)

### Test sample #17208 (Normal)
  - Neural meta score: **0.9369**
  - Symbolic (raw): 0.6884 (lambda=0.0)
  - Final score: **0.9369** vs threshold 0.73 → predicted **attack**
  - Rules fired:
    - `recon_dns_arp`: 1.0

### Test sample #10392 (Normal)
  - Neural meta score: **0.9626**
  - Symbolic (raw): 0.6884 (lambda=0.0)
  - Final score: **0.9626** vs threshold 0.73 → predicted **attack**
  - Rules fired:
    - `recon_dns_arp`: 1.0

### Test sample #3447 (Normal)
  - Neural meta score: **0.7514**
  - Symbolic (raw): 0.477 (lambda=0.0)
  - Final score: **0.7514** vs threshold 0.73 → predicted **attack**
  - No rules fired above threshold.

### Test sample #20099 (Normal)
  - Neural meta score: **0.9621**
  - Symbolic (raw): 0.6884 (lambda=0.0)
  - Final score: **0.9621** vs threshold 0.73 → predicted **attack**
  - Rules fired:
    - `recon_dns_arp`: 1.0

### Test sample #3324 (Normal)
  - Neural meta score: **0.9597**
  - Symbolic (raw): 0.6884 (lambda=0.0)
  - Final score: **0.9597** vs threshold 0.73 → predicted **attack**
  - Rules fired:
    - `recon_dns_arp`: 1.0


## True Negatives (correctly ignored normal traffic)

### Test sample #24402 (Normal)
  - Neural meta score: **0.0004**
  - Symbolic (raw): 0.714 (lambda=0.0)
  - Final score: **0.0004** vs threshold 0.73 → predicted **normal**
  - No rules fired above threshold.

### Test sample #2289 (Normal)
  - Neural meta score: **0.0004**
  - Symbolic (raw): 0.5897 (lambda=0.0)
  - Final score: **0.0004** vs threshold 0.73 → predicted **normal**
  - No rules fired above threshold.

### Test sample #13127 (Normal)
  - Neural meta score: **0.0004**
  - Symbolic (raw): 0.603 (lambda=0.0)
  - Final score: **0.0004** vs threshold 0.73 → predicted **normal**
  - No rules fired above threshold.

### Test sample #19056 (Normal)
  - Neural meta score: **0.0004**
  - Symbolic (raw): 0.4833 (lambda=0.0)
  - Final score: **0.0004** vs threshold 0.73 → predicted **normal**
  - No rules fired above threshold.

### Test sample #18422 (Normal)
  - Neural meta score: **0.0004**
  - Symbolic (raw): 0.5897 (lambda=0.0)
  - Final score: **0.0004** vs threshold 0.73 → predicted **normal**
  - No rules fired above threshold.
