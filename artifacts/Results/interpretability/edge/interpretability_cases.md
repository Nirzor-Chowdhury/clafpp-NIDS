# Symbolic Interpretability — Case Studies


## True Positives (correctly flagged attacks)

### Test sample #19289 (Vulnerability_scanner)
  - Neural meta score: **0.998**
  - Symbolic (raw): 0.9562 (lambda=0.0)
  - Final score: **0.998** vs threshold 0.58 → predicted **attack**
  - Rules fired:
    - `http_injection`: 1.0

### Test sample #11155 (DDoS_ICMP)
  - Neural meta score: **0.9975**
  - Symbolic (raw): 0.6483 (lambda=0.0)
  - Final score: **0.9975** vs threshold 0.58 → predicted **attack**
  - No rules fired above threshold.

### Test sample #16373 (XSS)
  - Neural meta score: **0.9951**
  - Symbolic (raw): 0.3608 (lambda=0.0)
  - Final score: **0.9951** vs threshold 0.58 → predicted **attack**
  - No rules fired above threshold.

### Test sample #2398 (DDoS_ICMP)
  - Neural meta score: **0.9975**
  - Symbolic (raw): 0.7594 (lambda=0.0)
  - Final score: **0.9975** vs threshold 0.58 → predicted **attack**
  - Rules fired:
    - `ddos_flag_storm`: 0.5507

### Test sample #10952 (DDoS_UDP)
  - Neural meta score: **0.9975**
  - Symbolic (raw): 0.477 (lambda=0.0)
  - Final score: **0.9975** vs threshold 0.58 → predicted **attack**
  - No rules fired above threshold.


## False Negatives (missed attacks)

### Test sample #10963 (Port_Scanning)
  - Neural meta score: **0.1171**
  - Symbolic (raw): 0.6884 (lambda=0.0)
  - Final score: **0.1171** vs threshold 0.58 → predicted **normal**
  - Rules fired:
    - `recon_dns_arp`: 1.0

### Test sample #5885 (Backdoor)
  - Neural meta score: **0.0008**
  - Symbolic (raw): 0.4833 (lambda=0.0)
  - Final score: **0.0008** vs threshold 0.58 → predicted **normal**
  - No rules fired above threshold.

### Test sample #9221 (Port_Scanning)
  - Neural meta score: **0.4062**
  - Symbolic (raw): 0.6884 (lambda=0.0)
  - Final score: **0.4062** vs threshold 0.58 → predicted **normal**
  - Rules fired:
    - `recon_dns_arp`: 1.0

### Test sample #8630 (Port_Scanning)
  - Neural meta score: **0.3419**
  - Symbolic (raw): 0.6884 (lambda=0.0)
  - Final score: **0.3419** vs threshold 0.58 → predicted **normal**
  - Rules fired:
    - `recon_dns_arp`: 1.0

### Test sample #16507 (Backdoor)
  - Neural meta score: **0.0007**
  - Symbolic (raw): 0.4833 (lambda=0.0)
  - Final score: **0.0007** vs threshold 0.58 → predicted **normal**
  - No rules fired above threshold.


## False Positives (misflagged normal traffic)

### Test sample #10392 (Normal)
  - Neural meta score: **0.9302**
  - Symbolic (raw): 0.6884 (lambda=0.0)
  - Final score: **0.9302** vs threshold 0.58 → predicted **attack**
  - Rules fired:
    - `recon_dns_arp`: 1.0

### Test sample #16829 (Normal)
  - Neural meta score: **0.6105**
  - Symbolic (raw): 0.477 (lambda=0.0)
  - Final score: **0.6105** vs threshold 0.58 → predicted **attack**
  - No rules fired above threshold.

### Test sample #8096 (Normal)
  - Neural meta score: **0.684**
  - Symbolic (raw): 0.477 (lambda=0.0)
  - Final score: **0.684** vs threshold 0.58 → predicted **attack**
  - No rules fired above threshold.

### Test sample #7055 (Normal)
  - Neural meta score: **0.6543**
  - Symbolic (raw): 0.477 (lambda=0.0)
  - Final score: **0.6543** vs threshold 0.58 → predicted **attack**
  - No rules fired above threshold.

### Test sample #4759 (Normal)
  - Neural meta score: **0.7773**
  - Symbolic (raw): 0.477 (lambda=0.0)
  - Final score: **0.7773** vs threshold 0.58 → predicted **attack**
  - No rules fired above threshold.


## True Negatives (correctly ignored normal traffic)

### Test sample #24400 (Normal)
  - Neural meta score: **0.0005**
  - Symbolic (raw): 0.714 (lambda=0.0)
  - Final score: **0.0005** vs threshold 0.58 → predicted **normal**
  - No rules fired above threshold.

### Test sample #2289 (Normal)
  - Neural meta score: **0.0004**
  - Symbolic (raw): 0.5897 (lambda=0.0)
  - Final score: **0.0004** vs threshold 0.58 → predicted **normal**
  - No rules fired above threshold.

### Test sample #13125 (Normal)
  - Neural meta score: **0.0005**
  - Symbolic (raw): 0.5897 (lambda=0.0)
  - Final score: **0.0005** vs threshold 0.58 → predicted **normal**
  - No rules fired above threshold.

### Test sample #19056 (Normal)
  - Neural meta score: **0.0005**
  - Symbolic (raw): 0.4833 (lambda=0.0)
  - Final score: **0.0005** vs threshold 0.58 → predicted **normal**
  - No rules fired above threshold.

### Test sample #18422 (Normal)
  - Neural meta score: **0.0004**
  - Symbolic (raw): 0.5897 (lambda=0.0)
  - Final score: **0.0004** vs threshold 0.58 → predicted **normal**
  - No rules fired above threshold.
