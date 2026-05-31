# Global rule statistics on test set

Dataset tag: `nslkdd` | seed: 42

Test set size: 22544 (12833 attacks, 9711 normals)

| rule                     |   learned_weight |   mean_confidence |   fires_on_attacks_pct |   fires_on_normals_pct |   standalone_precision |   standalone_recall |   standalone_f1 |
|:-------------------------|-----------------:|------------------:|-----------------------:|-----------------------:|-----------------------:|--------------------:|----------------:|
| service_sweep_dispersion |           0.3254 |            0.2454 |                  47.15 |                   2.34 |                 0.9638 |              0.4715 |          0.6332 |
| syn_flood_consensus      |           0.3188 |            0.0986 |                  17.2  |                   0.13 |                 0.9941 |              0.172  |          0.2932 |
| auth_compromise_pattern  |           0.1914 |            0.5313 |                  10.38 |                   0.8  |                 0.9447 |              0.1038 |          0.187  |
| low_entropy_repetition   |           0.1645 |            0.127  |                  11.93 |                   2.08 |                 0.8834 |              0.1193 |          0.2102 |