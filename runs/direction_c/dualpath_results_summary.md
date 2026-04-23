# DIOR 10-shot Dual-Path Experiment Results

| Run | Fusion Mode | gamma | mAP50 | mAP50-95 | mAP75 | Precision | Recall | Latency (ms) |
|-----|------------|-------|-------|----------|-------|-----------|--------|-------------|
| A_baseline | none | 0.0 | 0.2602 | 0.1769 | 0.1891 | 0.4181 | 0.2486 | 30.38 |
| B_fixed_g03 | fixed | 0.3 | 0.2407 | 0.1519 | 0.1564 | 0.2272 | 0.3044 | 130.27 |
| C_fixed_g05 | fixed | 0.5 | 0.2272 | 0.1415 | 0.1439 | 0.2252 | 0.2953 | 86.34 |
| D_fixed_g07 | fixed | 0.7 | 0.2142 | 0.1310 | 0.1313 | 0.2211 | 0.2940 | 85.61 |
| E_mult | multiplicative | 0.5 | **0.3086** | **0.2058** | **0.2180** | 0.2039 | **0.4216** | 85.87 |

## Key Findings

- **Best mAP50**: E_mult (multiplicative, gamma=0.5) achieves 0.3086, +18.6% over baseline (0.2602)
- **Fixed fusion degrades performance**: As gamma increases from 0.3 to 0.7, mAP50 drops from 0.2407 to 0.2142, all below baseline
- **Latency**: Baseline (none) is fastest at 30ms; all fusion modes add ~55-100ms CLIP verification overhead
- **Precision vs Recall tradeoff**: Multiplicative fusion sacrifices precision (0.204) but boosts recall (0.422) significantly vs baseline (P=0.418, R=0.249)
