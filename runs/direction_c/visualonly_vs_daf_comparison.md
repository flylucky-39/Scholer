# DIOR 1/5/10-shot Visual-Only vs DAF Dual-Path Comparison

| Shot | Model | Fusion | mAP50 | mAP50-95 | mAP75 | P | R | Latency |
|------|-------|--------|-------|----------|-------|------|------|---------|
| **1** | Visual-Only | none | 0.0710 | 0.0511 | 0.0624 | 0.006 | 0.216 | 29ms |
| **1** | Visual-Only | mult(γ=0.5,T=5) | 0.1363 | 0.0994 | 0.1218 | 0.099 | 0.167 | 87ms |
| **1** | DAF(α=0.12) | none | 0.0625 | 0.0483 | 0.0538 | 0.006 | 0.233 | 52ms |
| **1** | DAF(α=0.12) | mult(γ=0.5,T=5) | 0.0925 | 0.0651 | 0.0714 | 0.048 | 0.155 | 100ms |
| **5** | Visual-Only | none | 0.2222 | 0.1697 | 0.1766 | 0.089 | 0.383 | 25ms |
| **5** | Visual-Only | mult(γ=0.5,T=5) | 0.2395 | 0.1969 | 0.2199 | 0.210 | 0.191 | 91ms |
| **5** | DAF(α=0.12) | none | 0.1983 | 0.1529 | 0.1473 | 0.097 | 0.357 | 44ms |
| **5** | DAF(α=0.12) | mult(γ=0.5,T=5) | 0.1923 | 0.1632 | 0.1813 | 0.142 | 0.164 | 86ms |
| **10** | Visual-Only | none | 0.5985 | 0.4868 | 0.5288 | 0.432 | 0.552 | 34ms |
| **10** | Visual-Only | mult(γ=0.5,T=5) | 0.5146 | 0.4709 | 0.5084 | 0.396 | 0.404 | 99ms |
| **10** | DAF(α=0.12) | none | 0.8424 | 0.7447 | 0.8049 | 0.728 | 0.620 | 42ms |
| **10** | DAF(α=0.12) | mult(γ=0.5,T=5) | 0.8226 | 0.7500 | 0.8061 | 0.637 | 0.812 | 90ms |

## Dual-Path Gain over Baseline (Δ mAP50)

| Shot | Visual-Only Δ | DAF Δ |
|------|--------------|-------|
| 1 | **+92.0%** (0.071→0.136) | +48.0% (0.063→0.093) |
| 5 | +7.8% (0.222→0.240) | -3.0% (0.198→0.192) |
| 10 | -14.0% (0.599→0.515) | -2.4% (0.842→0.823) |

## Key Findings

- **1-shot**: CLIP dual-path helps dramatically on Visual-Only (+92% mAP50), and also boosts DAF (+48%). Visual-Only+dualpath (0.136) outperforms DAF+dualpath (0.093).
- **5-shot**: Dual-path gives modest gain on Visual-Only (+7.8%), slight hurt on DAF (-3.0%). Visual-Only baseline already stronger than DAF baseline.
- **10-shot**: Dual-path hurts both models on mAP50 (-14% Visual-Only, -2.4% DAF), but DAF+dualpath achieves best mAP50-95 (0.7500 vs 0.7447 baseline) and best mAP75 (0.8061).
- **Trend**: CLIP verification is most valuable in extreme low-shot regimes where visual features alone are unreliable. As shot count increases, the visual model becomes self-sufficient and CLIP noise dominates.
