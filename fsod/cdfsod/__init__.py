"""Cross-Domain Few-Shot Object Detection (CD-FSOD) extension package.

Implements the proposal in docs/proposal_cdfsod_realtime.md:
- Domain-Gap Estimator (DGE)
- Domain-Adaptive Fusion (DAF)
- Cross-Domain Prototype Calibration (CDPC)
- CD-FSOD-Bench dataset registry
"""
from __future__ import annotations

from .datasets import CDFSOD_DATASETS, get_dataset_spec  # noqa: F401
from .domain_gap import DomainGapEstimator, estimate_domain_gap  # noqa: F401
from .calibration import alpha_from_gap, calibrate_prototypes, fuse_prototypes  # noqa: F401
