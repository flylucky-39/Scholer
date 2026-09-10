from .cosine_head import CosineConv2d, FSODDetect
from .objectness_head import FSODDetectWithObjectness, FSODObjectnessLoss, add_objectness_callback  # noqa: F401
from .prototype import extract_prototypes, init_cosine_head_with_prototypes
from .adaptation import (
    AdaptationMLP,
    TextModulatedPrototype,
    extract_base_prototypes,
    extract_florence2_text_embeddings,
    extract_target_weights,
    generate_and_encode_descriptions,
    init_cosine_head_fused,
    init_cosine_head_modulated,
    init_cosine_head_with_florence2,
    train_adaptation_mlp,
    train_modulation_network,
)
from .dual_path_fusion import fuse_box_scores
from .vlm_verifier import CLIPVerifier
from .background_suppression import (
    extract_background_prototype,
    refine_prototypes_with_bg_suppression,
)
from .learnable_bg_suppression import (
    LearnableBackgroundSuppression,
    train_learnable_bg_suppression,
    apply_learnable_bg_suppression,
)

__all__ = [
    "CosineConv2d",
    "FSODDetect",
    "extract_prototypes",
    "init_cosine_head_with_prototypes",
    "AdaptationMLP",
    "TextModulatedPrototype",
    "extract_base_prototypes",
    "extract_florence2_text_embeddings",
    "extract_target_weights",
    "generate_and_encode_descriptions",
    "init_cosine_head_fused",
    "init_cosine_head_modulated",
    "init_cosine_head_with_florence2",
    "train_adaptation_mlp",
    "train_modulation_network",
    "CLIPVerifier",
    "fuse_box_scores",
    "extract_background_prototype",
    "refine_prototypes_with_bg_suppression",
    "LearnableBackgroundSuppression",
    "train_learnable_bg_suppression",
    "apply_learnable_bg_suppression",
]
