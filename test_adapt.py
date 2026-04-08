from fsod.modules.adaptation import (
    AdaptationMLP,
    extract_florence2_text_embeddings,
    extract_base_prototypes,
    train_adaptation_mlp,
    init_cosine_head_with_florence2,
)
print("All imports OK")

mlp = AdaptationMLP(1024, 256, 128)
import torch
x = torch.randn(5, 1024)
y = mlp(x)
print(f"MLP forward: input {x.shape} -> output {y.shape}")
