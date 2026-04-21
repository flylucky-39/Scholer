"""Domain-Gap Estimator (DGE).

Computes a scalar g ∈ [0, 1] measuring how far the target domain has drifted
from the source domain, using a frozen image encoder (CLIP by default).

g ≈ 0  → target ~ source distribution (in-domain)
g ≈ 1  → target completely orthogonal to source (severe domain shift)

Usage::

    dge = DomainGapEstimator(device="cuda:0")
    g = dge.estimate_gap(source_image_paths, target_image_paths)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import torch


class _StubImageEncoder:
    """Fallback encoder using a fixed random projection of pixel statistics.

    Used only when no real CLIP-style encoder is installed; results are NOT
    research-grade but allow the pipeline to be exercised end-to-end on a
    laptop without the open_clip / transformers heavy dependency.
    """

    def __init__(self, out_dim: int = 256, seed: int = 3407):
        gen = torch.Generator().manual_seed(seed)
        # Map a 4x4 colour histogram (= 48 bins) into out_dim
        self.proj = torch.randn(48, out_dim, generator=gen)
        self.proj /= self.proj.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.out_dim = out_dim

    def encode_image(self, img: "torch.Tensor") -> torch.Tensor:
        # img: (B, 3, H, W) in [0, 1]
        B, C, H, W = img.shape
        gh, gw = 4, 4
        cell_h, cell_w = max(H // gh, 1), max(W // gw, 1)
        feats = []
        for c in range(C):
            for i in range(gh):
                for j in range(gw):
                    patch = img[:, c, i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                    feats.append(patch.mean(dim=(1, 2)))
        flat = torch.stack(feats, dim=1)  # (B, 48)
        return flat @ self.proj.to(flat.device)


class DomainGapEstimator:
    """Encode source/target images and compute their distributional distance.

    Backend selection (in priority order):
      1. ``open_clip`` ViT-B/32 (laion2b_s34b_b79k)  — preferred, real CLIP space
      2. ``clip`` (OpenAI)                          — fallback if open_clip missing
      3. internal ``_StubImageEncoder``             — last-resort, dev-only

    The chosen backend is logged on first encode.
    """

    def __init__(self, device: str = "cuda:0", img_size: int = 224, max_images: int = 200):
        self.device = device
        self.img_size = img_size
        self.max_images = max_images
        self._backend: str | None = None
        self._model = None
        self._preprocess = None
        self._init_backend()

    # ------------------------------------------------------------------ setup
    def _init_backend(self) -> None:
        try:
            import open_clip  # type: ignore

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model = model.to(self.device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._model = model
            self._preprocess = preprocess
            self._backend = "open_clip:ViT-B-32"
            return
        except Exception:
            pass

        try:
            import clip  # type: ignore

            model, preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
            for p in model.parameters():
                p.requires_grad_(False)
            self._model = model.eval()
            self._preprocess = preprocess
            self._backend = "openai_clip:ViT-B/32"
            return
        except Exception:
            pass

        self._model = _StubImageEncoder(out_dim=256)
        self._backend = "stub_random_proj (dev only — install open_clip for research use)"

    @property
    def backend(self) -> str:
        return self._backend or "<uninitialized>"

    # ----------------------------------------------------------------- encode
    @torch.no_grad()
    def _encode_batch(self, paths: Sequence[Path]) -> torch.Tensor:
        from PIL import Image

        if isinstance(self._model, _StubImageEncoder):
            tensors = []
            for p in paths:
                img = Image.open(p).convert("RGB").resize((self.img_size, self.img_size))
                arr = torch.from_numpy(_pil_to_array(img)).permute(2, 0, 1).float() / 255.0
                tensors.append(arr)
            batch = torch.stack(tensors)
            feats = self._model.encode_image(batch)
        else:
            tensors = []
            for p in paths:
                img = Image.open(p).convert("RGB")
                tensors.append(self._preprocess(img))
            batch = torch.stack(tensors).to(self.device)
            feats = self._model.encode_image(batch).float()

        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats.cpu()

    def encode_set(self, paths: Iterable[str | Path], seed: int = 3407) -> torch.Tensor:
        """Return a (N, D) matrix of L2-normalised image features."""
        path_list = [Path(p) for p in paths if Path(p).exists()]
        if not path_list:
            raise FileNotFoundError("No valid image paths provided to DomainGapEstimator.")
        if len(path_list) > self.max_images:
            rng = random.Random(seed)
            path_list = rng.sample(path_list, self.max_images)
        # Encode in mini-batches of 32 to avoid OOM on small GPUs.
        chunks = []
        for i in range(0, len(path_list), 32):
            chunks.append(self._encode_batch(path_list[i:i + 32]))
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------- gap
    def estimate_gap(
        self,
        source_paths: Iterable[str | Path],
        target_paths: Iterable[str | Path],
    ) -> float:
        """Return scalar domain gap g ∈ [0, 1].

        g = 1 - mean cosine similarity between mean source feature and target features.
        We compare each target image against the *centroid* of source images so that
        small target sample sizes (few-shot) don't dominate the estimate with noise.
        """
        src_feats = self.encode_set(source_paths)
        tgt_feats = self.encode_set(target_paths)

        src_centroid = src_feats.mean(dim=0)
        src_centroid = src_centroid / src_centroid.norm().clamp_min(1e-8)

        cos_sims = (tgt_feats @ src_centroid).clamp(-1.0, 1.0)
        # Map cosine [-1,1] → gap [0,1]: gap=0 when cos=1 (same domain), gap=1 when cos=-1 (orthogonal+).
        gaps = (1.0 - cos_sims) / 2.0
        return float(gaps.mean().item())


def _pil_to_array(img):  # pragma: no cover - tiny helper
    import numpy as np

    return np.array(img, dtype=np.uint8)


def estimate_domain_gap(
    source_paths: Iterable[str | Path],
    target_paths: Iterable[str | Path],
    device: str = "cuda:0",
    max_images: int = 200,
) -> tuple[float, str]:
    """Convenience wrapper. Returns (g, backend_name)."""
    dge = DomainGapEstimator(device=device, max_images=max_images)
    return dge.estimate_gap(source_paths, target_paths), dge.backend
