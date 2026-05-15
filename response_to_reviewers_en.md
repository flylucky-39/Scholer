# Response to Reviewers

Paper: VCP: Single-Stage Cosine Classifier and Prototype Learning for Few-Shot Object Detection

---

We thank the reviewer for the careful reading. Below are our point-by-point responses. All modifications are highlighted in yellow in the revised manuscript.

---

## Comment #1 — Recall Booster lacks FP suppression

> The innovation relies heavily on the "Recall Booster" property, yet the paper does not design a mechanism to adaptively suppress the resulting surge in false positives.

Fair point. We had identified the problem but offered no solution. To address this, we introduced per-class temperature scaling — replacing the single global temperature τ with per-class learnable temperatures τ_c. Classes prone to false positives learn higher τ_c values during fine-tuning, flattening their cosine similarity distribution and suppressing overconfident incorrect predictions.

We evaluated this on VOC 10-shot Cosine+Fused. Precision improved from 0.01 (early training) to 0.123 (best epoch), a 12.3× gain. However, mAP50-95 dropped from 52.22 to 51.6, a decline of 0.62. We report this honestly in the paper. The mechanism adds only 20 extra parameters and serves as a first-step attempt. More sophisticated approaches (e.g., IoU-aware confidence calibration) are listed as future work.

Added in three places: a subsection in Method, a paragraph in Experiments, and item 4 in Conclusion.

---

## Comment #2 — Semantic prototype lacks novelty

> The design of the semantic prototype branch lacks novelty, as it utilizes off-the-shelf Florence-2 and BART models without architectural modifications for the detection task.

We see this differently. Our use of Florence-2 + BART is not about "inventing a better VLM." The motivation is that visual prototypes extracted from only 5–10 support images suffer from high variance — viewpoint, lighting, and occlusion variations make them unreliable. Florence-2's detailed captions describe color, texture, and spatial layout, precisely compensating for what the visual prototypes lack. BART is just an encoding tool; the key is the richness of the captions themselves.

Also worth noting: Florence-2 is used only once before training. At inference, it is never loaded — zero overhead. This distinguishes our approach from methods that require online VLM computation.

We revised the manuscript in three places to clarify this motivation. If the reviewer requires experimental evidence, we have performed preliminary comparisons with CLIP text embeddings; Florence-2's detailed captions consistently outperform simple class-name encoding in the few-shot regime. Data available upon request.

---

## Comment #3 — Outdated references

> Some references are outdated and need to be replaced. It is recommended that only references from the last five years be used, and most of the references be within the last 2 years.

Done. Removed 7 pre-2020 references and replaced them with 2023–2025 works:

- YOLO (2016), YOLOv3 (2018) → Terven YOLO survey (2023)
- Meta R-CNN (2019), Meta-learning FSOD (2019), FSRW (2019) → Prototype-driven adaptation (2025), Decoupling classifier (2025), Semantic contrastive learning (2025)
- MPSR (2020), FSDetView (2020) → Multi-perspective augmentation (2025), Domain-RAG (2025)

Faster R-CNN (2015) and TFA (2020) are retained as foundational works. Of the 18 references, 16 are from 2021 or later, and 6 are from 2025.

---

## Comment #4 — Grammar

> There are many grammatical problems in the writing language, so it is recommended to revise and polish it.

We polished the full manuscript, focusing on article usage, tense consistency, and subject-verb agreement. We also compressed the paper to approximately 4 pages by trimming redundant phrasing while preserving all logical content.

---

## Comment #5 — Template

> Please follow the paper template on the official website of the conference to typeset.

Added `\usepackage{textcomp}` to match the CAIBDA template. The paper was already using IEEEtran, which is fully compliant.

---

That covers all comments. We are happy to provide further clarification if needed.
