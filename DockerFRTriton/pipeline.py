from typing import Any, Tuple

import numpy as np

from triton_service import run_inference


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    a_norm = np.linalg.norm(vec_a)
    b_norm = np.linalg.norm(vec_b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (a_norm * b_norm))


def get_embeddings(client: Any, image_a: bytes, image_b: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Call Triton twice to obtain embeddings for two images.

    Extend this by adding detection/alignment/antispoof when those Triton models
    are available in the repository. For now we assume inputs are already aligned.
    """
    emb_a = run_inference(client, image_a)
    emb_b = run_inference(client, image_b)
    return emb_a.squeeze(0), emb_b.squeeze(0)


def calculate_face_similarity(client: Any, image_a: bytes, image_b: bytes) -> float:
    """
    Minimal end-to-end similarity using Triton-managed FR model.

    Students should swap in detection, alignment, and spoofing once those models
    are added to the Triton repository. This keeps all model execution on Triton.
    """
    emb_a, emb_b = get_embeddings(client, image_a, image_b)
    return _cosine_similarity(emb_a, emb_b)
