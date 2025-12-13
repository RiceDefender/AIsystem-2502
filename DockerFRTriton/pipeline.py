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


def process_single_image(client: Any, image_data: bytes) -> Tuple[np.ndarray, bool]:
    """
    Runs the full pipeline (Detection/Alignment/Antispoof/Embedding) for a single image.
    Returns the embedding and a boolean indicating if it passed the antispoof check.
    """
    input_data = image_data
    aligned_face = run_inference(client, "face_alignment", input_data)
    input_data = aligned_face

    # 2. Antispoofing Check
    is_real = True
    spoof_output = run_inference(client, "face_antispoof", input_data)
    is_real = spoof_output.item() == 1

    if not is_real:
        spoof_emb = np.zeros(256, dtype=np.float32)
        return spoof_emb, False

    emb_data = run_inference(client, "face_recognition", input_data)
    return emb_data.squeeze(0), is_real


def get_embeddings(client: Any, image_a: bytes, image_b: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Call Triton twice to obtain embeddings for two images.

    Extend this by adding detection/alignment/antispoof when those Triton models
    are available in the repository. For now we assume inputs are already aligned.
    """

    #emb_a = run_inference(client, image_a)
    #emb_b = run_inference(client, image_b)

    emb_a, is_real_a = process_single_image(client, image_a)
    emb_b, is_real_b = process_single_image(client, image_b)



    return emb_a.squeeze(0), emb_b.squeeze(0)


def calculate_face_similarity(client: Any, image_a: bytes, image_b: bytes) -> float:
    """
    Minimal end-to-end similarity using Triton-managed FR model.

    Students should swap in detection, alignment, and spoofing once those models
    are added to the Triton repository. This keeps all model execution on Triton.
    """
    emb_a, emb_b = get_embeddings(client, image_a, image_b)
    is_spoof_a = np.all(emb_a == 0)
    is_spoof_b = np.all(emb_b == 0)

    if is_spoof_a or is_spoof_b:
        print("Similarity check skipped: One or both embeddings are zero vectors (detected as spoof).")
        return 0.0
    return _cosine_similarity(emb_a, emb_b)