import numpy as np
import cv2
from typing import Any, Tuple, Optional
from triton_service import run_inference

# --- Configuration ---
# Thresholds
SPOOF_THRESHOLD = 0.5  # Score below this is considered a spoof
DETECTION_THRESHOLD = 0.5

# Model Names (Must match your Triton repo config.pbtxt files)
MODEL_DETECTOR = "face_detector"
MODEL_ALIGNMENT = "face_alignment"  # Or perform strictly client-side
MODEL_ANTISPOOF = "face_antispoof"
MODEL_RECOGNITION = "fr_model"


def detect_face(client: Any, image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Calls the detection model to find the primary face.
    Returns: 5 Landmarks points (eyes, nose, mouth) for the largest face found.
    """
    results = run_inference(
        client=client,
        model_name=MODEL_DETECTOR,
        image_bytes=image_bytes,
        input_name="input.1",
        output_names=["bbox", "kps"],
        model_image_size=(640, 640)
    )

    bboxes = results["bbox"]
    landmarks = results["kps"]

    # Filter
    valid_indices = np.where(bboxes[:, 4] > DETECTION_THRESHOLD)[0]
    if len(valid_indices) == 0:
        return None
    best_idx = -1
    max_area = 0
    for idx in valid_indices:
        box = bboxes[idx]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > max_area:
            max_area = area
            best_idx = idx

    return landmarks[best_idx]


def align_face(client: Any, image_bytes: bytes, landmarks: np.ndarray) -> bytes:
    """
    Calls an alignment model (or Python backend) to warp the face.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)

    if landmarks.shape == (10,):
        dst = landmarks.reshape(5, 2).astype(np.float32)
    else:
        dst = landmarks.astype(np.float32)

    tform = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)[0]
    aligned_img = cv2.warpAffine(img, tform, (112, 112), borderValue=0.0)

    success, encoded_img = cv2.imencode('.jpg', aligned_img)
    return encoded_img.tobytes()


def check_spoofing(client: Any, aligned_image_bytes: bytes) -> bool:
    """
    Calls the Anti-Spoofing model (e.g., MiniFASNet).
    Returns: True if Real, False if Spoof.
    """
    result = run_inference(
        client=client,
        model_name=MODEL_ANTISPOOF,
        image_bytes=aligned_image_bytes,
        input_name="input",  # Check your config.pbtxt
        output_names="score",
        model_image_size=(80, 80)  # Antispoof models often use smaller inputs (80x80 or 128x128)
    )
    spoof_score = result.squeeze()
    is_real = spoof_score > SPOOF_THRESHOLD
    return is_real


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    a_norm = np.linalg.norm(vec_a)
    b_norm = np.linalg.norm(vec_b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (a_norm * b_norm))


def get_secure_embedding(client: Any, image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Orchestrates the pipeline: Detect -> Align -> Spoof Check -> Recognize
    """
    landmarks = detect_face(client, image_bytes)
    if landmarks is None:
        print("No face detected.")
        return None
    aligned_bytes = align_face(client, image_bytes, landmarks)
    if not check_spoofing(client, aligned_bytes):
        print("Spoof detected! Rejecting image.")
        return None
    emb = run_inference(
        client=client,
        model_name=MODEL_RECOGNITION,
        image_bytes=aligned_bytes,
        input_name="input.1",
        output_names="516",
        model_image_size=(112, 112),
    )
    return emb.squeeze(0)


def calculate_face_similarity(client: Any, image_a: bytes, image_b: bytes) -> float:
    """
    Robust end-to-end similarity with full pipeline checks.
    """
    emb_a = get_secure_embedding(client, image_a)

    emb_b = get_secure_embedding(client, image_b)

    if emb_a is None or emb_b is None:
        return 0.0

    return _cosine_similarity(emb_a, emb_b)