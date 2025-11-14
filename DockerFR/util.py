import cv2
import numpy as np
from typing import Any, List
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# Initialize
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def detect_faces(image: Any) -> List[Any]:
    """
        Detect faces within the provided image.

        Return list of cropped faces.
        """
    if isinstance(image, bytes):
        npimg = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    faces = app.get(image)
    crops = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        crops.append(image[y1:y2, x1:x2])
    return crops

def compute_face_embedding(face_image: Any) -> Any:
    """
     Compute a numerical embedding vector for the provided face image.
    """
    faces = app.get(face_image)
    if not faces:
        return None
    return faces[0].embedding


def detect_face_keypoints(face_image: Any) -> Any:
    """
    Identify facial keypoints (landmarks) for alignment or analysis.
    """
    faces = app.get(face_image)
    if not faces:
        return None
    return faces[0].kps  # function not used.


def warp_face(image: Any, homography_matrix: Any) -> Any:
    """
    Warp the provided face image using the supplied homography matrix.
    """
    h, w = image.shape[:2]
    aligned = cv2.warpPerspective(image, homography_matrix, (w, h))
    return aligned # function not used.



def antispoof_check(face_image: Any) -> float:
    """
    Perform an anti-spoofing check and return a confidence score.

    A higher score should indicate a higher likelihood that the face is real.
    """
    raise NotImplementedError("Student implementation required for face anti-spoofing")


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    Full pipeline: detect, align, embed, compare.
    """
    # Decode if needed
    if isinstance(image_a, bytes):
        npimg = np.frombuffer(image_a, np.uint8)
        image_a = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if isinstance(image_b, bytes):
        npimg = np.frombuffer(image_b, np.uint8)
        image_b = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    faces_a = detect_faces(image_a)
    faces_b = detect_faces(image_b)
    if not faces_a or not faces_b:
        raise ValueError("No face detected in one or both images.")

    # Detect
    emb_a = compute_face_embedding(image_a)
    emb_b = compute_face_embedding(image_b)

    if emb_a is None or emb_b is None:
        raise ValueError("Face not detected in one or both images")

    # Compute cosine similarity (1 - distance)
    similarity = 1 - cosine(emb_a, emb_b)
    return float(similarity)
