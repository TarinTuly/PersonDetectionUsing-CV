import pickle
import numpy as np
from deepface import DeepFace

# ==========================================
# Configuration
# ==========================================

DATABASE_PATH = "embeddings/face_database.pkl"

MODEL_NAME = "ArcFace"

THRESHOLD = 0.70

# ==========================================
# Load Database
# ==========================================

with open(DATABASE_PATH, "rb") as f:
    database = pickle.load(f)

print("Face Database Loaded Successfully!")

# ==========================================
# Cosine Similarity
# ==========================================

def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ==========================================
# Face Recognition
# ==========================================

def recognize(face):

    if face is None:
        return "Unknown", 0.0

    if face.size == 0:
        return "Unknown", 0.0

    try:

        result = DeepFace.represent(

            img_path=face,

            model_name=MODEL_NAME,

            detector_backend="skip",
            enforce_detection=False,
            align=False

        )

        test_embedding = np.array(
            result[0]["embedding"]
        )

    except Exception:

        return "Unknown", 0.0

    scores = []

    # Compare against every person
    for person, embeddings in database.items():

        similarities = []

        for embedding in embeddings:

            similarity = cosine_similarity(
                test_embedding,
                embedding
            )

            similarities.append(similarity)

        # Take Top-3 Similarities
        top_k = min(3, len(similarities))

        score = np.mean(
            sorted(
                similarities,
                reverse=True
            )[:top_k]
        )

        scores.append(
            (person, score)
        )

    # Sort Descending
    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    best_person, best_score = scores[0]

    # Unknown Person
    if best_score < THRESHOLD:

        return "Unknown", float(best_score)

    return best_person, float(best_score)