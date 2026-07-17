import os
import pickle
import numpy as np
from deepface import DeepFace

# ==========================================
# Configuration
# ==========================================

DATABASE_PATH = "embeddings/face_database.pkl"
TEST_FOLDER = "test"

MODEL_NAME = "ArcFace"
THRESHOLD = 0.25

# ==========================================
# Load Database
# ==========================================

with open(DATABASE_PATH, "rb") as f:
    database = pickle.load(f)

print("Database Loaded Successfully!\n")

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
# Process Every Test Image
# ==========================================

for image_name in os.listdir(TEST_FOLDER):

    image_path = os.path.join(TEST_FOLDER, image_name)

    if not image_name.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ):
        continue

    print("=" * 60)
    print(f"Testing : {image_name}")
    print("=" * 60)

    try:

        result = DeepFace.represent(

            img_path=image_path,

            model_name=MODEL_NAME,

            detector_backend="opencv",

            enforce_detection=True,

            align=True

        )

        test_embedding = np.array(result[0]["embedding"])

    except Exception as e:

        print("Face not detected!")
        print(e)
        print()

        continue

    best_person = "Unknown"
    best_score = -1

    # Compare with every person

    for person in database:

        similarities = []

        for embedding in database[person]:

            score = cosine_similarity(
                test_embedding,
                embedding
            )

            similarities.append(score)

        average_score = np.mean(similarities)

        print(
            f"{person:<10} : {average_score:.4f}"
        )

        if average_score > best_score:

            best_score = average_score
            best_person = person

    print()

    if best_score >= THRESHOLD:

        print(f"Prediction : {best_person}")
        print(f"Similarity : {best_score:.4f}")
        print(f"Confidence : {best_score*100:.2f}%")

    else:

        print("Prediction : Unknown")
        print(f"Highest Similarity : {best_score:.4f}")

    print()