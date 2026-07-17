import os
import pickle
from deepface import DeepFace

# ==========================================
# Configuration
# ==========================================

KNOWN_FACES_DIR = "knownfaces"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "face_database.pkl")

MODEL_NAME = "ArcFace"

# ==========================================
# Create output folder
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Face Database
# ==========================================

face_database = {}

# ==========================================
# Process each person
# ==========================================

for person_name in os.listdir(KNOWN_FACES_DIR):

    person_path = os.path.join(KNOWN_FACES_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    print("\n===================================")
    print(f"Processing: {person_name}")
    print("===================================")

    embeddings = []

    # Process each image
    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        try:

            result = DeepFace.represent(

                img_path=image_path,

                model_name=MODEL_NAME,

                detector_backend="opencv",

                enforce_detection=True,

                align=True

            )

            embedding = result[0]["embedding"]

            embeddings.append(embedding)

            print(f"✓ {image_name}")

        except Exception as e:

            print(f"✗ {image_name}")
            print(e)

    face_database[person_name] = embeddings

    print(f"Saved {len(embeddings)} embeddings.")

# ==========================================
# Save Database
# ==========================================

with open(OUTPUT_FILE, "wb") as f:

    pickle.dump(face_database, f)

print("\n===================================")
print("Database Created Successfully!")
print(f"Saved to: {OUTPUT_FILE}")
print("===================================")

# ==========================================
# Statistics
# ==========================================

print("\nSummary")

total_people = len(face_database)

total_embeddings = sum(len(v) for v in face_database.values())

print(f"People      : {total_people}")
print(f"Embeddings  : {total_embeddings}")

for person in face_database:
    print(f"{person} -> {len(face_database[person])} images")