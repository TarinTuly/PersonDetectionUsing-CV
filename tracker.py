import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# ==========================================
# Load YOLO Model
# ==========================================

model = YOLO("yolov8n.pt")

# ==========================================
# Open Video
# ==========================================

cap = cv2.VideoCapture("video/walking.mp4")

if not cap.isOpened():
    print("Cannot open video")
    exit()

# ==========================================
# Store Track History
# ==========================================

track_history = defaultdict(list)

# ==========================================
# Main Loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        verbose=False

    )

    if len(results) == 0:
        continue

    boxes = results[0].boxes

    for box in boxes:

        if box.id is None:
            continue

        # ------------------------------
        # Bounding Box
        # ------------------------------

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # ------------------------------
        # Track ID
        # ------------------------------

        track_id = int(box.id[0])

        # ------------------------------
        # Confidence
        # ------------------------------

        confidence = float(box.conf[0])

        # ------------------------------
        # Center Point
        # ------------------------------

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # ------------------------------
        # Save History
        # ------------------------------

        history = track_history[track_id]

        history.append((center_x, center_y))

        # Keep only last 30 positions
        if len(history) > 30:
            history.pop(0)

        # ------------------------------
        # Random Color Based on ID
        # ------------------------------

        color = (

            (track_id * 37) % 255,

            (track_id * 17) % 255,

            (track_id * 97) % 255

        )

        # ------------------------------
        # Draw Bounding Box
        # ------------------------------

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            color,

            2

        )

        # ------------------------------
        # Draw Label
        # ------------------------------

        label = f"ID:{track_id}  {confidence:.2f}"

        cv2.putText(

            frame,

            label,

            (x1, y1 - 10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            color,

            2

        )

        # ------------------------------
        # Draw Current Center
        # ------------------------------

        cv2.circle(

            frame,

            (center_x, center_y),

            5,

            (0, 0, 255),

            -1

        )

        # ------------------------------
        # Draw Trail
        # ------------------------------

        points = np.array(
            history,
            dtype=np.int32
        )

        cv2.polylines(

            frame,

            [points],

            False,

            color,

            2

        )

    # ======================================
    # Show
    # ======================================

    cv2.imshow(
        "ByteTrack Motion Trail",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()