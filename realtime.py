import cv2
import time

from detector import detect
from recognize import recognize

# ==========================================
# Webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Cannot open webcam!")

    exit()

print("Press Q to Quit")

# ==========================================
# FPS
# ==========================================

prev_time = time.time()

# ==========================================
# Frame Skip
# ==========================================

FRAME_SKIP = 5

frame_count = 0

last_results = []

# ==========================================
# Main Loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        break

    frame_count += 1

    # --------------------------------------
    # Recognition Every N Frames
    # --------------------------------------

    if frame_count % FRAME_SKIP == 0:

        detections = detect(frame)

        last_results = []

        for detection in detections:

            face = detection["face"]

            x1, y1, x2, y2 = detection["box"]

            yolo_conf = detection["confidence"]

            name, similarity = recognize(face)

            last_results.append({

                "box": (x1, y1, x2, y2),

                "name": name,

                "similarity": similarity,

                "yolo_conf": yolo_conf

            })

    # --------------------------------------
    # Draw Results
    # --------------------------------------

    for result in last_results:

        x1, y1, x2, y2 = result["box"]

        name = result["name"]

        similarity = result["similarity"]

        yolo_conf = result["yolo_conf"]

        if name == "Unknown":

            color = (0, 0, 255)

        else:

            color = (0, 255, 0)

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            color,

            2

        )

        cv2.putText(

            frame,

            f"{name}",

            (x1, y1 - 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            color,

            2

        )

        cv2.putText(

            frame,

            f"Match : {similarity:.2f}",

            (x1, y1 - 25),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            color,

            2

        )

        cv2.putText(

            frame,

            f"YOLO : {yolo_conf:.2f}",

            (x1, y1 - 5),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            color,

            2

        )

    # --------------------------------------
    # FPS
    # --------------------------------------

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(

        frame,

        f"FPS : {int(fps)}",

        (20, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 0),

        2

    )

    cv2.putText(

        frame,

        f"Faces : {len(last_results)}",

        (20, 70),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 255),

        2

    )

    cv2.imshow(

        "Real-Time Face Recognition",

        frame

    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

cap.release()

cv2.destroyAllWindows()