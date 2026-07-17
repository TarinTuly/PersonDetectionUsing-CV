import cv2
from ultralytics import YOLO

# ======================================
# Load Model
# ======================================

model = YOLO("yolov8n.pt")

# ======================================
# Open Video
# ======================================

cap = cv2.VideoCapture("video/walking.mp4")

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

    annotated = results[0].plot()

    cv2.imshow(
        "Tracking",
        annotated
    )

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()