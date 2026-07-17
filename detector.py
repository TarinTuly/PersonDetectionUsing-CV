from ultralytics import YOLO

# ==========================================
# Load YOLO Face Detector
# ==========================================

model = YOLO("models/best.pt")


# ==========================================
# Face Detection
# ==========================================

def detect(frame):

    results = model.predict(
        source=frame,
        conf=0.50,
        verbose=False
    )

    detections = []

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face = frame[y1:y2, x1:x2]

            detections.append({

                "box": (x1, y1, x2, y2),

                "face": face,

                "confidence": float(box.conf[0])

            })

    return detections