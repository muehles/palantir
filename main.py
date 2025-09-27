import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque

# === CONFIGURATION === #
VIDEO_PATH = "test2.mp4"
OUTPUT_PATH = "test2out.mp4"
MODEL_PATH = "yolov8x.pt"  # Replace with yolov11.pt if available
FRAME_SIZE = (1280, 720)   # Output resolution
SMOOTH_WINDOW = 5          # Frames to average for smooth framing
PADDING = 60               # Padding around players

# === INITIALIZE YOLO + DeepSORT === #
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=30)

# === VIDEO SETUP === #
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    FRAME_SIZE
)

# For smoothing bounding boxes
bbox_history = deque(maxlen=SMOOTH_WINDOW)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0 and conf > 0.3:  # "person"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Run DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    player_boxes = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        ltrb = track.to_ltrb()  # left, top, right, bottom
        x1, y1, x2, y2 = map(int, ltrb)
        player_boxes.append([x1, y1, x2, y2])

    if player_boxes:
        player_boxes = np.array(player_boxes)
        x1_all, y1_all = np.min(player_boxes[:, 0]), np.min(player_boxes[:, 1])
        x2_all, y2_all = np.max(player_boxes[:, 2]), np.max(player_boxes[:, 3])

        # Padding and clamp to image bounds
        x1_all = max(0, x1_all - PADDING)
        y1_all = max(0, y1_all - PADDING)
        x2_all = min(width, x2_all + PADDING)
        y2_all = min(height, y2_all + PADDING)

        bbox_history.append([x1_all, y1_all, x2_all, y2_all])
        smoothed_box = np.mean(bbox_history, axis=0).astype(int)
        sx1, sy1, sx2, sy2 = smoothed_box

        # Crop and resize
        cropped = frame[sy1:sy2, sx1:sx2]
        resized = cv2.resize(cropped, FRAME_SIZE)

    else:
        # No players: fallback to full frame
        resized = cv2.resize(frame, FRAME_SIZE)

    out.write(resized)
    frame_idx += 1
    print(f"Processed frame {frame_idx}", end="\r")

cap.release()
out.release()
print("\n✅ Video processing complete!")