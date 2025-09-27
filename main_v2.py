import cv2
import numpy as np
import csv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import time

# from tqdm import tqdm

def tqdm(x, **kwargs): return x  # disable tqdm progress bar if not installed

# === CONFIGURATION ===
VIDEO_NAME = "stpeter2"

VIDEO_PATH = VIDEO_NAME + ".mp4"
OUTPUT_VIDEO_BOXES = VIDEO_NAME + "_out_boxes.mp4"   # with boxes
OUTPUT_VIDEO_NOBOX = VIDEO_NAME + "_out_nobox.mp4"   # without boxes
OUTPUT_CSV = "player_positions.csv"
YOLO_MODEL = "yolo11m.pt"  # Or yolov11.pt if you have it

OUTPUT_RESOLUTION = (1920, 1080)#(1280, 720)#
CONFIDENCE_THRESHOLD = 0.05
EXCLUDE_KEEPERS = 1  # exclude 1 player on each side (left/right)
DETECTION_INTERVAL = 30     # run YOLO every N frames

# === Global for clicks ===
pitch_points = []

def click_event(event, x, y, flags, param):
    global pitch_points
    if event == cv2.EVENT_LBUTTONDOWN:
        pitch_points.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Pitch Corners", param)


def get_pitch_polygon(video_path):
    global pitch_points
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Could not read first frame of video")

    temp_frame = frame.copy()
    cv2.imshow("Select Pitch Corners", temp_frame)
    cv2.setMouseCallback("Select Pitch Corners", click_event, temp_frame)

    print("👉 Click the 4 corners of the pitch (clockwise or counter-clockwise). Press ENTER when done.")

    while True:
        key = cv2.waitKey(1)
        if key == 13:  # ENTER
            break
    cv2.destroyAllWindows()

    # if len(pitch_points) != 4:
    #     raise ValueError(f"Expected 4 points, got {len(pitch_points)}")

    pitch_polygon = np.array(pitch_points, dtype=np.int32)
    return pitch_polygon



# === KALMAN FILTER CLASS (for horizontal pan only) ===
class KalmanFilterX:
    def __init__(self, q=0.001, r=30, smoothing_window=20):
        self.q = q
        self.r = r
        self.x = None
        self.p = None
        self.history = []
        self.smoothing_window = smoothing_window

    def update(self, z):
        """z = [x_center]"""
        z = np.array([z], dtype=float)
        if self.x is None:
            self.x = z
            self.p = np.eye(1) * 1000
        else:
            self.p += np.eye(1) * self.q
            k = self.p @ np.linalg.inv(self.p + np.eye(1) * self.r)
            self.x += k @ (z - self.x)
            self.p = (np.eye(1) - k) @ self.p

        self.history.append(self.x.copy()[0])
        if len(self.history) > self.smoothing_window:
            self.history.pop(0)

        smoothed = np.mean(self.history, axis=0)
        return int(smoothed)

def main():

    start_time = time.time()

    model = YOLO(YOLO_MODEL)  # CPU/GPU auto-detect if desired
    tracker = DeepSort(max_age=30)
    kf = KalmanFilterX()

    # === NEW: get pitch polygon ===
    pitch_polygon = get_pitch_polygon(VIDEO_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writers
    out_boxes = cv2.VideoWriter(
        OUTPUT_VIDEO_BOXES,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        OUTPUT_RESOLUTION,
    )
    out_nobox = cv2.VideoWriter(
        OUTPUT_VIDEO_NOBOX,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        OUTPUT_RESOLUTION,
    )

    csv_data = []
    last_detections = []

    for frame_idx in range(frames):
        ret, frame = cap.read()
        if not ret:
            break

        # --- Progress output ---
        if frame_idx % 5 == 0 or frame_idx == frames - 1:
            percent = (frame_idx + 1) / frames * 100
            print(f"Processing frame {frame_idx+1}/{frames} ({percent:.1f}%)")

        detections = []

        if frame_idx % DETECTION_INTERVAL == 0:
            results = model(frame, verbose=False, imgsz=640)[0]
            last_detections = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                if cls_id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    last_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        detections = last_detections

        tracks = tracker.update_tracks(detections, frame=frame)
        player_boxes = []

        # Make a copy of frame for drawing boxes
        frame_with_boxes = frame.copy()
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # === NEW: only keep if inside pitch polygon ===
            inside = cv2.pointPolygonTest(pitch_polygon, (x1, y1), False) or cv2.pointPolygonTest(pitch_polygon, (x2, y2), False)
            if inside >= 0:
                player_boxes.append([x1, y1, x2, y2])

            # Draw boxes on copy
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_with_boxes,
                f"ID:{track.track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            csv_data.append(
                {"frame": frame_idx, "id": track.track_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            )

        # --- Constant zoom (full height, horizontal pan only) ---
        if len(player_boxes) > 0:
            player_boxes = np.array(player_boxes)
            x_centers = (player_boxes[:, 0] + player_boxes[:, 2]) / 2
            sorted_indices = np.argsort(x_centers)
            if len(sorted_indices) > 2 * EXCLUDE_KEEPERS:
                filtered_boxes = player_boxes[sorted_indices[EXCLUDE_KEEPERS:-EXCLUDE_KEEPERS]]
            else:
                filtered_boxes = player_boxes

            # average horizontal center of players
            player_centers = (filtered_boxes[:, :2] + filtered_boxes[:, 2:]) / 2
            center_x = np.mean(player_centers[:, 0])

            # smooth horizontal pan only
            smoothed_center_x = kf.update(center_x)

            # crop box = full height, fixed width
            target_width = int(fh * (OUTPUT_RESOLUTION[0] / OUTPUT_RESOLUTION[1]))
            half_width = target_width // 2

            x1 = smoothed_center_x - half_width
            x2 = smoothed_center_x + half_width
            y1, y2 = 0, fh  # full height

            # clamp horizontally
            if x1 < 0:
                x1, x2 = 0, target_width
            if x2 > fw:
                x2, x1 = fw, fw - target_width

            cropped_box = frame_with_boxes[y1:y2, x1:x2]
            cropped_nobox = frame[y1:y2, x1:x2]

            resized_box = cv2.resize(cropped_box, OUTPUT_RESOLUTION)
            resized_nobox = cv2.resize(cropped_nobox, OUTPUT_RESOLUTION)

        else:
            resized_box = cv2.resize(frame_with_boxes, OUTPUT_RESOLUTION)
            resized_nobox = cv2.resize(frame, OUTPUT_RESOLUTION)

        out_boxes.write(resized_box)
        out_nobox.write(resized_nobox)

    # Save CSV data
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "id", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(csv_data)

    cap.release()
    out_boxes.release()
    out_nobox.release()

    print(f"\n✅ Output video with boxes saved: {OUTPUT_VIDEO_BOXES}")
    print(f"✅ Output video without boxes saved: {OUTPUT_VIDEO_NOBOX}")
    print(f"📄 Player positions CSV saved: {OUTPUT_CSV}")

    print(f"Speed: {frames/(time.time()-start_time)} fps // total time: {time.time()-start_time}")


if __name__ == "__main__":
    main()
