import cv2
import numpy as np
# import csv
from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import time

print(torch.cuda.is_available())       # should print True
print(torch.__version__)
print(torch.cuda.get_device_name(0))   # should print your GPU name

# from tqdm import tqdm

def tqdm(x, **kwargs): return x  # disable tqdm progress bar if not installed

# === CONFIGURATION ===
VIDEO_NAME = "stpeter5"
INPUT_DIR = "input_video"
OUTPUT_DIR = "output_video"

VIDEO_PATH = INPUT_DIR + "/" + VIDEO_NAME + ".mp4"
OUTPUT_VIDEO_BOXES = OUTPUT_DIR + "/" + VIDEO_NAME + "_out_boxes.mp4"   # with boxes
OUTPUT_VIDEO_NOBOX = OUTPUT_DIR + "/" + VIDEO_NAME + "_out_nobox.mp4"   # without boxes
YOLO_MODEL = "yolo11s.pt"#"runs/detect/train6/weights/best.pt" #"yolo11s.pt"  # Or yolov11.pt if you have it

OUTPUT_RESOLUTION = (1920, 1080)#(1280, 720)#
CONFIDENCE_THRESHOLD = 0.01
EXCLUDE_KEEPERS = 1  # exclude 1 player on each side (left/right)
DETECTION_INTERVAL = 30  # run YOLO every N frames

OUTPUT_BOXES = True
SPREAD_FACTOR = 1.2

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

    pitch_polygon = np.array(pitch_points, dtype=np.int32)
    return pitch_polygon



class KalmanFilterX:
    def __init__(self, q=0.003, r=25, smoothing_window=20):
        self.q = q
        self.r = r
        self.x = None
        self.p = None
        self.history = []
        self.smoothing_window = smoothing_window

    def update(self, z):
        z = np.array(z, dtype=float).reshape(-1)  # handle scalar or vector
        if self.x is None:
            self.x = z
            self.p = np.eye(len(z)) * 1000
        else:
            self.p += np.eye(len(z)) * self.q
            k = self.p @ np.linalg.inv(self.p + np.eye(len(z)) * self.r)
            self.x += k @ (z - self.x)
            self.p = (np.eye(len(z)) - k) @ self.p

        self.history.append(self.x.copy())
        if len(self.history) > self.smoothing_window:
            self.history.pop(0)

        smoothed = np.mean(self.history, axis=0)
        return smoothed if len(smoothed) > 1 else smoothed[0]

def main():

    start_time = time.time()
    prev_time = time.time()

    model = YOLO(YOLO_MODEL).to("cuda")  # CPU/GPU auto-detect if desired
    kf_pan = KalmanFilterX(smoothing_window=30)   # faster pan
    kf_zoom = KalmanFilterX(smoothing_window=60)  # slower zoom

    # === NEW: get pitch polygon ===
    pitch_polygon = get_pitch_polygon(VIDEO_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writers
    if OUTPUT_BOXES:
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

    last_detections = []
    last_crop_box = None
    prev_half_height = fh/2

    aspect_ratio = OUTPUT_RESOLUTION[0] / OUTPUT_RESOLUTION[1]
    crop_size = (fh*aspect_ratio, fh) 
    crop_center = (crop_size[0]/2, crop_size[1]/2)

    for frame_idx in range(frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % DETECTION_INTERVAL == 0:
            results = model(frame, verbose=False, imgsz=1920)[0]
            last_detections = []
            last_detections_outside = []

            # logging
            percent = (frame_idx + 1) / frames * 100
            print(f"Processing frame {frame_idx+1}/{frames} at {DETECTION_INTERVAL/(time.time()-prev_time):.2f} fps ({percent:.1f}%)")

            prev_time = time.time()

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                if cls_id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # === NEW: only keep if inside pitch polygon ===
                    inside = cv2.pointPolygonTest(pitch_polygon, (x1, y1), False) and cv2.pointPolygonTest(pitch_polygon, (x2, y2), False)
                    if inside >= 0:
                        # last_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))
                        last_detections.append([x1, y1, x2, y2])

                        # Draw boxes on copy
                        # cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    else:
                        last_detections_outside.append([x1, y1, x2, y2])
                        # cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
        player_boxes = last_detections

        if len(player_boxes) > 0:
            player_boxes = np.array(player_boxes)
            x_centers = (player_boxes[:, 0] + player_boxes[:, 2]) / 2
            sorted_indices = np.argsort(x_centers)
            if len(sorted_indices) > 2 * EXCLUDE_KEEPERS:
                filtered_boxes = player_boxes[sorted_indices[EXCLUDE_KEEPERS:-EXCLUDE_KEEPERS]]
            else:
                filtered_boxes = player_boxes

            # player center
            player_centers = (filtered_boxes[:, :2] + filtered_boxes[:, 2:]) / 2
            center_x = np.mean(player_centers[:, 0])
            center_y = np.mean(player_centers[:, 1])

            # spread for zoom
            distances = np.linalg.norm(player_centers - [center_x, center_y], axis=1)
            spread = np.max(distances) if len(distances) > 0 else fh // 2

            # --- Zoom logic ---
            aspect_ratio = OUTPUT_RESOLUTION[0] / OUTPUT_RESOLUTION[1]
            half_height = min(fh // 2, max(fh // 4, spread * SPREAD_FACTOR))

            # smooth separately
            smoothed_center_x, smoothed_center_y = kf_pan.update([center_x, center_y])

            if len(player_boxes) >= 3:
                smoothed_half_height = kf_zoom.update([half_height])
            else:
                smoothed_half_height = prev_half_height


            prev_half_height = smoothed_half_height
            smoothed_half_width  = smoothed_half_height * aspect_ratio

            # Crop size as float
            crop_size = (2 * smoothed_half_width, 2 * smoothed_half_height)  # (width, height)

            # Clamp crop center to avoid going out of frame
            clamped_center_x = np.clip(smoothed_center_x, smoothed_half_width, fw - smoothed_half_width)
            clamped_center_y = np.clip(smoothed_center_y, smoothed_half_height, fh - smoothed_half_height)
            crop_center = (clamped_center_x, clamped_center_y)

            # Subpixel crop using getRectSubPix
            cropped_nobox = cv2.getRectSubPix(frame, patchSize=(int(crop_size[0]), int(crop_size[1])), center=crop_center)

            # Resize to output resolution
            resized_nobox = cv2.resize(cropped_nobox, OUTPUT_RESOLUTION, interpolation=cv2.INTER_LINEAR)

            last_crop_box = (crop_center[0], crop_center[1], smoothed_half_width, smoothed_half_height)

        else:
            # Fallback to last known crop
            if last_crop_box is not None:
                cx, cy, hw, hh = last_crop_box
                crop_size = (2 * hw, 2 * hh)
                crop_center = (cx, cy)
                cropped_nobox = cv2.getRectSubPix(frame, patchSize=(int(crop_size[0]), int(crop_size[1])), center=crop_center)
                resized_nobox = cv2.resize(cropped_nobox, OUTPUT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
            else:
                # First frame fallback
                resized_nobox = cv2.resize(frame, OUTPUT_RESOLUTION, interpolation=cv2.INTER_LINEAR)

        
        if OUTPUT_BOXES:
            scale_x = OUTPUT_RESOLUTION[0] / crop_size[0]
            scale_y = OUTPUT_RESOLUTION[1] / crop_size[1]
            resized_box = resized_nobox.copy()
            for x1, y1, x2, y2 in last_detections:
                # transform coordinates to resized frame  
                x1_r = int((x1 - (crop_center[0] - crop_size[0]/2)) * scale_x)
                y1_r = int((y1 - (crop_center[1] - crop_size[1]/2)) * scale_y)
                x2_r = int((x2 - (crop_center[0] - crop_size[0]/2)) * scale_x)
                y2_r = int((y2 - (crop_center[1] - crop_size[1]/2)) * scale_y)
                cv2.rectangle(resized_box, (x1_r, y1_r), (x2_r, y2_r), (0,255,0), 2)

            for x1, y1, x2, y2 in last_detections_outside:
                x1_r = int((x1 - (crop_center[0] - crop_size[0]/2)) * scale_x)
                y1_r = int((y1 - (crop_center[1] - crop_size[1]/2)) * scale_y)
                x2_r = int((x2 - (crop_center[0] - crop_size[0]/2)) * scale_x)
                y2_r = int((y2 - (crop_center[1] - crop_size[1]/2)) * scale_y)
                cv2.rectangle(resized_box, (x1_r, y1_r), (x2_r, y2_r), (0,0,255), 2)
            
            out_boxes.write(resized_box)

        out_nobox.write(resized_nobox)

    cap.release()
    if OUTPUT_BOXES:
        out_boxes.release()
    out_nobox.release()

    if OUTPUT_BOXES:
        print(f"\n✅ Output video with boxes saved: {OUTPUT_VIDEO_BOXES}")

    print(f"✅ Output video without boxes saved: {OUTPUT_VIDEO_NOBOX}")
    print(f"Speed: {frames/(time.time()-start_time)} fps // total time: {time.time()-start_time}")


if __name__ == "__main__":
    main()
