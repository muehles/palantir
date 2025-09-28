import time
import os
from datetime import datetime
from picamera2 import Picamera2
import keyboard  # for keyboard input

# Folder for recordings
SAVE_DIR = "/home/pi/Videos/recordings"

# Global state
recording = False
cameras = []

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def start_recording():
    global cameras

    # Generate timestamped filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename0 = os.path.join(SAVE_DIR, f"camera0_{timestamp}.mp4")
    filename1 = os.path.join(SAVE_DIR, f"camera1_{timestamp}.mp4")

    # Initialize cameras
    cam0 = Picamera2(0)
    cam1 = Picamera2(1)

    config0 = cam0.create_video_configuration(main={"size": (1920, 1080)})
    config1 = cam1.create_video_configuration(main={"size": (1920, 1080)})

    cam0.configure(config0)
    cam1.configure(config1)

    cam0.start_recording(filename0)
    cam1.start_recording(filename1)

    cameras = [cam0, cam1]

    print(f"▶️ Recording started: {filename0}, {filename1}")

def stop_recording():
    global cameras

    for cam in cameras:
        cam.stop_recording()
        cam.close()

    cameras = []

    print("⏹ Recording stopped")

def toggle_recording():
    global recording

    if not recording:
        start_recording()
    else:
        stop_recording()

    recording = not recording


if __name__ == "__main__":
    print(f"Press 'r' to start/stop recording, 'q' to quit (videos saved in {SAVE_DIR})")

    try:
        while True:
            if keyboard.is_pressed("r"):
                toggle_recording()
                time.sleep(0.5)  # debounce so it doesn't trigger multiple times
            elif keyboard.is_pressed("q"):
                if recording:
                    stop_recording()
                print("Exiting.")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        if recording:
            stop_recording()
        print("Exiting.")