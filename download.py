"""
download_model.py

Downloads the MediaPipe FaceLandmarker model file (~4MB, free, from Google's
public model storage). Run this once before gaze_tracker.py.
"""

import urllib.request
import os

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"


def main():
    if os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} already exists, skipping download.")
        return
    print(f"Downloading model from {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()