import time
import csv
import math
import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from datetime import datetime

from analyze_plot import plot

# ---------------------------------------------------------------------------
# Landmark indices (same numbering as legacy Face Mesh with refine_landmarks)
# ---------------------------------------------------------------------------
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]

LEFT_EYE_CORNERS = (362, 263)   # inner, outer
RIGHT_EYE_CORNERS = (133, 33)   # inner, outer

LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468

EAR_BLINK_THRESHOLD = 0.21
CONSEC_FRAMES_FOR_BLINK = 2

MODEL_PATH = "face_landmarker.task"


def euclidean(p1, p2):
    return math.dist(p1, p2)


def eye_aspect_ratio(landmarks, idx):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idx]
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def gaze_ratio(landmarks, iris_idx, corner_idx):
    iris = landmarks[iris_idx]
    inner = landmarks[corner_idx[0]]
    outer = landmarks[corner_idx[1]]
    eye_width = outer[0] - inner[0]
    if eye_width == 0:
        return 0.5
    ratio = (iris[0] - inner[0]) / eye_width
    return max(0.0, min(1.0, ratio))


def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions/index.")
        return

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    filename = f"gaze_log_{timestamp}"

    csv_file = open(f"logs/{filename}.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "left_ear", "right_ear", "avg_ear",
                      "gaze_ratio_left", "gaze_ratio_right", "blink"])

    blink_counter = 0
    total_blinks = 0
    start_time = time.time()

    print("Starting capture. Press 'q' in the video window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks]

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_EAR_IDX)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_EAR_IDX)
            avg_ear = (left_ear + right_ear) / 2.0

            gaze_left = gaze_ratio(landmarks, LEFT_IRIS_CENTER, LEFT_EYE_CORNERS)
            gaze_right = gaze_ratio(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS)

            blink = 0
            if avg_ear < EAR_BLINK_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= CONSEC_FRAMES_FOR_BLINK:
                    total_blinks += 1
                    blink = 1
                blink_counter = 0

            writer.writerow([time.time() - start_time, left_ear, right_ear,
                              avg_ear, gaze_left, gaze_right, blink])

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {total_blinks}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Gaze L/R: {gaze_left:.2f}/{gaze_right:.2f}",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            for idx in LEFT_EYE_EAR_IDX + RIGHT_EYE_EAR_IDX:
                x, y = landmarks[idx]
                cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
            for idx in [LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER]:
                x, y = landmarks[idx]
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Gaze & Blink Tracker (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"Done. Logged data to {filename}.csv. Total blinks: {total_blinks}")
    plot(filename)


if __name__ == "__main__":
    main()