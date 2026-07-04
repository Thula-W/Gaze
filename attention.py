import time
import csv
import math
import os
import random
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from datetime import datetime

from gaze_tracker import (
    LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
    LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
    LEFT_EYE_VERT, RIGHT_EYE_VERT,
    EAR_BLINK_THRESHOLD, CONSEC_FRAMES_FOR_BLINK,
    MODEL_PATH, eye_aspect_ratio,
    get_absolute_gaze, vertical_gaze_ratio, plot
)

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
        print("ERROR: Could not open webcam.")
        return

    # Setup file logging
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    filename = f"attention_log_{timestamp}"

    csv_file = open(f"logs/{filename}.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "timestamp", "left_ear", "right_ear", "avg_ear",
        "gaze_ratio_left", "gaze_ratio_right", "blink", 
        "vertical_gaze_ratio_left", "vertical_gaze_ratio_right", "attention_score"
    ])

    blink_counter = 0
    total_blinks = 0
    
    # Initialize a dark canvas window
    canvas_w, canvas_h = 1280, 720
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Timers and states
    start_time = time.time()
    state = "CALIBRATION"

    baseline_h_samples = []
    baseline_v_samples = []
    anchor_h = 0.5
    anchor_v = 0.5

    # Random target coordinates (saved for future use)
    target_x = random.randint(100, canvas_w - 100)
    target_y = random.randint(100, canvas_h - 100)

    print("Experiment started. Focus on the Attention Stimulus Window.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)
        
        # Reset the canvas graphic window each frame
        canvas.fill(0) 
        elapsed = time.time() - start_time

        # Initialize telemetry metrics per frame
        gaze_left, gaze_right = 0.5, 0.5
        v_gaze_left, v_gaze_right = 0.5, 0.5
        avg_ear = 0.0
        left_ear, right_ear = 0.0, 0.0
        blink = 0
        attention_score = 0.0

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks]

            # Re-using original formulas via imports
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_EAR_IDX)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_EAR_IDX)
            avg_ear = (left_ear + right_ear) / 2.0

            gaze_right = get_absolute_gaze(landmarks[RIGHT_IRIS_CENTER], landmarks[33], landmarks[133])
            gaze_left = get_absolute_gaze(landmarks[LEFT_IRIS_CENTER], landmarks[362], landmarks[263])

            v_gaze_left = vertical_gaze_ratio(landmarks, LEFT_IRIS_CENTER, LEFT_EYE_VERT[0], LEFT_EYE_VERT[1])
            v_gaze_right = vertical_gaze_ratio(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_VERT[0], RIGHT_EYE_VERT[1])

            current_h = (gaze_left + gaze_right) / 2.0
            current_v = (v_gaze_left + v_gaze_right) / 2.0

            # Blink Detection tracking logic
            if avg_ear < EAR_BLINK_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= CONSEC_FRAMES_FOR_BLINK:
                    total_blinks += 1
                    blink = 1
                blink_counter = 0

            # ---------------------------------------------------------------------------
            # Core Timing & State Logic Execution
            # ---------------------------------------------------------------------------
            if elapsed < 10.0:
                state = "CALIBRATION"
                # Draw target green circle
                cv2.circle(canvas, (target_x, target_y), 25, (0, 255, 0), -1)
                
                # Dynamic countdown display on canvas
                remaining = 10.0 - elapsed
                cv2.putText(canvas, f"Focus on the Green Circle ({remaining:.1f}s)", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Collect baseline calibration telemetry points
                baseline_h_samples.append(current_h)
                baseline_v_samples.append(current_v)
                
            elif elapsed < 25.0:
                if state == "CALIBRATION":
                    # Calculate the absolute anchor baseline once as we transition states
                    if baseline_h_samples and baseline_v_samples:
                        anchor_h = sum(baseline_h_samples) / len(baseline_h_samples)
                        anchor_v = sum(baseline_v_samples) / len(baseline_v_samples)
                    state = "DISTRACTION"
                    print(f"Anchor Locked! H: {anchor_h:.4f}, V: {anchor_v:.4f}")
                    
                # Keep drawing the identical target circle at the saved position
                cv2.circle(canvas, (target_x, target_y), 25, (0, 255, 0), -1)
                cv2.putText(canvas, "Keep Focusing on the Target", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # --- ADD DISTRACTIONS ---
                # Draw a distracting flashing red circle somewhere else every half second
                if int(elapsed * 2) % 2 == 0:
                    cv2.circle(canvas, (200, 500), 40, (0, 0, 255), -1) 
                    cv2.putText(canvas, "LOOK HERE!", (150, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                # --- CALCULATE ATTENTION ERROR & SCORE ---
                error = math.dist([current_h, current_v], [anchor_h, anchor_v])
                
                # Tuning factor (0.15 represents maximum allowed gaze variance)
                scaled_error = error / 0.15 
                attention_score = max(0.0, 100.0 * (1.0 - scaled_error))
                
                cv2.putText(canvas, f"Attention: {attention_score:.1f}%", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                # End test automatically after 25 seconds
                break
        else:
            # Face tracking fallback display text
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(canvas, "Face Lost! Processing paused.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Log metrics to CSV
        writer.writerow([time.time() - start_time, left_ear, right_ear, avg_ear, 
                         gaze_left, gaze_right, blink, v_gaze_left, v_gaze_right, round(attention_score, 2)])

        # Render diagnostic tracking elements on primary webcam window
        if result.face_landmarks:
            for idx in LEFT_EYE_EAR_IDX + RIGHT_EYE_EAR_IDX:
                cv2.circle(frame, (int(landmarks[idx][0]), int(landmarks[idx][1])), 2, (255, 0, 0), -1)

        # Render both windows simultaneously
        cv2.imshow("Webcam Tracking Feed", frame)
        cv2.imshow("Attention Stimulus Window", canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    
    print(f"\nExperiment concluded! Data saved to logs/{filename}.csv. Running graph engine...")
    plot(filename)

if __name__ == "__main__":
    main()