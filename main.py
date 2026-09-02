# main.py
import os
import cv2
import time
import math
import numpy as np

try:
    from camera.capture_mock import MockCamera
    from actuation.valve_mock import MockValveController
except ImportError:
    from MockCamera import MockCamera
    from MockValveController import MockValveController

# Configuration constants
MOCK_MODE = True
DROP_DISTANCE_M = 0.08      # 8 cm from camera focal point to air nozzle
INITIAL_VELOCITY_M_S = 0.5  # Seed exit speed from chute
GRAVITY = 9.81              # m/s^2
VALVE_LATENCY_MS = 3.0      # Mechanical opening delay


def calculate_ejection_delay(y_pixel, frame_height=720):
    # Free-fall quadratic formula: t = (-v0 + sqrt(v0^2 + 2gd)) / g
    t_arrival = (-INITIAL_VELOCITY_M_S + math.sqrt(INITIAL_VELOCITY_M_S**2 + 2 * GRAVITY * DROP_DISTANCE_M)) / GRAVITY
    t_arrival_ms = t_arrival * 1000.0

    # Subtract valve opening mechanical latency
    delay_ms = max(0.0, t_arrival_ms - VALVE_LATENCY_MS)
    return delay_ms


def classify_seed(hsv_frame, contour):
    """Return whether a seed is good or bad based on its HSV brightness."""
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Ignore tiny or empty contours
    if np.count_nonzero(mask) == 0:
        return "unknown"

    mean_saturation = cv2.mean(hsv_frame[:, :, 1], mask=mask)[0]
    mean_value = cv2.mean(hsv_frame[:, :, 2], mask=mask)[0]

    # Bad seeds are typically dark/discolored and have low brightness.
    # Good seeds are brighter and more uniform than defective ones.
    if mean_value < 80 and mean_saturation < 200:
        return "bad"
    return "good"


def main():
    video_source = "realistic_mixed_seeds_different_pattern.mp4" if os.path.exists("realistic_mixed_seeds_different_pattern.mp4") else 0
    camera = MockCamera(video_source) if MOCK_MODE else None
    valve = MockValveController(pin=18)

    print(f"Starting pipeline in MOCK mode using source: {video_source}")
    while True:
        start_compute = time.time()
        ret, frame = camera.read()
        if not ret:
            break

        # --- Phase 2: Seed classification (good vs bad) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect candidate seeds in the scene, then classify each contour.
        # A wider threshold helps capture both normal and defective seeds.
        mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 180))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                seed_label = classify_seed(hsv, cnt)

                color = (0, 255, 0) if seed_label == "good" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, seed_label.upper(), (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if seed_label == "bad":
                    compute_latency_ms = (time.time() - start_compute) * 1000.0
                    ejection_delay = calculate_ejection_delay(y) - compute_latency_ms
                    valve.schedule_ejection(ejection_delay)

        cv2.imshow("Sorter Simulation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()