import math
import random
import time

import cv2
import mediapipe as mp
import numpy as np

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
CAMERA_INDEX = 0
TARGET_POINTS = 80
CAPTURE_POINTS = 120
PASS_THRESHOLD = 0.72
PATH_TOLERANCE = 0.13
MIN_TRACE_POINTS = 20

TARGET_COLOR = (0, 255, 255)
TRACE_COLOR = (0, 255, 0)
FAIL_COLOR = (0, 0, 255)
PASS_COLOR = (0, 255, 0)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def resample_path(points, count=TARGET_POINTS):
    """Resample a path to a fixed number of equally spaced points."""
    if len(points) < 2:
        return points

    pts = [np.array(p, dtype=np.float32) for p in points]
    total = sum(np.linalg.norm(pts[i] - pts[i - 1]) for i in range(1, len(pts)))

    if total == 0:
        return [tuple(pts[0])] * count

    interval = total / (count - 1)
    result = [pts[0]]
    current = pts[0].copy()
    remaining = interval

    for i in range(1, len(pts)):
        segment = pts[i] - current
        segment_len = float(np.linalg.norm(segment))

        while segment_len >= remaining and segment_len > 0:
            ratio = remaining / segment_len
            current = current + segment * ratio
            result.append(current.copy())
            segment = pts[i] - current
            segment_len = float(np.linalg.norm(segment))
            remaining = interval

        remaining -= segment_len
        if remaining <= 1e-6:
            remaining = interval
        current = pts[i].copy()

    while len(result) < count:
        result.append(pts[-1].copy())

    return [tuple(p) for p in result[:count]]


def normalize_path(points):
    """Translate and scale a trajectory for shape comparison."""
    pts = np.array(points, dtype=np.float32)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    size = np.maximum(max_xy - min_xy, 1.0)

    normalized = (pts - min_xy) / max(size[0], size[1])
    normalized -= normalized.mean(axis=0)
    return normalized


def path_similarity(target, user):
    """Compare the overall shape of two trajectories."""
    if len(target) < 2 or len(user) < 2:
        return 0.0

    a = normalize_path(resample_path(target))
    b = normalize_path(resample_path(user))
    error = float(np.mean(np.linalg.norm(a - b, axis=1)))
    return max(0.0, min(1.0, 1.0 - error / 0.75))


def path_accuracy(target, user):
    """Measure how closely the user's points follow the target path."""
    if len(target) < 2 or len(user) < 2:
        return 0.0

    target_norm = normalize_path(resample_path(target))
    user_norm = normalize_path(resample_path(user))

    errors = []
    for point in user_norm:
        errors.append(float(np.min(np.linalg.norm(target_norm - point, axis=1))))

    error = float(np.mean(errors))
    return max(0.0, min(1.0, 1.0 - error / PATH_TOLERANCE))


def direction_similarity(target, user):
    """Compare the direction changes along both paths."""
    a = normalize_path(resample_path(target))
    b = normalize_path(resample_path(user))

    va = np.diff(a, axis=0)
    vb = np.diff(b, axis=0)
    na = np.linalg.norm(va, axis=1)
    nb = np.linalg.norm(vb, axis=1)

    valid = (na > 1e-4) & (nb > 1e-4)
    if not np.any(valid):
        return 0.0

    dots = np.sum(va[valid] * vb[valid], axis=1) / (na[valid] * nb[valid])
    return float(np.clip((np.mean(dots) + 1.0) / 2.0, 0.0, 1.0))


def speed_consistency(points, timestamps):
    """Reward a controlled trace without extreme speed variation."""
    if len(points) < 3 or len(points) != len(timestamps):
        return 0.0

    speeds = []
    for i in range(1, len(points)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt > 0:
            speeds.append(distance(points[i - 1], points[i]) / dt)

    if len(speeds) < 2:
        return 0.0

    median_speed = float(np.median(speeds))
    if median_speed <= 1e-6:
        return 0.0

    variation = float(np.std(speeds) / median_speed)
    return float(np.clip(1.0 - variation / 2.0, 0.0, 1.0))


def generate_target(shape_name, width=520, height=360):
    """Generate a CAPTCHA trajectory in the center of the camera view."""
    cx, cy = CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2 + 30
    half_w, half_h = width // 2, height // 2

    if shape_name == "triangle":
        raw = [
            (cx, cy - half_h),
            (cx - half_w, cy + half_h),
            (cx + half_w, cy + half_h),
            (cx, cy - half_h),
        ]
    elif shape_name == "square":
        raw = [
            (cx - half_w, cy - half_h),
            (cx + half_w, cy - half_h),
            (cx + half_w, cy + half_h),
            (cx - half_w, cy + half_h),
            (cx - half_w, cy - half_h),
        ]
    elif shape_name == "zigzag":
        raw = [
            (cx - half_w, cy - half_h // 2),
            (cx - half_w // 3, cy + half_h),
            (cx + half_w // 3, cy - half_h),
            (cx + half_w, cy + half_h // 2),
        ]
    elif shape_name == "circle":
        raw = [
            (
                int(cx + (half_w * np.cos(angle))),
                int(cy + (half_h * np.sin(angle))),
            )
            for angle in np.linspace(0, 2 * np.pi, 70)
        ]
    elif shape_name == "arrow":
        raw = [
            (cx - half_w, cy),
            (cx + half_w // 3, cy),
            (cx + half_w // 3, cy - half_h // 2),
            (cx + half_w, cy),
            (cx + half_w // 3, cy + half_h // 2),
            (cx + half_w // 3, cy),
        ]
    else:
        raise ValueError(f"Unknown CAPTCHA shape: {shape_name}")

    return resample_path(raw)


def random_challenge():
    shape = random.choice(["triangle", "square", "circle", "zigzag", "arrow"])
    return shape, generate_target(shape)


def draw_polyline(image, points, color, thickness=4):
    if len(points) < 2:
        return
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], False, color, thickness, cv2.LINE_AA)


def draw_target(image, target):
    draw_polyline(image, target, TARGET_COLOR, 5)


def score_attempt(target, user_points, timestamps):
    shape = path_similarity(target, user_points)
    accuracy = path_accuracy(target, user_points)
    direction = direction_similarity(target, user_points)
    speed = speed_consistency(user_points, timestamps)

    final_score = (
        0.40 * shape
        + 0.30 * accuracy
        + 0.20 * direction
        + 0.10 * speed
    )

    return {
        "shape": shape,
        "accuracy": accuracy,
        "direction": direction,
        "speed": speed,
        "final": final_score,
    }


def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    shape_name, target = random_challenge()
    user_points = []
    timestamps = []
    state = "READY"
    result = None

    print("Gesture CAPTCHA started.")
    print("SPACE = start, R = new challenge, Q = quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame could not be captured.")
                break

            frame = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT))
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection = hands.process(rgb)

            if detection.multi_hand_landmarks:
                hand = detection.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                fingertip = (
                    int(index_tip.x * CANVAS_WIDTH),
                    int(index_tip.y * CANVAS_HEIGHT),
                )

                cv2.circle(frame, fingertip, 9, (255, 0, 255), -1)

                if state == "TRACING":
                    user_points.append(fingertip)
                    timestamps.append(time.perf_counter())

            cv2.rectangle(frame, (0, 0), (CANVAS_WIDTH, 85), (25, 25, 25), -1)
            cv2.putText(
                frame,
                "GESTURE CAPTCHA",
                (30, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Challenge: {shape_name.upper()}",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                TARGET_COLOR,
                2,
                cv2.LINE_AA,
            )

            draw_target(frame, target)

            if user_points:
                draw_polyline(
                    frame,
                    user_points,
                    TRACE_COLOR if state != "FAILED" else FAIL_COLOR,
                    4,
                )

            if state == "READY":
                message = "Press SPACE to start"
            elif state == "TRACING":
                message = "Trace the yellow path with your index finger"
            else:
                message = "Press R for another challenge"

            cv2.putText(
                frame,
                message,
                (CANVAS_WIDTH // 2 - 300, 680),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if state in ("PASSED", "FAILED") and result:
                label = "CAPTCHA VERIFIED" if state == "PASSED" else "VERIFICATION FAILED"
                color = PASS_COLOR if state == "PASSED" else FAIL_COLOR

                cv2.putText(
                    frame,
                    label,
                    (CANVAS_WIDTH // 2 - 210, 610),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Score: {result['final'] * 100:.1f}%",
                    (CANVAS_WIDTH // 2 - 100, 645),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Check.it - Gesture CAPTCHA", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                shape_name, target = random_challenge()
                user_points.clear()
                timestamps.clear()
                result = None
                state = "READY"

            if key == ord(" ") and state == "READY":
                user_points.clear()
                timestamps.clear()
                result = None
                state = "TRACING"

            if state == "TRACING" and len(user_points) >= CAPTURE_POINTS:
                if len(user_points) >= MIN_TRACE_POINTS:
                    result = score_attempt(target, user_points, timestamps)
                    state = "PASSED" if result["final"] >= PASS_THRESHOLD else "FAILED"

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
