import json
import math
import random
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# -----------------------------
# Configuration
# -----------------------------
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
CAMERA_INDEX = 0

ENROLLMENT_SAMPLES = 3
TARGET_POINTS = 80
MIN_TRACE_POINTS = 25
PASS_THRESHOLD = 0.72
PATH_TOLERANCE = 0.16

TEMPLATE_FILE = Path("gesture_template.json")

TARGET_COLOR = (0, 255, 255)
TRACE_COLOR = (0, 255, 0)
PASS_COLOR = (0, 255, 0)
FAIL_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)


# -----------------------------
# Path and feature utilities
# -----------------------------
def point_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def resample_path(points, count=TARGET_POINTS):
    """Resample a trajectory to a fixed number of equally spaced points."""
    if len(points) < 2:
        return points

    pts = np.asarray(points, dtype=np.float32)
    segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))

    if total_length <= 1e-6:
        return [tuple(pts[0])] * count

    distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    targets = np.linspace(0.0, total_length, count)

    x = np.interp(targets, distances, pts[:, 0])
    y = np.interp(targets, distances, pts[:, 1])

    return list(zip(x, y))


def normalize_path(points):
    """Remove translation and scale differences between attempts."""
    pts = np.asarray(points, dtype=np.float32)
    minimum = pts.min(axis=0)
    maximum = pts.max(axis=0)
    size = float(max(maximum - minimum))

    if size <= 1e-6:
        return np.zeros_like(pts)

    normalized = (pts - minimum) / size
    return normalized


def extract_features(points, timestamps):
    """Create a compact gesture template from fingertip coordinates."""
    sampled = resample_path(points)

    if len(sampled) < 2:
        raise ValueError("Not enough points for a gesture template.")

    normalized = normalize_path(sampled)

    vectors = np.diff(normalized, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)

    valid = lengths > 1e-5
    directions = np.zeros_like(vectors)
    directions[valid] = vectors[valid] / lengths[valid, None]

    # Normalize elapsed time to make the template independent of absolute duration.
    if timestamps and len(timestamps) == len(points):
        elapsed = np.asarray(timestamps, dtype=np.float32)
        elapsed = elapsed - elapsed[0]
        total = float(elapsed[-1])

        if total > 1e-6:
            original_distances = np.linspace(0.0, 1.0, len(elapsed))
            target_distances = np.linspace(0.0, 1.0, len(sampled))
            time_profile = np.interp(
                target_distances,
                original_distances,
                elapsed / total,
            )
        else:
            time_profile = np.linspace(0.0, 1.0, len(sampled))
    else:
        time_profile = np.linspace(0.0, 1.0, len(sampled))

    return {
        "path": normalized.tolist(),
        "directions": directions.tolist(),
        "time_profile": time_profile.tolist(),
    }


# -----------------------------
# Template persistence
# -----------------------------
def save_template(template):
    TEMPLATE_FILE.write_text(
        json.dumps(template, indent=2),
        encoding="utf-8",
    )


def load_template():
    if not TEMPLATE_FILE.exists():
        return None

    try:
        return json.loads(TEMPLATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def average_templates(templates):
    """Average multiple enrollment attempts into one personal template."""
    paths = np.asarray([t["path"] for t in templates], dtype=np.float32)
    directions = np.asarray(
        [t["directions"] for t in templates],
        dtype=np.float32,
    )
    times = np.asarray([t["time_profile"] for t in templates], dtype=np.float32)

    return {
        "path": np.mean(paths, axis=0).tolist(),
        "directions": np.mean(directions, axis=0).tolist(),
        "time_profile": np.mean(times, axis=0).tolist(),
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# -----------------------------
# Verification
# -----------------------------
def path_score(template, attempt):
    a = np.asarray(template["path"], dtype=np.float32)
    b = np.asarray(attempt["path"], dtype=np.float32)

    error = float(np.mean(np.linalg.norm(a - b, axis=1)))
    return float(np.clip(1.0 - error / PATH_TOLERANCE, 0.0, 1.0))


def direction_score(template, attempt):
    a = np.asarray(template["directions"], dtype=np.float32)
    b = np.asarray(attempt["directions"], dtype=np.float32)

    lengths_a = np.linalg.norm(a, axis=1)
    lengths_b = np.linalg.norm(b, axis=1)
    valid = (lengths_a > 1e-5) & (lengths_b > 1e-5)

    if not np.any(valid):
        return 0.0

    a = a[valid]
    b = b[valid]

    cosine = np.sum(a * b, axis=1)
    return float(np.clip((np.mean(cosine) + 1.0) / 2.0, 0.0, 1.0))


def timing_score(template, attempt):
    a = np.asarray(template["time_profile"], dtype=np.float32)
    b = np.asarray(attempt["time_profile"], dtype=np.float32)

    error = float(np.mean(np.abs(a - b)))
    return float(np.clip(1.0 - error * 2.0, 0.0, 1.0))


def bounding_box_score(points):
    """Reject nearly stationary or extremely tiny movements."""
    pts = np.asarray(points, dtype=np.float32)

    if len(pts) < 2:
        return 0.0

    width, height = np.ptp(pts, axis=0)
    area = float(width * height)

    return 1.0 if area >= 2500 else 0.0


def verify_gesture(template, points, timestamps):
    if len(points) < MIN_TRACE_POINTS:
        return {
            "path": 0.0,
            "direction": 0.0,
            "timing": 0.0,
            "movement": 0.0,
            "final": 0.0,
        }

    attempt = extract_features(points, timestamps)

    scores = {
        "path": path_score(template, attempt),
        "direction": direction_score(template, attempt),
        "timing": timing_score(template, attempt),
        "movement": bounding_box_score(points),
    }

    scores["final"] = (
        0.50 * scores["path"]
        + 0.25 * scores["direction"]
        + 0.15 * scores["timing"]
        + 0.10 * scores["movement"]
    )

    return scores


# -----------------------------
# Camera / UI helpers
# -----------------------------
def get_fingertip(detection, hands_module):
    if not detection.multi_hand_landmarks:
        return None

    hand = detection.multi_hand_landmarks[0]
    tip = hand.landmark[hands_module.HandLandmark.INDEX_FINGER_TIP]

    if not (0.0 <= tip.x <= 1.0 and 0.0 <= tip.y <= 1.0):
        return None

    return (
        int(tip.x * CANVAS_WIDTH),
        int(tip.y * CANVAS_HEIGHT),
    )


def draw_trace(frame, points, color=TRACE_COLOR):
    if len(points) < 2:
        return

    pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], False, color, 5, cv2.LINE_AA)


def draw_random_challenge(frame, sequence, points):
    """Display randomized numbered points for the enrolled gesture."""
    for i, point in enumerate(points):
        p = tuple(map(int, point))
        cv2.circle(frame, p, 28, TARGET_COLOR, 3)
        cv2.putText(
            frame,
            str(sequence[i]),
            (p[0] - 10, p[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TARGET_COLOR,
            2,
            cv2.LINE_AA,
        )


def create_challenge(sequence_length=4):
    """Randomize point locations while preserving the user's secret order."""
    margin_x = 220
    margin_y = 170

    positions = []
    attempts = 0

    while len(positions) < sequence_length and attempts < 500:
        attempts += 1
        candidate = (
            random.randint(margin_x, CANVAS_WIDTH - margin_x),
            random.randint(margin_y, CANVAS_HEIGHT - margin_y),
        )

        if all(point_distance(candidate, p) > 150 for p in positions):
            positions.append(candidate)

    if len(positions) < sequence_length:
        raise RuntimeError("Could not generate a sufficiently spaced challenge.")

    # The labels represent the secret order. Positions change each round.
    labels = list(range(1, sequence_length + 1))
    random.shuffle(labels)

    return labels, positions


def draw_header(frame, title, subtitle):
    cv2.rectangle(frame, (0, 0), (CANVAS_WIDTH, 95), (25, 25, 25), -1)

    cv2.putText(
        frame,
        title,
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        subtitle,
        (30, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        TARGET_COLOR,
        2,
        cv2.LINE_AA,
    )


# -----------------------------
# Enrollment
# -----------------------------
def enroll(hands, mp_hands, mp_drawing, cap):
    templates = []

    print("\nPERSONAL GESTURE ENROLLMENT")
    print("Perform your preferred gesture three times.")
    print("Press SPACE to start each recording and Q to quit.")

    for sample in range(ENROLLMENT_SAMPLES):
        points = []
        timestamps = []
        state = "READY"

        while True:
            ret, frame = cap.read()
            if not ret:
                return None

            frame = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT))
            frame = cv2.flip(frame, 1)

            detection = hands.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            fingertip = get_fingertip(detection, mp_hands)

            if detection.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    detection.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                )

            if fingertip and state == "RECORDING":
                points.append(fingertip)
                timestamps.append(time.perf_counter())

            draw_header(
                frame,
                "CHECK.IT - GESTURE ENROLLMENT",
                f"Sample {sample + 1}/{ENROLLMENT_SAMPLES}",
            )

            if points:
                draw_trace(frame, points)

            if state == "READY":
                message = "Press SPACE, then perform your personal gesture"
            else:
                message = "Recording... press ENTER when finished"

            cv2.putText(
                frame,
                message,
                (220, 650),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Check.it", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                return None

            if key == ord(" ") and state == "READY":
                points.clear()
                timestamps.clear()
                state = "RECORDING"

            if key == 13 and state == "RECORDING":
                if len(points) >= MIN_TRACE_POINTS:
                    templates.append(extract_features(points, timestamps))
                    break

    return average_templates(templates)


# -----------------------------
# Verification loop
# -----------------------------
def verify(hands, mp_hands, mp_drawing, cap, template):
    sequence, positions = create_challenge()
    target_lookup = {number: positions[i] for i, number in enumerate(sequence)}

    # The user must visit the randomized points in the secret order.
    ordered_points = [target_lookup[n] for n in sorted(sequence)]

    points = []
    timestamps = []
    state = "READY"
    result = None
    current_target_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT))
        frame = cv2.flip(frame, 1)

        detection = hands.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        fingertip = get_fingertip(detection, mp_hands)

        if detection.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                detection.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
            )

        if fingertip:
            cv2.circle(frame, fingertip, 10, (255, 0, 255), -1)

            if state == "TRACING":
                points.append(fingertip)
                timestamps.append(time.perf_counter())

                if current_target_index < len(ordered_points):
                    if point_distance(
                        fingertip,
                        ordered_points[current_target_index],
                    ) < 65:
                        current_target_index += 1

                if current_target_index == len(ordered_points):
                    result = verify_gesture(template, points, timestamps)
                    state = (
                        "PASSED"
                        if result["final"] >= PASS_THRESHOLD
                        else "FAILED"
                    )

        draw_header(
            frame,
            "CHECK.IT - PERSONAL GESTURE CAPTCHA",
            "Follow the numbered points in your secret order",
        )

        draw_random_challenge(frame, sequence, positions)

        if points:
            draw_trace(
                frame,
                points,
                PASS_COLOR if state == "PASSED" else TRACE_COLOR,
            )

        if state == "READY":
            message = "Press SPACE to start verification"
        elif state == "TRACING":
            message = f"Progress: {current_target_index}/{len(ordered_points)}"
        else:
            message = "Press R for another challenge"

        cv2.putText(
            frame,
            message,
            (300, 650),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

        if result:
            label = (
                "CAPTCHA VERIFIED"
                if state == "PASSED"
                else "VERIFICATION FAILED"
            )
            color = PASS_COLOR if state == "PASSED" else FAIL_COLOR

            cv2.putText(
                frame,
                label,
                (430, 560),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Score: {result['final'] * 100:.1f}%",
                (500, 600),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Check.it", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return

        if key == ord("r"):
            sequence, positions = create_challenge()
            target_lookup = {
                number: positions[i] for i, number in enumerate(sequence)
            }
            ordered_points = [target_lookup[n] for n in sorted(sequence)]
            points.clear()
            timestamps.clear()
            current_target_index = 0
            result = None
            state = "READY"

        if key == ord(" ") and state == "READY":
            points.clear()
            timestamps.clear()
            current_target_index = 0
            result = None
            state = "TRACING"


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

    try:
        template = load_template()

        if template is None:
            template = enroll(hands, mp_hands, mp_drawing, cap)
            if template is None:
                return
            save_template(template)
            print(f"Personal gesture template saved to {TEMPLATE_FILE}")

        verify(hands, mp_hands, mp_drawing, cap, template)

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
