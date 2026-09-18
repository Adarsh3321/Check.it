import random
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


# Check.it — Letter-in-Shape Gesture CAPTCHA
# The user traces the visible letters inside a randomly generated shape.
# This is a prototype: thresholds must be calibrated with real evaluation data.


WIDTH = 1280
HEIGHT = 720
CAMERA_INDEX = 0

PASS_THRESHOLD = 0.70
MIN_POINTS = 35
TARGET_THICKNESS = 7
TRACE_THICKNESS = 7

BACKGROUND = (18, 24, 32)
WHITE = (245, 245, 245)
BLUE = (255, 170, 40)
CYAN = (255, 220, 80)
GREEN = (60, 220, 90)
RED = (70, 70, 235)
PURPLE = (220, 80, 180)

SHAPES = ("circle", "square", "triangle", "hexagon", "star")
LETTER_POOL = "ABCDEFGHJKLMNPQRSTUVWXYZ"


@dataclass
class Challenge:
    shape: str
    letters: str
    target_mask: np.ndarray
    shape_mask: np.ndarray
    letter_boxes: list


def polygon_points(cx, cy, radius, sides, rotation=-np.pi / 2):
    return np.array(
        [
            (
                int(cx + radius * np.cos(rotation + 2 * np.pi * i / sides)),
                int(cy + radius * np.sin(rotation + 2 * np.pi * i / sides)),
            )
            for i in range(sides)
        ],
        dtype=np.int32,
    )


def draw_shape(mask, shape, center, radius, thickness=5):
    cx, cy = center

    if shape == "circle":
        cv2.circle(mask, center, radius, 255, thickness, cv2.LINE_AA)
    elif shape == "square":
        cv2.rectangle(
            mask,
            (cx - radius, cy - radius),
            (cx + radius, cy + radius),
            255,
            thickness,
            cv2.LINE_AA,
        )
    elif shape == "triangle":
        pts = polygon_points(cx, cy, radius, 3)
        cv2.polylines(mask, [pts], True, 255, thickness, cv2.LINE_AA)
    elif shape == "hexagon":
        pts = polygon_points(cx, cy, radius, 6)
        cv2.polylines(mask, [pts], True, 255, thickness, cv2.LINE_AA)
    elif shape == "star":
        pts = []
        for i in range(10):
            angle = -np.pi / 2 + i * np.pi / 5
            r = radius if i % 2 == 0 else radius * 0.42
            pts.append(
                (
                    int(cx + r * np.cos(angle)),
                    int(cy + r * np.sin(angle)),
                )
            )
        cv2.polylines(mask, [np.array(pts, dtype=np.int32)], True, 255, thickness, cv2.LINE_AA)


def put_centered_text(mask, text, center, font_scale, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x = int(center[0] - w / 2)
    y = int(center[1] + h / 2)

    cv2.putText(
        mask,
        text,
        (x, y),
        font,
        font_scale,
        255,
        thickness,
        cv2.LINE_AA,
    )

    return (x, y - h, x + w, y + baseline)


def generate_challenge():
    target_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    shape_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    center = (WIDTH // 2, HEIGHT // 2 + 30)
    radius = 245
    shape = random.choice(SHAPES)

    draw_shape(shape_mask, shape, center, radius, thickness=6)

    letters = "".join(random.sample(LETTER_POOL, 2))

    # Keep the letters inside the shape and clearly separated.
    font_scale = 4.6
    text_thickness = 8
    spacing = 155

    boxes = []
    letter_centers = [
        (center[0] - spacing, center[1]),
        (center[0] + spacing, center[1]),
    ]

    for letter, letter_center in zip(letters, letter_centers):
        box = put_centered_text(
            target_mask,
            letter,
            letter_center,
            font_scale,
            text_thickness,
        )
        boxes.append(box)

    return Challenge(
        shape=shape,
        letters=letters,
        target_mask=target_mask,
        shape_mask=shape_mask,
        letter_boxes=boxes,
    )


def draw_ui(frame, challenge, trace, state, score=None):
    overlay = frame.copy()

    # Shape
    shape_contours, _ = cv2.findContours(
        challenge.shape_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, shape_contours, -1, BLUE, 5, cv2.LINE_AA)

    # Letters
    letter_pixels = challenge.target_mask > 0
    overlay[letter_pixels] = CYAN

    # User trace
    if len(trace) >= 2:
        pts = np.asarray(trace, dtype=np.int32).reshape((-1, 1, 2))
        trace_color = GREEN if state == "PASSED" else PURPLE
        cv2.polylines(overlay, [pts], False, trace_color, TRACE_THICKNESS, cv2.LINE_AA)

    frame[:] = overlay

    cv2.rectangle(frame, (0, 0), (WIDTH, 95), BACKGROUND, -1)
    cv2.putText(
        frame,
        "CHECK.IT",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        WHITE,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "LETTER-IN-SHAPE CAPTCHA",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        CYAN,
        2,
        cv2.LINE_AA,
    )

    if state == "READY":
        message = "Press SPACE to start — trace the letters from left to right"
    elif state == "TRACING":
        message = f"Trace: {challenge.letters}"
    elif state == "PASSED":
        message = f"CAPTCHA VERIFIED  |  Score: {score * 100:.1f}%  |  Press R for another"
    else:
        message = f"TRY AGAIN  |  Score: {score * 100:.1f}%  |  Press R for another"

    cv2.putText(
        frame,
        message,
        (210, 680),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        GREEN if state == "PASSED" else (RED if state == "FAILED" else WHITE),
        2,
        cv2.LINE_AA,
    )


def get_fingertip(results, mp_hands):
    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    if not (0 <= tip.x <= 1 and 0 <= tip.y <= 1):
        return None

    return int(tip.x * WIDTH), int(tip.y * HEIGHT)


def point_inside_box(point, box, margin=55):
    x, y = point
    x1, y1, x2, y2 = box
    return (
        x1 - margin <= x <= x2 + margin
        and y1 - margin <= y <= y2 + margin
    )


def path_distance_score(target_mask, points):
    if len(points) < MIN_POINTS:
        return 0.0

    # Distance from every captured fingertip point to the nearest target-letter pixel.
    inverse = cv2.bitwise_not(target_mask)
    distance_map = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)

    distances = []
    for x, y in points:
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            distances.append(float(distance_map[y, x]))

    if not distances:
        return 0.0

    # 25 px is treated as the edge of the useful tracing tolerance.
    mean_distance = float(np.mean(np.minimum(distances, 25.0)))
    return max(0.0, min(1.0, 1.0 - mean_distance / 25.0))


def coverage_score(target_mask, points):
    if len(points) < MIN_POINTS:
        return 0.0

    trace_mask = np.zeros_like(target_mask)
    pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(trace_mask, [pts], False, 255, TRACE_THICKNESS, cv2.LINE_AA)

    target = target_mask > 0
    covered = cv2.dilate(trace_mask, np.ones((9, 9), np.uint8), iterations=1) > 0

    target_pixels = int(np.count_nonzero(target))
    if target_pixels == 0:
        return 0.0

    covered_target = int(np.count_nonzero(target & covered))
    return float(np.clip(covered_target / target_pixels, 0.0, 1.0))


def order_score(points, boxes):
    if len(points) < MIN_POINTS:
        return 0.0

    visited = []
    for point in points:
        for index, box in enumerate(boxes):
            if index not in visited and point_inside_box(point, box):
                visited.append(index)
                break

    # Require the two letters to be encountered left-to-right.
    if visited == list(range(len(boxes))):
        return 1.0

    if visited and visited == [0]:
        return 0.45

    return 0.0


def verify_attempt(challenge, points):
    path = path_distance_score(challenge.target_mask, points)
    coverage = coverage_score(challenge.target_mask, points)
    order = order_score(points, challenge.letter_boxes)

    final = (
        0.55 * path
        + 0.30 * coverage
        + 0.15 * order
    )

    return {
        "path": path,
        "coverage": coverage,
        "order": order,
        "final": float(final),
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
        raise RuntimeError("Could not open the camera.")

    challenge = generate_challenge()
    trace = []
    state = "READY"
    result = None

    print("Check.it — Letter-in-Shape CAPTCHA")
    print("SPACE = start | R = new challenge | Q = quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frame = cv2.flip(frame, 1)

            results = hands.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            fingertip = get_fingertip(results, mp_hands)

            if results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                )

            if fingertip and state == "TRACING":
                trace.append(fingertip)

                # Finish when the user has reached the second letter area.
                if len(trace) >= MIN_POINTS and point_inside_box(
                    fingertip,
                    challenge.letter_boxes[1],
                    margin=60,
                ):
                    result = verify_attempt(challenge, trace)
                    state = (
                        "PASSED"
                        if result["final"] >= PASS_THRESHOLD
                        else "FAILED"
                    )

            draw_ui(
                frame,
                challenge,
                trace,
                state,
                result["final"] if result else None,
            )

            cv2.imshow("Check.it", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                challenge = generate_challenge()
                trace.clear()
                result = None
                state = "READY"

            if key == ord(" ") and state == "READY":
                trace.clear()
                result = None
                state = "TRACING"

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
