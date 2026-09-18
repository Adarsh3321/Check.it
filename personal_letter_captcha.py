import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# Check.it — Personalized Letter-in-Shape CAPTCHA
#
# First run:
#   The user chooses two personal letters (for example, their initials).
#
# Each challenge then randomizes:
#   - the enclosing shape
#   - letter positions
#   - letter rotation
#
# The user traces the letters with a live index finger.
#
# This is a prototype. Thresholds must be calibrated with controlled testing
# before the system is used as a real security control.


WIDTH = 1280
HEIGHT = 720
CAMERA_INDEX = 0

CONFIG_FILE = Path("user_profile.json")

PASS_THRESHOLD = 0.70
MIN_POINTS = 30
TARGET_THICKNESS = 8
TRACE_THICKNESS = 7

SHAPES = ("circle", "square", "triangle", "hexagon", "star")
LETTER_POOL = "ABCDEFGHJKLMNPQRSTUVWXYZ"

BACKGROUND = (18, 24, 32)
WHITE = (245, 245, 245)
SHAPE_COLOR = (255, 170, 40)
LETTER_COLOR = (255, 220, 80)
TRACE_COLOR = (220, 80, 180)
PASS_COLOR = (60, 220, 90)
FAIL_COLOR = (70, 70, 235)


@dataclass
class Challenge:
    shape: str
    letters: str
    target_mask: np.ndarray
    letter_masks: list
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


def draw_shape(mask, shape, center, radius, thickness=6):
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
                (int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle)))
            )
        cv2.polylines(
            mask,
            [np.array(pts, dtype=np.int32)],
            True,
            255,
            thickness,
            cv2.LINE_AA,
        )


def render_letter_mask(letter, center, angle):
    """Render one personal letter and rotate it around its center."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4.8
    thickness = TARGET_THICKNESS

    (text_w, text_h), baseline = cv2.getTextSize(
        letter, font, font_scale, thickness
    )

    x = int(center[0] - text_w / 2)
    y = int(center[1] + text_h / 2)

    cv2.putText(
        mask,
        letter,
        (x, y),
        font,
        font_scale,
        255,
        thickness,
        cv2.LINE_AA,
    )

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        mask,
        rotation_matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
    )


def create_profile():
    print("\nCHECK.IT — PERSONAL SETUP")
    print("Choose two letters that are meaningful to you, such as your initials.")
    print("Use two different A-Z letters.")

    while True:
        letters = input("Enter your two letters: ").strip().upper()

        if len(letters) == 2 and letters[0] in LETTER_POOL and letters[1] in LETTER_POOL and letters[0] != letters[1]:
            profile = {
                "letters": letters,
                "created_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "version": 1,
            }

            CONFIG_FILE.write_text(
                json.dumps(profile, indent=2),
                encoding="utf-8",
            )

            print(f"Personal challenge letters saved: {letters}")
            return profile

        print("Please enter exactly two different letters, e.g. AR.")


def load_profile():
    if not CONFIG_FILE.exists():
        return create_profile()

    try:
        profile = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        letters = profile.get("letters", "")

        if (
            isinstance(letters, str)
            and len(letters) == 2
            and all(letter in LETTER_POOL for letter in letters)
        ):
            return profile
    except (json.JSONDecodeError, OSError):
        pass

    return create_profile()


def generate_challenge(personal_letters):
    shape = random.choice(SHAPES)
    letters = personal_letters

    target_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    shape_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    center = (WIDTH // 2, HEIGHT // 2 + 30)
    radius = 245

    draw_shape(shape_mask, shape, center, radius)

    # Randomized positions stay inside the enclosing shape.
    offset = random.randint(125, 165)
    positions = [
        (center[0] - offset, center[1] + random.randint(-25, 25)),
        (center[0] + offset, center[1] + random.randint(-25, 25)),
    ]

    letter_masks = []
    boxes = []

    for letter, position in zip(letters, positions):
        angle = random.randint(-12, 12)
        letter_mask = render_letter_mask(letter, position, angle)
        letter_masks.append(letter_mask)

        ys, xs = np.where(letter_mask > 0)
        if len(xs):
            boxes.append(
                (
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                )
            )
        else:
            boxes.append((0, 0, 0, 0))

        target_mask = cv2.bitwise_or(target_mask, letter_mask)

    return Challenge(
        shape=shape,
        letters=letters,
        target_mask=target_mask,
        letter_masks=letter_masks,
        letter_boxes=boxes,
    )


def point_inside_box(point, box, margin=55):
    x, y = point
    x1, y1, x2, y2 = box
    return (
        x1 - margin <= x <= x2 + margin
        and y1 - margin <= y <= y2 + margin
    )


def draw_trace(frame, points, color=TRACE_COLOR):
    if len(points) < 2:
        return

    pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], False, color, TRACE_THICKNESS, cv2.LINE_AA)


def draw_challenge(frame, challenge, state, points):
    overlay = frame.copy()

    contours, _ = cv2.findContours(
        challenge.shape_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        overlay,
        contours,
        -1,
        SHAPE_COLOR,
        5,
        cv2.LINE_AA,
    )

    overlay[challenge.target_mask > 0] = LETTER_COLOR

    if points:
        draw_trace(
            overlay,
            points,
            PASS_COLOR if state == "PASSED" else TRACE_COLOR,
        )

    frame[:] = overlay


def draw_header(frame, challenge, state, result):
    cv2.rectangle(frame, (0, 0), (WIDTH, 105), BACKGROUND, -1)

    cv2.putText(
        frame,
        "CHECK.IT",
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"PERSONAL CAPTCHA  |  {challenge.shape.upper()}",
        (30, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        LETTER_COLOR,
        2,
        cv2.LINE_AA,
    )

    if state == "READY":
        message = "SPACE: start  |  Trace your letters from left to right"
        color = WHITE
    elif state == "TRACING":
        message = f"Trace your letters: {challenge.letters}"
        color = WHITE
    elif state == "PASSED":
        message = f"CAPTCHA VERIFIED  |  {result['final'] * 100:.1f}%  |  R: new challenge"
        color = PASS_COLOR
    else:
        message = f"VERIFICATION FAILED  |  {result['final'] * 100:.1f}%  |  R: try again"
        color = FAIL_COLOR

    cv2.putText(
        frame,
        message,
        (185, 675),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
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


def path_distance_score(target_mask, points):
    if len(points) < MIN_POINTS:
        return 0.0

    inverse = cv2.bitwise_not(target_mask)
    distance_map = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)

    distances = [
        float(distance_map[y, x])
        for x, y in points
        if 0 <= x < WIDTH and 0 <= y < HEIGHT
    ]

    if not distances:
        return 0.0

    mean_distance = float(np.mean(np.minimum(distances, 30.0)))
    return float(np.clip(1.0 - mean_distance / 30.0, 0.0, 1.0))


def coverage_score(target_mask, points):
    if len(points) < MIN_POINTS:
        return 0.0

    trace_mask = np.zeros_like(target_mask)
    pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))

    cv2.polylines(
        trace_mask,
        [pts],
        False,
        255,
        TRACE_THICKNESS,
        cv2.LINE_AA,
    )

    covered = cv2.dilate(
        trace_mask,
        np.ones((11, 11), dtype=np.uint8),
        iterations=1,
    ) > 0

    target = target_mask > 0
    target_count = int(np.count_nonzero(target))

    if target_count == 0:
        return 0.0

    return float(
        np.clip(
            np.count_nonzero(target & covered) / target_count,
            0.0,
            1.0,
        )
    )


def order_score(points, boxes):
    """Check that both personal letters are actually visited left-to-right."""
    if len(points) < MIN_POINTS or len(boxes) < 2:
        return 0.0

    first_seen = False
    second_seen = False

    for point in points:
        if not first_seen and point_inside_box(point, boxes[0]):
            first_seen = True

        if first_seen and point_inside_box(point, boxes[1]):
            second_seen = True
            break

    if second_seen:
        return 1.0

    if first_seen:
        return 0.35

    return 0.0


def movement_score(points):
    if len(points) < MIN_POINTS:
        return 0.0

    pts = np.asarray(points, dtype=np.float32)
    width, height = np.ptp(pts, axis=0)

    if width < 100 or height < 40:
        return 0.0

    return 1.0


def verify_attempt(challenge, points):
    path = path_distance_score(challenge.target_mask, points)
    coverage = coverage_score(challenge.target_mask, points)
    order = order_score(points, challenge.letter_boxes)
    movement = movement_score(points)

    final = (
        0.50 * path
        + 0.30 * coverage
        + 0.15 * order
        + 0.05 * movement
    )

    return {
        "path": path,
        "coverage": coverage,
        "order": order,
        "movement": movement,
        "final": float(final),
    }


def main():
    profile = load_profile()
    personal_letters = profile["letters"]

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

    challenge = generate_challenge(personal_letters)
    trace = []
    state = "READY"
    result = None

    print("\nCheck.it — Personalized Letter-in-Shape CAPTCHA")
    print(f"Personal letters: {personal_letters}")
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

                # The attempt ends after the second personal letter is reached.
                if (
                    len(trace) >= MIN_POINTS
                    and point_inside_box(
                        fingertip,
                        challenge.letter_boxes[1],
                        margin=60,
                    )
                ):
                    result = verify_attempt(challenge, trace)
                    state = (
                        "PASSED"
                        if result["final"] >= PASS_THRESHOLD
                        else "FAILED"
                    )

            draw_challenge(frame, challenge, state, trace)
            draw_header(frame, challenge, state, result)

            cv2.imshow("Check.it", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                challenge = generate_challenge(personal_letters)
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
