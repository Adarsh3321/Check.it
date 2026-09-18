# Check.it — Personalized Letter-in-Shape CAPTCHA

A computer-vision CAPTCHA prototype that asks a user to trace their personal letters inside a randomly generated shape using live index-finger movement.

## Concept

The challenge combines two ideas:

1. **Personalization** — the user chooses two meaningful letters, such as their initials.
2. **Dynamic CAPTCHA** — every attempt changes the enclosing shape, letter position, and letter rotation.

Example:

```text
          ╭──────────────╮
       ╱                    ╲
      │         A    R        │
       ╲                    ╱
          ╰──────────────╯

             Trace A → R
```

The letters remain personal while the visual challenge changes between attempts.

## How It Works

```text
User chooses personal letters
            ↓
      Random shape
            ↓
  Random letter placement
            ↓
       Webcam input
            ↓
    MediaPipe hand tracking
            ↓
     Index-finger path
            ↓
 ┌──────────┼───────────┐
 ↓          ↓           ↓
Path     Coverage      Order
 ↓          ↓           ↓
 └──────────┼───────────┘
            ↓
      Verification score
            ↓
       PASS / FAIL
```

## Supported Shapes

- Circle
- Square
- Triangle
- Hexagon
- Star

Each challenge contains the user's two selected letters.

## Verification

The prototype currently evaluates:

| Signal | Weight |
|---|---:|
| Path proximity to letters | 50% |
| Letter-path coverage | 30% |
| Letter order | 15% |
| Overall movement | 5% |

The current development threshold is **70%**. This value is not a security guarantee and must be calibrated using controlled human and automated testing.

## Technology Stack

- Python
- OpenCV
- MediaPipe
- NumPy

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python personal_letter_captcha.py
```

On first launch, Check.it asks the user to choose two personal letters.

Example:

```text
Enter your two letters: AR
```

The choice is stored locally in `user_profile.json`.

### Controls

- `SPACE` — start tracing
- `R` — generate a new challenge
- `Q` — quit

## Privacy

The prototype does not need to save camera frames or video. It stores the selected letters locally and processes the live camera stream for fingertip tracking.

This local prototype should not be treated as a production identity or biometric-storage system.

## Project Structure

```text
Check.it/
├── personal_letter_captcha.py   # Current personalized CAPTCHA
├── personal_gesture_captcha.py  # Earlier gesture-template prototype
├── letter_shape_captcha.py      # Earlier letter-in-shape prototype
├── gesture_captcha.py           # Initial trajectory prototype
├── airDraw.py                   # Original air-drawing implementation
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Webcam hand tracking
- [x] Index-finger trajectory capture
- [x] Random shape generation
- [x] Letter-in-shape CAPTCHA
- [x] Personal letter selection
- [x] Random letter placement
- [x] Random letter rotation
- [x] Path scoring
- [x] Coverage scoring
- [x] Letter-order verification
- [ ] Stroke-level letter matching
- [ ] Multiple difficulty levels
- [ ] Stronger liveness / anti-replay checks
- [ ] Flask backend
- [ ] Browser camera interface
- [ ] Session-based CAPTCHA generation
- [ ] Human vs automated benchmark
- [ ] Automated tests

## Security Note

Check.it is a computer-vision prototype for exploring gesture-based CAPTCHA verification. The current implementation has not undergone security evaluation and should not be presented as a production replacement for established CAPTCHA systems.
