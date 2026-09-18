# Check.it — Letter-in-Shape Gesture CAPTCHA

A computer-vision CAPTCHA prototype that asks users to trace letters displayed inside a randomly generated shape using their live index-finger movement.

## Concept

Instead of a conventional text or image CAPTCHA, Check.it creates a visual challenge such as:

- Circle + two letters
- Square + two letters
- Triangle + two letters
- Hexagon + two letters
- Star + two letters

The user uses their finger in front of a webcam to trace the displayed letters from left to right.

## How It Works

```text
Random Shape + Letters
        ↓
Webcam
        ↓
MediaPipe Hand Detection
        ↓
Index Finger Tracking
        ↓
Trajectory Capture
        ↓
Letter Path Analysis
        ↓
Coverage + Path + Order Score
        ↓
PASS / FAIL
```

## Verification

The current prototype evaluates three signals:

| Signal | Weight |
|---|---:|
| Path proximity to letters | 55% |
| Letter-path coverage | 30% |
| Letter order | 15% |

The prototype currently passes an attempt at a score of **70% or higher**. This threshold is a development setting and must be calibrated with real human and automated test data before being used for security decisions.

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

Run the letter-in-shape CAPTCHA:

```bash
python letter_shape_captcha.py
```

### Controls

- `SPACE` — start tracing
- `R` — generate a new challenge
- `Q` — quit

## Example Challenge

A generated challenge might look like:

```text
        ╭────────────────╮
       /                  \
      /       A    R       \
     |                      |
      \                    /
       \__________________/
```

The user must trace the letters with their index finger.

## Project Structure

```text
Check.it/
├── letter_shape_captcha.py       # Current CAPTCHA prototype
├── personal_gesture_captcha.py   # Earlier personalized gesture prototype
├── gesture_captcha.py            # Initial trajectory prototype
├── airDraw.py                    # Original air-drawing implementation
├── requirements.txt
└── README.md
```

## Current Limitations

This is a computer-vision research/prototype implementation.

- The current application is a desktop OpenCV prototype.
- Challenge letters are generated from an OpenCV font rather than a dedicated vector-stroke model.
- The scoring threshold has not been statistically calibrated.
- Replay and advanced anti-spoofing resistance require dedicated security testing.
- A browser frontend and backend API are not yet integrated.

## Roadmap

- [x] Webcam hand tracking
- [x] Index-finger trajectory capture
- [x] Random shapes
- [x] Random letters
- [x] Letter-in-shape CAPTCHA
- [x] Path scoring
- [x] Coverage scoring
- [x] Letter-order verification
- [ ] Improve stroke-level letter matching
- [ ] Add rotation and difficulty levels
- [ ] Add stronger liveness / anti-replay checks
- [ ] Build Flask backend
- [ ] Build browser camera interface
- [ ] Add session-based CAPTCHA generation
- [ ] Benchmark false-accept and false-reject rates
- [ ] Add automated tests

## Security Note

Check.it is intended as a prototype for exploring gesture-based CAPTCHA verification. It should not be presented as a production security control until its detection, scoring, replay resistance, and attack resilience have been independently evaluated.
