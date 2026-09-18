# Check.it — Personalized Gesture CAPTCHA

A computer-vision prototype that adds a personalized finger-gesture verification layer to CAPTCHA challenges.

## Concept

Instead of asking a user to solve a traditional visual CAPTCHA, Check.it learns a user's personal finger gesture during enrollment and verifies a live attempt using a webcam.

The verification combines:

- Normalized fingertip trajectory
- Path similarity
- Movement direction
- Timing profile
- Minimum movement / liveness-related checks

## How It Works

### 1. Enrollment

The user performs their preferred gesture three times.

The application extracts a compact gesture template from the fingertip trajectory rather than storing camera footage.

### 2. Personalized Challenge

A verification challenge presents randomized numbered points.

The user must visit the points in the expected sequence while the camera tracks their index finger.

### 3. Verification

The captured trajectory is compared with the enrolled gesture template.

The prototype produces a weighted verification score:

| Component | Weight |
|---|---:|
| Path similarity | 50% |
| Direction similarity | 25% |
| Timing similarity | 15% |
| Movement check | 10% |

The current prototype uses a 72% passing threshold. This threshold is a prototype setting and should be calibrated with real evaluation data before being used for security decisions.

## Technology Stack

- Python
- OpenCV
- MediaPipe
- NumPy

## Project Structure

\`\`\`text
Check.it/
├── personal_gesture_captcha.py
├── gesture_captcha.py
├── airDraw.py
├── requirements.txt
└── gesture_template.json   # created locally after enrollment
\`\`\`

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Run

\`\`\`bash
python personal_gesture_captcha.py
\`\`\`

If no local gesture template exists, the program starts enrollment automatically.

### Controls

- \`SPACE\` — start enrollment / verification
- \`ENTER\` — finish an enrollment recording
- \`R\` — generate another verification challenge
- \`Q\` — quit

## Current Limitations

This is a research/prototype implementation rather than a production CAPTCHA service.

- The prototype uses a local JSON template instead of a user database.
- The camera application currently runs as a desktop OpenCV application.
- The verification threshold has not yet been statistically calibrated.
- Replay resistance and advanced anti-spoofing require additional testing.
- A web frontend and backend API are still to be integrated.

## Planned Architecture

\`\`\`text
Browser
   │
   ▼
CAPTCHA Challenge
   │
   ▼
Camera / Finger Tracking
   │
   ▼
Trajectory Extraction
   │
   ▼
Feature Normalization
   │
   ├── Path Similarity
   ├── Direction
   ├── Timing
   └── Movement / Liveness Signals
   │
   ▼
Verification Score
   │
   ├── PASS
   └── FAIL
\`\`\`

## Roadmap

- [x] Webcam finger tracking
- [x] Trajectory extraction
- [x] Personal gesture enrollment
- [x] Gesture template generation
- [x] Randomized challenge points
- [x] Multi-factor trajectory scoring
- [ ] Improve challenge-response design
- [ ] Add stronger anti-replay / liveness checks
- [ ] Add Flask backend
- [ ] Add browser-based JavaScript client
- [ ] Add user/session management
- [ ] Benchmark false-accept and false-reject rates
- [ ] Add automated tests

## Disclaimer

The current implementation is a computer-vision security prototype. It should not be represented as a proven replacement for established CAPTCHA or biometric security systems without controlled evaluation and security testing.
