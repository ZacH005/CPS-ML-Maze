# CPS-ML Maze

Autonomous marble maze project using a camera, a two-axis servo-actuated wooden labyrinth board, and a classical vision/control software stack. The system successfully completed the full physical maze for the final project demo/report.

The target system is not a direct CyberRunner clone. It uses the same high-level idea of camera-based state estimation and motorized board control, but the implementation is adapted for this project's hardware: a USB global-shutter camera, Arduino UNO R4 Minima, PCA9685 PWM servo driver, and hobby servos.

## Goal and Outcome

The goal was to solve the full physical maze without user input after manual ball placement/reset.

This milestone has been achieved: the autonomous stack completed the maze using camera-based ball tracking, calibrated board coordinates, path following, classical control, and Arduino/PCA9685-driven servo actuation.

The working control loop is:

```text
camera -> ball tracking -> maze coordinates -> path planner -> controller -> Arduino -> servos
```

Reinforcement learning was intentionally not required; the final solver uses the classical-control approach developed in this repository.

## Repository Map

- `src/cps_maze/` - Python package for camera, vision, planning, control, serial hardware interface, and logging.
- `firmware/arduino/maze_servo_controller/` - Arduino firmware for UNO R4 + PCA9685.
- `configs/` - Runtime configuration files.
- `calibration/` - Camera/board calibration files and notes.
- `data/` - Local run data, videos, and processed logs.
- `scripts/` - Operator scripts for camera checks, servo tests, and autonomous runs.
- `docs/` - System context, architecture, hardware notes, validation plan, and AI-agent handoff docs.
- `logs/agent/` - Human/AI development handoff logs.
- `tests/` - Unit tests for pure Python logic.

## Quick Start

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Upload the Arduino firmware from:

```text
firmware/arduino/maze_servo_controller/maze_servo_controller.ino
```

Run initial checks:

```bash
python scripts/check_camera.py --config configs/default.yaml
python scripts/manual_servo_test.py --config configs/default.yaml --neutral
python scripts/servo_sweep_test.py --config configs/default.yaml --axis yaw --amplitude 0.10
```

> Set the serial port for your machine in `configs/default.yaml` (e.g. `COM10`
> on Windows, `/dev/ttyACM0` on Linux), or pass `--port` to the scripts that
> support it. Close the Arduino IDE Serial Monitor first — only one program can
> hold the serial port at a time.

### Manual keyboard control (teleop)

Drive the board tilt live with the arrow keys. Useful for finding safe tilt
limits and checking axis direction before autonomous runs. Windows only (uses
`msvcrt`).

```bash
python scripts/keyboard_teleop.py --limit 0.3
```

Controls:

| Key | Action |
| --- | --- |
| Left / Right | yaw (channel 0) |
| Up / Down | pitch (channel 1) |
| Space | return to neutral |
| `+` / `-` | larger / smaller step per press |
| `q` / Esc | quit (returns to neutral first) |

Each arrow press nudges the tilt and the script streams the command
continuously so the board holds position against the firmware's 500 ms
watchdog.

Useful flags:

- `--limit 0.3` caps the maximum tilt (0–1). **Start low** to avoid stalling a
  servo into the board's mechanical stops.
- `--step 0.05` sets the tilt change per key press.
- `--invert-yaw` / `--invert-pitch` flip a reversed axis direction.
- `--swap-axes` swaps which arrow pair drives each channel (use if the servo
  connectors are wired to the opposite channels).
- `--port COM10` overrides the serial port from the config.

Create an initial board homography from measured correspondences:

```bash
python scripts/create_homography_from_csv.py \
  --points-csv calibration/example_marker_points.csv \
  --output calibration/board_homography.npz
```

Or calibrate directly from the CharUco board in the camera view:

```bash
python scripts/calibrate_charuco_homography.py \
  --config configs/default.yaml \
  --output calibration/board_homography.npz
```

## Current Project Status

The maze completion milestone is done. The repository now contains the working software stack used to drive the physical maze through the final route:

1. Camera capture and bright-blob ball tracking.
2. Image-to-board coordinate calibration through homography.
3. Annotated maze path following with speed profiling and recovery behavior.
4. PID/classical control for yaw and pitch board commands.
5. Arduino serial command bridge and PCA9685 servo output.
6. Run logging and visualization tools for tuning and post-run analysis.

The remaining documentation and logs preserve the development history, hardware notes, calibration assumptions, and tuning workflow used to reach the successful run. See [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [logs/agent/](logs/agent/).

The initial scaffold validation is documented in [docs/SCAFFOLD_VALIDATION.md](docs/SCAFFOLD_VALIDATION.md).
