# CPS-ML Maze

Autonomous marble maze project using a camera, a two-axis servo-actuated wooden labyrinth board, and a classical vision/control software stack.

The target system is not a direct CyberRunner clone. It uses the same high-level idea of camera-based state estimation and motorized board control, but the implementation is adapted for this project's hardware: a USB global-shutter camera, Arduino UNO R4 Minima, PCA9685 PWM servo driver, and hobby servos.

## Goal

Solve the full physical maze without user input after manual ball placement/reset.

The first working target is a reliable classical-control solver:

```text
camera -> ball tracking -> maze coordinates -> path planner -> controller -> Arduino -> servos
```

Reinforcement learning is intentionally not required for the initial demo.

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

The project is at scaffold stage. Before autonomous runs, the team must:

1. Verify servo PWM direction and safe travel limits.
2. Mount servos and camera rigidly.
3. Calibrate camera intrinsics and board homography.
4. Annotate the maze path and holes.
5. Tune the first segment controller.

See [TODO.md](TODO.md) and [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md).

The initial scaffold validation is documented in [docs/SCAFFOLD_VALIDATION.md](docs/SCAFFOLD_VALIDATION.md).
