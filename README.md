# CPS-LabyrinTUM

CPS-LabyrinTUM is an autonomous cyber-physical marble maze solver. The system uses an overhead camera to estimate the ball position, converts image pixels into board coordinates, plans along an annotated maze path, and commands a two-axis servo-actuated wooden labyrinth through an Arduino and PCA9685 servo driver.

The project successfully completed the full physical maze for the final demo/report using a classical vision and control stack. Reinforcement learning was not required.

## Final Outcome

The completed system solves the physical marble maze autonomously after manual ball placement/reset. The final run uses:

```text
camera -> ball tracking -> maze coordinates -> path planner -> controller -> Arduino -> servos
```

The repository contains the software, firmware, configuration, calibration workflow, run logging, and visualization tools used to reach the successful full-maze run.

---

## Why CPS-LabyrinTUM?

A wooden labyrinth maze is a compact cyber-physical systems problem: perception, calibration, planning, control, real-time hardware actuation, and mechanical safety all interact. CPS-LabyrinTUM demonstrates that a reliable autonomous solver can be built without a large reinforcement-learning setup by combining calibrated computer vision, classical control, careful servo limits, and iterative hardware testing.

---

## Features

- **Autonomous full-maze solving** after manual ball placement/reset.
- **Overhead camera tracking** for a reflective marble on the wooden maze.
- **Image-to-board homography calibration** for maze coordinates in millimeters.
- **Annotated waypoint path following** with speed profiling and recovery behavior.
- **Classical yaw/pitch control** for a two-axis servo-actuated board.
- **Arduino/PCA9685 firmware layer** with serial commands, servo limits, ramping, and watchdog neutral behavior.
- **Operator tools** for camera checks, threshold tuning, keyboard teleop, phone-tilt teleop, servo testing, calibration, autonomous runs, and run visualization.
- **Run logging and replay support** for debugging failures and documenting successful runs.

---

## Hardware

- USB2 UVC global-shutter mono camera, OV9281 class.
- Arduino UNO R4 Minima.
- PCA9685 16-channel I2C PWM servo driver.
- 2 x hobby digital metal-gear servos.
- 5 V 10 A servo power supply.
- Wooden labyrinth board.
- M3 rods, ball joints, servo horns, and camera mounting hardware.
- Reflective metal marble.

---

## Software Stack

- Python 3.10+.
- OpenCV - camera access, image processing, calibration, overlays, and video tools.
- NumPy - geometry, controller math, and log analysis.
- PySerial - PC-to-Arduino serial communication.
- PyYAML - runtime configuration files.
- pynput - keyboard teleoperation support.
- cryptography - local HTTPS certificate support for phone-tilt teleop.
- Arduino firmware for UNO R4 + PCA9685 servo output.

---

## Dependencies & Setup

1. Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

2. Upload the Arduino firmware:

```text
firmware/arduino/maze_servo_controller/maze_servo_controller.ino
```

3. Configure the serial port in `configs/default.yaml`, or pass `--port` to supported scripts. Examples: `COM10` on Windows or `/dev/ttyACM0` on Linux. Close the Arduino IDE Serial Monitor before running Python scripts because only one process can hold the serial port.

4. Check the camera and servo connection:

```bash
python scripts/check_camera.py --config configs/default.yaml
python scripts/manual_servo_test.py --config configs/default.yaml --neutral
python scripts/servo_sweep_test.py --config configs/default.yaml --axis yaw --amplitude 0.10
```

5. Calibrate the board coordinate transform. The proven workflow is the interactive corner-click homography:

```bash
python scripts/calibrate_homography.py --config configs/default.yaml
```

A CharUco-based option is also available:

```bash
python scripts/calibrate_charuco_homography.py \
  --config configs/default.yaml \
  --output calibration/board_homography.npz
```

6. Use teleoperation tools when checking axis direction, trim, and safe tilt limits:

```bash
python scripts/keyboard_teleop.py --limit 0.3
python scripts/phone_tilt_teleop.py
```

Start with conservative limits to avoid driving a servo into the maze's mechanical stops.

7. Run the autonomous solver:

```bash
python scripts/run_autonomous.py --config configs/default.yaml
```

Run videos and CSV logs can be reviewed with:

```bash
python scripts/visualize_run.py
```

---

## Repository Map

- `src/cps_maze/` - Python package for camera, vision, planning, control, serial hardware interface, networking, and logging.
- `firmware/arduino/maze_servo_controller/` - Arduino firmware for UNO R4 + PCA9685.
- `configs/` - Runtime configuration files and tuned control parameters.
- `calibration/` - Camera/board calibration files, ROI data, path data, and calibration notes.
- `data/` - Local run data, videos, and processed logs.
- `scripts/` - Operator scripts for calibration, camera checks, servo tests, teleop, autonomous runs, and analysis.
- `docs/` - Architecture notes, hardware notes, validation plans, paper material, and project handoff docs.
- `docs/video/` - Final demonstration video assets.
- `logs/agent/` - Human/AI development handoff logs.
- `tests/` - Unit tests for pure Python logic.

---

## Demo Video

[Final autonomous maze demo](docs/video/demo.mp4)

---

## Acknowledgements

- Developed as part of the Embedded Systems, Cyber-Physical Systems and Robotics (INHN0018) course at TUM.
- Built on OpenCV, NumPy, PySerial, PyYAML, Arduino, and the PCA9685 servo-control ecosystem.
- Inspired by the high-level CyberRunner idea of camera-based maze control, but implemented for this project's own hardware, maze layout, and classical-control approach.
