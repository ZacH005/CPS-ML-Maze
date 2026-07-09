#!/usr/bin/env python3
"""Run the full fixed-camera recalibration workflow in order.

This is an operator orchestrator, not a headless calibration algorithm. Most
steps still open their normal interactive windows and require human approval.

Default flow:
  1. Back up current generated artifacts.
  2. Check the camera view.
  3. Rebuild homography, holes, and path.
  4. Record a fresh tracking video while launching scripts/launcher.py for
     manual board control.
  5. Rebuild ROI/confusers and validate tracker offline.
  6. Rebuild axis map.
  7. Open autonomous dry-run overlay.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
GENERATED_ARTIFACTS = [
    Path("calibration/board_homography.npz"),
    Path("calibration/live_roi.json"),
    Path("calibration/live_confusers.json"),
    Path("calibration/live_roi_overlay.png"),
    Path("calibration/axis_map.npz"),
    Path("configs/maze_holes.csv"),
    Path("configs/maze_path_auto.csv"),
]


def env_with_src() -> dict[str, str]:
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not old else f"{src}{os.pathsep}{old}"
    return env


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def prompt(message: str, assume_yes: bool) -> None:
    if assume_yes:
        print(message)
        return
    input(f"\n{message}\nPress Enter to continue, or Ctrl-C to stop. ")


def run_step(
    title: str,
    command: list[str],
    assume_yes: bool,
    instructions: str | None = None,
    allow_failure: bool = False,
) -> bool:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    if instructions:
        print(instructions.strip())
    print("\nCommand:")
    print("  " + " ".join(command))
    prompt("Start this step.", assume_yes)
    result = subprocess.run(command, cwd=REPO_ROOT, env=env_with_src(), check=False)
    if result.returncode == 0:
        return True
    print(f"\nStep failed with exit code {result.returncode}: {title}")
    if allow_failure:
        print("Continuing because this step is marked optional.")
        return False
    raise SystemExit(result.returncode)


def backup_artifacts(timestamp: str) -> Path:
    backup_dir = REPO_ROOT / "calibration" / f"refresh_backup_{timestamp}"
    copied = 0
    for artifact in GENERATED_ARTIFACTS:
        src = REPO_ROOT / artifact
        if not src.exists():
            continue
        dst = backup_dir / artifact
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"Backed up {copied} artifact(s) to {rel(backup_dir)}")
    return backup_dir


def record_with_launcher(args: argparse.Namespace, timestamp: str) -> Path:
    if args.video:
        video = Path(args.video)
        if not video.is_absolute():
            video = REPO_ROOT / video
        if not video.exists():
            raise SystemExit(f"--video does not exist: {video}")
        print(f"Using existing recording: {video}")
        return video

    output = REPO_ROOT / "data" / "raw" / f"recalibration_{timestamp}.avi"
    launcher_proc: subprocess.Popen | None = None
    try:
        print("\n" + "=" * 78)
        print("Record Tracking Video With Board Control")
        print("=" * 78)
        print(
            "A launcher window will open. Choose Keyboard or Touchpad control, "
            "then move the ball through representative maze regions while the "
            "camera records. Keep lighting and camera pose unchanged."
        )
        prompt("Open board-control launcher.", args.yes)
        launcher_proc = subprocess.Popen(
            [sys.executable, str(SCRIPTS / "launcher.py")],
            cwd=REPO_ROOT,
            env=env_with_src(),
        )
        prompt(
            "After selecting a control mode in the launcher and placing the ball, "
            "start recording.",
            assume_yes=False,
        )
        run_step(
            "Record Camera Footage",
            [
                sys.executable,
                str(SCRIPTS / "record_camera.py"),
                "--config",
                args.config,
                "--output",
                str(output),
                "--duration-s",
                str(args.record_duration_s),
            ],
            assume_yes=True,
            instructions=(
                "Use the board-control process launched by launcher.py during "
                "this recording. Press q in the recording preview to stop early."
            ),
        )
    finally:
        if launcher_proc is not None and launcher_proc.poll() is None:
            print("\nRecording step finished. Closing the launcher window/process.")
            launcher_proc.terminate()
            try:
                launcher_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                launcher_proc.kill()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full recalibration sequence for run_autonomous.py."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--video",
        default=None,
        help="Existing tracking video to use instead of recording a new one.",
    )
    parser.add_argument("--record-duration-s", type=float, default=60.0)
    parser.add_argument("--homography", default="calibration/board_homography.npz")
    parser.add_argument("--holes", default="configs/maze_holes.csv")
    parser.add_argument("--path", default="configs/maze_path_auto.csv")
    parser.add_argument("--roi", default="calibration/live_roi.json")
    parser.add_argument("--roi-overlay", default="calibration/live_roi_overlay.png")
    parser.add_argument("--confusers", default="calibration/live_confusers.json")
    parser.add_argument("--axis-map", default="calibration/axis_map.npz")
    parser.add_argument(
        "--manual-path",
        action="store_true",
        help="Use scripts/annotate_path.py instead of scripts/auto_trace_path.py.",
    )
    parser.add_argument("--skip-camera-check", action="store_true")
    parser.add_argument("--skip-tracker-validation", action="store_true")
    parser.add_argument("--skip-axis-check", action="store_true")
    parser.add_argument("--skip-dry-run", action="store_true")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Do not pause before each step. Interactive tool windows still require input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Full recalibration workflow")
    print(f"repo: {REPO_ROOT}")
    print(f"config: {args.config}")
    backup_artifacts(timestamp)

    if not args.skip_camera_check:
        run_step(
            "Camera Check",
            [sys.executable, str(SCRIPTS / "check_camera.py"), "--config", args.config],
            args.yes,
            "Confirm the live view, tracker preview, and camera convention. Press q to close.",
        )

    run_step(
        "Rebuild Board Homography",
        [
            sys.executable,
            str(SCRIPTS / "calibrate_homography.py"),
            "--config",
            args.config,
            "--output",
            args.homography,
        ],
        args.yes,
        "Click inside playable corners in order, press s to save, verify the grid, then q.",
    )

    run_step(
        "Rebuild Hole CSV",
        [
            sys.executable,
            str(SCRIPTS / "auto_detect_holes.py"),
            "--config",
            args.config,
            "--homography",
            args.homography,
            "--output",
            args.holes,
        ],
        args.yes,
        "Adjust threshold, add/remove holes if needed, press s to save, then q.",
    )

    if args.manual_path:
        path_command = [
            sys.executable,
            str(SCRIPTS / "annotate_path.py"),
            "--config",
            args.config,
            "--homography",
            args.homography,
            "--path-output",
            args.path,
            "--holes-output",
            args.holes,
        ]
        path_instructions = "Click path waypoints from start to finish, press s to save, then q."
    else:
        path_command = [
            sys.executable,
            str(SCRIPTS / "auto_trace_path.py"),
            "--config",
            args.config,
            "--homography",
            args.homography,
            "--output",
            args.path,
        ]
        path_instructions = "Click the start of the printed line, inspect trace, press s to save, then q."
    run_step("Rebuild Path CSV", path_command, args.yes, path_instructions)

    video = record_with_launcher(args, timestamp)

    run_step(
        "Select Live Tracker ROI",
        [
            sys.executable,
            str(SCRIPTS / "select_maze_roi.py"),
            "--source",
            str(video),
            "--output",
            args.roi,
            "--overlay-output",
            args.roi_overlay,
        ],
        args.yes,
        "Click a polygon around only the playable maze area, press s to save, then q.",
    )

    run_step(
        "Rebuild Static Confusers",
        [
            sys.executable,
            str(SCRIPTS / "pipeline.py"),
            "--calibrate",
            str(video),
            "--confusers-file",
            args.confusers,
            "--roi-file",
            args.roi,
        ],
        args.yes,
    )

    if not args.skip_tracker_validation:
        run_step(
            "Validate Tracker Offline",
            [
                sys.executable,
                str(SCRIPTS / "pipeline.py"),
                str(video),
                "--auto-seed",
                "--confusers-file",
                args.confusers,
                "--out-video",
                str(REPO_ROOT / "data" / "processed" / f"recalibration_tracker_{timestamp}.mp4"),
                "--out-csv",
                str(REPO_ROOT / "data" / "processed" / f"recalibration_tracker_{timestamp}.csv"),
            ],
            args.yes,
            "Review the generated annotated video after this step.",
            allow_failure=True,
        )

    if not args.skip_axis_check:
        run_step(
            "Rebuild Axis Map",
            [
                sys.executable,
                str(SCRIPTS / "axis_check.py"),
                "--config",
                args.config,
                "--homography",
                args.homography,
                "--output",
                args.axis_map,
                "--amplitude",
                "0.4",
                "--max-amplitude",
                "1.0",
                "--pulse-seconds",
                "1.2",
                "--measure-timeout-s",
                "10",
            ],
            args.yes,
            "Place the ball in an open area, click to seed each pulse, and stop if unsafe.",
        )

    if not args.skip_dry_run:
        run_step(
            "Autonomous Dry Run Overlay",
            [
                sys.executable,
                str(SCRIPTS / "run_autonomous.py"),
                "--config",
                args.config,
                "--homography",
                args.homography,
                "--path",
                args.path,
                "--holes",
                args.holes,
                "--axis-map",
                args.axis_map,
                "--dry-run",
            ],
            args.yes,
            "Click the ball, press Space, and verify path/target/command overlay. No servos move.",
        )

    print("\nRecalibration workflow complete.")
    print("Primary artifacts:")
    for artifact in GENERATED_ARTIFACTS:
        print(f"  {artifact}")
    print(f"Tracking video: {rel(video)}")


if __name__ == "__main__":
    main()
