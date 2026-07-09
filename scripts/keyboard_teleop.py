#!/usr/bin/env python3
"""Manual keyboard control of the maze servos — pynput multi-key mode.

Uses real key-down / key-up events so any combination of keys can be
held simultaneously (true diagonal movement).

Hold key(s) → board accelerates toward max tilt on each active axis.
Release key  → that axis snaps instantly back to neutral.
              The other axis keeps going unaffected.
"""
from __future__ import annotations

import argparse
import sys
import time
import threading

from cps_maze.config import load_config
from cps_maze.hardware.serial_link import ArduinoServoLink, ServoCommand

try:
    from pynput import keyboard as kb
except ImportError:
    sys.exit("pynput is required: pip3 install pynput")


HELP = """
Keyboard teleop  (hold keys — diagonals supported)
----------------------------------------------------
  W / S      : tilt front / back
  A / D      : tilt left  / right
  W+A, W+D … : hold any two (or more) for diagonal
  Space      : force neutral
  q or Esc   : quit
"""

# Map pynput key objects → direction tokens
_KEY_MAP: dict = {}  # built after parse so we can also check char

# Direction token → (yaw_sign, pitch_sign)
_DIR: dict[str, tuple[int, int]] = {
    "fwd":   (+1,  0),   # W  → yaw negative
    "back":  (-1,  0),   # S  → yaw positive
    "left":  ( 0, -1),   # A  → pitch positive
    "right": ( 0, +1),   # D  → pitch negative
}

# On first frame of a key press, velocity is set to max_vel directly —
# board moves at full speed from tick one, zero ramp-up delay.
_INITIAL_KICK_FRACTION = 1.0  # fraction of max_vel applied on frame 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--port", default=None)
    parser.add_argument(
        "--accel", type=float, default=40.0,
        help="Acceleration in units/sec² while key held.",
    )
    parser.add_argument(
        "--max-vel", type=float, default=12.0,
        help="Max tilt speed in units/sec.",
    )
    parser.add_argument(
        "--max-tilt", type=float, default=0.9,
        help="Maximum tilt angle (0-1).",
    )
    parser.add_argument(
        "--rate-hz", type=float, default=200.0,
        help="Command stream rate in Hz.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    port = args.port or config.serial["port"]
    period = 1.0 / args.rate_hz

    # ── Shared key state (written by pynput thread, read by control loop) ──
    pressed: set[str] = set()   # set of active direction tokens
    quit_flag = threading.Event()
    lock = threading.Lock()

    def _token(key) -> str | None:
        char = getattr(key, "char", None)
        if char:
            return {"w": "fwd", "W": "fwd",
                    "s": "back", "S": "back",
                    "a": "left", "A": "left",
                    "d": "right", "D": "right"}.get(char)
        return None  # non-character key

    def on_press(key):
        char = getattr(key, "char", None)
        if char in ("q", "Q") or key == kb.Key.esc:
            quit_flag.set()
            return False  # stop listener
        if char == " ":
            with lock:
                pressed.clear()
            return
        tok = _token(key)
        if tok:
            with lock:
                pressed.add(tok)

    def on_release(key):
        tok = _token(key)
        if tok:
            with lock:
                pressed.discard(tok)

    print(HELP)
    print(f"Connecting on {port}  accel={args.accel}  max_vel={args.max_vel}  "
          f"max_tilt=±{args.max_tilt}  {args.rate_hz:.0f} Hz ...")

    with ArduinoServoLink(
        port=port,
        baudrate=int(config.serial["baudrate"]),
        timeout_s=float(config.serial["timeout_s"]),
    ) as link:
        time.sleep(2.0)
        link.neutral()
        print("Ready. Hold WASD (combine for diagonals). q / Esc = quit.\n")

        listener = kb.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        yaw   = 0.0
        pitch = 0.0
        yaw_vel   = 0.0
        pitch_vel = 0.0
        prev_pressed: set[str] = set()

        next_tick = time.monotonic()
        try:
            while not quit_flag.is_set():
                with lock:
                    cur = set(pressed)

                # Detect freshly pressed keys → jump straight to max velocity
                fresh = cur - prev_pressed
                for tok in fresh:
                    dy, dp = _DIR[tok]
                    if dy != 0:
                        yaw_vel = dy * args.max_vel * _INITIAL_KICK_FRACTION
                    if dp != 0:
                        pitch_vel = dp * args.max_vel * _INITIAL_KICK_FRACTION

                prev_pressed = cur

                # Net direction per axis
                yaw_sign   = max(-1, min(1, sum(_DIR[t][0] for t in cur)))
                pitch_sign = max(-1, min(1, sum(_DIR[t][1] for t in cur)))

                # Yaw axis
                if yaw_sign != 0:
                    yaw_vel = max(-args.max_vel, min(args.max_vel,
                                  yaw_vel + yaw_sign * args.accel * period))
                    yaw = max(-args.max_tilt, min(args.max_tilt,
                               yaw + yaw_vel * period))
                else:
                    yaw = 0.0
                    yaw_vel = 0.0

                # Pitch axis
                if pitch_sign != 0:
                    pitch_vel = max(-args.max_vel, min(args.max_vel,
                                    pitch_vel + pitch_sign * args.accel * period))
                    pitch = max(-args.max_tilt, min(args.max_tilt,
                                 pitch + pitch_vel * period))
                else:
                    pitch = 0.0
                    pitch_vel = 0.0

                print(f"\ryaw={yaw:+.3f}  pitch={pitch:+.3f}   ", end="", flush=True)
                link.send(ServoCommand(yaw=yaw, pitch=pitch))

                next_tick += period
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

        finally:
            listener.stop()
            link.neutral()
            print("\nReturned to neutral. Bye.")


if __name__ == "__main__":
    main()
