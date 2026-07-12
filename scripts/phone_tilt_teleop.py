#!/usr/bin/env python3
"""Phone-gyroscope joystick control of the maze servos.

Serves a small HTTPS page over the local Wi-Fi network; the phone's browser
streams its DeviceOrientationEvent (beta/gamma) back as HTTP POSTs to that
same page's origin, which this script maps to yaw/pitch tilt exactly like
the mouse-driven joystick in touchpad_teleop.py.

Hold the phone in portrait, flat like a tray, top edge pointing away from
you. Tap "Recenter" on the phone page any time to re-zero the reference
orientation. Press q or Esc in this terminal (or "Stop Streaming" on the
phone) to stop.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from cps_maze.config import load_config
from cps_maze.control.phone_tilt import PhoneTiltConfig, is_stale, map_orientation
from cps_maze.hardware.serial_link import ArduinoServoLink, ServoCommand
from cps_maze.logging.run_logger import CsvRunLogger
from cps_maze.net.local_ip import get_local_ip
from cps_maze.net.phone_orientation_server import PhoneOrientationServer
from cps_maze.net.tls_cert import build_ssl_context, ensure_self_signed_cert

try:
    from pynput import keyboard as kb
except ImportError:
    sys.exit("pynput is required: pip3 install pynput")


HELP = """
Phone tilt teleop
------------------
  Open the printed URL on your phone's browser (same Wi-Fi as this PC).
  Tap "Enable Tilt Control", then tilt the phone to steer.
  Recenter (on phone) : re-zero the reference orientation
  Stop Streaming (phone) or q / Esc (here) : stop
"""


class _DryLink:
    """Stand-in for ArduinoServoLink under --dry-run: no serial port opened."""

    def send(self, command: ServoCommand) -> None:
        pass  # the control loop already prints yaw/pitch every tick

    def neutral(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "_DryLink":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phone-gyroscope joystick control for the maze servos.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--port", default=None)
    parser.add_argument("--deadzone-deg", type=float, default=3.0,
                         help="Degrees around the recentered zero treated as neutral. Default 3.")
    parser.add_argument("--max-tilt-deg", type=float, default=20.0,
                         help="Phone tilt (degrees) from zero for full tilt. Default 20.")
    parser.add_argument("--smooth", type=float, default=0.3,
                         help="Exponential smoothing factor 0-1 (0=frozen, 1=raw). Default 0.3.")
    parser.add_argument("--max-tilt", type=float, default=0.6,
                         help="Maximum tilt angle sent to servos (0-1). Default 0.6.")
    parser.add_argument("--rate-hz", type=float, default=200.0,
                         help="Command stream rate in Hz. Default 200.")
    parser.add_argument("--invert-yaw", action="store_true", help="Flip front/back direction.")
    parser.add_argument("--no-invert-pitch", dest="invert_pitch", action="store_false", default=True,
                         help="Disable the left/right inversion (on by default for this rig's wiring).")
    parser.add_argument("--swap-axes", action="store_true",
                         help="Swap which phone axis (beta/gamma) drives yaw vs pitch.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind the phone server on.")
    parser.add_argument("--http-port", type=int, default=8443)
    parser.add_argument("--stale-timeout-ms", type=float, default=350.0,
                         help="Decay to neutral if no phone sample arrives within this window.")
    parser.add_argument("--cert-dir", default="certs/phone_tls",
                         help="Where to cache the self-signed HTTPS certificate.")
    parser.add_argument("--log", default=None, help="Optional CSV path to log yaw/pitch each tick.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Skip serial output; print computed commands instead.")
    args = parser.parse_args()

    if not 0.0 < args.smooth <= 1.0:
        sys.exit("--smooth must be in (0, 1].")
    if not 0.0 < args.max_tilt <= 1.0:
        sys.exit("--max-tilt must be in (0, 1].")

    cfg = PhoneTiltConfig(
        deadzone_deg=args.deadzone_deg,
        max_tilt_deg=args.max_tilt_deg,
        max_tilt=args.max_tilt,
        yaw_sign=-1.0 if args.invert_yaw else 1.0,
        pitch_sign=-1.0 if args.invert_pitch else 1.0,
        swap_axes=args.swap_axes,
    )

    config = load_config(args.config)
    port = args.port or config.serial["port"]
    period = 1.0 / args.rate_hz
    stale_timeout_s = args.stale_timeout_ms / 1000.0

    local_ip = get_local_ip()
    cert_path, key_path = ensure_self_signed_cert(args.cert_dir, local_ip)
    ssl_context = build_ssl_context(cert_path, key_path)

    html_path = Path(__file__).parent / "phone_tilt_teleop_page.html"
    server = PhoneOrientationServer(
        host=args.host,
        http_port=args.http_port,
        ssl_context=ssl_context,
        html_path=html_path,
    )
    server.start()

    print(HELP)
    print("=" * 60)
    print(" Point your phone's browser at:")
    print(f"     https://{local_ip}:{args.http_port}")
    print(' Same Wi-Fi as this PC. Accept the "not private" warning once.')
    print("=" * 60)
    print(f"Connecting on {port}  deadzone={args.deadzone_deg}deg  max_tilt_deg={args.max_tilt_deg}  "
          f"smooth={args.smooth}  max_tilt=±{args.max_tilt}  {args.rate_hz:.0f} Hz "
          f"{'(dry run — no serial)' if args.dry_run else ''}")

    quit_flag = threading.Event()

    def on_press(key):
        char = getattr(key, "char", None)
        if char in ("q", "Q") or key == kb.Key.esc:
            quit_flag.set()
            return False

    listener = kb.Listener(on_press=on_press)
    listener.start()

    logger = CsvRunLogger(args.log, ["timestamp_s", "yaw", "pitch"]) if args.log else None

    link_cm = _DryLink() if args.dry_run else ArduinoServoLink(
        port=port,
        baudrate=int(config.serial["baudrate"]),
        timeout_s=float(config.serial["timeout_s"]),
    )

    start_time = time.monotonic()
    with link_cm as link:
        if not args.dry_run:
            time.sleep(2.0)
        link.neutral()
        print("Ready. Open the URL above on your phone, tap Enable Tilt Control. q / Esc = quit.\n")

        smoothed_yaw = 0.0
        smoothed_pitch = 0.0
        alpha = args.smooth
        next_tick = time.monotonic()

        try:
            while not quit_flag.is_set():
                sample = server.latest()
                now = time.monotonic()
                if sample is None:
                    raw_yaw, raw_pitch = 0.0, 0.0
                else:
                    beta, gamma, recv_time = sample
                    if is_stale(now - recv_time, stale_timeout_s):
                        raw_yaw, raw_pitch = 0.0, 0.0
                    else:
                        raw_yaw, raw_pitch = map_orientation(beta, gamma, cfg)

                smoothed_yaw = alpha * raw_yaw + (1.0 - alpha) * smoothed_yaw
                smoothed_pitch = alpha * raw_pitch + (1.0 - alpha) * smoothed_pitch

                print(f"\ryaw={smoothed_yaw:+.3f}  pitch={smoothed_pitch:+.3f}   ",
                      end="", flush=True)
                link.send(ServoCommand(yaw=smoothed_yaw, pitch=smoothed_pitch))
                if logger:
                    logger.write({
                        "timestamp_s": now - start_time,
                        "yaw": smoothed_yaw,
                        "pitch": smoothed_pitch,
                    })

                next_tick += period
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            listener.stop()
            server.stop()
            link.neutral()
            if logger:
                logger.close()
            print("\nReturned to neutral. Bye.")


if __name__ == "__main__":
    main()
