#!/usr/bin/env python3
"""Maze controller launcher — choose keyboard or touchpad control."""
from __future__ import annotations

import subprocess
import sys
import signal
import tkinter as tk
from pathlib import Path

SCRIPTS = Path(__file__).parent

# Colors
BG       = "#1a1a2e"
BTN_KEY  = "#16213e"
BTN_PAD  = "#0f3460"
ACCENT   = "#e94560"
FG       = "#eaeaea"
FG_DIM   = "#888888"


def launch(script: str, proc_holder: list) -> None:
    # Kill any running session first
    if proc_holder and proc_holder[0].poll() is None:
        proc_holder[0].terminate()
        proc_holder.clear()

    p = subprocess.Popen([sys.executable, str(SCRIPTS / script)])
    proc_holder.append(p)


def build_ui() -> None:
    proc_holder: list = []

    def stop_running_session() -> None:
        if proc_holder and proc_holder[0].poll() is None:
            proc_holder[0].terminate()
            try:
                proc_holder[0].wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc_holder[0].kill()
        proc_holder.clear()

    def handle_signal(_signum, _frame) -> None:
        stop_running_session()
        root.destroy()

    root = tk.Tk()
    root.title("Maze Controller")
    root.configure(bg=BG)
    root.resizable(False, False)

    # Center window on screen
    root.update_idletasks()
    w, h = 380, 280
    x = (root.winfo_screenwidth()  - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    tk.Label(root, text="Maze Controller", font=("Helvetica", 18, "bold"),
             bg=BG, fg=FG).pack(pady=(28, 4))
    tk.Label(root, text="Choose your control method", font=("Helvetica", 11),
             bg=BG, fg=FG_DIM).pack(pady=(0, 24))

    def btn(parent, label, sublabel, color, command):
        frame = tk.Frame(parent, bg=color, cursor="hand2")
        frame.pack(fill="x", padx=32, pady=6, ipady=10)
        tk.Label(frame, text=label, font=("Helvetica", 13, "bold"),
                 bg=color, fg=FG).pack()
        tk.Label(frame, text=sublabel, font=("Helvetica", 10),
                 bg=color, fg=FG_DIM).pack()
        frame.bind("<Button-1>", lambda _: command())
        for child in frame.winfo_children():
            child.bind("<Button-1>", lambda _: command())

    btn(root, "⌨  Keyboard", "WASD keys · diagonal combos · hold to tilt",
        BTN_KEY, lambda: launch("keyboard_teleop.py", proc_holder))

    btn(root, "🖱  Touchpad", "Slide finger · joystick feel · lift to neutral",
        BTN_PAD, lambda: launch("touchpad_teleop.py", proc_holder))

    tk.Label(root, text="Press q or Esc inside the terminal to stop",
             font=("Helvetica", 9), bg=BG, fg=FG_DIM).pack(pady=(18, 0))

    def on_close():
        stop_running_session()
        root.destroy()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    build_ui()
