"""
app.py

Main GUI entry point for the Face Recognition Attendance System.
Wires together registration, training, and attendance into one window.
"""

import tkinter as tk

from utils import ensure_project_folders
from register import register_student
from train import train_model
from attendance import start_attendance

WINDOW_TITLE = "Face Recognition Attendance System"
WINDOW_SIZE = "900x600"
BG_COLOR = "#E8F0FE"
BUTTON_FONT = ("Arial", 16)
BUTTON_SIZE = {"width": 20, "height": 2}


def build_window() -> tk.Tk:
    window = tk.Tk()
    window.title(WINDOW_TITLE)
    window.geometry(WINDOW_SIZE)
    window.configure(bg=BG_COLOR)

    tk.Label(
        window,
        text="FACE RECOGNITION ATTENDANCE SYSTEM",
        font=("Arial", 22, "bold"),
        bg=BG_COLOR,
        fg="navy"
    ).pack(pady=20)

    tk.Button(
        window, text="Register Student", font=BUTTON_FONT,
        command=register_student, **BUTTON_SIZE
    ).pack(pady=10)

    tk.Button(
        window, text="Train Model", font=BUTTON_FONT,
        command=train_model, **BUTTON_SIZE
    ).pack(pady=10)

    tk.Button(
        window, text="Start Attendance", font=BUTTON_FONT,
        command=start_attendance, **BUTTON_SIZE
    ).pack(pady=10)

    tk.Button(
        window, text="Exit", font=BUTTON_FONT,
        command=window.destroy, **BUTTON_SIZE
    ).pack(pady=10)

    return window


if __name__ == "__main__":
    ensure_project_folders()
    app_window = build_window()
    app_window.mainloop()
