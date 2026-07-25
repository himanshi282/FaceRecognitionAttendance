"""
database.py

All persistent-data I/O for the attendance system:
    - labels.json           (label_id -> "roll_name", written by train.py)
    - Attendance/<date>.csv (daily attendance log, written by attendance.py)

Keeping this in one module means every other file reads/writes these
files the exact same way instead of re-implementing the format.
"""

from __future__ import annotations

import os
import csv
import json
from datetime import datetime

LABELS_FILE = "labels.json"
TRAINER_FILE = "trainer.yml"
ATTENDANCE_DIR = "Attendance"

CSV_HEADER = ["Roll Number", "Name", "Date", "Time"]


# --------------------------------------------------------------------------
# labels.json
# --------------------------------------------------------------------------

def save_labels(label_map: dict) -> None:
    """Persist the label_id -> 'roll_name' mapping produced during training."""
    with open(LABELS_FILE, "w") as file:
        json.dump(label_map, file)


def load_labels() -> dict:
    """Load the label map. Returns {} if the file doesn't exist."""
    if not os.path.exists(LABELS_FILE):
        return {}
    with open(LABELS_FILE, "r") as file:
        return json.load(file)


def labels_exist() -> bool:
    return os.path.exists(LABELS_FILE)


def trainer_exists() -> bool:
    return os.path.exists(TRAINER_FILE)


# --------------------------------------------------------------------------
# Attendance/<date>.csv
# --------------------------------------------------------------------------

def get_today_attendance_file() -> str:
    """Return today's attendance CSV path, creating the folder/header if needed."""
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)

    today = datetime.now().strftime("%d-%m-%Y")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"{today}.csv")

    if not os.path.exists(attendance_file):
        with open(attendance_file, "w", newline="") as file:
            csv.writer(file).writerow(CSV_HEADER)

    return attendance_file


def load_marked_rolls(attendance_file: str) -> set:
    """Return the set of roll numbers already marked present today."""
    marked = set()
    with open(attendance_file, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                marked.add(row[0])
    return marked


def mark_attendance(attendance_file: str, roll: str, name: str) -> None:
    """Append one attendance row for the given student."""
    today = datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.now().strftime("%H:%M:%S")

    with open(attendance_file, "a", newline="") as file:
        csv.writer(file).writerow([roll, name, today, current_time])
