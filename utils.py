"""
utils.py

Shared, reusable helpers used across the project:
    - camera access
    - Haar Cascade face detector loading
    - student folder naming/parsing
    - face image preprocessing
    - one-time project folder setup

Centralizing these here keeps register.py, train.py, and attendance.py
free of duplicated logic and consistent in how they behave.
"""

from __future__ import annotations

import os
import cv2

IMAGES_DIR = "Images"
ATTENDANCE_DIR = "Attendance"
FACE_SIZE = (200, 200)


def ensure_project_folders() -> None:
    """Create Images/ and Attendance/ if they don't already exist."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)


def get_face_detector() -> cv2.CascadeClassifier:
    """
    Load OpenCV's bundled Haar Cascade face detector.

    Raises:
        RuntimeError: if the cascade file fails to load. CascadeClassifier
        does not raise on a bad path by itself -- it silently returns an
        empty, non-functional classifier -- so we check explicitly.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            "Failed to load the Haar Cascade face detector.\n"
            f"Expected file at:\n{cascade_path}\n\n"
            "Reinstall opencv-contrib-python to fix this:\n"
            "pip install --force-reinstall opencv-contrib-python"
        )

    return detector


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """
    Open the webcam at the given index.

    Raises:
        RuntimeError: if the camera cannot be accessed.
    """
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError(
            "Cannot access the webcam.\n"
            "Make sure it's connected and not being used by another app."
        )
    return camera


def student_folder_name(roll: str, name: str) -> str:
    """Build the folder name used to store a student's captured face images."""
    return f"{roll}_{name}"


def student_folder_path(roll: str, name: str) -> str:
    """Full path under Images/ for a given student, e.g. Images/101_Aditi."""
    return os.path.join(IMAGES_DIR, student_folder_name(roll, name))


def parse_student_folder(folder_name: str) -> tuple[str, str]:
    """
    Reverse of student_folder_name(): given 'roll_name', return (roll, name).

    Raises:
        ValueError: if the folder name doesn't contain an underscore
        separating roll number and name.
    """
    roll, name = folder_name.split("_", 1)
    return roll, name


def preprocess_face(gray_frame, x: int, y: int, w: int, h: int):
    """Crop a face region from a grayscale frame and resize it to FACE_SIZE."""
    face = gray_frame[y:y + h, x:x + w]
    return cv2.resize(face, FACE_SIZE)
