"""
attendance.py

Runs the webcam, recognizes a student's face using the trained LBPH
model, marks their attendance in today's CSV, briefly confirms it on
screen, then closes the camera automatically -- no key press required.
"""

from __future__ import annotations

import cv2
from tkinter import messagebox

from database import (
    trainer_exists,
    labels_exist,
    load_labels,
    get_today_attendance_file,
    load_marked_rolls,
    mark_attendance,
)
from utils import get_face_detector, open_camera, preprocess_face

WINDOW_NAME = "Face Recognition Attendance System"
CONFIDENCE_THRESHOLD = 60          # lower = better match, LBPH scale
CONFIRMATION_DISPLAY_MS = 1500     # how long the "Marked" banner stays up


def start_attendance() -> None:
    """Run one attendance session: recognize one student, mark, and exit."""
    if not trainer_exists():
        messagebox.showerror(
            "Error", "trainer.yml not found.\nPlease train the model first."
        )
        return

    if not labels_exist():
        messagebox.showerror(
            "Error", "labels.json not found.\nPlease train the model first."
        )
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    try:
        detector = get_face_detector()
    except RuntimeError as e:
        messagebox.showerror("Error", str(e))
        return

    labels = load_labels()
    attendance_file = get_today_attendance_file()
    marked_today = load_marked_rolls(attendance_file)

    try:
        camera = open_camera()
    except RuntimeError as e:
        messagebox.showerror("Error", str(e))
        return

    marked_person = None

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face = preprocess_face(gray, x, y, w, h)
            label, confidence = recognizer.predict(face)

            if confidence < CONFIDENCE_THRESHOLD:
                student = labels.get(str(label), "Unknown_Unknown")
                roll, name = student.split("_", 1)

                cv2.putText(
                    frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                cv2.putText(
                    frame, f"Confidence: {100 - confidence:.1f}%",
                    (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                )

                if roll not in marked_today:
                    mark_attendance(attendance_file, roll, name)
                    marked_today.add(roll)
                    marked_person = name
                    print(f"{name} Attendance Marked")
            else:
                cv2.putText(
                    frame, "Unknown", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

        cv2.imshow(WINDOW_NAME, frame)

        if marked_person is not None:
            confirm_frame = frame.copy()
            cv2.putText(
                confirm_frame, f"{marked_person} - Attendance Marked",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.imshow(WINDOW_NAME, confirm_frame)
            cv2.waitKey(CONFIRMATION_DISPLAY_MS)
            break  # session complete -- closes automatically

        # Esc is an emergency-only exit (e.g. nobody gets recognized).
        # Not required in normal use -- attendance still closes itself
        # automatically the moment a student is marked.
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
