"""
register.py

Student registration UI: collects Name + Roll Number, then captures face
images automatically from the webcam and saves them under
Images/<roll>_<name>/ for later use by train.py.
"""

from __future__ import annotations

import os
import cv2
import tkinter as tk
from tkinter import messagebox

from utils import get_face_detector, open_camera, student_folder_path

NUM_IMAGES_TO_CAPTURE = 100
CAPTURE_WINDOW_NAME = "Capturing Faces"

try:
    face_detector = get_face_detector()
    _detector_error = None
except RuntimeError as e:
    face_detector = None
    _detector_error = str(e)


def register_student() -> None:
    """Open the registration window and handle the full capture workflow."""
    if face_detector is None:
        messagebox.showerror("Error", _detector_error)
        return

    window = tk.Toplevel()
    window.title("Register Student")
    window.geometry("400x250")

    tk.Label(window, text="Student Name").pack(pady=5)
    name_entry = tk.Entry(window, width=30)
    name_entry.pack()

    tk.Label(window, text="Roll Number").pack(pady=5)
    roll_entry = tk.Entry(window, width=30)
    roll_entry.pack()

    def capture_images() -> None:
        name = name_entry.get().strip()
        roll = roll_entry.get().strip()

        if not name or not roll:
            messagebox.showerror("Error", "Please enter Name and Roll Number.")
            return

        student_folder = student_folder_path(roll, name)
        os.makedirs(student_folder, exist_ok=True)

        try:
            camera = open_camera()
        except RuntimeError as e:
            messagebox.showerror("Error", str(e))
            return

        count = 0

        while count < NUM_IMAGES_TO_CAPTURE:
            ret, frame = camera.read()
            if not ret:
                messagebox.showerror("Error", "Failed to read from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                count += 1

                face = gray[y:y + h, x:x + w]
                cv2.imwrite(os.path.join(student_folder, f"{count}.jpg"), face)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if count >= NUM_IMAGES_TO_CAPTURE:
                    break

            cv2.putText(
                frame, f"Images : {count}/{NUM_IMAGES_TO_CAPTURE}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
            )
            cv2.imshow(CAPTURE_WINDOW_NAME, frame)
            cv2.waitKey(1)  # required for OpenCV to render the frame

        # Capture is fully automatic -- closes itself once the target count
        # is reached, no key press required from the user.
        camera.release()
        cv2.destroyAllWindows()

        messagebox.showinfo(
            "Success",
            f"{count} images captured successfully.\nSaved in:\n{student_folder}"
        )
        window.destroy()

    tk.Button(
        window, text="Capture Images", command=capture_images, width=20, height=2
    ).pack(pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    register_student()
    root.mainloop()
