"""
train.py

Trains an LBPH face recognizer on all images captured by register.py.
Saves the trained model to trainer.yml and the label map to labels.json
(via database.py) so attendance.py can use them later.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from tkinter import messagebox

from database import save_labels
from utils import IMAGES_DIR, parse_student_folder

TRAINER_FILE = "trainer.yml"


def train_model() -> None:
    """Train the LBPH recognizer on every registered student's images."""
    if not os.path.exists(IMAGES_DIR) or not os.listdir(IMAGES_DIR):
        messagebox.showerror(
            "Error",
            "No registered students found.\nPlease register a student first."
        )
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces: list = []
    labels: list = []
    label_map: dict = {}

    student_folders = sorted(os.listdir(IMAGES_DIR))

    for label, folder_name in enumerate(student_folders):
        folder_path = os.path.join(IMAGES_DIR, folder_name)

        if not os.path.isdir(folder_path):
            continue  # skip stray files like .DS_Store

        try:
            parse_student_folder(folder_name)  # validate "roll_name" format
        except ValueError:
            continue

        label_map[label] = folder_name

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue  # skip unreadable/non-image files

            image = cv2.resize(image, (200, 200))
            faces.append(image)
            labels.append(label)

    if not faces:
        messagebox.showerror(
            "Error",
            "No face images found.\nPlease register at least one student with captured images."
        )
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_FILE)
    save_labels(label_map)

    print("Model trained successfully!")
    messagebox.showinfo("Success", "Model trained successfully!")


if __name__ == "__main__":
    train_model()
