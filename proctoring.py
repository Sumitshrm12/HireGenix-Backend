# proctoring.py
import cv2
import numpy as np
from deepface import DeepFace  # for emotion
from typing import Dict

# Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_faces(frame: bytes) -> int:
    """Return number of faces detected in the frame."""
    img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces)


def detect_face(frame: bytes) -> bool:
    """Return True if at least one face is detected."""
    return detect_faces(frame) > 0


def detect_motion(prev_frame: bytes, curr_frame: bytes, threshold: int = 50000) -> bool:
    """Return True if significant motion is detected between two frames."""
    prev = cv2.imdecode(np.frombuffer(prev_frame, np.uint8), cv2.IMREAD_GRAYSCALE)
    curr = cv2.imdecode(np.frombuffer(curr_frame, np.uint8), cv2.IMREAD_GRAYSCALE)
    # Compute absolute difference between frames
    diff = cv2.absdiff(prev, curr)
    # Threshold the difference
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    # Count non-zero pixels
    non_zero = np.count_nonzero(thresh)
    return non_zero > threshold


def analyze_emotion(frame: bytes) -> str:
    """Return the dominant emotion detected in the frame."""
    img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    analysis = DeepFace.analyze(
        img,
        actions=["emotion"],
        enforce_detection=False
    )

    if isinstance(analysis, list):
        analysis = analysis[0]

    return analysis.get("dominant_emotion", "unknown")


def detect_gaze(frame: bytes) -> str:
    """Return 'on_screen' if eyes are detected, else 'looking_away'."""
    img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        if len(eyes) >= 2:
            return "on_screen"
    return "looking_away"


def detect_mobile(frame: bytes) -> bool:
    """Placeholder: return False. Integrate an object detection model for actual mobile detection."""
    return False
