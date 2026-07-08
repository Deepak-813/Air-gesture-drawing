"""
Hand tracking module using MediaPipe Tasks (v0.10+).
"""

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


@dataclass
class HandState:
    index_tip: Tuple[int, int]
    raised_finger_count: int
    is_drawing_gesture: bool
    is_pause_gesture: bool


class HandTracker:
    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
    ) -> None:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr: Any) -> Optional[HandState]:
        self._timestamp_ms += 33  # ~30 fps

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]
        height, width = frame_bgr.shape[:2]

        index_tip = (int(landmarks[8].x * width), int(landmarks[8].y * height))
        fingers = self._finger_states(landmarks)
        raised_count = sum(fingers.values())

        index_up = fingers["index"]
        middle_up = fingers["middle"]
        others_down = not fingers["ring"] and not fingers["pinky"]

        return HandState(
            index_tip=index_tip,
            raised_finger_count=raised_count,
            is_drawing_gesture=index_up and not middle_up and others_down,
            is_pause_gesture=index_up and middle_up and others_down,
        )

    @staticmethod
    def _finger_states(landmarks: Any) -> dict:
        def is_up(tip: int, pip: int) -> bool:
            return landmarks[tip].y < landmarks[pip].y

        return {
            "index": is_up(8, 6),
            "middle": is_up(12, 10),
            "ring": is_up(16, 14),
            "pinky": is_up(20, 18),
        }

    def close(self) -> None:
        self._landmarker.close()
