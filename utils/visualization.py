"""
utils/visualization.py
======================
Drawing utilities for bounding boxes, labels, and window management.
"""

import cv2
import numpy as np


# Color palette per position
POSITION_COLORS = {
    "LEFT":    (255, 100, 0),    # Blue-orange
    "FORWARD": (0, 220, 100),    # Green
    "RIGHT":   (0, 100, 255),    # Red-orange
}
DEFAULT_COLOR = (200, 200, 200)


def setup_fullscreen_window(name: str, fullscreen: bool = True) -> None:
    """
    Create and optionally fullscreen an OpenCV display window.

    Args:
        name:       Window title.
        fullscreen: If True, set window to fullscreen mode.
    """
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
    if fullscreen:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes and info labels for each detection on the frame.

    Box color encodes positional direction (LEFT/FORWARD/RIGHT).
    A semi-transparent fill highlights the nearest object.

    Args:
        frame:      The video frame.
        detections: List of detection dicts from detect_persons().

    Returns:
        Annotated frame.
    """
    if not detections:
        return frame

    # Find the nearest detection
    nearest_dist = min(d["distance"] for d in detections)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        distance = det["distance"]
        position = det["position"]
        is_nearest = distance == nearest_dist

        color = POSITION_COLORS.get(position, DEFAULT_COLOR)

        # Semi-transparent fill for nearest object
        if is_nearest:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Bounding box
        thickness = 3 if is_nearest else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label background
        label_text = f"{label} {distance:.1f}m | {position}"
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_y = max(y1 - 10, th + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - th - 4),
            (x1 + tw + 4, label_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label_text,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )

        # "NEAREST" badge
        if is_nearest:
            cv2.putText(
                frame,
                "NEAREST",
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    return frame
