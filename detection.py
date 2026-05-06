"""
utils/detection.py
==================
Object detection, distance estimation, position classification,
and privacy blurring using YOLOv8.
"""

import numpy as np
import cv2
from ultralytics import YOLO

# Camera horizontal FOV in degrees (adjust for your camera)
CAMERA_FOV_DEG = 70

# Average known widths for object classes (used to refine distance estimates)
CLASS_AVG_SIZES = {
    "person": {"width_ratio": 2.5},
}


def load_model(weights: str = "yolov8n.pt") -> YOLO:
    """Load and return a YOLOv8 model."""
    return YOLO(weights)


def calculate_distance(
    box,
    frame_width: int,
    label: str,
    fov_deg: float = CAMERA_FOV_DEG,
) -> float:
    """
    Estimate distance (meters) to a detected object using focal length
    approximation from bounding box width.

    Args:
        box:         YOLO bounding box object.
        frame_width: Width of the video frame in pixels.
        label:       Detected class name (used for width ratio lookup).
        fov_deg:     Horizontal field of view of the camera in degrees.

    Returns:
        Estimated distance in meters (rounded to 2 decimal places).
    """
    object_width = box.xyxy[0, 2].item() - box.xyxy[0, 0].item()

    if label in CLASS_AVG_SIZES:
        object_width *= CLASS_AVG_SIZES[label]["width_ratio"]

    focal_length = frame_width / (2 * np.tan(np.radians(fov_deg / 2)))
    distance = focal_length / (object_width + 1e-6)
    return round(distance, 2)


def get_position(frame_width: int, box_x1: float) -> str:
    """
    Classify horizontal position of an object within the frame.

    Args:
        frame_width: Width of the frame in pixels.
        box_x1:     Left edge x-coordinate of the bounding box.

    Returns:
        One of "LEFT", "FORWARD", or "RIGHT".
    """
    if box_x1 < frame_width // 3:
        return "LEFT"
    elif box_x1 < 2 * (frame_width // 3):
        return "FORWARD"
    else:
        return "RIGHT"


def blur_person(image: np.ndarray, box) -> np.ndarray:
    """
    Apply Gaussian blur to the top portion (head region) of a detected person
    for privacy protection.

    Args:
        image: The video frame (modified in-place).
        box:   YOLO bounding box object.

    Returns:
        Frame with the head region blurred.
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    h = y2 - y1

    top = max(0, y1)
    bottom = min(image.shape[0], y1 + int(0.08 * h))
    left = max(0, x1)
    right = min(image.shape[1], x2)

    region = image[top:bottom, left:right]
    if region.size > 0:
        image[top:bottom, left:right] = cv2.GaussianBlur(region, (15, 15), 0)

    return image


def detect_persons(
    frame: np.ndarray,
    model: YOLO,
    conf: float = 0.25,
    iou_thresh: float = 0.4,
    blur: bool = True,
    target_class: str = "person",
) -> tuple[list[dict], dict | None]:
    """
    Run YOLOv8 inference on a frame, filter for the target class,
    estimate distances, determine positions, and optionally blur heads.

    Args:
        frame:        Video frame as a NumPy array.
        model:        Loaded YOLO model instance.
        conf:         Confidence threshold for detections.
        iou_thresh:   IoU threshold for NMS.
        blur:         Whether to blur the head region of detected persons.
        target_class: Object class to detect (default: "person").

    Returns:
        Tuple of:
          - detections: List of dicts with keys label, box, distance, position.
          - nearest:    Dict for the nearest detected object, or None.
    """
    results = model(frame, conf=conf, iou=iou_thresh)
    result = results[0]

    detections = []
    nearest = None
    min_distance = float("inf")

    for box in result.boxes:
        label = result.names[int(box.cls[0])]
        if label != target_class:
            continue

        coords = [round(x) for x in box.xyxy[0].tolist()]
        distance = calculate_distance(box, frame.shape[1], label)
        position = get_position(frame.shape[1], coords[0])

        det = {
            "label": label,
            "box": coords,
            "distance": distance,
            "position": position,
        }
        detections.append(det)

        if blur:
            frame = blur_person(frame, box)

        if distance < min_distance:
            min_distance = distance
            nearest = det

    return detections, nearest
