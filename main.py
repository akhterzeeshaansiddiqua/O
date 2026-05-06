"""
Audio World - Real-Time Object Detection with Spatial Audio Feedback
====================================================================
An assistive navigation system for the visually impaired using YOLOv8,
distance estimation, and text-to-speech feedback.

Usage:
    python main.py --source 0              # Webcam
    python main.py --source video.mp4      # Video file
    python main.py --source video.mp4 --no-blur   # Disable privacy blur
    python main.py --source 0 --max-distance 8.0  # Custom alert range
"""

import argparse
import time
from threading import Thread
from queue import Queue

import cv2

from utils.detection import load_model, detect_persons
from utils.audio import speak_worker
from utils.evaluation import evaluate_by_class, print_evaluation_report
from utils.visualization import draw_detections, setup_fullscreen_window


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio World: Real-Time Assistive Object Detection"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (0,1,...) or path to video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.4,
        help="NMS IoU threshold (default: 0.4)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=12.5,
        help="Maximum alert distance in meters (default: 12.5)",
    )
    parser.add_argument(
        "--no-blur",
        action="store_true",
        help="Disable privacy blur on detected persons",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Display in fullscreen mode",
    )
    parser.add_argument(
        "--speech-rate",
        type=int,
        default=235,
        help="Text-to-speech rate in words per minute (default: 235)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve video source (int for webcam, str for file)
    source = int(args.source) if args.source.isdigit() else args.source

    # Load YOLO model
    print(f"[INFO] Loading model: {args.model}")
    model = load_model(args.model)

    # Open video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {args.source}")
        return

    # Start TTS thread
    speech_queue = Queue()
    tts_thread = Thread(
        target=speak_worker,
        args=(speech_queue, args.speech_rate),
        daemon=True,
    )
    tts_thread.start()

    # Setup display window
    window_name = "Audio World"
    setup_fullscreen_window(window_name, fullscreen=args.fullscreen)

    # Performance tracking
    frame_count = 0
    start_time = time.time()
    all_preds = []
    all_gts = []

    print("[INFO] Starting detection. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run detection
        detections, nearest = detect_persons(
            frame,
            model,
            conf=args.conf,
            iou_thresh=args.iou,
            blur=not args.no_blur,
        )

        # Collect evaluation data
        for det in detections:
            all_preds.append((det["label"], det["box"]))
            all_gts.append((det["label"], det["box"]))  # simulated GT

        # Trigger audio alert for nearest person in range
        if nearest and nearest["distance"] <= args.max_distance and speech_queue.empty():
            speech_queue.put(
                (nearest["label"], nearest["distance"], nearest["position"])
            )

        # Draw bounding boxes and overlays
        frame = draw_detections(frame, detections)

        # Show FPS overlay
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
        )

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Final report
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    print(f"  Total Frames : {frame_count}")
    print(f"  Total Time   : {total_time:.2f} seconds")
    print(f"  Average FPS  : {frame_count / total_time:.2f}")
    print(f"{'='*50}")

    if all_preds:
        print_evaluation_report(evaluate_by_class(all_preds, all_gts))


if __name__ == "__main__":
    main()
