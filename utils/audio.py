"""
utils/audio.py
==============
Text-to-speech worker for announcing detected objects.
Runs in a dedicated thread to avoid blocking frame processing.
"""

import time
from queue import Queue

import pyttsx3


def speak_worker(q: Queue, rate: int = 235) -> None:
    """
    Background thread worker that reads (label, distance, position) tuples
    from a queue and announces them via text-to-speech.

    Distance is rounded to the nearest 0.5 m for natural speech output.
    After speaking, the queue is flushed to avoid stale announcements.

    Args:
        q:    Shared queue populated by the main detection loop.
        rate: TTS speech rate in words per minute.
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", 1.0)

    while True:
        if not q.empty():
            label, distance, position = q.get()

            # Round distance to nearest 0.5 m
            rounded = round(distance * 2) / 2
            dist_str = (
                str(int(rounded)) if rounded == int(rounded) else str(rounded)
            )

            engine.say(f"{label} is {dist_str} meters on {position}")
            engine.runAndWait()

            # Clear any queued messages that built up during speech
            with q.mutex:
                q.queue.clear()
        else:
            time.sleep(0.1)
