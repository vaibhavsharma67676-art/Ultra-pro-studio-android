#!/usr/bin/env python3
"""Standalone watermark remover for videos using FFmpeg delogo.

Supports manual coordinates or automatic guess from frame variance + edge map.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def get_auto_box(video_path: str, sample_count: int = 25) -> Optional[Tuple[int, int, int, int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // sample_count, 1)
    frames = []

    for i in range(sample_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(frames) < 5:
        return None

    stacked = np.array(frames, dtype=np.float32)
    variance_map = stacked.std(axis=0)

    norm = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inverted = 255 - norm

    mid_frame = frames[len(frames) // 2]
    edges = cv2.Canny(mid_frame, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    candidate = cv2.bitwise_and(inverted, edges)
    _, mask = cv2.threshold(candidate, 80, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    h_frame, w_frame = mask.shape
    coverage = (w * h) / (w_frame * h_frame)
    if coverage < 0.001 or coverage > 0.25:
        return None

    return x, y, w, h


def process_video(input_path: str, output_path: str, box: Optional[Tuple[int, int, int, int]], auto: bool) -> None:
    if auto:
        box = get_auto_box(input_path)
        if not box:
            raise RuntimeError("Auto-detect failed. Provide manual x,y,w,h.")

    if not box:
        raise RuntimeError("Missing box coordinates.")

    x, y, w, h = box
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        f"delogo=x={x}:y={y}:w={w}:h={h}:show=0",
        "-c:a",
        "copy",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove video watermark")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--box", help="x,y,w,h")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    parsed_box = None
    if args.box:
        try:
            parsed_box = tuple(int(v.strip()) for v in args.box.split(","))
            if len(parsed_box) != 4:
                raise ValueError
        except ValueError:
            print("Invalid --box. Expected x,y,w,h", file=sys.stderr)
            sys.exit(2)

    try:
        process_video(args.input, args.output, parsed_box, args.auto)
        print(f"DONE: {Path(args.output).resolve()}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
