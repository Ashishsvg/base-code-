import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime
import sqlite3
from pathlib import Path
from typing import Tuple, List



def _dependency_check() -> None:
    # Install help:
    # 1) python -m venv .venv
    # 2) .venv\Scripts\activate
    # 3) python -m pip install --upgrade pip
    # 4) pip install -r requirements.txt
    try:
        import torch  # noqa: F401
        print("Torch loaded")
    except ImportError as exc:
        raise RuntimeError(
            "Torch not installed. Activate venv and run: "
            "pip install -r requirements.txt ; "
            "or CPU install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    try:
        import cv2  # noqa: F401
        import easyocr  # noqa: F401
        import ultralytics  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = str(exc).replace("No module named ", "").strip("'")
        raise RuntimeError(
            f"Missing dependency: {missing}. Install dependencies with: pip install -r requirements.txt"
        ) from exc


def save_to_db(track_id: int, plat_license: str, violence_category: str, vehicle: str, open_photo: str, speed: float, max_speed: float, location: str = "Bandung", plate_text: str = "UNKNOWN") -> None:
    def normalize_static_path(path: str) -> str:
        if not path:
            return ""
        path = str(path).replace("\\", "/")
        path = path.replace("database/static/", "")
        path = path.replace("static/", "")
        return path.lstrip("/")

    plat_license = normalize_static_path(plat_license)
    open_photo = normalize_static_path(open_photo)
    db_path = Path("database") / "datalog.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    try:
        try:
            cursor.execute(
                """
                INSERT INTO datalog (
                    track_id, date, plat_license, plate_text, violence_category, vehicle, open_photo, speed, max_speed, location
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    track_id,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    plat_license,
                    plate_text,
                    violence_category,
                    vehicle,
                    open_photo,
                    speed,
                    max_speed,
                    location,
                ),
            )
        except sqlite3.OperationalError:
            cursor.execute(
                """
                INSERT INTO datalog (
                    track_id, date, plat_license, violence_category, vehicle, open_photo, speed, max_speed, location
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    track_id,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    plat_license,
                    violence_category,
                    vehicle,
                    open_photo,
                    speed,
                    max_speed,
                    location,
                ),
            )
        conn.commit()
        print(f"Inserted into DB: track_id={track_id}, plate={plat_license}, violation={violence_category}")
        print("DB plate path:", plat_license)
        print("DB violation path:", open_photo)
    except Exception as exc:
        print(f"DB insert failed: {exc}")
        raise
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    # Install help:
    # 1) python -m venv .venv
    # 2) .venv\Scripts\activate
    # 3) python -m pip install --upgrade pip
    # 4) pip install -r requirements.txt
    try:
        import torch  # noqa: F401
        print("Torch loaded")
    except ImportError as exc:
        raise RuntimeError(
            "Torch not installed. Activate venv and run: "
            "pip install -r requirements.txt ; "
            "or CPU install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    try:
        import cv2  # noqa: F401
        import easyocr  # noqa: F401
        import ultralytics  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = str(exc).replace("No module named ", "").strip("'")
        raise RuntimeError(
            f"Missing dependency: {missing}. Install dependencies with: pip install -r requirements.txt"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 traffic violation detection (CPU)")
    parser.add_argument("--video", required=True, help="Input video path, e.g. 5.mp4")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--show", action="store_true", help="Show preview window (legacy flag)")
    parser.add_argument("--no-show", action="store_true", help="Run without preview window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _dependency_check()
    print(f"Python interpreter: {sys.executable}")

    import cv2

    from config import AppConfig
    from detector import YOLODetector
    from utils import draw_bbox
    from violation_handler import ViolationHandler

    config = AppConfig(model_path=args.model)
    config.ensure_output_dirs()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    print("Video started")

    try:
        detector = YOLODetector(config)
        violation_handler = ViolationHandler(config)
    except Exception as exc:
        cap.release()
        raise RuntimeError(f"Pipeline startup failed: {exc}") from exc

    show_window = not args.no_show

    if show_window:
        try:
            cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            cap.release()
            raise RuntimeError(
                "OpenCV GUI is not available. Ensure 'opencv-python' is installed (not headless)."
            ) from exc

    def _scale_bbox_to_original(bbox, scale_x: float, scale_y: float):
        x1, y1, x2, y2 = bbox
        return (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        )

    frame_index = 0
    fps_window = deque(maxlen=30)
    processed = 0
    violation_total = 0
    last_violations = []
    last_detection_count = 0
    frame_count = 0
    vehicle_counts = {"car": 0, "motorbike": 0, "truck": 0}

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_count += 1
        if frame_count % max(1, int(config.frame_process_interval)) != 0:
            continue
        verbose_log = getattr(config, "verbose_pipeline_logs", False)
        if verbose_log:
            print("Frame received")
        original_frame = frame.copy()
        detect_frame = cv2.resize(frame, (960, 540))
        display_frame = original_frame.copy()
        if verbose_log:
            print("Original:", original_frame.shape)
            print("Detection:", detect_frame.shape)

        orig_h, orig_w = original_frame.shape[:2]
        det_h, det_w = detect_frame.shape[:2]
        to_orig_x = orig_w / float(det_w)
        to_orig_y = orig_h / float(det_h)
        start = time.perf_counter()
        try:
            secondary_every = max(1, int(getattr(config, "secondary_model_interval", 2)))
            run_secondary_models = (processed % secondary_every) == 0
            if verbose_log:
                print("Detection running")
            detections = detector.detect(detect_frame, run_secondary_models=run_secondary_models)
            if not detections and verbose_log:
                print("No detections in this frame.")
            detections_for_processing: List = []
            for det in detections:
                scaled = type(det)(
                    bbox=_scale_bbox_to_original(det.bbox, to_orig_x, to_orig_y),
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    track_id=det.track_id,
                )
                detections_for_processing.append(scaled)

            violations, db_inserts = violation_handler.process_frame(original_frame, detections_for_processing, frame_count=frame_count)
            last_violations = violations
            last_detection_count = len(detections)
            processed += 1
            violation_total += len(violations)
            vehicle_counts = {"car": 0, "motorbike": 0, "truck": 0}

            # Save to DB
            for data in db_inserts:
                save_to_db(**data)
                print("Inserted into DB")

            for det in detections:
                tid = det.track_id if det.track_id is not None else "-"
                label = f"ID {tid} | {det.class_name} {det.confidence:.2f}"
                dname = str(det.class_name).lower().replace("-", "_").replace(" ", "_")
                if dname == "car":
                    vehicle_counts["car"] += 1
                elif dname in {"motorcycle", "bike", "bicycle"}:
                    vehicle_counts["motorbike"] += 1
                elif dname == "truck":
                    vehicle_counts["truck"] += 1
                if det.track_id is not None:
                    speed_kmh = violation_handler.track_speed_kmh.get(det.track_id)
                    if speed_kmh is not None and speed_kmh > 0:
                        label = f"{label} | {speed_kmh:.1f} km/h"
                    elif violation_handler.is_track_fast(det.track_id):
                        label = f"{label} | Speed: FAST"
                scaled_bbox = _scale_bbox_to_original(det.bbox, to_orig_x, to_orig_y)
                draw_bbox(display_frame, scaled_bbox, label, color=(0, 255, 0))

            for v in violations:
                draw_bbox(display_frame, v["bbox"], v["label"], color=(0, 0, 255))
                plate_bbox = v.get("plate_bbox")
                if plate_bbox:
                    px1, py1, px2, py2 = plate_bbox
                    cv2.rectangle(display_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
            if verbose_log:
                print("Frame processed")
        except Exception as err:
            print(f"Error: {err}")
            continue

        overlay = violation_handler.get_debug_overlay(original_frame.shape)
        line_y1 = int(overlay["line_y1"])
        line_y2 = int(overlay["line_y2"])
        stop_line_y = int(overlay["stop_line_y"])
        if overlay.get("show_debug_lines"):
            cv2.line(display_frame, (0, line_y1), (display_frame.shape[1], line_y1), (0, 255, 0), 1)
            cv2.line(display_frame, (0, line_y2), (display_frame.shape[1], line_y2), (0, 255, 255), 1)
            cv2.putText(
                display_frame,
                f"Speed Limit: {int(config.speed_limit_kmph)} km/h",
                (10, max(20, line_y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            if overlay.get("red_light_enabled"):
                cv2.line(display_frame, (0, stop_line_y), (display_frame.shape[1], stop_line_y), (0, 0, 255), 2)
                cv2.putText(display_frame, "Red Light Mode: ON", (10, max(20, stop_line_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        elapsed = max(1e-6, time.perf_counter() - start)
        fps_window.append(1.0 / elapsed)
        fps = sum(fps_window) / len(fps_window)

        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(
            display_frame,
            f"Detections: {last_detection_count} | Violations: {violation_total}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            "ESC: Exit | YOLOv8n CPU mode",
            (10, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            f"Cars: {vehicle_counts['car']} | Bikes: {vehicle_counts['motorbike']} | Trucks: {vehicle_counts['truck']}",
            (10, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 255, 180),
            2,
        )

        if last_violations:
            if any(v["type"] == "no_helmet" for v in last_violations):
                cv2.putText(
                    display_frame,
                    "NO HELMET VIOLATION",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3,
                )
            if any(v["type"] == "triple_riding" for v in last_violations):
                cv2.putText(
                    display_frame,
                    "TRIPLE RIDING VIOLATION",
                    (10, 186),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            if any(v["type"] == "overspeed" for v in last_violations):
                cv2.putText(
                    display_frame,
                    "OVER SPEED VIOLATION",
                    (10, 214),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            if any(v["type"] == "red_light_violation" for v in last_violations):
                cv2.putText(
                    display_frame,
                    "RED LIGHT VIOLATION",
                    (10, 242),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

        if violation_handler.should_show_recorded():
            cv2.putText(
                display_frame,
                "VIOLATION RECORDED",
                (10, 158),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            if last_violations:
                cv2.putText(
                    display_frame,
                    f"Last: {last_violations[-1]['type']}",
                    (10, 104),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        if show_window:
            cv2.imshow(config.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished. Processed frames: {processed}, total violations: {violation_total}")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"[ERROR] {err}")
