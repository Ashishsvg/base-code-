import csv
import json
import logging
import math
import re
import os
import queue
import threading
import uuid
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

from config import AppConfig
from db_writer import insert_violation
from utils import Detection, clip_bbox_to_frame, iou


logger = logging.getLogger(__name__)


class ViolationHandler:
    """Handles violation logic, image saving, OCR, CSV logging, and backend post."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.config.ensure_output_dirs()
        self.reader = easyocr.Reader(list(config.easyocr_languages), gpu=False)
        self.last_saved_at: Dict[str, float] = {}
        self.track_memory: Dict[int, dict] = {}
        self.tracked_objects: Dict[int, float] = {}
        self.track_plate_text: Dict[int, str] = {}
        self.track_last_violation: Dict[int, float] = {}
        self.last_violation_time: Dict[str, float] = {}
        self.track_last_centroid: Dict[int, tuple[int, int]] = {}
        self.track_last_seen_at: Dict[int, float] = {}
        self.track_speed_kmh: Dict[int, float] = {}
        self.track_line_cross_time: Dict[int, dict] = {}
        self.track_crossed_red_light: Dict[int, bool] = {}
        self.track_wrong_side_frames: Dict[int, int] = {}
        self.track_wrong_side_flagged: Dict[int, bool] = {}
        self.track_wrong_side_reported: Dict[int, bool] = {}
        self.vehicle_history: Dict[int, List[tuple[int, int]]] = {}
        self.triple_riding_cache: Dict[int, float] = {}
        self.recent_violation_until: Dict[int, float] = {}
        self.last_saved_time: float = 0.0
        self.last_saved_time_by_key: Dict[str, float] = {}
        self.violation_cooldown: Dict[str, int] = {}
        self.speed_logged: set[int] = set()
        self.detection_count: Dict[int, int] = {}
        self.best_capture: Dict[int, dict] = {}
        self.triple_riding_frames: Dict[int, int] = {}
        self.red_light_frames: Dict[int, int] = {}
        self.vehicle_motion: Dict[int, dict] = {}
        self.track_fallback_fast: Dict[int, bool] = {}
        self.plate_model = None
        self.next_track_id = 1
        self.jobs: "queue.Queue[dict]" = queue.Queue(maxsize=128)
        self.lock = threading.Lock()
        self._image_io_lock = threading.Lock()
        self._last_plate_yolo_time: Dict[int, float] = {}
        self._ocr_plate_cache: Dict[str, str] = {}
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._init_plate_model()
        self._init_log_file()

    def _init_plate_model(self) -> None:
        try:
            model_path = self.config.plate_model_path
            if model_path and os.path.exists(model_path):
                self.plate_model = YOLO(model_path)
                print(f"✅ Plate model loaded: {model_path}")
            else:
                print("❌ Plate model missing at models/plate_model.pt")
        except Exception as exc:
            print(f"❌ Plate model missing at models/plate_model.pt")
            print(f"Plate model init failed: {exc}")
            self.plate_model = None

    def _init_log_file(self) -> None:
        log_path = Path(self.config.csv_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_path.exists():
            with log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "violation_type", "plate_text", "plate_image_path", "full_image_path"])

    def get_violations(self, detections: List[Detection]) -> List[dict]:
        """
        Current logic:
        - no_helmet: rider without helmet (person + motorcycle and no helmet on head)
        - no_seatbelt: placeholder support for future classes
        """
        violations: List[dict] = []
        normalized = [(self._normalize_name(d.class_name), d) for d in detections]

        persons = [d for name, d in normalized if name == "person"]
        bikes = [d for name, d in normalized if name in {"motorcycle", "bicycle", "bike"}]
        helmets = [d for name, d in normalized if name == "helmet"]
        explicit_no_helmet = [d for name, d in normalized if name == "no_helmet"]

        if self.config.enable_helmet_detection:
            # Primary logic: person riding bike and no helmet around upper body/head.
            for person in persons:
                matched_bike = None
                for bike in bikes:
                    if iou(person.bbox, bike.bbox) >= self.config.rider_bike_iou_threshold:
                        matched_bike = bike
                        break
                if matched_bike is None:
                    continue

                if not self._helmet_on_person(person, helmets):
                    combined_box = self._merge_bboxes(person.bbox, matched_bike.bbox)
                    synthetic = Detection(
                        bbox=combined_box,
                        class_id=-1,
                        class_name="rider_no_helmet",
                        confidence=min(person.confidence, matched_bike.confidence),
                        track_id=person.track_id if person.track_id is not None else matched_bike.track_id,
                    )
                    violations.append({"type": "no_helmet", "detection": synthetic})

            # Backward-compatible path if model directly emits no_helmet class.
            for det in explicit_no_helmet:
                violations.append({"type": "no_helmet", "detection": det})

        # Placeholder for custom violation classes from models.
        for name, det in normalized:
            if name in {"no_seatbelt", "seatbelt_violation"}:
                violations.append({"type": "no_seatbelt", "detection": det})
            elif name == "overspeed":
                violations.append({"type": "overspeed", "detection": det})
            elif name in {"wrong_direction", "wrongdirection", "wrong_direction_violation"}:
                violations.append({"type": "wrong_direction", "detection": det})

        # Triple riding: 3+ people associated with one motorcycle for a sustained window.
        persons = [d for name, d in normalized if name == "person"]
        bikes = [d for name, d in normalized if name in {"motorcycle", "bike"}]
        active_bike_ids = set()
        now = time.time()
        for bike in bikes:
            if bike.track_id is None:
                continue
            bike_id = bike.track_id
            active_bike_ids.add(bike_id)
            if now - self.triple_riding_cache.get(bike_id, 0.0) < self.config.violation_cooldown_sec:
                continue

            associated_persons = [person for person in persons if self._is_person_on_bike(person.bbox, bike.bbox)]
            person_count = len(associated_persons)
            if person_count >= self.config.triple_riding_min_riders:
                self.triple_riding_frames[bike_id] = self.triple_riding_frames.get(bike_id, 0) + 1
            else:
                self.triple_riding_frames[bike_id] = 0

            if self.triple_riding_frames.get(bike_id, 0) >= self.config.triple_riding_required_frames:
                combined_bbox = bike.bbox
                for person in associated_persons:
                    combined_bbox = self._merge_bboxes(combined_bbox, person.bbox)
                synthetic = Detection(
                    bbox=combined_bbox,
                    class_id=-1,
                    class_name="triple_riding",
                    confidence=min([bike.confidence] + [p.confidence for p in associated_persons]) if associated_persons else bike.confidence,
                    track_id=bike_id,
                )
                violations.append({"type": "triple_riding", "detection": synthetic})
        stale_bikes = [bid for bid in self.triple_riding_frames.keys() if bid not in active_bike_ids]
        for bid in stale_bikes:
            self.triple_riding_frames.pop(bid, None)
            self.triple_riding_cache.pop(bid, None)

        # Line-based overspeed and optional red-light violations populated from tracking state.
        for _, det in normalized:
            if self._normalize_name(det.class_name) not in {"car", "motorcycle", "bicycle", "bike", "bus", "truck"}:
                continue
            if det.track_id is None:
                continue
            speed = self.track_speed_kmh.get(det.track_id)
            track_hits = self.detection_count.get(det.track_id, 0)
            track_cross = self.track_line_cross_time.get(det.track_id, {})
            has_two_line_timestamps = track_cross.get("start") is not None and track_cross.get("end") is not None
            if (
                bool(getattr(self.config, "enable_speed_detection", True))
                and speed is not None
                and has_two_line_timestamps
                and 5.0 < float(speed) < 180.0
                and track_hits >= 2
                and float(speed) >= float(self.config.speed_limit_kmph)
                and det.track_id not in self.speed_logged
            ):
                violations.append({"type": "overspeed", "detection": det})
            elif (
                bool(getattr(self.config, "enable_speed_fallback", True))
                and self.track_fallback_fast.get(det.track_id, False)
                and det.track_id not in self.speed_logged
            ):
                violations.append({"type": "overspeed", "detection": det})
            if (
                self.config.enable_wrong_side_detection
                and self.track_wrong_side_flagged.get(det.track_id, False)
                and not self.track_wrong_side_reported.get(det.track_id, False)
            ):
                violations.append({"type": "wrong_direction", "detection": det})
            if self.config.enable_red_light_detection:
                if self.track_crossed_red_light.get(det.track_id, False):
                    self.red_light_frames[det.track_id] = self.red_light_frames.get(det.track_id, 0) + 1
                else:
                    self.red_light_frames[det.track_id] = 0
                if self.red_light_frames.get(det.track_id, 0) >= 2:
                    violations.append({"type": "red_light_violation", "detection": det})

        return violations

    def _normalize_name(self, name: str) -> str:
        return name.lower().replace("-", "_").replace(" ", "_")

    def _normalize_static_path(self, path: str) -> str:
        if not path:
            return ""
        path = str(path).replace("\\", "/")
        path = path.replace("database/static/", "")
        path = path.replace("static/", "")
        return path.lstrip("/")

    def _log_verbose(self, msg: str) -> None:
        if bool(getattr(self.config, "verbose_pipeline_logs", False)):
            print(msg)

    @staticmethod
    def _frame_ok(frame: Optional[np.ndarray]) -> bool:
        try:
            if frame is None or not isinstance(frame, np.ndarray):
                return False
            if frame.size == 0 or len(frame.shape) < 2:
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def _verify_saved_image(path: str, min_bytes: int = 48) -> bool:
        try:
            if not path or not os.path.isfile(path):
                return False
            return os.path.getsize(path) >= min_bytes
        except OSError:
            return False

    def _cache_ocr_lookup(self, static_relative_path: str) -> Optional[str]:
        key = self._normalize_static_path(static_relative_path)
        val = self._ocr_plate_cache.get(key)
        return val

    def _cache_ocr_store(self, static_relative_path: str, text: str) -> None:
        key = self._normalize_static_path(static_relative_path)
        if len(self._ocr_plate_cache) > 200:
            self._ocr_plate_cache.clear()
        self._ocr_plate_cache[key] = text

    def _helmet_on_person(self, person: Detection, helmets: List[Detection]) -> bool:
        px1, py1, px2, py2 = person.bbox
        person_h = max(1, py2 - py1)
        person_w = max(1, px2 - px1)
        expand_x = int(person_w * 0.08)
        head_box = (
            px1 - expand_x,
            py1,
            px2 + expand_x,
            py1 + int(0.45 * person_h),
        )
        for helmet in helmets:
            if iou(head_box, helmet.bbox) >= self.config.helmet_head_iou_threshold:
                return True
        return False

    def _merge_bboxes(self, a, b):
        return (
            min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[2], b[2]),
            max(a[3], b[3]),
        )

    def _is_person_on_bike(self, person_bbox, bike_bbox) -> bool:
        px1, py1, px2, py2 = person_bbox
        bx1, by1, bx2, by2 = bike_bbox
        if px2 <= px1 or py2 <= py1 or bx2 <= bx1 or by2 <= by1:
            return False

        pcx = int((px1 + px2) / 2)
        pcy = int((py1 + py2) / 2)
        bottom_cx = int((px1 + px2) / 2)
        bottom_cy = py2

        if bx1 <= bottom_cx <= bx2 and by1 <= bottom_cy <= by2:
            return True
        if bx1 <= pcx <= bx2 and by1 <= pcy <= by2:
            return True

        return iou(person_bbox, bike_bbox) >= self.config.rider_bike_iou_threshold

    def process_frame(self, frame: np.ndarray, detections: List[Detection], frame_count: int = 0) -> tuple[list[dict], list[dict]]:
        if len(detections) == 0:
            self._cleanup_stale_tracks()
            return [], []
        self._assign_track_ids(detections)
        self._update_line_speed_and_signals(detections, frame.shape, frame_count=frame_count)
        violations = self.get_violations(detections)
        frame_events: List[dict] = []
        db_inserts: List[dict] = []

        for item in violations:
            violation_type = item["type"]
            det = item["detection"]
            track_id = det.track_id if det.track_id is not None else -1
            event_key = f"{violation_type}:{track_id}"
            cooldown_key = f"{violation_type}_{track_id}"
            if violation_type == "no_helmet":
                self._log_verbose("Helmet violation confirmed")
            if violation_type == "overspeed" and track_id in self.speed_logged:
                continue

            if track_id >= 0:
                x1, y1, x2, y2 = det.bbox
                area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                best = self.best_capture.get(track_id)
                if best is None or area > best["area"]:
                    self.best_capture[track_id] = {
                        "area": area,
                        "frame": frame.copy(),
                        "bbox": det.bbox,
                    }

            if not self._allowed_by_cooldown(event_key):
                self._log_verbose(f"Duplicate ignored for ID: {track_id} / type: {violation_type}")
                continue

            if track_id >= 0 and not self._allowed_by_track(track_id):
                self._log_verbose(f"Duplicate ignored for same vehicle ID: {track_id}")
                continue

            if track_id >= 0:
                last_frame = self.violation_cooldown.get(cooldown_key, -9999)
                if frame_count - last_frame < int(self.config.violation_cooldown_frames):
                    self._log_verbose(f"Duplicate ignored by frame cooldown: {cooldown_key}")
                    continue

            # Track + violation duplicate window (5 sec).
            key = f"{track_id}:{violation_type}"
            now = time.time()
            if now - self.last_saved_time_by_key.get(key, 0.0) < 5.0:
                self._log_verbose(f"Duplicate ignored in 5s window for key: {key}")
                continue

            # New cooldown: per track_id + violation_type
            if now - self.last_saved_time < 3.0:
                self._log_verbose("Duplicate save skipped (global cooldown)")
                continue

            save_frame = frame
            save_bbox = det.bbox
            if track_id >= 0 and track_id in self.best_capture:
                save_frame = self.best_capture[track_id]["frame"]
                save_bbox = self.best_capture[track_id]["bbox"]

            full_image_path = self._save_violation_image(save_frame, save_bbox, violation_type)
            plate_image_path, plate_draw_bbox = self._save_plate_image(save_frame, save_bbox, track_id)
            if not full_image_path:
                continue
            plate_text = self._extract_plate_from_image(plate_image_path)
            if track_id >= 0:
                with self.lock:
                    self.track_plate_text[track_id] = plate_text

            # Prepare DB data
            db_data = self._prepare_db_data(violation_type, plate_image_path, full_image_path, track_id, plate_text)
            db_inserts.append(db_data)

            self._log_verbose(f"New violation saved for ID: {track_id} ({violation_type})")

            frame_events.append(
                {
                    "type": violation_type,
                    "bbox": det.bbox,
                    "track_id": track_id,
                    "label": f"ID {track_id} | {violation_type} | PROCESSING",
                    "image_path": full_image_path,
                    "plate_image_path": plate_image_path,
                    "plate_bbox": plate_draw_bbox,
                }
            )
            self.last_saved_time = now
            self.last_saved_time_by_key[key] = now
            if track_id >= 0:
                self.violation_cooldown[cooldown_key] = frame_count
            self.last_saved_at[event_key] = now
            if track_id >= 0:
                if violation_type == "triple_riding":
                    self.triple_riding_cache[track_id] = now
                self.track_last_violation[track_id] = now
                if violation_type == "overspeed":
                    self.speed_logged.add(track_id)
                    motion = self.vehicle_motion.get(track_id)
                    if isinstance(motion, dict):
                        motion["violation_logged"] = True
                    self.track_fallback_fast[track_id] = False
                if violation_type == "wrong_direction":
                    self.track_wrong_side_reported[track_id] = True
            self.recent_violation_until[track_id] = now + 1.5

        self._cleanup_stale_tracks()
        return frame_events, db_inserts

    def _allowed_by_cooldown(self, event_key: str) -> bool:
        now = time.time()
        last = self.last_saved_at.get(event_key, 0.0)
        return (now - last) >= self.config.violation_cooldown_sec

    def _allowed_by_track(self, track_id: int) -> bool:
        now = time.time()
        last = self.track_last_violation.get(track_id, 0.0)
        return (now - last) >= self.config.violation_cooldown_sec

    def _assign_track_ids(self, detections: List[Detection]) -> None:
        now = time.time()
        for det in detections:
            if det.track_id is not None:
                self.track_memory[det.track_id] = {
                    "bbox": det.bbox,
                    "last_seen": now,
                    "class_name": det.class_name,
                }
                continue

            best_id = None
            best_iou = 0.0

            for track_id, meta in self.track_memory.items():
                if meta["class_name"] != det.class_name:
                    continue
                overlap = iou(meta["bbox"], det.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_id = track_id

            if best_id is not None and best_iou >= self.config.tracker_match_iou_threshold:
                det.track_id = best_id
                self.track_memory[best_id]["bbox"] = det.bbox
                self.track_memory[best_id]["last_seen"] = now
            else:
                det.track_id = self.next_track_id
                self.track_memory[self.next_track_id] = {
                    "bbox": det.bbox,
                    "last_seen": now,
                    "class_name": det.class_name,
                }
                self.next_track_id += 1

    def _cleanup_stale_tracks(self) -> None:
        now = time.time()
        stale_ids = [
            track_id
            for track_id, meta in self.track_memory.items()
            if (now - meta["last_seen"]) > self.config.tracker_stale_seconds
        ]
        for track_id in stale_ids:
            self.track_memory.pop(track_id, None)
            self.track_plate_text.pop(track_id, None)
            self.recent_violation_until.pop(track_id, None)
            self.track_last_centroid.pop(track_id, None)
            self.track_last_seen_at.pop(track_id, None)
            self.track_speed_kmh.pop(track_id, None)
            self.track_line_cross_time.pop(track_id, None)
            self.track_crossed_red_light.pop(track_id, None)
            self.track_wrong_side_frames.pop(track_id, None)
            self.track_wrong_side_flagged.pop(track_id, None)
            self.track_wrong_side_reported.pop(track_id, None)
            self.vehicle_history.pop(track_id, None)
            self.red_light_frames.pop(track_id, None)
            self.last_saved_time_by_key.pop(f"{track_id}:overspeed", None)
            self.last_saved_time_by_key.pop(f"{track_id}:red_light_violation", None)
            self.last_saved_time_by_key.pop(f"{track_id}:triple_riding", None)
            self.triple_riding_cache.pop(track_id, None)
            self.detection_count.pop(track_id, None)
            self.best_capture.pop(track_id, None)
            self.speed_logged.discard(track_id)
            self.vehicle_motion.pop(track_id, None)
            self.track_fallback_fast.pop(track_id, None)
            self._last_plate_yolo_time.pop(track_id, None)

    def _save_violation_image(self, frame: np.ndarray, bbox, violation_type: str) -> str:
        if not self._frame_ok(frame):
            logger.warning("Violation image not saved (%s): invalid or empty frame", violation_type)
            return ""
        rel_out = ""
        try:
            x1, y1, x2, y2 = self._expand_bbox(bbox, frame.shape, padding=0.05)
            if x2 <= x1 or y2 <= y1:
                logger.warning("Violation image not saved (%s): invalid bbox after expand", violation_type)
                return ""

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            uid = uuid.uuid4().hex[:8]
            safe_vt = re.sub(r"[^a-z0-9_]+", "_", str(violation_type).lower())[:40]
            filename = f"violation_{safe_vt}_{timestamp}_{uid}.jpg"
            violation_dir = "database/static/violations"
            os.makedirs(violation_dir, exist_ok=True)
            full_image_path = os.path.join(violation_dir, filename)
            rel_out = f"violations/{filename}"

            marked = np.ascontiguousarray(frame.copy())
            try:
                cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except cv2.error as rect_err:
                logger.warning("cv2.rectangle failed for violation %s: %s", violation_type, rect_err)

            jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            ok = False
            with self._image_io_lock:
                ok = bool(cv2.imwrite(full_image_path, marked, jpeg_params))

            if not ok or not self._verify_saved_image(full_image_path):
                logger.error("Violation JPEG write invalid or missing: %s", full_image_path)
                fallback_name = f"violation_fallback_{safe_vt}_{timestamp}_{uid}.jpg"
                fallback_path = os.path.join(violation_dir, fallback_name)
                with self._image_io_lock:
                    ok_fb = bool(cv2.imwrite(fallback_path, np.ascontiguousarray(frame.copy()), jpeg_params))
                if ok_fb and self._verify_saved_image(fallback_path):
                    logger.info("Saved violation fallback full-frame image: %s", fallback_path)
                    return f"violations/{fallback_name}"
                return ""

            self._log_verbose(f"Saved violation: {full_image_path}")
            return rel_out
        except Exception:
            logger.exception("Violation image save failed (%s)", violation_type)
            return ""

    def _save_plate_image(
        self, frame: np.ndarray, bbox, track_id: int = -1
    ) -> tuple[str, Optional[tuple[int, int, int, int]]]:
        if not self._frame_ok(frame):
            logger.warning("Plate image not saved (track=%s): invalid frame", track_id)
            return "Not Detected", None
        try:
            x1, y1, x2, y2 = self._expand_bbox(bbox, frame.shape, padding=0.05)
            if x2 <= x1 or y2 <= y1:
                return "Not Detected", None

            plate_region: Optional[np.ndarray] = None
            plate_bbox: Optional[tuple[int, int, int, int]] = None
            plate_source = "none"
            now_t = time.time()
            gap = float(getattr(self.config, "plate_yolo_min_interval_sec", 1.5))
            throttle = track_id >= 0 and (now_t - self._last_plate_yolo_time.get(track_id, 0.0)) < gap
            if throttle:
                fb = self._fallback_plate_crop(frame, x1, y1, x2, y2)
                if fb[0] is not None and fb[0].size > 0:
                    plate_region, plate_bbox = fb[0], fb[1]
                    plate_source = "fallback_throttled"
            if plate_region is None:
                plate_region, plate_source, plate_bbox = self._detect_plate_region(frame, x1, y1, x2, y2)
                if plate_source == "model" and track_id >= 0:
                    self._last_plate_yolo_time[track_id] = time.time()

            if plate_region is None or plate_region.size == 0:
                return "Not Detected", None

            pad = 10
            if plate_bbox is not None:
                bx1, by1, bx2, by2 = plate_bbox
                bx1 = max(0, bx1 - pad)
                by1 = max(0, by1 - pad)
                bx2 = min(frame.shape[1], bx2 + pad)
                by2 = min(frame.shape[0], by2 + pad)
                if bx2 > bx1 and by2 > by1:
                    plate_region = frame[by1:by2, bx1:bx2]
                    plate_bbox = (bx1, by1, bx2, by2)

            self._log_verbose(f"Saving plate crop shape={plate_region.shape} src={plate_source}")

            plate_original = np.ascontiguousarray(plate_region.copy())
            resized = cv2.resize(plate_region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            if len(resized.shape) == 3 and resized.shape[2] >= 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            elif len(resized.shape) == 2:
                gray = resized
            else:
                logger.warning("Unexpected plate crop shape %s", getattr(resized, "shape", None))
                gray = resized[:, :, 0] if resized.ndim == 3 else resized
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            plate = cv2.filter2D(gray, -1, kernel)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            uid = uuid.uuid4().hex[:8]
            filename = f"plate_{timestamp}_{uid}.png"
            enhanced_filename = f"enhanced_plate_{timestamp}_{uid}.png"
            plate_dir = "database/static/plates"
            os.makedirs(plate_dir, exist_ok=True)
            raw_plate_path = os.path.join(plate_dir, f"plate_raw_{timestamp}_{uid}.png")
            plate_image_path = os.path.join(plate_dir, filename)
            enhanced_plate_path = os.path.join(plate_dir, enhanced_filename)

            with self._image_io_lock:
                raw_ok = bool(cv2.imwrite(raw_plate_path, plate_original))
                ok = bool(cv2.imwrite(plate_image_path, np.ascontiguousarray(plate)))
                enhanced_ok = bool(cv2.imwrite(enhanced_plate_path, np.ascontiguousarray(plate)))

            if not raw_ok:
                logger.warning("Failed to save raw plate image: %s", raw_plate_path)
            if not ok or not self._verify_saved_image(plate_image_path, min_bytes=24):
                logger.error("Plate PNG write failed: %s", plate_image_path)
                return "Not Detected", plate_bbox
            if not enhanced_ok:
                logger.warning("Failed to save enhanced plate image: %s", enhanced_plate_path)

            rel = f"plates/{filename}"
            self._log_verbose(f"Plate saved {rel} ({plate_source})")
            return rel, plate_bbox
        except Exception:
            logger.exception("Plate pipeline failed (track=%s)", track_id)
            return "Not Detected", None

    def _detect_plate_region(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[Optional[np.ndarray], str, Optional[tuple[int, int, int, int]]]:
        try:
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop is None or vehicle_crop.size == 0:
                self._log_verbose("Plate source: none")
                return None, "none", None
            if self.plate_model is not None:
                try:
                    self._log_verbose("Running plate model...")
                    plate_results = self.plate_model(vehicle_crop, verbose=False)
                    best_box = None
                    best_conf = -1.0
                    box_count = 0
                    for r in plate_results:
                        if r.boxes is None:
                            continue
                        for idx, box in enumerate(r.boxes.xyxy):
                            box_count += 1
                            px1, py1, px2, py2 = map(int, box.tolist())
                            conf = float(r.boxes.conf[idx].item()) if r.boxes.conf is not None else 0.0
                            if conf > best_conf:
                                best_conf = conf
                                best_box = (px1, py1, px2, py2)
                    self._log_verbose(f"Plate boxes found: {box_count}")
                    if best_box is not None:
                        px1, py1, px2, py2 = best_box
                        plate_x1 = max(0, x1 + px1)
                        plate_y1 = max(0, y1 + py1)
                        plate_x2 = min(frame.shape[1], x1 + px2)
                        plate_y2 = min(frame.shape[0], y1 + py2)
                        if plate_x2 > plate_x1 and plate_y2 > plate_y1:
                            plate = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                            if plate is not None and plate.size > 0:
                                self._log_verbose("Plate detected by model")
                                return plate, "model", (plate_x1, plate_y1, plate_x2, plate_y2)
                except Exception:
                    pass
            else:
                self._log_verbose("Plate model missing, plate extraction disabled")
            self._log_verbose("Plate source: none")
            return None, "none", None
        except Exception:
            logger.debug("Plate detection exception", exc_info=True)
            return None, "none", None

    def _fallback_plate_crop(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]]]:
        try:
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            rear_w = int(w * 0.5)
            fx1 = x1 + max(0, (w - rear_w) // 2)
            fx2 = fx1 + rear_w
            fy1 = y1 + int(h * 0.62)
            fy2 = y1 + int(h * 0.95)
            fx1, fy1, fx2, fy2 = clip_bbox_to_frame((fx1, fy1, fx2, fy2), frame.shape)
            if fx2 <= fx1 or fy2 <= fy1:
                return None, None
            crop = frame[fy1:fy2, fx1:fx2]
            if crop is None or crop.size == 0:
                return None, None
            return crop, (fx1, fy1, fx2, fy2)
        except Exception:
            return None, None

    def _plate_bbox_from_violation(self, bbox, frame_shape):
        try:
            x1, y1, x2, y2 = bbox
            h = max(1, y2 - y1)
            plate_start = int(y1 + h * 0.6)
            plate_box = (x1, plate_start, x2, y2)
            return clip_bbox_to_frame(plate_box, frame_shape)
        except Exception:
            return clip_bbox_to_frame(bbox, frame_shape)

    def _expand_bbox(self, bbox, frame_shape, padding: float = 0.1):
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        return clip_bbox_to_frame((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), frame_shape)

    def _process_plate_crop(self, crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(gray, -1, kernel)
        enhanced = cv2.convertScaleAbs(sharpened, alpha=1.7, beta=20)
        resized = cv2.resize(enhanced, (400, 150), interpolation=cv2.INTER_LINEAR)
        return resized

    def _extract_plate_from_image(self, plate_image_path: str) -> str:
        try:
            if "Not Detected" in str(plate_image_path):
                return "Not Detected"
            rel_key = self._normalize_static_path(str(plate_image_path))
            cached = self._cache_ocr_lookup(rel_key)
            if cached is not None:
                return cached

            img_path = str(plate_image_path).replace("\\", "/")
            if not img_path.startswith("database/static/"):
                img_path = f"database/static/{img_path}"
            crop = cv2.imread(img_path)
            if crop is None:
                self._cache_ocr_store(rel_key, "Not Detected")
                return "Not Detected"
            if crop.size == 0:
                self._cache_ocr_store(rel_key, "Not Detected")
                return "Not Detected"

            results = self.reader.readtext(crop)
            self._log_verbose(f"OCR results length: {len(results) if results else 0}")
            if not results:
                self._cache_ocr_store(rel_key, "Not Detected")
                return "Not Detected"

            filtered = []
            for (_, text, conf) in results:
                cleaned = "".join(ch for ch in text.replace(" ", "").upper() if ch.isalnum())
                if conf < self.config.ocr_confidence_threshold:
                    continue
                if len(cleaned) < self.config.plate_min_chars or len(cleaned) > self.config.plate_max_chars:
                    continue
                filtered.append(cleaned)
            if not filtered:
                self._cache_ocr_store(rel_key, "Not Detected")
                return "Not Detected"
            self._cache_ocr_store(rel_key, filtered[0])
            return filtered[0]
        except Exception:
            logger.warning("OCR failed for plate image path %s", plate_image_path, exc_info=False)
            return "Not Detected"

    def _append_log(self, violation_type: str, plate: str, plate_image_path: str, full_image_path: str) -> None:
        with Path(self.config.csv_log_path).open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    violation_type,
                    plate,
                    plate_image_path,
                    full_image_path,
                ]
            )

    def _send_to_backend(
        self,
        violation_type: str,
        plate: str,
        plate_image_path: str,
        full_image_path: str,
        event_time: Optional[str] = None,
    ) -> None:
        if not self.config.enable_backend_post:
            return
        payload = {
            "type": violation_type,
            "plate": plate,
            "plate_image": plate_image_path,
            "image": full_image_path,
            "time": event_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.config.backend_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            print("Sending to server...")
            with urllib.request.urlopen(req, timeout=self.config.backend_timeout_sec):
                print("Sent to server")
        except (urllib.error.URLError, TimeoutError, Exception):
            # Backend is optional; never crash detection loop.
            return

    def _enqueue_job(self, payload: dict) -> None:
        try:
            self.jobs.put_nowait(payload)
        except queue.Full:
            # Keep real-time behavior; skip if overloaded.
            return

    def _worker_loop(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    continue

                track_id = int(job["track_id"])
                violation_type = str(job["violation_type"])
                plate_image_path = str(job["plate_image_path"])
                full_image_path = str(job["full_image_path"])
                event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                plate_text = self._extract_plate_from_image(plate_image_path)
                print(f"Plate detected: {plate_text}")

                with self.lock:
                    self.track_plate_text[track_id] = plate_text

                self._append_log(violation_type, plate_text, plate_image_path, full_image_path)
                self._send_to_backend(violation_type, plate_text, plate_image_path, full_image_path, event_time)
            except Exception as err:
                print(f"Worker error: {err}")
            finally:
                self.jobs.task_done()

    def _prepare_db_data(self, violation_type: str, plate_image_path: str, full_image_path: str, track_id: int, plate_text: str) -> dict:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db_violation = {
            "no_helmet": "Helmet",
            "no_seatbelt": "Seatbelt",
            "overspeed": "Overspeed",
            "triple_riding": "Triple Riding",
            "red_light_violation": "Red Light",
            "wrong_direction": "Wrong Direction",
        }.get(violation_type, violation_type.replace("_", " ").title())
        vehicle = "motorcycle" if violation_type in {"no_helmet", "no_seatbelt", "triple_riding"} else "vehicle"
        speed_value = float(self.track_speed_kmh.get(track_id, 0.0)) if violation_type == "overspeed" else 0.0
        plate_rel = (
            "Not Detected"
            if "Not Detected" in str(plate_image_path)
            else self._normalize_static_path(str(plate_image_path))
        )
        full_rel = self._normalize_static_path(str(full_image_path))
        actual_plate_path = os.path.join("database", "static", plate_rel).replace("\\", "/") if plate_rel != "Not Detected" else "Not Detected"
        actual_violation_path = os.path.join("database", "static", full_rel).replace("\\", "/")
        self._log_verbose(
            f"DB paths plate_ok={os.path.exists(actual_plate_path) if plate_rel != 'Not Detected' else False} "
            f"viol_ok={os.path.exists(actual_violation_path)} plate={plate_rel} viol={full_rel}"
        )

        return {
            "track_id": track_id,
            "plat_license": plate_rel,
            "plate_text": plate_text,
            "violence_category": db_violation,
            "vehicle": vehicle,
            "open_photo": full_rel,
            "speed": speed_value,
            "max_speed": float(self.config.speed_limit_kmph),
            "location": "Bandung",
        }

    def _update_line_speed_and_signals(self, detections: List[Detection], frame_shape, frame_count: int = 0) -> None:
        now = time.time()
        frame_h = frame_shape[0]
        frame_w = frame_shape[1]
        line_y1 = int(frame_h * float(self.config.line1_y_ratio))
        line_y2 = int(frame_h * float(self.config.line2_y_ratio))
        stop_line_y = int(frame_h * 0.75)
        traffic_light_state = "RED"

        for det in detections:
            if det.track_id is None:
                continue
            class_name = self._normalize_name(det.class_name)
            if class_name not in {"car", "motorcycle", "bicycle", "bike", "bus", "truck"}:
                continue
            self.detection_count[det.track_id] = self.detection_count.get(det.track_id, 0) + 1
            x1, y1, x2, y2 = det.bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            motion = self.vehicle_motion.setdefault(
                det.track_id,
                {"positions": [], "frames": [], "last_frame": frame_count, "violation_logged": False},
            )
            positions = motion.setdefault("positions", [])
            frames = motion.setdefault("frames", [])
            positions.append((cx, cy))
            frames.append(frame_count)
            if len(positions) > 10:
                positions.pop(0)
            if len(frames) > 10:
                frames.pop(0)
            motion["last_frame"] = frame_count
            if (
                bool(getattr(self.config, "enable_speed_fallback", True))
                and not bool(motion.get("violation_logged", False))
                and len(positions) >= int(getattr(self.config, "min_movement_frames", 5))
                and len(frames) >= 2
            ):
                start_x, start_y = positions[0]
                end_x, end_y = positions[-1]
                dx = float(end_x - start_x)
                dy = float(end_y - start_y)
                distance_pixels = math.sqrt((dx * dx) + (dy * dy))
                frame_gap = max(1, int(frames[-1] - frames[0]))
                pixel_speed = distance_pixels / float(frame_gap)
                self._log_verbose(f"[DEBUG] Track {det.track_id} pixel speed: {pixel_speed:.2f}")
                if pixel_speed > float(getattr(self.config, "pixel_speed_threshold", 25.0)):
                    self.track_fallback_fast[det.track_id] = True

            prev = self.track_last_centroid.get(det.track_id)
            prev_t = self.track_last_seen_at.get(det.track_id)
            if prev is not None and prev_t is not None:
                prev_cy = prev[1]
                if self.config.enable_wrong_side_detection:
                    history = self.vehicle_history.setdefault(det.track_id, [])
                    history.append((cx, cy))
                    if len(history) > 30:
                        history.pop(0)
                    lane = self._get_lane_for_x(cx, frame_w)
                    if lane is not None and len(history) >= 10:
                        first_x, first_y = history[0]
                        last_x, last_y = history[-1]
                        dy_total = last_y - first_y
                        if abs(dy_total) < int(self.config.wrong_side_min_displacement):
                            self.track_wrong_side_frames[det.track_id] = 0
                        else:
                            direction = "down" if dy_total > 0 else "up"
                            allowed = str(lane.get("allowed_direction", "")).strip().lower()
                            moving_opposite = direction != allowed
                            if moving_opposite:
                                self.track_wrong_side_frames[det.track_id] = self.track_wrong_side_frames.get(det.track_id, 0) + 1
                            else:
                                self.track_wrong_side_frames[det.track_id] = 0
                            if self.track_wrong_side_frames.get(det.track_id, 0) >= int(self.config.wrong_side_min_frames):
                                self.track_wrong_side_flagged[det.track_id] = True
                    else:
                        self.track_wrong_side_frames[det.track_id] = 0
                track_cross = self.track_line_cross_time.setdefault(det.track_id, {"start": None, "end": None})
                if track_cross["start"] is None and (
                    (prev_cy < line_y1 <= cy) or (prev_cy > line_y1 >= cy)
                ):
                    track_cross["start"] = now
                    self._log_verbose(f"Track {det.track_id} crossed line 1")
                if track_cross["start"] is not None and track_cross["end"] is None and (
                    (prev_cy < line_y2 <= cy) or (prev_cy > line_y2 >= cy)
                ):
                    track_cross["end"] = now
                    self._log_verbose(f"Track {det.track_id} crossed line 2")
                    time_taken = abs(track_cross["end"] - track_cross["start"])
                    if time_taken > 1e-6:
                        speed_mps = float(self.config.distance_meters_between_lines) / time_taken
                        speed_kmh = speed_mps * 3.6
                        self._log_verbose(f"Speed: {speed_kmh:.1f} km/h")
                        if 5.0 < speed_kmh < 180.0:
                            self.track_speed_kmh[det.track_id] = speed_kmh

                crossed_stop_line = prev_cy < stop_line_y <= cy
                if self.config.enable_red_light_detection and crossed_stop_line and traffic_light_state == "RED":
                    self.track_crossed_red_light[det.track_id] = True

            self.track_last_centroid[det.track_id] = (cx, cy)
            self.track_last_seen_at[det.track_id] = now

    def is_track_fast(self, track_id: Optional[int]) -> bool:
        if track_id is None:
            return False
        return bool(self.track_fallback_fast.get(track_id, False))

    def _get_lane_for_x(self, cx: int, frame_w: int) -> Optional[dict]:
        lanes = list(getattr(self.config, "wrong_side_lanes", ()) or ())
        if not lanes:
            return None
        max_lane_x = max(int(l.get("x2", 0)) for l in lanes)
        scale = float(frame_w) / float(max_lane_x) if max_lane_x > 0 else 1.0
        for lane in lanes:
            lx1 = int(float(lane.get("x1", 0)) * scale)
            lx2 = int(float(lane.get("x2", 0)) * scale)
            if lx1 <= cx <= lx2:
                return lane
        return None

    def get_debug_overlay(self, frame_shape) -> dict:
        frame_h = frame_shape[0]
        return {
            "line_y1": int(frame_h * float(self.config.line1_y_ratio)),
            "line_y2": int(frame_h * float(self.config.line2_y_ratio)),
            "stop_line_y": int(frame_h * 0.75),
            "traffic_light_state": "RED",
            "show_debug_lines": bool(self.config.debug_line_overlay),
            "red_light_enabled": bool(self.config.enable_red_light_detection),
        }

    def get_plate_for_track(self, track_id: Optional[int]) -> Optional[str]:
        if track_id is None:
            return None
        with self.lock:
            return self.track_plate_text.get(track_id)

    def should_show_recorded(self) -> bool:
        now = time.time()
        return any(until > now for until in self.recent_violation_until.values())
