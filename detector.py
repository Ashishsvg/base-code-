from typing import List

import numpy as np
import os
from ultralytics import YOLO

from config import AppConfig
from utils import Detection
from typing import Tuple

class YOLODetector:
    """YOLOv8 detector only (CPU-friendly)."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = "cpu" if config.use_cpu else "0"
        model_path = config.model_path or "yolov8n.pt"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.helmet_model = None
        self.seatbelt_model = None
        self.helmet_class_aliases = {
            "helmet",
            "helm",
            "no_helmet",
            "no_helm",
            "nohelm",
            "without_helmet",
            "with_helmet",
            "head",
            "rider_helmet",
            "rider_no_helmet",
        }
        self.helmet_negative_aliases = {"no_helmet", "no_helm", "nohelm", "without_helmet", "rider_no_helmet"}
        self.seatbelt_positive_aliases = {"seatbelt", "seat_belt", "with_seatbelt", "belt"}
        self.seatbelt_negative_aliases = {"no_seatbelt", "without_seatbelt", "no_belt"}
        self._warned_missing_helmet_classes = False
        print(f"✅ Main model loaded: {model_path}")
        self._init_helmet_model()
        self._init_seatbelt_model()

    def _normalize_class_name(self, name: str) -> str:
        return str(name).lower().replace("-", "_").replace(" ", "_").strip()

    def find_class_ids(self, model_names: dict, positive_names: set[str], negative_names: set[str]) -> tuple[int | None, int | None]:
        normalized = {}
        for class_id, name in model_names.items():
            normalized_name = self._normalize_class_name(name)
            normalized[class_id] = normalized_name

        positive_id = None
        negative_id = None
        for class_id, name in normalized.items():
            if name in positive_names and positive_id is None:
                positive_id = class_id
            if name in negative_names and negative_id is None:
                negative_id = class_id
        return positive_id, negative_id

    def _extract_model_class_names(self, model) -> dict:
        names = model.names if hasattr(model, "names") else {}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {idx: str(v) for idx, v in enumerate(names)}

    def _init_helmet_model(self) -> None:
        helmet_model_path = self.config.helmet_model_path
        if not (helmet_model_path and os.path.exists(helmet_model_path)):
            self.config.enable_helmet_detection = False
            print("⚠️ Helmet model missing: models/helmet_model.pt (helmet detection disabled)")
            return
        try:
            self.helmet_model = YOLO(helmet_model_path)
            class_map = self._extract_model_class_names(self.helmet_model)
            print(f"✅ Helmet model loaded: {helmet_model_path}")
            print(f"Helmet classes: {class_map}")
            helmet_id, no_helmet_id = self.find_class_ids(class_map, self.helmet_class_aliases - self.helmet_negative_aliases, self.helmet_negative_aliases)
            if helmet_id is not None or no_helmet_id is not None:
                self.config.enable_helmet_detection = True
                print("✅ Helmet detection enabled")
                print(f"Helmet class id: {helmet_id}")
                print(f"No-helmet class id: {no_helmet_id}")
            else:
                self.config.enable_helmet_detection = False
                self.helmet_model = None
                print("⚠️ Helmet model loaded but no helmet/no-helmet compatible classes found.")
                print(f"Helmet classes found: {class_map}")
                print("Helmet detection disabled.")
        except Exception as exc:
            self.config.enable_helmet_detection = False
            self.helmet_model = None
            print(f"⚠️ Helmet model init failed: {exc}")

    def _init_seatbelt_model(self) -> None:
        seatbelt_candidates = [
            "models/seatbelt_model.pt",
            "models/seat_belt_model.pt",
            "models/seatbelt.pt",
            "models/seat_belt.pt",
            "models/best_seatbelt.pt",
        ]
        seatbelt_model_path = None
        for candidate in seatbelt_candidates:
            if os.path.exists(candidate):
                seatbelt_model_path = candidate
                break
        if seatbelt_model_path is None:
            self.config.enable_seatbelt_detection = False
            print("⚠️ Seatbelt model missing: models/seatbelt_model.pt (seatbelt detection disabled)")
            return
        try:
            self.seatbelt_model = YOLO(seatbelt_model_path)
            class_map = self._extract_model_class_names(self.seatbelt_model)
            print(f"✅ Seatbelt model loaded: {seatbelt_model_path}")
            print(f"Seatbelt classes: {class_map}")
            seatbelt_id, no_seatbelt_id = self.find_class_ids(class_map, self.seatbelt_positive_aliases, self.seatbelt_negative_aliases)
            if seatbelt_id is not None or no_seatbelt_id is not None:
                self.config.enable_seatbelt_detection = True
                print("✅ Seatbelt detection enabled")
                print(f"Seatbelt class id: {seatbelt_id}")
                print(f"No-seatbelt class id: {no_seatbelt_id}")
            else:
                self.config.enable_seatbelt_detection = False
                self.seatbelt_model = None
                print("⚠️ Seatbelt model loaded but no compatible classes found. Seatbelt detection disabled.")
                print(f"Seatbelt classes found: {class_map}")
        except Exception as exc:
            self.config.enable_seatbelt_detection = False
            self.seatbelt_model = None
            print(f"⚠️ Seatbelt model init failed: {exc}")

    def detect(self, frame: np.ndarray, run_secondary_models: bool = True) -> List[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.inference_imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        verbose = bool(getattr(self.config, "verbose_pipeline_logs", False))
        if not results or len(results) == 0:
            if verbose:
                print("Detection count: 0")
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            if verbose:
                print("Detection count: 0")
            return detections

        for box in result.boxes:
            cls_id = int(box.cls.item())
            if isinstance(result.names, dict):
                cls_name = result.names.get(cls_id, str(cls_id))
            else:
                cls_name = result.names[cls_id] if 0 <= cls_id < len(result.names) else str(cls_id)
            conf = float(box.conf.item())
            if conf < 0.3:
                continue
            if verbose:
                print("Detected class:", cls_name, "confidence:", conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if x2 <= x1 or y2 <= y1:
                continue
            track_id = int(box.id.item()) if getattr(box, "id", None) is not None else None
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    track_id=track_id,
                )
            )

        if run_secondary_models and self.helmet_model is not None:
            helmet_results = self.helmet_model.predict(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.inference_imgsz,
                device=self.device,
                verbose=False,
            )
            if helmet_results and len(helmet_results) > 0 and helmet_results[0].boxes is not None:
                h_result = helmet_results[0]
                for box in h_result.boxes:
                    cls_id = int(box.cls.item())
                    if isinstance(h_result.names, dict):
                        cls_name = h_result.names.get(cls_id, str(cls_id))
                    else:
                        cls_name = h_result.names[cls_id] if 0 <= cls_id < len(h_result.names) else str(cls_id)
                    norm_name = self._normalize_class_name(cls_name)
                    if norm_name not in self.helmet_class_aliases:
                        continue
                    mapped_name = "no_helmet" if norm_name in self.helmet_negative_aliases else "helmet"
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if x2 <= x1 or y2 <= y1:
                        continue
                    detections.append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            class_id=cls_id,
                            class_name=mapped_name,
                            confidence=conf,
                        )
                    )
        if run_secondary_models and self.seatbelt_model is not None:
            seatbelt_results = self.seatbelt_model.predict(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.inference_imgsz,
                device=self.device,
                verbose=False,
            )
            if seatbelt_results and len(seatbelt_results) > 0 and seatbelt_results[0].boxes is not None:
                s_result = seatbelt_results[0]
                for box in s_result.boxes:
                    cls_id = int(box.cls.item())
                    if isinstance(s_result.names, dict):
                        cls_name = s_result.names.get(cls_id, str(cls_id))
                    else:
                        cls_name = s_result.names[cls_id] if 0 <= cls_id < len(s_result.names) else str(cls_id)
                    norm_name = self._normalize_class_name(cls_name)
                    if norm_name not in (self.seatbelt_positive_aliases | self.seatbelt_negative_aliases):
                        continue
                    mapped_name = "no_seatbelt" if norm_name in self.seatbelt_negative_aliases else "seatbelt"
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if x2 <= x1 or y2 <= y1:
                        continue
                    detections.append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            class_id=cls_id,
                            class_name=mapped_name,
                            confidence=conf,
                        )
                    )
        if verbose:
            print(f"Detection count: {len(detections)}")
        return detections
