import math
from collections import deque
import logging

logger = logging.getLogger(__name__)

class DistractionObjectDetector:
    """주의산만 객체 실시간 감지 시스템"""

    def __init__(self):
        self.distraction_objects = {
            "cell phone": {"risk_level": 0.9, "description": "휴대폰"},
            "cup": {"risk_level": 0.6, "description": "컵"},
            "bottle": {"risk_level": 0.6, "description": "물병"},
            "sandwich": {"risk_level": 0.7, "description": "음식"},
            "book": {"risk_level": 0.8, "description": "책"},
            "laptop": {"risk_level": 0.9, "description": "노트북"},
            "remote": {"risk_level": 0.5, "description": "리모컨"},
        }
        self.detection_history = deque(maxlen=90)
        logger.info("DistractionObjectDetector 초기화 완료")

    def analyze_detections(self, object_results, hand_positions, timestamp):
        detected_objects = []
        risk_score = 0.0

        if object_results and object_results.detections:
            for detection in object_results.detections:
                category = detection.categories[0].category_name
                confidence = detection.categories[0].score
                bbox = detection.bounding_box

                if category in self.distraction_objects:
                    obj_info = self.distraction_objects[category]
                    hand_proximity = self._calculate_hand_proximity(bbox, hand_positions)
                    object_risk = obj_info["risk_level"] * confidence * hand_proximity
                    risk_score = max(risk_score, object_risk)

                    detected_objects.append(
                        {
                            "category": category,
                            "description": obj_info["description"],
                            "confidence": confidence,
                            "risk_level": object_risk,
                            "hand_proximity": hand_proximity,
                            "bbox": bbox,
                        }
                    )

        self.detection_history.append(
            {
                "timestamp": timestamp,
                "objects": detected_objects,
                "risk_score": risk_score,
            }
        )

        persistent_risk = self._detect_persistent_risk()

        return {
            "detected_objects": detected_objects,
            "immediate_risk": risk_score,
            "persistent_risk": persistent_risk,
            "object_count": len(detected_objects),
        }

    def _calculate_hand_proximity(self, bbox, hand_positions):
        if not hand_positions:
            return 0.0

        bbox_center_x = bbox.origin_x + bbox.width / 2
        bbox_center_y = bbox.origin_y + bbox.height / 2
        min_distance = float("inf")

        for hand in hand_positions:
            hand_x = hand.get("x", 0.5)
            hand_y = hand.get("y", 0.5)
            distance = math.sqrt((bbox_center_x - hand_x) ** 2 + (bbox_center_y - hand_y) ** 2)
            min_distance = min(min_distance, distance)

        proximity = max(0.0, 1.0 - min_distance / 0.5)
        return proximity

    def _detect_persistent_risk(self):
        if len(self.detection_history) < 30:
            return 0.0
        recent_detections = list(self.detection_history)[-30:]
        risk_frames = sum(1 for detection in recent_detections if detection["risk_score"] > 0.5)
        persistence_ratio = risk_frames / len(recent_detections)
        return persistence_ratio
