import numpy as np
import math
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DriverIdentificationSystem:
    """운전자 신원 확인 및 개인화 시스템"""

    def __init__(self):
        self.profiles_dir = Path("driver_profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        self.current_driver = None
        self.identification_confidence = 0.0
        self.face_encodings = {}
        self._load_driver_profiles()
        logger.info("DriverIdentificationSystem 초기화 완료")

    def identify_driver(self, face_landmarks):
        if not face_landmarks:
            return {"driver_id": "unknown", "confidence": 0.0, "is_new_driver": False}

        face_features = self._extract_face_features(face_landmarks)
        best_match = None
        best_similarity = 0.0

        for driver_id, profile in self.face_encodings.items():
            similarity = self._calculate_similarity(face_features, profile["features"])
            if similarity > best_similarity and similarity > 0.7:
                best_similarity = similarity
                best_match = driver_id

        if best_match:
            self.current_driver = best_match
            self.identification_confidence = best_similarity
            return {
                "driver_id": best_match,
                "confidence": best_similarity,
                "is_new_driver": False,
            }
        else:
            new_driver_id = f"driver_{len(self.face_encodings) + 1}"
            self._register_new_driver(new_driver_id, face_features)
            return {
                "driver_id": new_driver_id,
                "confidence": 1.0,
                "is_new_driver": True,
            }

    def _extract_face_features(self, landmarks):
        try:
            key_points = {
                "left_eye": landmarks[33],
                "right_eye": landmarks[263],
                "nose_tip": landmarks[1],
                "mouth_left": landmarks[61],
                "mouth_right": landmarks[291],
                "chin": landmarks[175],
            }
            features = []
            eye_distance = self._euclidean_distance(key_points["left_eye"], key_points["right_eye"])
            features.append(eye_distance)
            nose_mouth_distance = self._euclidean_distance(key_points["nose_tip"], key_points["mouth_left"])
            features.append(nose_mouth_distance)
            mouth_width = self._euclidean_distance(key_points["mouth_left"], key_points["mouth_right"])
            features.append(mouth_width)
            face_height = self._euclidean_distance(key_points["nose_tip"], key_points["chin"])
            features.append(face_height)
            return np.array(features)
        except (IndexError, AttributeError) as e:
            logger.debug(f"얼굴 특징 추출 실패: {e}")
            return np.zeros(4)

    def _euclidean_distance(self, point1, point2):
        try:
            return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
        except AttributeError as e:
            logger.debug(f"유클리드 거리 계산 실패: {e}")
            return 0.0

    def _calculate_similarity(self, features1, features2):
        try:
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)
        except (TypeError, ValueError) as e:
            # 배열 형태나 값 오류
            logger.debug(f"특징 유사도 계산 실패 (데이터 문제): {e}")
            return 0.0
        except Exception as e:
            # 기타 예상치 못한 오류
            logger.warning(f"특징 유사도 계산 중 예상치 못한 오류: {e}")
            return 0.0

    def _load_driver_profiles(self):
        profile_file = self.profiles_dir / "face_encodings.json"
        if profile_file.exists():
            try:
                with open(profile_file, "r") as f:
                    data = json.load(f)
                for driver_id, profile_data in data.items():
                    self.face_encodings[driver_id] = {
                        "features": np.array(profile_data["features"]),
                        "created_at": profile_data["created_at"],
                    }
                logger.info(f"{len(self.face_encodings)}명의 운전자 프로필 로드됨")
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"프로필 로드 실패: {e}")

    def _register_new_driver(self, driver_id, features):
        self.face_encodings[driver_id] = {
            "features": features,
            "created_at": datetime.now().isoformat(),
        }
        self.current_driver = driver_id
        self.identification_confidence = 1.0
        self._save_driver_profiles()
        logger.info(f"새 운전자 등록됨: {driver_id}")

    def _save_driver_profiles(self):
        profile_file = self.profiles_dir / "face_encodings.json"
        try:
            data = {}
            for driver_id, profile in self.face_encodings.items():
                data[driver_id] = {
                    "features": profile["features"].tolist(),
                    "created_at": profile["created_at"],
                }
            with open(profile_file, "w") as f:
                json.dump(data, f, indent=2)
        except (IOError, TypeError) as e:
            logger.error(f"프로필 저장 실패: {e}")

    def get_current_driver(self):
        return {
            "driver_id": self.current_driver or "unknown",
            "confidence": self.identification_confidence,
        }
