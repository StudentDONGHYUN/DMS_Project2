import numpy as np
import math
from collections import deque
import logging

logger = logging.getLogger(__name__)

class EnhancedDrowsinessDetector:
    """향상된 EAR 기반 졸음 감지 시스템"""

    def __init__(self):
        self.ear_history = deque(maxlen=900)
        self.personalized_threshold = None
        self.calibration_frames = 300
        self.temporal_attention = TemporalAttentionModel()
        self.microsleep_detector = MicrosleepDetector()
        self.calibration_ears = deque(maxlen=300)
        self.is_calibrated = False
        logger.info("EnhancedDrowsinessDetector 초기화 완료")

    def detect_drowsiness(self, face_landmarks, timestamp):
        if not face_landmarks:
            return {"status": "no_face", "confidence": 0.0}

        left_ear = self._calculate_enhanced_ear(face_landmarks, "left")
        right_ear = self._calculate_enhanced_ear(face_landmarks, "right")
        avg_ear = (left_ear + right_ear) / 2.0
        head_pose = self._estimate_head_pose_simple(face_landmarks)
        corrected_ear = self._correct_for_head_pose(avg_ear, head_pose)

        self.ear_history.append(
            {
                "timestamp": timestamp,
                "ear": corrected_ear,
                "raw_ear": avg_ear,
                "head_pose": head_pose,
            }
        )

        if len(self.ear_history) >= self.calibration_frames and not self.is_calibrated:
            self._update_personalized_threshold()
            self.is_calibrated = True

        drowsiness_probability = self.temporal_attention.predict(
            self.ear_history, self.personalized_threshold or 0.25
        )
        microsleep_result = self.microsleep_detector.detect(self.ear_history)
        perclos = self._calculate_perclos()

        return {
            "status": self._determine_drowsiness_level(drowsiness_probability),
            "confidence": drowsiness_probability,
            "enhanced_ear": corrected_ear,
            "threshold": self.personalized_threshold or 0.25,
            "microsleep": microsleep_result,
            "perclos": perclos,
            "temporal_attention_score": drowsiness_probability,
        }

    def _calculate_enhanced_ear(self, landmarks, eye_side):
        if eye_side == "left":
            eye_points = [33, 7, 163, 144, 145, 153]
        else:
            eye_points = [362, 382, 381, 380, 374, 373]

        try:
            eye_landmarks = [landmarks[i] for i in eye_points]
            vertical_1 = self._euclidean_distance(eye_landmarks[1], eye_landmarks[5])
            vertical_2 = self._euclidean_distance(eye_landmarks[2], eye_landmarks[4])
            vertical_3 = self._euclidean_distance(eye_landmarks[0], eye_landmarks[3])
            horizontal = self._euclidean_distance(eye_landmarks[0], eye_landmarks[3])
            if horizontal > 0:
                ear = (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)
            else:
                ear = 0.0
            return ear
        except (IndexError, AttributeError):
            return 0.0

    def _euclidean_distance(self, point1, point2):
        try:
            return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
        except AttributeError:
            return 0.0

    def _estimate_head_pose_simple(self, landmarks):
        try:
            nose = landmarks[1]
            left_ear = landmarks[234]
            right_ear = landmarks[454]
            ear_center_x = (left_ear.x + right_ear.x) / 2
            yaw = math.degrees(math.atan2(nose.x - ear_center_x, 0.1))
            ear_center_y = (left_ear.y + right_ear.y) / 2
            pitch = math.degrees(math.atan2(nose.y - ear_center_y, 0.5))
            return {"yaw": yaw, "pitch": pitch, "roll": 0.0}
        except (IndexError, AttributeError) as e:
            # 랜드마크 인덱스 오류 또는 속성 접근 오류
            logger.debug(f"헤드 포즈 추정 실패 (랜드마크 문제): {e}")
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        except Exception as e:
            # 기타 예상치 못한 오류
            logger.warning(f"헤드 포즈 추정 중 예상치 못한 오류: {e}")
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    def _correct_for_head_pose(self, ear, head_pose):
        pitch_correction = np.cos(np.radians(head_pose["pitch"]))
        yaw_correction = np.cos(np.radians(abs(head_pose["yaw"])))
        pitch_correction = max(0.5, min(1.0, pitch_correction))
        yaw_correction = max(0.5, min(1.0, yaw_correction))
        corrected_ear = ear / (pitch_correction * yaw_correction)
        return corrected_ear

    def _update_personalized_threshold(self):
        recent_ears = [frame["ear"] for frame in list(self.ear_history)[-300:]]
        if len(recent_ears) < 50:
            return
        self.personalized_threshold = np.percentile(recent_ears, 5)
        self.personalized_threshold = max(0.15, min(0.35, self.personalized_threshold))
        logger.info(f"개인화 임계값 업데이트: {self.personalized_threshold:.3f}")

    def _calculate_perclos(self):
        if len(self.ear_history) < 30:
            return 0.0
        recent_ears = [frame["ear"] for frame in list(self.ear_history)[-30:]]
        threshold = self.personalized_threshold or 0.25
        closed_count = sum(1 for ear in recent_ears if ear < threshold)
        perclos = closed_count / len(recent_ears)
        return perclos

    def _determine_drowsiness_level(self, probability):
        if probability > 0.8:
            return "critical_drowsiness"
        elif probability > 0.6:
            return "high_drowsiness"
        elif probability > 0.4:
            return "moderate_drowsiness"
        elif probability > 0.2:
            return "mild_drowsiness"
        else:
            return "alert"

class TemporalAttentionModel:
    def __init__(self):
        self.window_size = 30
        self.attention_weights = self._create_attention_weights()

    def predict(self, ear_history, threshold):
        if len(ear_history) < self.window_size:
            return 0.0
        recent_ears = [frame["ear"] for frame in list(ear_history)[-self.window_size :]]
        weighted_ears = np.array(recent_ears) * self.attention_weights
        below_threshold_ratio = np.sum(weighted_ears < threshold) / len(weighted_ears)
        consecutive_count = self._count_consecutive_low_ears(recent_ears, threshold)
        trend_score = self._analyze_trend(recent_ears)
        drowsiness_prob = (
            below_threshold_ratio * 0.4
            + min(consecutive_count / 15, 1.0) * 0.4
            + trend_score * 0.2
        )
        return min(1.0, drowsiness_prob)

    def _create_attention_weights(self):
        weights = np.exp(np.linspace(-2, 0, self.window_size))
        return weights / np.sum(weights)

    def _count_consecutive_low_ears(self, ears, threshold):
        consecutive = 0
        max_consecutive = 0
        for ear in reversed(ears):
            if ear < threshold:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        return max_consecutive

    def _analyze_trend(self, ears):
        if len(ears) < 10:
            return 0.0
        x = np.arange(len(ears))
        coeffs = np.polyfit(x, ears, 1)
        slope = coeffs[0]
        trend_score = max(0.0, -slope * 10)
        return min(1.0, trend_score)

class MicrosleepDetector:
    def __init__(self):
        self.microsleep_threshold = 0.15
        self.min_duration = 0.5
        self.max_duration = 3.0

    def detect(self, ear_history):
        if len(ear_history) < 15:
            return {"detected": False, "duration": 0.0, "confidence": 0.0}
        recent_data = list(ear_history)[-90:]
        for i in range(len(recent_data) - 15):
            low_ear_duration = 0
            start_idx = i
            for j in range(i, min(i + 90, len(recent_data))):
                if recent_data[j]["ear"] < self.microsleep_threshold:
                    low_ear_duration = (j - start_idx) / 30.0
                else:
                    break
            if self.min_duration <= low_ear_duration <= self.max_duration:
                confidence = min(1.0, low_ear_duration / self.max_duration)
                return {
                    "detected": True,
                    "duration": low_ear_duration,
                    "confidence": confidence,
                }
        return {"detected": False, "duration": 0.0, "confidence": 0.0}
