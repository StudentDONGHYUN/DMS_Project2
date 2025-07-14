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

        # --- 개선: 변화 감지 기반 임계값 재캘리브레이션 ---
        if len(self.ear_history) >= self.calibration_frames and not self.is_calibrated:
            self._update_personalized_threshold()
            self.is_calibrated = True
        elif self.is_calibrated and len(self.ear_history) >= 1800:  # 10분 이상 데이터가 쌓였을 때만
            # 최근 5분(900프레임)과 그 이전 5분(900프레임) 분포 비교
            recent_ears = [frame["ear"] for frame in list(self.ear_history)[-900:]]
            prev_ears = [frame["ear"] for frame in list(self.ear_history)[-1800:-900]]
            if len(prev_ears) == 900:
                mean_recent, std_recent, var_recent = np.mean(recent_ears), np.std(recent_ears), np.var(recent_ears)
                mean_prev, std_prev, var_prev = np.mean(prev_ears), np.std(prev_ears), np.var(prev_ears)
                # 변화 임계값: 평균 0.03, 표준편차 0.02, 분산 0.001
                if (
                    abs(mean_recent - mean_prev) > 0.03 or
                    abs(std_recent - std_prev) > 0.02 or
                    abs(var_recent - var_prev) > 0.001
                ):
                    self._update_personalized_threshold()
                    logger.info(
                        f"EAR 분포 변화 감지: 재캘리브레이션 수행 "
                        f"(mean {mean_prev:.3f}->{mean_recent:.3f}, "
                        f"std {std_prev:.3f}->{std_recent:.3f}, "
                        f"var {var_prev:.4f}->{var_recent:.4f})"
                    )

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
        self.min_frames = int(self.min_duration * 30)  # 프레임 단위로 계산
        self.max_frames = int(self.max_duration * 30)

    def detect(self, ear_history):
        if len(ear_history) < self.min_frames:
            return {"detected": False, "duration": 0.0, "confidence": 0.0}
        
        recent_data = list(ear_history)[-90:]  # 최근 90프레임 (약 3초)
        
        # 최적화된 슬라이딩 윈도우 알고리즘 - O(n) 복잡도
        return self._sliding_window_detection(recent_data)
    
    def _sliding_window_detection(self, data):
        """슬라이딩 윈도우를 이용한 최적화된 마이크로슬립 감지"""
        if len(data) < self.min_frames:
            return {"detected": False, "duration": 0.0, "confidence": 0.0}
        
        # 연속된 낮은 EAR 값의 시작점을 찾기
        consecutive_low_start = -1
        consecutive_low_count = 0
        
        for i, frame in enumerate(data):
            if frame["ear"] < self.microsleep_threshold:
                if consecutive_low_start == -1:
                    consecutive_low_start = i
                consecutive_low_count += 1
            else:
                # 연속된 낮은 EAR 구간이 끝남
                if consecutive_low_start != -1:
                    # 마이크로슬립 기준 확인
                    if self.min_frames <= consecutive_low_count <= self.max_frames:
                        duration = consecutive_low_count / 30.0  # 초 단위 변환
                        confidence = min(1.0, duration / self.max_duration)
                        
                        return {
                            "detected": True,
                            "duration": duration,
                            "confidence": confidence,
                            "start_frame": consecutive_low_start,
                            "end_frame": i - 1
                        }
                
                # 상태 초기화
                consecutive_low_start = -1
                consecutive_low_count = 0
        
        # 마지막까지 연속된 낮은 EAR 구간이 있는 경우
        if consecutive_low_start != -1:
            if self.min_frames <= consecutive_low_count <= self.max_frames:
                duration = consecutive_low_count / 30.0
                confidence = min(1.0, duration / self.max_duration)
                
                return {
                    "detected": True,
                    "duration": duration,
                    "confidence": confidence,
                    "start_frame": consecutive_low_start,
                    "end_frame": len(data) - 1
                }
        
        return {"detected": False, "duration": 0.0, "confidence": 0.0}
