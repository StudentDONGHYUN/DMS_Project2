import numpy as np
from collections import deque
import logging
from core.definitions import EmotionState

logger = logging.getLogger(__name__)

class EmotionRecognitionSystem:
    """52개 블렌드셰이프 기반 감정 인식 시스템"""

    def __init__(self):
        self.emotion_history = deque(maxlen=150)
        self.emotion_classifier = EmotionClassifier()
        self.stress_detector = StressDetector()
        self.current_emotion = EmotionState.NEUTRAL
        self.emotion_confidence = 0.0
        logger.info("EmotionRecognitionSystem 초기화 완료")

    def analyze_emotion(self, face_blendshapes, timestamp):
        if not face_blendshapes:
            return {
                "emotion": EmotionState.NEUTRAL,
                "confidence": 0.0,
                "arousal": 0.5,
                "valence": 0.5,
                "stress_level": 0.0,
            }

        blendshapes_dict = {cat.category_name: cat.score for cat in face_blendshapes}
        au_features = self._map_blendshapes_to_aus(blendshapes_dict)
        arousal = self._calculate_arousal(au_features, blendshapes_dict)
        valence = self._calculate_valence(au_features, blendshapes_dict)
        stress_level = self.stress_detector.detect(au_features, blendshapes_dict)

        # --- 계층적(2단계) 감정 분석 구조 적용 ---
        # 1단계: Valence/Arousal만으로 중립/안정 판단 (0.4~0.6 범위)
        if 0.4 < valence < 0.6 and 0.4 < arousal < 0.6:
            # 중립/안정 상태로 간주, 세부 감정 분석 생략
            emotion_result = {"emotion": EmotionState.NEUTRAL, "confidence": 1.0}
        else:
            # 2단계: 세부 감정 분석
            emotion_result = self.emotion_classifier.classify(au_features)

        self.emotion_history.append(
            {
                "timestamp": timestamp,
                "emotion": emotion_result["emotion"],
                "confidence": emotion_result["confidence"],
                "arousal": arousal,
                "valence": valence,
                "stress_level": stress_level,
            }
        )

        smoothed_result = self._temporal_smoothing()
        return smoothed_result

    def _map_blendshapes_to_aus(self, blendshapes):
        au_mapping = {
            "AU1": blendshapes.get("browInnerUp", 0),
            "AU2": blendshapes.get("browOuterUpLeft", 0) + blendshapes.get("browOuterUpRight", 0),
            "AU4": blendshapes.get("browDownLeft", 0) + blendshapes.get("browDownRight", 0),
            "AU5": blendshapes.get("eyeWideLeft", 0) + blendshapes.get("eyeWideRight", 0),
            "AU6": blendshapes.get("cheekSquintLeft", 0) + blendshapes.get("cheekSquintRight", 0),
            "AU7": blendshapes.get("eyeSquintLeft", 0) + blendshapes.get("eyeSquintRight", 0),
            "AU9": blendshapes.get("noseSneerLeft", 0) + blendshapes.get("noseSneerRight", 0),
            "AU10": blendshapes.get("mouthUpperUpLeft", 0) + blendshapes.get("mouthUpperUpRight", 0),
            "AU12": blendshapes.get("mouthSmileLeft", 0) + blendshapes.get("mouthSmileRight", 0),
            "AU15": blendshapes.get("mouthFrownLeft", 0) + blendshapes.get("mouthFrownRight", 0),
            "AU17": blendshapes.get("mouthDimpleLeft", 0) + blendshapes.get("mouthDimpleRight", 0),
            "AU20": blendshapes.get("mouthStretchLeft", 0) + blendshapes.get("mouthStretchRight", 0),
            "AU23": blendshapes.get("mouthPressLeft", 0) + blendshapes.get("mouthPressRight", 0),
            "AU25": blendshapes.get("jawOpen", 0),
            "AU26": blendshapes.get("jawOpen", 0),
        }
        return list(au_mapping.values())

    def _calculate_arousal(self, au_features, blendshapes):
        high_arousal_features = [
            blendshapes.get("eyeWideLeft", 0),
            blendshapes.get("eyeWideRight", 0),
            blendshapes.get("browInnerUp", 0),
            blendshapes.get("jawOpen", 0),
            au_features[4],
            au_features[0],
        ]
        arousal = np.mean(high_arousal_features)
        return min(1.0, max(0.0, arousal))

    def _calculate_valence(self, au_features, blendshapes):
        positive_features = [
            blendshapes.get("mouthSmileLeft", 0),
            blendshapes.get("mouthSmileRight", 0),
            au_features[8],
            au_features[5],
        ]
        negative_features = [
            blendshapes.get("mouthFrownLeft", 0),
            blendshapes.get("mouthFrownRight", 0),
            au_features[9],
            au_features[2],
        ]
        positive_score = np.mean(positive_features)
        negative_score = np.mean(negative_features)
        valence = 0.5 + (positive_score - negative_score) * 0.5
        return min(1.0, max(0.0, valence))

    def _temporal_smoothing(self):
        if len(self.emotion_history) < 5:
            return {
                "emotion": EmotionState.NEUTRAL,
                "confidence": 0.0,
                "arousal": 0.5,
                "valence": 0.5,
                "stress_level": 0.0,
            }
        recent_history = list(self.emotion_history)[-30:]
        avg_arousal = np.mean([h["arousal"] for h in recent_history])
        avg_valence = np.mean([h["valence"] for h in recent_history])
        avg_stress = np.mean([h["stress_level"] for h in recent_history])
        emotions = [h["emotion"] for h in recent_history]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        emotion_confidence = dominant_emotion[1] / len(recent_history)
        return {
            "emotion": dominant_emotion[0],
            "confidence": emotion_confidence,
            "arousal": avg_arousal,
            "valence": avg_valence,
            "stress_level": avg_stress,
        }

class EmotionClassifier:
    def __init__(self):
        self.emotion_rules = self._define_emotion_rules()

    def classify(self, au_features):
        scores = {}
        for emotion, rules in self.emotion_rules.items():
            score = 0.0
            for au_idx, weight in rules.items():
                if au_idx < len(au_features):
                    score += au_features[au_idx] * weight
            scores[emotion] = score
        best_emotion = max(scores.items(), key=lambda x: x[1])
        return {
            "emotion": best_emotion[0],
            "confidence": min(1.0, best_emotion[1]),
            "all_scores": scores,
        }

    def _define_emotion_rules(self):
        return {
            EmotionState.HAPPINESS: {8: 1.0, 5: 0.8},
            EmotionState.SADNESS: {9: 1.0, 2: 0.6, 0: 0.4},
            EmotionState.ANGER: {2: 1.0, 6: 0.8, 12: 0.6},
            EmotionState.FEAR: {0: 1.0, 4: 0.8, 13: 0.6},
            EmotionState.SURPRISE: {0: 1.0, 1: 0.8, 4: 0.8, 13: 0.6},
            EmotionState.DISGUST: {7: 1.0, 6: 0.6},
            EmotionState.NEUTRAL: {},
        }

class StressDetector:
    def detect(self, au_features, blendshapes):
        stress_indicators = []
        brow_furrow = blendshapes.get("browDownLeft", 0) + blendshapes.get("browDownRight", 0)
        stress_indicators.append(brow_furrow)
        lip_tension = blendshapes.get("mouthPressLeft", 0) + blendshapes.get("mouthPressRight", 0)
        stress_indicators.append(lip_tension)
        eye_tension = blendshapes.get("eyeSquintLeft", 0) + blendshapes.get("eyeSquintRight", 0)
        stress_indicators.append(eye_tension)
        jaw_tension = blendshapes.get("jawForward", 0)
        stress_indicators.append(jaw_tension)
        stress_level = np.mean(stress_indicators)
        return min(1.0, stress_level)
