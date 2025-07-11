import numpy as np
from collections import deque
import logging
from core.definitions import EmotionState

logger = logging.getLogger(__name__)

class PredictiveSafetySystem:
    """30초 전 위험 예측 시스템"""

    def __init__(self):
        self.prediction_window = 30.0
        self.feature_history = deque(maxlen=900)
        self.risk_predictor = RiskPredictor()
        logger.info("PredictiveSafetySystem 초기화 완료")

    def predict_risk(self, current_metrics, timestamp):
        features = self._extract_features(current_metrics)
        self.feature_history.append({"timestamp": timestamp, "features": features})

        if len(self.feature_history) < 300:
            return {
                "risk_probability": 0.0,
                "prediction_horizon": self.prediction_window,
                "risk_factors": [],
                "confidence": 0.0,
                "alert_level": "none",
            }

        prediction_result = self.risk_predictor.predict(list(self.feature_history))
        risk_factors = self._identify_risk_factors(current_metrics)
        alert_level = self._determine_alert_level(prediction_result["risk_probability"])

        return {
            "risk_probability": prediction_result["risk_probability"],
            "prediction_horizon": self.prediction_window,
            "risk_factors": risk_factors,
            "confidence": prediction_result["confidence"],
            "alert_level": alert_level,
            "predicted_event": prediction_result.get("predicted_event", "unknown"),
        }

    def _extract_features(self, metrics):
        features = [
            metrics.fatigue_risk_score,
            metrics.distraction_risk_score,
            metrics.enhanced_ear,
            metrics.perclos,
            metrics.temporal_attention_score,
            abs(metrics.head_yaw) / 90.0,
            abs(metrics.head_pitch) / 60.0,
            metrics.arousal_level,
            1.0 - metrics.attention_focus_score,
            1.0 if metrics.phone_detected else 0.0,
            len(metrics.distraction_objects) / 5.0,
            metrics.emotion_confidence if metrics.emotion_state == EmotionState.STRESS else 0.0,
        ]
        return np.array(features)

    def _identify_risk_factors(self, metrics):
        risk_factors = []
        if metrics.fatigue_risk_score > 0.6:
            risk_factors.append("높은 피로도")
        if metrics.perclos > 0.7:
            risk_factors.append("빈번한 눈 감음")
        if metrics.distraction_risk_score > 0.5:
            risk_factors.append("주의 분산")
        if abs(metrics.head_yaw) > 45:
            risk_factors.append("과도한 고개 돌림")
        if metrics.phone_detected:
            risk_factors.append("휴대폰 사용")
        if metrics.emotion_state == EmotionState.STRESS and metrics.emotion_confidence > 0.7:
            risk_factors.append("스트레스 상태")
        if metrics.attention_focus_score < 0.3:
            risk_factors.append("주의력 저하")
        return risk_factors

    def _determine_alert_level(self, risk_probability):
        if risk_probability > 0.8:
            return "critical"
        elif risk_probability > 0.6:
            return "high"
        elif risk_probability > 0.4:
            return "medium"
        elif risk_probability > 0.2:
            return "low"
        else:
            return "none"

class RiskPredictor:
    def __init__(self):
        self.window_size = 300
        self.trend_analyzer = TrendAnalyzer()

    def predict(self, feature_history):
        if len(feature_history) < self.window_size:
            return {"risk_probability": 0.0, "confidence": 0.0}

        recent_features = [entry["features"] for entry in feature_history[-self.window_size :]]
        feature_matrix = np.array(recent_features)
        trends = []
        for i in range(feature_matrix.shape[1]):
            trend = self.trend_analyzer.analyze_trend(feature_matrix[:, i])
            trends.append(trend)

        weights = [0.2, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02]
        risk_score = 0.0
        for i, (trend, weight) in enumerate(zip(trends, weights)):
            if i < len(trends):
                risk_score += trend * weight

        current_features = feature_matrix[-1]
        current_risk = np.mean(current_features[:4])
        final_risk = risk_score * 0.7 + current_risk * 0.3
        confidence = min(1.0, len(feature_history) / 900.0)

        return {
            "risk_probability": min(1.0, max(0.0, final_risk)),
            "confidence": confidence,
            "trend_contribution": risk_score,
            "current_contribution": current_risk,
        }

class TrendAnalyzer:
    def analyze_trend(self, data):
        if len(data) < 10:
            return 0.0

        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        slope = coeffs[0]

        if len(data) >= 20:
            first_half = np.mean(data[: len(data) // 2])
            second_half = np.mean(data[len(data) // 2 :])
            acceleration = (second_half - first_half) / (len(data) // 2)
        else:
            acceleration = 0.0

        volatility = np.std(data) / (np.mean(data) + 1e-6)
        trend_risk = (max(0.0, slope) * 0.5 + max(0.0, acceleration) * 0.3 + min(1.0, volatility) * 0.2)
        return min(1.0, trend_risk)
