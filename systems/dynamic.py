import logging
import time

logger = logging.getLogger(__name__)

class DynamicAnalysisEngine:
    """동적 상황인지 분석 엔진"""

    def __init__(self):
        self.vehicle_objects = {
            "steering_wheel": {"x1": 0.3, "y1": 0.4, "x2": 0.7, "y2": 0.8},
            "dashboard_area": {"x1": 0.2, "y1": 0.0, "x2": 0.8, "y2": 0.3},
            "gear_lever": {"x1": 0.4, "y1": 0.7, "x2": 0.6, "y2": 0.9},
            "side_mirror_left": {"x1": 0.0, "y1": 0.2, "x2": 0.2, "y2": 0.4},
        }
        self.reset()
        logger.info("DynamicAnalysisEngine 초기화 완료")

    async def initialize(self):
        """
        비동기 초기화: 동적 분석 상태 리셋 및 향후 비동기 확장 고려
        """
        self.reset()
        logger.info("DynamicAnalysisEngine for async initialized.")

    def reset(self):
        self.analysis_mode = "primary"
        self.trigger_durations = {"face_lost": 0.0, "pose_lost": 0.0, "hand_out": 0.0}
        self.last_detection_times = {
            "face": time.time(),
            "pose": time.time(),
            "hand": time.time(),
        }
        self.interaction_states = {}
        logger.info("DynamicAnalysisEngine 상태 초기화됨.")

    def should_expand_analysis(self, face_available, pose_available, hand_positions):
        current_time = time.time()

        self.trigger_durations["face_lost"] = (
            0 if face_available else current_time - self.last_detection_times["face"]
        )
        if face_available:
            self.last_detection_times["face"] = current_time

        self.trigger_durations["pose_lost"] = (
            0 if pose_available else current_time - self.last_detection_times["pose"]
        )
        if pose_available:
            self.last_detection_times["pose"] = current_time

        hands_in_bounds = self._check_hands_in_bounds(hand_positions)
        self.trigger_durations["hand_out"] = (
            0 if hands_in_bounds else current_time - self.last_detection_times["hand"]
        )
        if hands_in_bounds:
            self.last_detection_times["hand"] = current_time

        expand_needed = (
            self.trigger_durations["face_lost"] > 2.0
            or self.trigger_durations["pose_lost"] > 2.0
            or self.trigger_durations["hand_out"] > 1.0
        )

        if expand_needed and self.analysis_mode == "primary":
            self.analysis_mode = "expanded"
            logger.info("확장 분석 모드 활성화")
        elif not expand_needed and self.analysis_mode == "expanded":
            self.analysis_mode = "primary"
            logger.info("기본 분석 모드 복귀")

        return expand_needed

    def _check_hands_in_bounds(self, hand_positions):
        if not hand_positions:
            return False
        wheel = self.vehicle_objects["steering_wheel"]
        return any(
            wheel["x1"] <= h.get("x", 0) <= wheel["x2"]
            and wheel["y1"] <= h.get("y", 0) <= wheel["y2"]
            for h in hand_positions
        )

    def analyze_body_object_interaction(self, hand_positions, object_detections):
        interactions = []
        if not hand_positions:
            return interactions

        current_interactions = set()
        for hand_pos in hand_positions:
            x, y = hand_pos.get("x", 0.5), hand_pos.get("y", 0.5)
            for obj_name, bounds in self.vehicle_objects.items():
                if (
                    bounds["x1"] <= x <= bounds["x2"]
                    and bounds["y1"] <= y <= bounds["y2"]
                ):
                    key = f"hand_with_{obj_name}"
                    current_interactions.add(key)
                    if key not in self.interaction_states:
                        self.interaction_states[key] = {
                            "start_time": time.time(),
                            "duration": 0.0,
                        }

                    self.interaction_states[key]["duration"] = (
                        time.time() - self.interaction_states[key]["start_time"]
                    )
                    interactions.append(
                        {
                            "type": key,
                            "duration": self.interaction_states[key]["duration"],
                        }
                    )

        expired_keys = set(self.interaction_states.keys()) - current_interactions
        for key in expired_keys:
            del self.interaction_states[key]

        return interactions

    def update_performance_metrics(self, processing_time=None, fps=None, **kwargs):
        """프레임 처리 시간, FPS 등 성능 메트릭을 기록/갱신"""
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {'processing_times': [], 'fps': []}
        if processing_time is not None:
            self.performance_metrics['processing_times'].append(processing_time)
            # 최근 100개만 유지
            self.performance_metrics['processing_times'] = self.performance_metrics['processing_times'][-100:]
        if fps is not None:
            self.performance_metrics['fps'].append(fps)
            self.performance_metrics['fps'] = self.performance_metrics['fps'][-100:]

    def get_analysis_mode(self):
        return self.analysis_mode

    def get_trigger_status(self):
        return self.trigger_durations.copy()
