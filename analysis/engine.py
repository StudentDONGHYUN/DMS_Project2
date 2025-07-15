import asyncio
import time
from collections import deque
import logging
import numpy as np
from cachetools import cached, TTLCache
from mediapipe.framework.formats import landmark_pb2

from mediapipe import solutions
import math
from enum import Enum

from core.definitions import (
    CameraPosition, AdvancedMetrics, TimeWindowConfig, GazeZone,
    AnalysisEvent, RiskLevel, EmotionState
)
from utils.memory import MemoryManager
from systems.personalization import PersonalizationEngine
from systems.dynamic import DynamicAnalysisEngine
from systems.backup import SensorBackupManager
from analysis.gaze import GazeZoneClassifier
from analysis.drowsiness import EnhancedDrowsinessDetector
from analysis.emotion import EmotionRecognitionSystem
from analysis.distraction import DistractionObjectDetector
from analysis.identity import DriverIdentificationSystem
from analysis.prediction import PredictiveSafetySystem
from io_handler.ui import SClassAdvancedUIManager

logger = logging.getLogger(__name__)

class EnhancedAnalysisEngine:
    """대폭 향상된 분석 엔진"""

    def __init__(
        self,
        state_manager,
        user_id: str = "default",
        camera_position: CameraPosition = CameraPosition.REARVIEW_MIRROR,
        calibration_manager=None,
        enable_calibration: bool = True,
    ):
        self.state_manager = state_manager
        self.user_id = user_id
        self.latest_results = {}
        self.metrics = AdvancedMetrics()
        self.calibration_manager = calibration_manager
        self.enable_calibration = enable_calibration
        self.memory_manager = MemoryManager()
        self.personalization = PersonalizationEngine(user_id)
        self.dynamic_analyzer = DynamicAnalysisEngine()
        self.sensor_backup = SensorBackupManager()
        self.counter_analyzer = CounterBasedAnalyzer(TimeWindowConfig())
        # 시스템 사양에 따라 시선 분류 모드 자동 설정
        mode_map = {
            'HIGH_PERFORMANCE': '3d',
            'RESEARCH': '3d',
            'STANDARD': 'lut',
            'LOW_RESOURCE': 'bbox',
        }
        # system_type이 문자열 또는 Enum일 수 있음
        stype = getattr(self, 'system_type', 'STANDARD')
        if isinstance(stype, Enum):
            stype_str = stype.name
        elif isinstance(stype, str):
            stype_str = stype
        else:
            stype_str = str(stype)
        gaze_mode = mode_map.get(stype_str.upper(), 'lut')
        self.gaze_classifier = GazeZoneClassifier(mode=gaze_mode)
        self.drowsiness_detector = EnhancedDrowsinessDetector()
        self.emotion_recognizer = EmotionRecognitionSystem()
        self.distraction_detector = DistractionObjectDetector()
        self.driver_identifier = DriverIdentificationSystem()
        self.predictive_safety = PredictiveSafetySystem()
        self.multimodal_analyzer = EnhancedMultiModalAnalyzer()
        self.current_gaze_zone, self.gaze_zone_start_time = GazeZone.FRONT, time.time()
        self.frame_buffer, self.result_buffer = {}, {}
        self.processed_data_queue = deque(maxlen=5)
        self.ui_manager = SClassAdvancedUIManager()
        self.calibration_mode = enable_calibration
        if (
            self.enable_calibration
            and self.calibration_manager
            and self.calibration_manager.should_skip_calibration()
        ):
            shared_data = self.calibration_manager.get_shared_calibration_data()
            if shared_data:
                self.safe_zone = shared_data
                self.calibration_mode = False
                logger.info("공유 캘리브레이션 데이터 적용됨 - 캘리브레이션 건너뜀")

    def on_face_result(self, result, image, timestamp_ms: int):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[진단] engine.on_face_result 진입: ts={timestamp_ms}, result={type(result)}")
        if timestamp_ms not in self.result_buffer:
            self.result_buffer[timestamp_ms] = {}
        self.result_buffer[timestamp_ms]["face"] = result
        logger.info(f"[진단] engine.result_buffer keys: {list(self.result_buffer.keys())}, frame_buffer keys: {list(self.frame_buffer.keys())}")
        logger.info(f"[진단] engine.result_buffer[{timestamp_ms}]: {list(self.result_buffer[timestamp_ms].keys())}")
        self._try_queue_results(timestamp_ms)

    def on_pose_result(self, result, image, timestamp_ms: int):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[진단] engine.on_pose_result 진입: ts={timestamp_ms}, result={type(result)}")
        if timestamp_ms not in self.result_buffer:
            self.result_buffer[timestamp_ms] = {}
        self.result_buffer[timestamp_ms]["pose"] = result
        logger.info(f"[진단] engine.result_buffer keys: {list(self.result_buffer.keys())}, frame_buffer keys: {list(self.frame_buffer.keys())}")
        logger.info(f"[진단] engine.result_buffer[{timestamp_ms}]: {list(self.result_buffer[timestamp_ms].keys())}")
        self._try_queue_results(timestamp_ms)

    def on_hand_result(self, result, image, timestamp_ms: int):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[진단] engine.on_hand_result 진입: ts={timestamp_ms}, result={type(result)}")
        if timestamp_ms not in self.result_buffer:
            self.result_buffer[timestamp_ms] = {}
        self.result_buffer[timestamp_ms]["hand"] = result
        logger.info(f"[진단] engine.result_buffer keys: {list(self.result_buffer.keys())}, frame_buffer keys: {list(self.frame_buffer.keys())}")
        logger.info(f"[진단] engine.result_buffer[{timestamp_ms}]: {list(self.result_buffer[timestamp_ms].keys())}")
        self._try_queue_results(timestamp_ms)

    def on_object_result(self, result, image, timestamp_ms: int):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[진단] engine.on_object_result 진입: ts={timestamp_ms}, result={type(result)}")
        if timestamp_ms not in self.result_buffer:
            self.result_buffer[timestamp_ms] = {}
        self.result_buffer[timestamp_ms]["object"] = result
        logger.info(f"[진단] engine.result_buffer keys: {list(self.result_buffer.keys())}, frame_buffer keys: {list(self.frame_buffer.keys())}")
        logger.info(f"[진단] engine.result_buffer[{timestamp_ms}]: {list(self.result_buffer[timestamp_ms].keys())}")
        self._try_queue_results(timestamp_ms)

    def _try_queue_results(self, timestamp_ms: int):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[진단] engine._try_queue_results 진입: ts={timestamp_ms}")
        if (
            timestamp_ms in self.result_buffer
            and "face" in self.result_buffer[timestamp_ms]
            and "pose" in self.result_buffer[timestamp_ms]
        ):
            frame = self.frame_buffer.pop(timestamp_ms, None)
            results = self.result_buffer.pop(timestamp_ms, None)
            logger.info(f"[진단] engine._try_queue_results: frame={type(frame)}, results={type(results)}, keys={list(results.keys()) if results else None}")
            if results:
                for k, v in results.items():
                    logger.info(f"[진단] engine._try_queue_results: results[{k}]={type(v)}")
            if frame is not None and results is not None:
                self.processed_data_queue.append((frame, results))
                try:
                    ui_manager = SClassAdvancedUIManager()
                    import cv2
                    annotated = ui_manager.draw_enhanced_results(
                        frame,
                        self.get_latest_metrics(),
                        self.state_manager.get_current_state(),
                        results,
                        self.gaze_classifier,
                        self.dynamic_analyzer,
                        self.sensor_backup,
                        None,  # perf_stats (None 또는 실제 값)
                        None,  # playback_info (None 또는 실제 값)
                        self.driver_identifier,
                        self.predictive_safety,
                        self.emotion_recognizer,
                    )
                    logger.info("[진단] engine._try_queue_results: draw_enhanced_results 호출 완료")
                    try:
                        frame_to_show = cv2.UMat(annotated) if not isinstance(annotated, cv2.UMat) else annotated
                    except (cv2.error, TypeError) as e:
                        logger.warning(f"Frame conversion for display failed: {e}")
                        frame_to_show = annotated
                    cv2.imshow("DMS", frame_to_show)
                    logger.info("[진단] engine._try_queue_results: cv2.imshow 호출 완료")
                except (AttributeError, TypeError, ValueError, cv2.error) as e:
                    logger.error(f"[진단] engine._try_queue_results: 시각화 예외: {e}")
        logger.info("[진단] engine._try_queue_results 종료")

    def _prune_buffers(self):
        current_time_ms = int(time.time() * 1000)
        for ts in [ts for ts in self.frame_buffer if current_time_ms - ts > 2000]:
            self.frame_buffer.pop(ts, None)
            self.result_buffer.pop(ts, None)

    async def process_and_annotate_frame(self, frame, results, perf_stats, playback_info):
        timestamp = time.time()
        face_result = results.get("face")
        pose_result = results.get("pose")
        hand_result = results.get("hand")
        object_result = results.get("object")
        
        # 비동기 작업 생성 및 예외 처리 강화
        created_tasks = []
        try:
            face_task = asyncio.create_task(self._process_face_data_async(face_result, timestamp))
            pose_task = asyncio.create_task(self._process_pose_data_async(pose_result))
            hand_task = asyncio.create_task(self._process_hand_data_async(hand_result))
            created_tasks = [face_task, pose_task, hand_task]
            
            # 손 처리 완료 대기 (객체 처리에 필요)
            hand_positions = await hand_task
            
            # 객체 처리 (손 위치 정보 필요)
            await self._process_object_data_async(object_result, hand_positions, timestamp)
            
            # 나머지 얼굴 및 자세 처리 완료 대기
            await asyncio.gather(face_task, pose_task, return_exceptions=True)
            
        except (asyncio.CancelledError, AttributeError, TypeError, ValueError) as e:
            logger.error(f"비동기 작업 처리 중 오류: {e}", exc_info=True)
            # 실패한 작업들 정리
            for task in created_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception as task_error:
                        logger.error(f"작업 정리 중 오류: {task_error}", exc_info=True)
            
            # 폴백 처리 - 동기 방식으로 기본 처리
            await self._process_face_data_async(face_result, timestamp)
            await self._process_pose_data_async(pose_result)
            hand_positions = await self._process_hand_data_async(hand_result)
            await self._process_object_data_async(object_result, hand_positions, timestamp)
        
        # 나머지 처리 과정
        self._perform_multimodal_fusion_analysis(timestamp)
        self._run_predictive_analysis(timestamp)
        self._update_driver_state()
        self.memory_manager.check_and_cleanup()
        
        annotated_frame = self.ui_manager.draw_enhanced_results(
            frame,
            self.get_latest_metrics(),
            self.state_manager.get_current_state(),
            self.latest_results,
            self.gaze_classifier,
            self.dynamic_analyzer,
            self.sensor_backup,
            perf_stats,
            playback_info,
            self.driver_identifier,
            self.predictive_safety,
            self.emotion_recognizer,
        )
        return annotated_frame

    @cached(cache=TTLCache(maxsize=5, ttl=300))
    def _cached_identify_driver(self, landmarks_tuple):
        landmarks = [landmark_pb2.NormalizedLandmark(x=lm[0], y=lm[1], z=lm[2]) for lm in landmarks_tuple]
        return self.driver_identifier.identify_driver(landmarks)

    async def _process_face_data_async(self, face_result, timestamp):
        if not face_result or not face_result.face_landmarks:
            self.latest_results.pop("face", None)
            self.sensor_backup.activate_backup("face_backup_active")
            return
        self.latest_results["face"] = face_result
        self.sensor_backup.deactivate_backup("face_backup_active")
        landmarks = face_result.face_landmarks[0]
        drowsiness_result = self.drowsiness_detector.detect_drowsiness(landmarks, timestamp)
        self.metrics.enhanced_ear = drowsiness_result["enhanced_ear"]
        self.metrics.perclos = drowsiness_result["perclos"]
        self.metrics.temporal_attention_score = drowsiness_result["temporal_attention_score"]
        self.metrics.personalized_threshold = drowsiness_result["threshold"]
        if face_result.face_blendshapes:
            emotion_result = self.emotion_recognizer.analyze_emotion(face_result.face_blendshapes[0].categories, timestamp)
            self.metrics.emotion_state = emotion_result["emotion"]
            self.metrics.emotion_confidence = emotion_result["confidence"]
            self.metrics.arousal_level = emotion_result["arousal"]
            self.metrics.valence_level = emotion_result["valence"]
            self._analyze_blendshapes(face_result.face_blendshapes[0].categories)
        landmarks_tuple = tuple((lm.x, lm.y, lm.z) for lm in landmarks)
        driver_info = self._cached_identify_driver(landmarks_tuple)
        self.metrics.driver_identity = driver_info["driver_id"]
        self.metrics.driver_confidence = driver_info["confidence"]
        if face_result.facial_transformation_matrixes:
            self._analyze_enhanced_head_pose(face_result.facial_transformation_matrixes[0])

    async def _process_pose_data_async(self, pose_result):
        if not pose_result or not pose_result.pose_landmarks:
            self.latest_results.pop("pose", None)
            return
        self.latest_results["pose"] = pose_result
        if pose_result.pose_world_landmarks:
            self._analyze_enhanced_pose(pose_result.pose_world_landmarks[0])

    async def _process_hand_data_async(self, hand_result):
        hand_positions = []
        if not hand_result or not hand_result.hand_landmarks:
            self.latest_results.pop("hand", None)
            self.sensor_backup.activate_backup("hand_backup_active")
        else:
            self.latest_results["hand"] = hand_result
            self.sensor_backup.deactivate_backup("hand_backup_active")
            for i, h_lm in enumerate(hand_result.hand_landmarks):
                handedness = hand_result.handedness[i][0].category_name
                wrist = h_lm[0]
                hand_positions.append({"handedness": handedness, "x": wrist.x, "y": wrist.y, "landmarks": h_lm})
        self.latest_results["hand_positions"] = hand_positions
        return hand_positions

    async def _process_object_data_async(self, object_result, hand_positions, timestamp):
        if not object_result or not object_result.detections:
            self.latest_results.pop("object", None)
            self.metrics.phone_detected = False
            self.metrics.distraction_objects = []
            return
        self.latest_results["object"] = object_result
        distraction_analysis = self.distraction_detector.analyze_detections(object_result, hand_positions, timestamp)
        self.metrics.distraction_objects = [obj["description"] for obj in distraction_analysis["detected_objects"]]
        self.metrics.phone_detected = any("휴대폰" in obj for obj in self.metrics.distraction_objects)

    def _perform_multimodal_fusion_analysis(self, timestamp: float):
        face_data = {
            "available": "face" in self.latest_results,
            "perclos": self.metrics.perclos,
            "enhanced_ear": self.metrics.enhanced_ear,
            "temporal_attention_score": self.metrics.temporal_attention_score,
            "gaze_deviation_score": self._calculate_enhanced_gaze_deviation_score(),
            "attention_focus_score": self.gaze_classifier.get_attention_focus_score(),
        }
        pose_data = {
            "available": "pose" in self.latest_results,
            "head_nod_score": min(1.0, self.counter_analyzer.get_event_counts()["head_nods_2min"] / 5.0),
            "pose_complexity_score": self.metrics.pose_complexity_score,
        }
        hand_data = {
            "available": "hand" in self.latest_results,
            "hands_on_wheel_confidence": self._calculate_hands_on_wheel_confidence(),
        }
        object_data = {
            "available": "object" in self.latest_results,
            "distraction_score": len(self.metrics.distraction_objects) / 5.0,
            "phone_usage_score": 1.0 if self.metrics.phone_detected else 0.0,
        }
        emotion_data = {
            "available": self.metrics.emotion_confidence > 0.5,
            "emotion": self.metrics.emotion_state,
            "confidence": self.metrics.emotion_confidence,
            "arousal": self.metrics.arousal_level,
            "valence": self.metrics.valence_level,
        }
        self.metrics.fatigue_risk_score = self.multimodal_analyzer.fuse_drowsiness_signals(face_data, pose_data, emotion_data)
        self.metrics.distraction_risk_score = self.multimodal_analyzer.fuse_distraction_signals(face_data, hand_data, object_data, emotion_data)
        self.metrics.attention_focus_score = face_data["attention_focus_score"]

    def _run_predictive_analysis(self, timestamp: float):
        prediction_result = self.predictive_safety.predict_risk(self.metrics, timestamp)
        self.metrics.predictive_risk_score = prediction_result["risk_probability"]
        if prediction_result["alert_level"] in ["high", "critical"]:
            self.state_manager.handle_event(AnalysisEvent.PREDICTIVE_RISK_HIGH)

    def _analyze_blendshapes(self, blendshapes_list):
        blendshapes = {cat.category_name: cat.score for cat in blendshapes_list}
        self.metrics.yawn_score = blendshapes.get("jawOpen", 0.0)
        self.metrics.left_eye_closure = blendshapes.get("eyeBlinkLeft", 0.0)
        self.metrics.right_eye_closure = blendshapes.get("eyeBlinkRight", 0.0)
        if self.metrics.yawn_score > 0.6:
            self.counter_analyzer.add_event("yawns")
        if (self.metrics.left_eye_closure + self.metrics.right_eye_closure) / 2 > 0.8:
            self.counter_analyzer.add_event("blinks")

    def _analyze_enhanced_head_pose(self, transform_matrix):
        if transform_matrix is None:
            self.latest_results.update({"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0})
            self.metrics.current_gaze_zone = GazeZone.FRONT
            return
        R = np.array(transform_matrix.data).reshape(4, 4)[:3, :3]
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        head_yaw, head_pitch, head_roll = -math.degrees(y), -math.degrees(x), math.degrees(z)
        self.latest_results.update({"head_yaw": head_yaw, "head_pitch": head_pitch, "head_roll": head_roll})
        new_gaze_zone = self.gaze_classifier.classify(head_yaw, head_pitch, time.time())
        if new_gaze_zone != self.current_gaze_zone:
            self.current_gaze_zone, self.gaze_zone_start_time = new_gaze_zone, time.time()
            self.metrics.gaze_zone_duration = 0.0
        else:
            self.metrics.gaze_zone_duration = time.time() - self.gaze_zone_start_time
        self.metrics.current_gaze_zone = self.current_gaze_zone
        self.metrics.head_yaw = head_yaw
        self.metrics.head_pitch = head_pitch
        self.metrics.head_roll = head_roll

    def _analyze_enhanced_pose(self, world_landmarks):
        if not world_landmarks or len(world_landmarks) < 33:
            return
        ls, rs = world_landmarks[11], world_landmarks[12]
        vec = np.array([ls.x - rs.x, ls.z - rs.z])
        if np.linalg.norm(vec) > 0:
            self.latest_results["shoulder_yaw"] = np.degrees(np.arctan2(vec[0], -vec[1]))
        torso_landmarks = [world_landmarks[i] for i in [11, 12, 23, 24]]
        pose_variance = self._calculate_pose_variance(torso_landmarks)
        self.metrics.pose_complexity_score = min(1.0, pose_variance * 10)
        hip_center = [(world_landmarks[23].x + world_landmarks[24].x) / 2, (world_landmarks[23].y + world_landmarks[24].y) / 2]
        shoulder_center = [(world_landmarks[11].x + world_landmarks[12].x) / 2, (world_landmarks[11].y + world_landmarks[12].y) / 2]
        torso_angle = math.degrees(math.atan2(shoulder_center[1] - hip_center[1], abs(shoulder_center[0] - hip_center[0]) + 0.1))
        self.latest_results["slouch_factor"] = max(0.0, min(1.0, (90 - abs(torso_angle)) / 90))

    def _calculate_pose_variance(self, landmarks):
        if len(landmarks) < 2:
            return 0.0
        positions = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        return np.var(positions)

    def _calculate_enhanced_gaze_deviation_score(self) -> float:
        base_score = 0.0
        if abs(self.metrics.head_yaw) > 60.0:
            base_score = min(1.0, 0.5 + self.metrics.gaze_zone_duration / 2.0)
        gaze_stability = self.gaze_classifier.get_gaze_stability()
        instability_penalty = (1.0 - gaze_stability) * 0.3
        zone_weights = {
            GazeZone.FRONT: 0.0, GazeZone.REARVIEW_MIRROR: 0.2, GazeZone.LEFT_SIDE_MIRROR: 0.2,
            GazeZone.RIGHT_SIDE_MIRROR: 0.2, GazeZone.INSTRUMENT_CLUSTER: 0.1, GazeZone.CENTER_STACK: 0.4,
            GazeZone.FLOOR: 0.8, GazeZone.ROOF: 0.6, GazeZone.PASSENGER: 0.7, GazeZone.DRIVER_WINDOW: 0.5,
            GazeZone.BLIND_SPOT_LEFT: 0.9,
        }
        zone_weight = zone_weights.get(self.current_gaze_zone, 0.5)
        duration_factor = min(1.0, self.metrics.gaze_zone_duration / 3.0)
        final_score = max(base_score, zone_weight * duration_factor) + instability_penalty
        return min(1.0, final_score)

    def _calculate_hands_on_wheel_confidence(self) -> float:
        hand_positions = self.latest_results.get("hand_positions", [])
        if not hand_positions:
            return 0.0
        wheel_zone = {"x1": 0.3, "y1": 0.4, "x2": 0.7, "y2": 0.8}
        hands_on_wheel = 0
        for hand in hand_positions:
            x, y = hand.get("x", 0.5), hand.get("y", 0.5)
            if wheel_zone["x1"] <= x <= wheel_zone["x2"] and wheel_zone["y1"] <= y <= wheel_zone["y2"]:
                hands_on_wheel += 1
        return hands_on_wheel / 2.0

    def _update_driver_state(self):
        self.metrics.overall_risk_level = self._determine_enhanced_overall_risk()
        self._check_enhanced_state_transitions()
        event_counts = self.counter_analyzer.get_event_counts()
        self.metrics.blink_count_1min = event_counts["blinks_1min"]
        self.metrics.yawn_count_5min = event_counts["yawns_5min"]
        self.metrics.head_nod_count_2min = event_counts["head_nods_2min"]
        self.metrics.gaze_deviation_count_1min = event_counts["gaze_deviations_1min"]

    def _determine_enhanced_overall_risk(self) -> RiskLevel:
        combined_risk = max(self.metrics.fatigue_risk_score, self.metrics.distraction_risk_score)
        predictive_weight = 0.3
        combined_risk = combined_risk * (1 - predictive_weight) + self.metrics.predictive_risk_score * predictive_weight
        if self.metrics.emotion_state == EmotionState.STRESS and self.metrics.emotion_confidence > 0.7:
            combined_risk = min(1.0, combined_risk + 0.2)
        if self.metrics.pose_complexity_score > 0.7:
            combined_risk = min(1.0, combined_risk + 0.1)
        if abs(self.metrics.head_roll) > 25.0 and self.metrics.gaze_zone_duration > 1.0:
            combined_risk = max(combined_risk, 0.7)
        if combined_risk > 0.8:
            return RiskLevel.CRITICAL
        if combined_risk > 0.6:
            return RiskLevel.HIGH
        if combined_risk > 0.4:
            return RiskLevel.MEDIUM
        if combined_risk > 0.2:
            return RiskLevel.LOW
        return RiskLevel.SAFE

    def _check_enhanced_state_transitions(self):
        if self.metrics.fatigue_risk_score > 0.8:
            self.state_manager.handle_event(AnalysisEvent.FATIGUE_ACCUMULATION)
        elif self.metrics.distraction_risk_score > 0.7:
            self.state_manager.handle_event(AnalysisEvent.ATTENTION_DECLINE)
        if self.metrics.emotion_state == EmotionState.STRESS and self.metrics.emotion_confidence > 0.7:
            self.state_manager.handle_event(AnalysisEvent.EMOTION_STRESS_DETECTED)
        if self.metrics.distraction_objects:
            self.state_manager.handle_event(AnalysisEvent.DISTRACTION_OBJECT_DETECTED)
        if not any([self.metrics.fatigue_risk_score > 0.5, self.metrics.distraction_risk_score > 0.5, self.metrics.predictive_risk_score > 0.5]):
            self.state_manager.handle_event(AnalysisEvent.NORMAL_BEHAVIOR)

    def _process_backup_face_data(self, backup_data):
        if "head_pose" in backup_data:
            self._analyze_head_pose_from_backup(backup_data["head_pose"])

    def _analyze_head_pose_from_backup(self, head_pose):
        self.latest_results.update({"head_yaw": head_pose["yaw"], "head_pitch": head_pose["pitch"], "head_roll": head_pose["roll"]})
        new_gaze_zone = self.gaze_classifier.classify(head_pose["yaw"], head_pose["pitch"])
        if new_gaze_zone != self.current_gaze_zone:
            self.current_gaze_zone, self.gaze_zone_start_time = new_gaze_zone, time.time()
        self.metrics.current_gaze_zone = new_gaze_zone

    def get_latest_metrics(self) -> AdvancedMetrics:
        return self.metrics

class CounterBasedAnalyzer:
    def __init__(self, config: TimeWindowConfig):
        self.event_buffers = {
            "blinks": deque(maxlen=300),
            "yawns": deque(maxlen=100),
            "head_nods": deque(maxlen=120),
            "gaze_deviations": deque(maxlen=60),
        }

    def add_event(self, event_type: str):
        if event_type in self.event_buffers:
            self.event_buffers[event_type].append(time.time())

    def get_event_counts(self) -> dict:
        now = time.time()
        return {
            "blinks_1min": sum(1 for ts in self.event_buffers["blinks"] if now - ts <= 60),
            "yawns_5min": sum(1 for ts in self.event_buffers["yawns"] if now - ts <= 300),
            "head_nods_2min": sum(1 for ts in self.event_buffers["head_nods"] if now - ts <= 120),
            "gaze_deviations_1min": sum(1 for ts in self.event_buffers["gaze_deviations"] if now - ts <= 60),
        }

class EnhancedMultiModalAnalyzer:
    """향상된 멀티모달 분석기 - 0값 문제 해결"""
    
    def __init__(self):
        self.weights = {"face": 0.35, "pose": 0.25, "hand": 0.20, "object": 0.10, "emotion": 0.10}
        # 디버깅을 위한 로거 추가
        logger.info("EnhancedMultiModalAnalyzer 초기화 완료")

    def fuse_drowsiness_signals(self, face_data, pose_data, emotion_data) -> float:
        score, total_weight = 0.0, 0.0
        
        # 디버깅 로그 추가
        logger.debug(f"Drowsiness fusion input - Face: {face_data}, Pose: {pose_data}, Emotion: {emotion_data}")

        # 1. 얼굴 데이터 처리 (조건 완화)
        face_available = face_data.get("available", True)  # 기본값을 True로 변경
        if face_available or any(face_data.get(key, 0) > 0.001 for key in ["perclos", "enhanced_ear", "temporal_attention_score"]):
            face_drowsiness = (
                face_data.get("perclos", 0.0) * 0.4 + 
                face_data.get("enhanced_ear", 0.0) * 0.3 + 
                face_data.get("temporal_attention_score", 0.0) * 0.3
            )
            
            # 최소 임계값 적용 - 너무 작은 값도 의미있게 처리
            if face_drowsiness > 0.001:
                score += face_drowsiness * self.weights["face"]
                total_weight += self.weights["face"]
                logger.debug(f"Face drowsiness added: {face_drowsiness}")

        # 2. 자세 데이터 처리 (조건 완화)
        pose_available = pose_data.get("available", True)  # 기본값을 True로 변경
        if pose_available or pose_data.get("head_nod_score", 0) > 0.001:
            pose_drowsiness = pose_data.get("head_nod_score", 0.0)
            
            # 자세 기반 피로 신호 강화
            pose_complexity = pose_data.get("pose_complexity_score", 0.0)
            head_stability = max(0.0, 1.0 - pose_complexity) * 0.3  # 자세 불안정성도 피로 신호
            pose_drowsiness = max(pose_drowsiness, head_stability)
            
            if pose_drowsiness > 0.001:
                score += pose_drowsiness * self.weights["pose"]
                total_weight += self.weights["pose"]
                logger.debug(f"Pose drowsiness added: {pose_drowsiness}")

        # 3. 감정 데이터 처리 (조건 대폭 완화)
        emotion_available = emotion_data.get("available", False)
        emotion_confidence = emotion_data.get("confidence", 0.0)
        
        if emotion_available or emotion_confidence > 0.1:  # 임계값 대폭 낮춤 (0.5 → 0.1)
            emotion_fatigue = 0.0
            
            # 피로 관련 감정 상태 고려
            emotion_state = emotion_data.get("emotion")
            if emotion_state == EmotionState.FATIGUE:
                emotion_fatigue = emotion_confidence
            elif emotion_data.get("arousal", 0.5) < 0.4:  # 임계값 완화 (0.3 → 0.4)
                arousal = emotion_data.get("arousal", 0.5)
                emotion_fatigue = (0.4 - arousal) * 2.5  # 스케일링 조정
            
            if emotion_fatigue > 0.001:
                score += emotion_fatigue * self.weights["emotion"]
                total_weight += self.weights["emotion"]
                logger.debug(f"Emotion fatigue added: {emotion_fatigue}")

        # 4. 폴백 메커니즘 - 모든 데이터가 없으면 기본 분석 수행
        if total_weight == 0:
            logger.warning("No modalities available for drowsiness fusion - using fallback")
            # 기본적인 분석만으로라도 결과 제공
            fallback_score = self._calculate_fallback_drowsiness(face_data, pose_data, emotion_data)
            logger.debug(f"Fallback drowsiness score: {fallback_score}")
            return fallback_score
        
        final_score = score / total_weight
        logger.debug(f"Final drowsiness score: {final_score} (total_weight: {total_weight})")
        
        return final_score

    def fuse_distraction_signals(self, face_data, hand_data, object_data, emotion_data) -> float:
        score, total_weight = 0.0, 0.0
        
        # 디버깅 로그 추가
        logger.debug(f"Distraction fusion input - Face: {face_data}, Hand: {hand_data}, Object: {object_data}, Emotion: {emotion_data}")

        # 1. 얼굴/시선 데이터 처리 (조건 완화)
        face_available = face_data.get("available", True)  # 기본값을 True로 변경
        if face_available or any(face_data.get(key, 0) > 0.001 for key in ["gaze_deviation_score", "attention_focus_score"]):
            gaze_score = face_data.get("gaze_deviation_score", 0.0)
            attention_score = 1.0 - face_data.get("attention_focus_score", 1.0)
            face_distraction = max(gaze_score, attention_score)
            
            if face_distraction > 0.001:
                score += face_distraction * self.weights["face"]
                total_weight += self.weights["face"]
                logger.debug(f"Face distraction added: {face_distraction}")

        # 2. 손 데이터 처리 (조건 완화)
        hand_available = hand_data.get("available", True)  # 기본값을 True로 변경
        if hand_available or hand_data.get("hands_on_wheel_confidence") is not None:
            hands_on_wheel_confidence = hand_data.get("hands_on_wheel_confidence", 0.5)  # 기본값 0.5
            hand_distraction = max(0.0, 1.0 - hands_on_wheel_confidence)
            
            if hand_distraction > 0.001:
                score += hand_distraction * self.weights["hand"]
                total_weight += self.weights["hand"]
                logger.debug(f"Hand distraction added: {hand_distraction}")

        # 3. 객체 데이터 처리 (조건 완화)
        object_available = object_data.get("available", True)  # 기본값을 True로 변경
        if object_available or any(object_data.get(key, 0) > 0.001 for key in ["distraction_score", "phone_usage_score"]):
            object_distraction = object_data.get("distraction_score", 0.0)
            phone_usage = object_data.get("phone_usage_score", 0.0)
            combined_object_distraction = max(object_distraction, phone_usage)
            
            if combined_object_distraction > 0.001:
                score += combined_object_distraction * self.weights["object"]
                total_weight += self.weights["object"]
                logger.debug(f"Object distraction added: {combined_object_distraction}")

        # 4. 감정 데이터 처리 (조건 대폭 완화)
        emotion_available = emotion_data.get("available", False)
        emotion_confidence = emotion_data.get("confidence", 0.0)
        
        if emotion_available or emotion_confidence > 0.1:  # 임계값 대폭 낮춤
            emotion_distraction = 0.0
            emotion_state = emotion_data.get("emotion")
            
            # 스트레스나 분노 상태는 주의 분산 위험 증가
            if emotion_state in [EmotionState.STRESS, EmotionState.ANGER]:
                emotion_distraction = emotion_confidence
            elif emotion_confidence > 0.3:  # 기본적인 감정 불안정성도 고려
                emotion_distraction = emotion_confidence * 0.5
            
            if emotion_distraction > 0.001:
                score += emotion_distraction * self.weights["emotion"]
                total_weight += self.weights["emotion"]
                logger.debug(f"Emotion distraction added: {emotion_distraction}")

        # 5. 폴백 메커니즘 - 모든 데이터가 없으면 기본 분석 수행
        if total_weight == 0:
            logger.warning("No modalities available for distraction fusion - using fallback")
            fallback_score = self._calculate_fallback_distraction(face_data, hand_data, object_data, emotion_data)
            logger.debug(f"Fallback distraction score: {fallback_score}")
            return fallback_score
        
        final_score = score / total_weight
        logger.debug(f"Final distraction score: {final_score} (total_weight: {total_weight})")
        
        return final_score

    def _calculate_fallback_drowsiness(self, face_data, pose_data, emotion_data) -> float:
        """모든 모달리티가 사용 불가능할 때의 폴백 졸음 분석"""
        fallback_signals = []
        
        # 기본적인 얼굴 신호들 확인
        if face_data:
            if face_data.get("perclos", 0) > 0:
                fallback_signals.append(face_data["perclos"])
            if face_data.get("enhanced_ear", 0) > 0:
                fallback_signals.append(1.0 - face_data["enhanced_ear"])
        
        # 기본적인 자세 신호들 확인  
        if pose_data:
            if pose_data.get("head_nod_score", 0) > 0:
                fallback_signals.append(pose_data["head_nod_score"])
            if pose_data.get("pose_complexity_score", 0) > 0:
                fallback_signals.append(pose_data["pose_complexity_score"] * 0.5)
        
        # 기본적인 감정 신호들 확인
        if emotion_data:
            arousal = emotion_data.get("arousal", 0.5)
            if arousal < 0.5:  # 낮은 각성도는 피로 신호
                fallback_signals.append((0.5 - arousal) * 2.0)
        
        # 신호가 있으면 평균값 반환, 없으면 0.0
        if fallback_signals:
            result = sum(fallback_signals) / len(fallback_signals)
            logger.info(f"Fallback drowsiness signals found: {len(fallback_signals)}, average: {result}")
            return min(1.0, result)
        else:
            logger.info("No fallback drowsiness signals available")
            return 0.0

    def _calculate_fallback_distraction(self, face_data, hand_data, object_data, emotion_data) -> float:
        """모든 모달리티가 사용 불가능할 때의 폴백 주의산만 분석"""
        fallback_signals = []
        
        # 기본적인 시선 신호들 확인
        if face_data:
            if face_data.get("gaze_deviation_score", 0) > 0:
                fallback_signals.append(face_data["gaze_deviation_score"])
            attention_focus = face_data.get("attention_focus_score", 1.0)
            if attention_focus < 1.0:
                fallback_signals.append(1.0 - attention_focus)
        
        # 기본적인 손 신호들 확인
        if hand_data:
            hands_confidence = hand_data.get("hands_on_wheel_confidence", 0.5)
            if hands_confidence < 1.0:
                fallback_signals.append(1.0 - hands_confidence)
        
        # 기본적인 객체 신호들 확인
        if object_data:
            distraction_score = object_data.get("distraction_score", 0.0)
            if distraction_score > 0:
                fallback_signals.append(distraction_score)
            phone_usage = object_data.get("phone_usage_score", 0.0)
            if phone_usage > 0:
                fallback_signals.append(phone_usage)
        
        # 기본적인 감정 신호들 확인
        if emotion_data:
            emotion_state = emotion_data.get("emotion")
            if emotion_state in [EmotionState.STRESS, EmotionState.ANGER]:
                confidence = emotion_data.get("confidence", 0.0)
                if confidence > 0:
                    fallback_signals.append(confidence)
        
        # 신호가 있으면 평균값 반환, 없으면 0.0
        if fallback_signals:
            result = sum(fallback_signals) / len(fallback_signals)
            logger.info(f"Fallback distraction signals found: {len(fallback_signals)}, average: {result}")
            return min(1.0, result)
        else:
            logger.info("No fallback distraction signals available")
            return 0.0