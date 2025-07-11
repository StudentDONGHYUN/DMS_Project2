import cv2
import numpy as np
import math
from mediapipe.python.solutions import drawing_utils as mp_drawing, face_mesh as mp_face_mesh, pose as mp_pose
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from core.definitions import DriverState

class EnhancedUIManager:
    """대폭 향상된 UI 관리자"""

    def __init__(self):
        self.colors = {
            "safe": (0, 200, 0),
            "low_risk": (0, 255, 255),
            "medium_risk": (0, 165, 255),
            "high_risk": (0, 100, 255),
            "critical": (0, 0, 255),
            "text": (255, 255, 255),
            "backup": (0, 165, 255),
            "calibration": (255, 255, 0),
            "emotion_positive": (0, 255, 0),
            "emotion_negative": (0, 0, 255),
            "emotion_neutral": (128, 128, 128),
            "prediction_warning": (255, 165, 0),
            "driver_identified": (255, 255, 0),
        }

    def draw_enhanced_results(self, frame, metrics, state, results, gaze_classifier, dynamic_analyzer, sensor_backup, perf_stats, playback_info, driver_identifier, predictive_safety, emotion_recognizer):
        annotated_frame = frame.copy()
        self._draw_enhanced_status_info(annotated_frame, metrics, state)
        self._draw_emotion_status(annotated_frame, metrics)
        self._draw_driver_identity(annotated_frame, metrics)
        self._draw_predictive_warnings(annotated_frame, metrics)
        self._draw_enhanced_gaze_analysis(annotated_frame, metrics, gaze_classifier)
        self._draw_distraction_objects(annotated_frame, metrics, results)
        self._draw_dynamic_analysis_status(annotated_frame, dynamic_analyzer, metrics)
        self._draw_backup_status(annotated_frame, sensor_backup)
        self._draw_enhanced_state_alerts(annotated_frame, state, metrics)
        self._draw_enhanced_performance_info(annotated_frame, perf_stats, playback_info)
        self._draw_landmarks(annotated_frame, results)
        return annotated_frame

    def _draw_enhanced_status_info(self, frame, metrics, state):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        state_color = self.colors.get(state.name.lower().split("_")[0], self.colors["text"])
        risk_color = self.colors.get(metrics.overall_risk_level.name.lower(), self.colors["text"])
        y, font_scale, thickness = 25, 0.5, 1
        cv2.putText(frame, f"Driver State: {state.value}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        y += 25
        cv2.putText(frame, f"Risk Level: {metrics.overall_risk_level.name}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, risk_color, thickness)
        y += 20
        cv2.putText(frame, f"Enhanced EAR: {metrics.enhanced_ear:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.colors["text"], thickness)
        y += 18
        cv2.putText(frame, f"PERCLOS: {metrics.perclos:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.colors["text"], thickness)
        y += 18
        cv2.putText(frame, f"Temporal Attention: {metrics.temporal_attention_score:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.colors["text"], thickness)
        y += 18
        cv2.putText(frame, f"Personal Threshold: {metrics.personalized_threshold:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.colors["text"], thickness)
        y += 20
        cv2.putText(frame, f"Gaze: {metrics.current_gaze_zone.name} ({metrics.gaze_zone_duration:.1f}s)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.colors["text"], thickness)
        y += 18
        cv2.putText(frame, f"Focus Score: {metrics.attention_focus_score:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.colors["text"], thickness)
        y += 20
        cv2.putText(frame, f"Head: Y{metrics.head_yaw:.1f} P{metrics.head_pitch:.1f} R{metrics.head_roll:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, risk_color, thickness)
        y += 18
        cv2.putText(frame, f"Pose Complexity: {metrics.pose_complexity_score:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.colors["text"], thickness)

    def _draw_emotion_status(self, frame, metrics):
        if metrics.emotion_confidence < 0.3: return
        emotion_color = self.colors["emotion_neutral"]
        if metrics.valence_level > 0.6: emotion_color = self.colors["emotion_positive"]
        elif metrics.valence_level < 0.4: emotion_color = self.colors["emotion_negative"]
        x_start, y_start = frame.shape[1] - 200, 30
        cv2.putText(frame, f"Emotion: {metrics.emotion_state.value}", (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 2)
        cv2.putText(frame, f"Confidence: {metrics.emotion_confidence:.2f}", (x_start, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors["text"], 1)
        cv2.putText(frame, f"Arousal: {metrics.arousal_level:.2f}", (x_start, y_start + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors["text"], 1)
        cv2.putText(frame, f"Valence: {metrics.valence_level:.2f}", (x_start, y_start + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors["text"], 1)

    def _draw_driver_identity(self, frame, metrics):
        if metrics.driver_confidence < 0.5: return
        x_start, y_start = frame.shape[1] - 200, 100
        cv2.putText(frame, f"Driver: {metrics.driver_identity}", (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["driver_identified"], 2)
        cv2.putText(frame, f"ID Confidence: {metrics.driver_confidence:.2f}", (x_start, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors["text"], 1)

    def _draw_predictive_warnings(self, frame, metrics):
        if metrics.predictive_risk_score < 0.4: return
        h, w = frame.shape[:2]
        warning_text = f"PREDICTIVE RISK: {metrics.predictive_risk_score:.2f}"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        x, y = (w - text_size[0]) // 2, 50
        cv2.rectangle(frame, (x - 10, y - 25), (x + text_size[0] + 10, y + 5), (0, 0, 0), -1)
        warning_color = self.colors["prediction_warning"]
        if metrics.predictive_risk_score > 0.7: warning_color = self.colors["critical"]
        cv2.putText(frame, warning_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2)

    def _draw_enhanced_gaze_analysis(self, frame, metrics, gaze_classifier):
        h, w, _ = frame.shape
        center, radius = (w - 120, 150), 90
        stability = gaze_classifier.get_gaze_stability()
        stability_color = self.colors["safe"] if stability > 0.7 else self.colors["medium_risk"]
        yaw, pitch = math.radians(-metrics.head_yaw), math.radians(metrics.head_pitch)
        Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
        R = Ry @ Rx
        for i in range(-90, 91, 30):
            points = []
            for j in range(-180, 181, 15):
                lat, lon = math.radians(i), math.radians(j)
                x, y, z = radius * math.cos(lat) * math.cos(lon), radius * math.cos(lat) * math.sin(lon), radius * math.sin(lat)
                p = R @ [x, y, z]
                points.append((int(center[0] + p[0]), int(center[1] - p[1])))
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, stability_color, 1)
        cv2.putText(frame, f"Stability: {stability:.2f}", (center[0] - 60, center[1] + radius + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, stability_color, 1)

    def _draw_distraction_objects(self, frame, metrics, results):
        if not metrics.distraction_objects: return
        h, w = frame.shape[:2]
        y_start = h - 100
        cv2.putText(frame, "Distraction Objects:", (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["high_risk"], 2)
        for i, obj in enumerate(metrics.distraction_objects[:3]):
            cv2.putText(frame, f"- {obj}", (10, y_start + 20 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors["text"], 1)
        if "object" in results and results["object"].detections:
            for detection in results["object"].detections:
                bbox = detection.bounding_box
                category, confidence = detection.categories[0].category_name, detection.categories[0].score
                x1, y1, x2, y2 = int(bbox.origin_x * w), int(bbox.origin_y * h), int((bbox.origin_x + bbox.width) * w), int((bbox.origin_y + bbox.height) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["high_risk"], 2)
                cv2.putText(frame, f"{category}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["high_risk"], 2)

    def _draw_dynamic_analysis_status(self, frame, dynamic_analyzer, metrics):
        mode = dynamic_analyzer.get_analysis_mode()
        text = f"Analysis: {mode.upper()}"
        color = self.colors.get(metrics.overall_risk_level.name.lower(), self.colors["text"]) if mode == "expanded" else self.colors["safe"]
        h, w = frame.shape[:2]
        cv2.putText(frame, text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_backup_status(self, frame, sensor_backup):
        active_backups = sensor_backup.get_backup_status().get("active_backups", [])
        if not active_backups: return
        backup_map = {"face_backup_active": "Face->Pose", "hand_backup_active": "Hand->Pose"}
        text = "Backup: " + " | ".join(backup_map.get(b, "") for b in active_backups if b in backup_map)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        pos = ((frame.shape[1] - text_size[0]) // 2, 40)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["backup"], 2)

    def _draw_landmarks(self, frame, results):
        invisible_spec = mp_drawing.DrawingSpec(thickness=0, circle_radius=0)
        if "pose" in results and results["pose"].pose_landmarks:
            for pose_landmarks in results["pose"].pose_landmarks:
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_landmarks])
                solutions.drawing_utils.draw_landmarks(image=frame, landmark_list=pose_landmarks_proto, connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=solutions.drawing_styles.get_default_pose_landmarks_style())
        if "hand" in results and results["hand"].hand_landmarks:
            for hand_landmarks in results["hand"].hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks])
                solutions.drawing_utils.draw_landmarks(image=frame, landmark_list=hand_landmarks_proto, connections=solutions.hands.HAND_CONNECTIONS, landmark_drawing_spec=solutions.drawing_styles.get_default_hand_landmarks_style(), connection_drawing_spec=solutions.drawing_styles.get_default_hand_connections_style())
        if "face" in results and results["face"].face_landmarks:
            for face_landmarks in results["face"].face_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in face_landmarks])
                solutions.drawing_utils.draw_landmarks(image=frame, landmark_list=face_landmarks_proto, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=invisible_spec, connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                solutions.drawing_utils.draw_landmarks(image=frame, landmark_list=face_landmarks_proto, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=invisible_spec, connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_contours_style())
                solutions.drawing_utils.draw_landmarks(image=frame, landmark_list=face_landmarks_proto, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=invisible_spec, connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    def _draw_enhanced_state_alerts(self, frame, state, metrics):
        alert_map = {
            DriverState.FATIGUE_HIGH: "FATIGUE DETECTED!", DriverState.DISTRACTION_DANGER: "DANGEROUS DISTRACTION!",
            DriverState.PHONE_USAGE: "PHONE USAGE!", DriverState.MULTIPLE_RISK: "MULTIPLE RISKS!",
            DriverState.MICROSLEEP: "MICROSLEEP DETECTED!", DriverState.EMOTIONAL_STRESS: "EMOTIONAL STRESS!",
            DriverState.PREDICTIVE_WARNING: "PREDICTIVE WARNING!",
        }
        alert_text = alert_map.get(state)
        if alert_text:
            risk_color = self.colors.get(metrics.overall_risk_level.name.lower(), self.colors["critical"])
            h, w = frame.shape[:2]
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            pos = ((w - text_size[0]) // 2, h // 2)
            cv2.rectangle(frame, (pos[0] - 20, pos[1] - 40), (pos[0] + text_size[0] + 20, pos[1] + 10), (0, 0, 0), -1)
            cv2.putText(frame, alert_text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, risk_color, 3)

    def _draw_enhanced_performance_info(self, frame, perf_stats, playback_info):
        h, w = frame.shape[:2]
        texts = [f"FPS: {perf_stats.get('fps', 0.0):.1f}", f"Health: {perf_stats.get('system_health', 1.0):.1%}"]
        if "performance_status" in perf_stats:
            perf_status = perf_stats["performance_status"]
            if perf_status.get("active", False): texts.append("OPTIMIZED")
            texts.append(f"Perf: {perf_status.get('performance_score', 1.0):.1%}")
        if playback_info["mode"] == "video":
            texts.append(f"Video: {playback_info.get('current_video', 0)}/{playback_info.get('total_videos', 0)}")
        y_pos = h - (len(texts) * 25)
        for text in texts:
            cv2.putText(frame, text, (w - 200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 1)
            y_pos += 25
