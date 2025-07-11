import cv2
import numpy as np
import math
import time
import json
from pathlib import Path
from mediapipe.python.solutions import drawing_utils as mp_drawing, face_mesh as mp_face_mesh, pose as mp_pose
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from core.definitions import DriverState
from models.data_structures import UIMode, UIState

class SClassAdvancedUIManager:
    """S-Class DMS v18+ 고급 UI 매니저 - 차세대 시각적 인터페이스"""

    def __init__(self):
        # 적응형 UI 모드 설정
        self.current_ui_mode = UIMode.STANDARD
        self.manual_mode_override = False  # 'M' 키로 수동 모드 전환 시 True
        
        # UI 테마 로드
        self._load_ui_theme()
        
        # S-Class 전용 색상 팔레트 (config에서 로드 또는 기본값)
        self.colors = getattr(self, 'theme_colors', {
            "primary_blue": (255, 191, 0),       # #00BFFF -> BGR
            "accent_cyan": (255, 255, 0),        # #00FFFF -> BGR
            "warning_amber": (7, 193, 255),      # #FFC107 -> BGR
            "danger_red": (0, 69, 255),          # #FF4500 -> BGR
            "critical_magenta": (255, 0, 255),   # #FF00FF -> BGR
            "success_green": (127, 255, 0),      # #00FF7F -> BGR
            "text_white": (255, 255, 255),       # #FFFFFF -> BGR
            "text_silver": (138, 138, 138),      # #8A8A8A -> BGR
            "bg_dark": (46, 26, 26),             # #1a1a2e -> BGR
            "bg_panel": (62, 33, 22),            # #16213e -> BGR
            "border_glow": (255, 191, 0),        # 글로우 테두리
            "chart_line": (255, 255, 0),         # 차트 라인
            "pulse_effect": (255, 0, 255),       # 펄스 효과
        })
        
        # S-Class 시스템 상태별 색상
        self.status_colors = {
            "HEALTHY": self.colors["success_green"],
            "WARNING": self.colors["warning_amber"],
            "DANGER": self.colors["danger_red"],
            "CRITICAL": self.colors["critical_magenta"],
            "OPTIMAL": self.colors["primary_blue"],
            "DEGRADED": self.colors["warning_amber"],
        }
        
        # 위험도별 색상 매핑
        self.risk_colors = {
            "SAFE": self.colors["success_green"],
            "LOW": self.colors["primary_blue"],
            "MEDIUM": self.colors["warning_amber"],
            "HIGH": self.colors["danger_red"],
            "CRITICAL": self.colors["critical_magenta"],
        }
        
        # 실시간 데이터 버퍼
        self.data_history = {
            "fatigue_scores": [],
            "attention_scores": [],
            "heart_rates": [],
            "risk_scores": [],
            "timestamps": []
        }
        self.max_history = 100  # 최대 100개 데이터 포인트
        
        # 애니메이션 상태
        self.animation_time = time.time()
        self.pulse_phase = 0
        self.glow_intensity = 0
        
    def _load_ui_theme(self):
        """UI 테마 설정 로드"""
        try:
            theme_path = Path("config/ui_theme.json")
            if theme_path.exists():
                with open(theme_path, 'r', encoding='utf-8') as f:
                    theme_data = json.load(f)
                    
                # 색상 변환 (HEX -> BGR)
                self.theme_colors = {}
                colors = theme_data.get('colors', {})
                for name, hex_color in colors.items():
                    # HEX -> RGB -> BGR 변환
                    hex_color = hex_color.lstrip('#')
                    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    bgr = (rgb[2], rgb[1], rgb[0])  # RGB -> BGR
                    self.theme_colors[name.replace('_', '_')] = bgr
                
                # UI 모드별 설정
                self.ui_mode_settings = theme_data.get('ui_modes', {})
                
        except Exception as e:
            print(f"테마 로드 실패: {e}, 기본 테마 사용")
            
    def update_ui_mode(self, ui_state: UIState):
        """UIState에 따라 UI 모드 업데이트"""
        if not self.manual_mode_override:
            # 자동 모드: risk_score에 따라 UI 모드 결정
            ui_state.update_ui_mode_from_risk()
            self.current_ui_mode = ui_state.ui_mode
    
    def handle_key_input(self, key):
        """키 입력 처리 (적응형 UI 모드 순환)"""
        if key == ord('m') or key == ord('M'):
            # 'M' 키: UI 모드 순환
            self.manual_mode_override = True
            
            if self.current_ui_mode == UIMode.MINIMAL:
                self.current_ui_mode = UIMode.STANDARD
            elif self.current_ui_mode == UIMode.STANDARD:
                self.current_ui_mode = UIMode.ALERT
            else:  # ALERT
                self.current_ui_mode = UIMode.MINIMAL
                
            print(f"UI Mode changed to: {self.current_ui_mode.value}")
        
        elif key == ord('a') or key == ord('A'):
            # 'A' 키: 자동 모드로 복귀
            self.manual_mode_override = False
            print("UI Mode: AUTO (based on risk score)")
            
        return key  # 다른 키는 그대로 반환
    
    def render_ui_state(self, frame, ui_state: UIState):
        """UIState 기반 통합 렌더링 (지침서 요구사항)"""
        # UI 모드 업데이트
        self.update_ui_mode(ui_state)
        
        # 모드별 렌더링
        if self.current_ui_mode == UIMode.MINIMAL:
            return self._render_minimal_mode(frame, ui_state)
        elif self.current_ui_mode == UIMode.ALERT:
            return self._render_alert_mode(frame, ui_state)
        else:  # STANDARD
            return self._render_standard_mode(frame, ui_state)
    
    def _render_minimal_mode(self, frame, ui_state: UIState):
        """MINIMAL 모드: 핵심 정보만 표시"""
        h, w = frame.shape[:2]
        
        # 최소한의 상태 표시 (우측 상단)
        status_text = f"SAFE" if ui_state.risk_score < 0.3 else f"RISK: {ui_state.risk_score:.1f}"
        status_color = self.colors["success_green"] if ui_state.risk_score < 0.3 else self.colors["warning_amber"]
        
        cv2.putText(frame, status_text, (w - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 필수 경고만 표시
        if ui_state.active_alert_type.value != "none":
            self._draw_minimal_alert(frame, ui_state)
        
        # 모드 표시
        cv2.putText(frame, "MINIMAL", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text_silver"], 1)
        
        return frame
    
    def _render_standard_mode(self, frame, ui_state: UIState):
        """STANDARD 모드: 주요 분석 정보 표시"""
        # 기존 draw_enhanced_results와 유사하지만 UIState 기반
        annotated_frame = frame.copy()
        self._update_animation_state()
        
        # 메인 패널들
        self._draw_main_status_panel_uistate(annotated_frame, ui_state)
        self._draw_biometric_panel_uistate(annotated_frame, ui_state)
        self._draw_system_health_panel(annotated_frame, ui_state)
        
        # 고급 시각화 (축소)
        if ui_state.gaze.attention_score < 0.7:
            self._draw_attention_warning(annotated_frame, ui_state)
        
        # 모드 표시
        h = frame.shape[0]
        cv2.putText(annotated_frame, "STANDARD", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text_silver"], 1)
        
        return annotated_frame
    
    def _render_alert_mode(self, frame, ui_state: UIState):
        """ALERT 모드: 위험 요소 강조, 시각적 경고 활성화"""
        annotated_frame = frame.copy()
        self._update_animation_state()
        
        # 강화된 배경 효과
        self._apply_alert_background_effects(annotated_frame, ui_state)
        
        # 중앙 경고 (확대)
        self._draw_critical_warning_center(annotated_frame, ui_state)
        
        # 핵심 위험 지표만 표시 (큰 글씨)
        self._draw_critical_metrics(annotated_frame, ui_state)
        
        # 가장자리 펄스 효과
        self._apply_edge_pulse_effect(annotated_frame, ui_state.risk_score)
        
        # 모드 표시
        h = frame.shape[0]
        cv2.putText(annotated_frame, "ALERT MODE", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["critical_magenta"], 2)
        
        return annotated_frame
    
    def _draw_minimal_alert(self, frame, ui_state: UIState):
        """최소 모드용 간단한 경고"""
        h, w = frame.shape[:2]
        alert_text = ui_state.get_primary_concern()
        
        # 중앙 하단에 간단한 경고
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        x = (w - text_size[0]) // 2
        y = h - 60
        
        cv2.putText(frame, alert_text, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["danger_red"], 2)
    
    def _draw_main_status_panel_uistate(self, frame, ui_state: UIState):
        """UIState 기반 메인 상태 패널"""
        h, w = frame.shape[:2]
        panel_w, panel_h = 400, 280
        
        # 패널 배경
        overlay = frame.copy()
        panel_points = np.array([[10, 10], [panel_w, 10], [panel_w, panel_h], [10, panel_h]], np.int32)
        cv2.fillPoly(overlay, [panel_points], self.colors["bg_panel"])
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.polylines(frame, [panel_points], True, self.colors["border_glow"], 2)
        
        # 헤더
        cv2.putText(frame, "S-CLASS DMS v18+", (20, 35), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors["primary_blue"], 2)
        
        y_current = 70
        line_height = 22
        
        # 위험 점수
        risk_color = self._get_risk_score_color(ui_state.risk_score)
        cv2.putText(frame, f"RISK SCORE: {ui_state.risk_score:.2f}", (20, y_current), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
        y_current += line_height
        
        # 안전 상태
        cv2.putText(frame, f"STATUS: {ui_state.overall_safety_status.upper()}", (20, y_current), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
        y_current += line_height
        
        # 주요 우려사항
        concern = ui_state.get_primary_concern()
        cv2.putText(frame, f"PRIMARY: {concern}", (20, y_current), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["accent_cyan"], 1)
        y_current += line_height
        
        # 생체 정보
        if ui_state.biometrics.heart_rate:
            hr_color = self._get_pulse_color(ui_state.biometrics.heart_rate)
            cv2.putText(frame, f"♥ HR: {ui_state.biometrics.heart_rate:.0f} BPM", 
                       (20, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hr_color, 1)
            y_current += line_height
        
        # 주의집중도
        cv2.putText(frame, f"👁 ATTENTION: {ui_state.gaze.attention_score:.2f}", 
                   (20, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   self.colors["success_green"] if ui_state.gaze.attention_score > 0.7 else self.colors["warning_amber"], 1)
    
    def _draw_biometric_panel_uistate(self, frame, ui_state: UIState):
        """UIState 기반 생체측정 패널"""
        h, w = frame.shape[:2]
        panel_x, panel_y = w - 280, 10
        panel_w, panel_h = 270, 200
        
        # 패널 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     self.colors["bg_panel"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     self.colors["border_glow"], 2)
        
        # 헤더
        cv2.putText(frame, "BIOMETRICS", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors["primary_blue"], 2)
        
        y_current = panel_y + 50
        line_height = 20
        
        # 심박수
        if ui_state.biometrics.heart_rate:
            pulse_color = self._get_pulse_color(ui_state.biometrics.heart_rate)
            cv2.putText(frame, f"♥ {ui_state.biometrics.heart_rate:.0f} BPM", 
                       (panel_x + 10, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pulse_color, 2)
            y_current += line_height
        
        # 스트레스 레벨
        if ui_state.biometrics.stress_level is not None:
            stress_color = self._get_stress_color(ui_state.biometrics.stress_level)
            cv2.putText(frame, f"📊 STRESS: {ui_state.biometrics.stress_level:.2f}", 
                       (panel_x + 10, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stress_color, 1)
            y_current += line_height
        
        # 감정 상태
        emotion_color = self._get_emotion_state_color(ui_state.face.emotion_state)
        cv2.putText(frame, f"😐 {ui_state.face.emotion_state.value.upper()}", 
                   (panel_x + 10, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 1)
    
    def _draw_system_health_panel(self, frame, ui_state: UIState):
        """시스템 건강도 패널"""
        h, w = frame.shape[:2]
        panel_x, panel_y = 10, h - 120
        panel_w, panel_h = 300, 110
        
        # 패널 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     self.colors["bg_panel"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     self.colors["border_glow"], 2)
        
        # 헤더
        cv2.putText(frame, "SYSTEM HEALTH", (panel_x + 10, panel_y + 20), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors["primary_blue"], 2)
        
        y_current = panel_y + 45
        
        # FPS
        fps_color = self.colors["success_green"] if ui_state.system_health.processing_fps > 15 else self.colors["warning_amber"]
        cv2.putText(frame, f"FPS: {ui_state.system_health.processing_fps:.1f}", 
                   (panel_x + 10, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        
        # 전체 상태
        status_color = self.status_colors.get(ui_state.system_health.overall_status.upper(), self.colors["text_white"])
        cv2.putText(frame, f"STATUS: {ui_state.system_health.overall_status.upper()}", 
                   (panel_x + 10, y_current + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    def _draw_critical_warning_center(self, frame, ui_state: UIState):
        """ALERT 모드용 중앙 경고"""
        h, w = frame.shape[:2]
        
        # 주요 위험 요소 강조
        warning_text = ui_state.get_primary_concern()
        font_scale = 2.0
        thickness = 6
        
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
        x = (w - text_size[0]) // 2
        y = h // 2
        
        # 펄스 효과가 있는 텍스트
        pulse_intensity = 0.6 + 0.4 * math.sin(self.pulse_phase * 6)
        warning_color = tuple(int(c * pulse_intensity) for c in self.colors["critical_magenta"])
        
        cv2.putText(frame, warning_text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 
                   font_scale, warning_color, thickness)
        
        # 위험 점수 표시
        risk_text = f"RISK: {ui_state.risk_score:.1f}"
        cv2.putText(frame, risk_text, (x, y + 60), cv2.FONT_HERSHEY_DUPLEX, 
                   1.2, warning_color, 4)
    
    def _draw_critical_metrics(self, frame, ui_state: UIState):
        """ALERT 모드용 핵심 지표"""
        h, w = frame.shape[:2]
        
        # 좌측에 핵심 지표들
        x, y = 20, h // 2 + 100
        line_height = 30
        
        if ui_state.biometrics.heart_rate:
            cv2.putText(frame, f"♥ {ui_state.biometrics.heart_rate:.0f}", 
                       (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors["danger_red"], 3)
            y += line_height
        
        cv2.putText(frame, f"👁 {ui_state.gaze.attention_score:.2f}", 
                   (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors["warning_amber"], 3)
    
    def _apply_alert_background_effects(self, frame, ui_state: UIState):
        """ALERT 모드 배경 효과"""
        # 가장자리 빨간색 테두리
        h, w = frame.shape[:2]
        border_thickness = 10
        
        # 펄스 효과
        pulse_intensity = 0.3 + 0.7 * math.sin(self.pulse_phase * 4)
        border_color = tuple(int(c * pulse_intensity) for c in self.colors["critical_magenta"])
        
        # 상하좌우 테두리
        cv2.rectangle(frame, (0, 0), (w, border_thickness), border_color, -1)
        cv2.rectangle(frame, (0, h-border_thickness), (w, h), border_color, -1)
        cv2.rectangle(frame, (0, 0), (border_thickness, h), border_color, -1)
        cv2.rectangle(frame, (w-border_thickness, 0), (w, h), border_color, -1)
    
    def _apply_edge_pulse_effect(self, frame, risk_score):
        """가장자리 펄스 효과"""
        # 위험도에 따른 펄스 강도
        pulse_rate = 2 + risk_score * 4  # 위험할수록 빠른 펄스
        pulse_intensity = 0.2 + 0.8 * math.sin(self.pulse_phase * pulse_rate)
        
        # 오버레이로 가장자리 효과
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # 그라데이션 효과
        for i in range(20):
            alpha = pulse_intensity * (20 - i) / 20 * 0.1
            color = tuple(int(c * alpha) for c in self.colors["critical_magenta"])
            cv2.rectangle(overlay, (i, i), (w-i, h-i), color, 1)
        
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    def _get_risk_score_color(self, risk_score):
        """위험 점수에 따른 색상"""
        if risk_score < 0.3:
            return self.colors["success_green"]
        elif risk_score < 0.7:
            return self.colors["warning_amber"]
        else:
            return self.colors["danger_red"]
    
    def _get_stress_color(self, stress_level):
        """스트레스 레벨에 따른 색상"""
        if stress_level < 0.3:
            return self.colors["success_green"]
        elif stress_level < 0.7:
            return self.colors["warning_amber"]
        else:
            return self.colors["danger_red"]
    
    def _get_emotion_state_color(self, emotion_state):
        """감정 상태에 따른 색상"""
        from models.data_structures import EmotionState
        
        color_map = {
            EmotionState.NEUTRAL: self.colors["text_white"],
            EmotionState.HAPPY: self.colors["success_green"],
            EmotionState.STRESSED: self.colors["danger_red"],
            EmotionState.ANGRY: self.colors["critical_magenta"],
            EmotionState.FATIGUE: self.colors["warning_amber"],
            EmotionState.DROWSY: self.colors["danger_red"],
        }
        return color_map.get(emotion_state, self.colors["text_white"])
    
    def _draw_attention_warning(self, frame, ui_state: UIState):
        """주의집중도 경고"""
        if ui_state.gaze.attention_score < 0.7:
            h, w = frame.shape[:2]
            warning_text = "ATTENTION WARNING"
            
            # 중앙 상단에 경고
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            x = (w - text_size[0]) // 2
            y = 60
            
            cv2.putText(frame, warning_text, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors["warning_amber"], 2)
         
    def draw_enhanced_results(self, frame, metrics, state, results, gaze_classifier, dynamic_analyzer, sensor_backup, perf_stats, playback_info, driver_identifier, predictive_safety, emotion_recognizer):
        """S-Class 메인 렌더링 함수"""
        annotated_frame = frame.copy()
        self._update_animation_state()
        
        # 메인 패널들 그리기
        self._draw_main_status_panel(annotated_frame, metrics, state)
        self._draw_s_class_expert_systems_panel(annotated_frame, metrics)
        self._draw_real_time_charts(annotated_frame, metrics)
        self._draw_biometric_panel(annotated_frame, metrics)
        self._draw_advanced_warnings(annotated_frame, metrics, state)
        self._draw_performance_hud(annotated_frame, perf_stats)
        
        # 고급 시각화
        self._draw_holographic_gaze_tracker(annotated_frame, metrics, gaze_classifier)
        self._draw_3d_pose_skeleton(annotated_frame, results)
        self._draw_neural_network_visualization(annotated_frame, metrics)
        
        # 상태별 특수 효과
        self._apply_status_effects(annotated_frame, state, metrics)
        
        # MediaPipe 랜드마크 (고급 스타일)
        self._draw_enhanced_landmarks(annotated_frame, results)
        
        return annotated_frame

    def _update_animation_state(self):
        """애니메이션 상태 업데이트"""
        current_time = time.time()
        self.pulse_phase = (current_time * 2) % (2 * math.pi)
        self.glow_intensity = int(50 + 30 * math.sin(current_time * 3))

    def _draw_main_status_panel(self, frame, metrics, state):
        """메인 상태 패널 - 좌측 상단"""
        h, w = frame.shape[:2]
        panel_w, panel_h = 420, 320
        
        # 글래스모피즘 패널 배경
        overlay = frame.copy()
        panel_points = np.array([[10, 10], [panel_w, 10], [panel_w, panel_h], [10, panel_h]], np.int32)
        cv2.fillPoly(overlay, [panel_points], self.colors["bg_panel"])
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 네온 테두리
        cv2.polylines(frame, [panel_points], True, self.colors["border_glow"], 2)
        cv2.polylines(frame, [panel_points], True, self.colors["primary_blue"], 1)
        
        # S-Class 로고 헤더
        cv2.putText(frame, "S-CLASS DMS v18+", (20, 35), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors["primary_blue"], 2)
        cv2.putText(frame, "EXPERT SYSTEMS ACTIVE", (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["accent_cyan"], 1)
        
        # 상태 정보 (개선된 레이아웃)
        y_start = 85
        line_height = 22
        
        # 드라이버 상태
        state_color = self._get_state_color(state)
        cv2.putText(frame, f"◆ DRIVER STATE: {state.value}", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        # 위험도 레벨 (진행 바 포함)
        risk_level = getattr(metrics, 'overall_risk_level', None)
        if risk_level:
            risk_text = f"◆ RISK LEVEL: {risk_level.name}"
            risk_color = self.risk_colors.get(risk_level.name, self.colors["text_white"])
            cv2.putText(frame, risk_text, (20, y_start + line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
            
            # 위험도 진행 바
            bar_x, bar_y = 20, y_start + line_height + 10
            bar_w, bar_h = 200, 8
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                         self.colors["bg_dark"], -1)
            
            risk_value = self._risk_level_to_value(risk_level.name)
            fill_w = int(bar_w * risk_value)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), 
                         risk_color, -1)
        
        # S-Class 메트릭스
        y_current = y_start + line_height * 3
        
        # Enhanced EAR
        ear_value = getattr(metrics, 'enhanced_ear', 0.0)
        self._draw_metric_line(frame, "EAR", f"{ear_value:.3f}", 
                              (20, y_current), ear_value > 0.25)
        
        # PERCLOS
        perclos_value = getattr(metrics, 'perclos', 0.0)
        self._draw_metric_line(frame, "PERCLOS", f"{perclos_value:.2f}", 
                              (220, y_current), perclos_value < 0.15)
        
        y_current += line_height
        
        # 주의집중도
        attention_score = getattr(metrics, 'attention_focus_score', 0.0)
        self._draw_metric_line(frame, "ATTENTION", f"{attention_score:.2f}", 
                              (20, y_current), attention_score > 0.7)
        
        # 시간적 주의
        temporal_attention = getattr(metrics, 'temporal_attention_score', 0.0)
        self._draw_metric_line(frame, "TEMPORAL ATT", f"{temporal_attention:.2f}", 
                              (220, y_current), temporal_attention > 0.6)
        
        y_current += line_height
        
        # 시선 정보
        gaze_zone = getattr(metrics, 'current_gaze_zone', None)
        if gaze_zone:
            gaze_duration = getattr(metrics, 'gaze_zone_duration', 0.0)
            cv2.putText(frame, f"◆ GAZE: {gaze_zone.name} ({gaze_duration:.1f}s)", 
                       (20, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       self.colors["accent_cyan"], 1)
        
        # 머리 자세 (3D 시각화)
        y_current += line_height
        head_yaw = getattr(metrics, 'head_yaw', 0.0)
        head_pitch = getattr(metrics, 'head_pitch', 0.0) 
        head_roll = getattr(metrics, 'head_roll', 0.0)
        cv2.putText(frame, f"◆ HEAD POSE: Y{head_yaw:.1f}° P{head_pitch:.1f}° R{head_roll:.1f}°", 
                   (20, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   self.colors["text_silver"], 1)

    def _draw_s_class_expert_systems_panel(self, frame, metrics):
        """S-Class 전문가 시스템 패널 - 우측 상단"""
        h, w = frame.shape[:2]
        panel_x, panel_y = w - 380, 10
        panel_w, panel_h = 370, 280
        
        # 패널 배경
        overlay = frame.copy()
        panel_points = np.array([[panel_x, panel_y], [panel_x + panel_w, panel_y], 
                                [panel_x + panel_w, panel_y + panel_h], 
                                [panel_x, panel_y + panel_h]], np.int32)
        cv2.fillPoly(overlay, [panel_points], self.colors["bg_panel"])
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.polylines(frame, [panel_points], True, self.colors["border_glow"], 2)
        
        # 헤더
        cv2.putText(frame, "EXPERT SYSTEMS", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors["primary_blue"], 2)
        
        y_start = panel_y + 50
        line_height = 18
        
        # FaceProcessor (디지털 심리학자)
        self._draw_expert_system_status(frame, "FACE PROCESSOR", "Digital Psychologist", 
                                       (panel_x + 10, y_start), 
                                       self._get_face_processor_status(metrics))
        
        # PoseProcessor (생체역학 전문가)
        self._draw_expert_system_status(frame, "POSE PROCESSOR", "Biomechanics Expert", 
                                       (panel_x + 10, y_start + line_height * 3), 
                                       self._get_pose_processor_status(metrics))
        
        # HandProcessor (모터 제어 분석가)
        self._draw_expert_system_status(frame, "HAND PROCESSOR", "Motor Control Analyst", 
                                       (panel_x + 10, y_start + line_height * 6), 
                                       self._get_hand_processor_status(metrics))
        
        # ObjectProcessor (행동 예측 전문가)
        self._draw_expert_system_status(frame, "OBJECT PROCESSOR", "Behavior Prediction Expert", 
                                       (panel_x + 10, y_start + line_height * 9), 
                                       self._get_object_processor_status(metrics))

    def _draw_real_time_charts(self, frame, metrics):
        """실시간 차트 패널 - 좌측 하단"""
        h, w = frame.shape[:2]
        panel_x, panel_y = 10, h - 200
        panel_w, panel_h = 400, 190
        
        # 패널 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     self.colors["bg_panel"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     self.colors["border_glow"], 2)
        
        # 헤더
        cv2.putText(frame, "REAL-TIME ANALYTICS", (panel_x + 10, panel_y + 20), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors["primary_blue"], 2)
        
        # 데이터 업데이트
        self._update_data_history(metrics)
        
        # 차트 영역
        chart_x, chart_y = panel_x + 10, panel_y + 35
        chart_w, chart_h = panel_w - 20, panel_h - 45
        
        # 위험도 차트
        self._draw_line_chart(frame, "RISK SCORE", self.data_history["risk_scores"], 
                             (chart_x, chart_y), (chart_w//2 - 5, chart_h//2 - 5), 
                             self.colors["danger_red"])
        
        # 주의집중도 차트
        self._draw_line_chart(frame, "ATTENTION", self.data_history["attention_scores"], 
                             (chart_x + chart_w//2 + 5, chart_y), (chart_w//2 - 5, chart_h//2 - 5), 
                             self.colors["accent_cyan"])
        
        # 심박수 차트 (하단)
        self._draw_line_chart(frame, "HEART RATE", self.data_history["heart_rates"], 
                             (chart_x, chart_y + chart_h//2 + 5), (chart_w - 10, chart_h//2 - 5), 
                             self.colors["chart_line"])

    def _draw_biometric_panel(self, frame, metrics):
        """생체측정 패널 - 우측 하단"""
        h, w = frame.shape[:2]
        panel_x, panel_y = w - 300, h - 150
        panel_w, panel_h = 290, 140
        
        # 패널 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     self.colors["bg_panel"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     self.colors["border_glow"], 2)
        
        # 헤더
        cv2.putText(frame, "BIOMETRICS", (panel_x + 10, panel_y + 20), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors["primary_blue"], 2)
        
        y_current = panel_y + 45
        line_height = 18
        
        # rPPG 심박수
        if hasattr(metrics, 'face_analysis') and 'rppg' in getattr(metrics, 'face_analysis', {}):
            rppg_data = metrics.face_analysis['rppg']
            hr_bpm = rppg_data.get('estimated_hr_bpm', 0)
            signal_quality = rppg_data.get('signal_quality', 0)
            
            # 심박수 표시 (맥박 애니메이션 포함)
            pulse_color = self._get_pulse_color(hr_bpm)
            pulse_intensity = int(255 * (0.5 + 0.5 * math.sin(self.pulse_phase * 2)))
            animated_color = tuple(int(c * pulse_intensity / 255) for c in pulse_color)
            
            cv2.putText(frame, f"♥ HEART RATE: {hr_bpm:.0f} BPM", 
                       (panel_x + 10, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       animated_color, 2)
            cv2.putText(frame, f"  Signal Quality: {signal_quality:.2f}", 
                       (panel_x + 10, y_current + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                       self.colors["text_silver"], 1)
            y_current += line_height * 2
        
        # 사카드 분석
        if hasattr(metrics, 'face_analysis') and 'saccade' in getattr(metrics, 'face_analysis', {}):
            saccade_data = metrics.face_analysis['saccade']
            saccade_velocity = saccade_data.get('saccade_velocity_norm', 0)
            cv2.putText(frame, f"👁 SACCADE VEL: {saccade_velocity:.3f}", 
                       (panel_x + 10, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       self.colors["accent_cyan"], 1)
            y_current += line_height
        
        # 감정 상태
        if hasattr(metrics, 'emotion_state'):
            emotion_color = self._get_emotion_color(metrics)
            cv2.putText(frame, f"😐 EMOTION: {metrics.emotion_state.value}", 
                       (panel_x + 10, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       emotion_color, 1)

    def _draw_advanced_warnings(self, frame, metrics, state):
        """고급 경고 시스템 - 중앙"""
        h, w = frame.shape[:2]
        
        # 위험 상태별 중앙 경고
        warning_states = [
            state for state in [DriverState.FATIGUE_HIGH, DriverState.DISTRACTION_DANGER,
                               DriverState.PHONE_USAGE, DriverState.MULTIPLE_RISK,
                               DriverState.MICROSLEEP, DriverState.EMOTIONAL_STRESS]
            if state == state
        ]
        
        if warning_states:
            warning_text = self._get_warning_text(state)
            if warning_text:
                # 홀로그래픽 효과가 있는 중앙 경고
                self._draw_holographic_warning(frame, warning_text, state)
        
        # 예측적 위험 경고
        if hasattr(metrics, 'predictive_risk_score') and metrics.predictive_risk_score > 0.6:
            self._draw_predictive_warning(frame, metrics.predictive_risk_score)

    def _draw_holographic_warning(self, frame, text, state):
        """홀로그래픽 효과가 있는 경고"""
        h, w = frame.shape[:2]
        
        # 텍스트 크기 계산
        font_scale = 1.5
        thickness = 4
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
        
        # 중앙 위치
        x = (w - text_size[0]) // 2
        y = h // 2 - 50
        
        # 글로우 효과
        for i in range(10, 0, -2):
            alpha = 0.1 * (10 - i)
            glow_color = tuple(int(c * alpha) for c in self.colors["critical_magenta"])
            cv2.putText(frame, text, (x - i, y - i), cv2.FONT_HERSHEY_DUPLEX, 
                       font_scale, glow_color, thickness + i)
        
        # 메인 텍스트 (펄스 효과)
        pulse_intensity = 0.7 + 0.3 * math.sin(self.pulse_phase * 4)
        main_color = tuple(int(c * pulse_intensity) for c in self.colors["critical_magenta"])
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, main_color, thickness)
        
        # 배경 박스
        padding = 20
        cv2.rectangle(frame, (x - padding, y - text_size[1] - padding), 
                     (x + text_size[0] + padding, y + padding), 
                     self.colors["bg_dark"], -1)
        cv2.rectangle(frame, (x - padding, y - text_size[1] - padding), 
                     (x + text_size[0] + padding, y + padding), 
                     main_color, 3)

    # 유틸리티 메서드들
    def _get_state_color(self, state):
        """상태별 색상 반환"""
        state_color_map = {
            DriverState.SAFE: self.colors["success_green"],
            DriverState.FATIGUE_LOW: self.colors["warning_amber"],
            DriverState.FATIGUE_HIGH: self.colors["danger_red"],
            DriverState.DISTRACTION_NORMAL: self.colors["warning_amber"],
            DriverState.DISTRACTION_DANGER: self.colors["danger_red"],
            DriverState.MICROSLEEP: self.colors["critical_magenta"],
            DriverState.MULTIPLE_RISK: self.colors["critical_magenta"],
        }
        return state_color_map.get(state, self.colors["text_white"])

    def _risk_level_to_value(self, risk_level):
        """위험도 레벨을 0-1 값으로 변환"""
        risk_map = {"SAFE": 0.1, "LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.8, "CRITICAL": 1.0}
        return risk_map.get(risk_level, 0.0)

    def _draw_metric_line(self, frame, label, value, pos, is_good):
        """메트릭 라인 그리기"""
        color = self.colors["success_green"] if is_good else self.colors["warning_amber"]
        cv2.putText(frame, f"{label}: {value}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _get_face_processor_status(self, metrics):
        """얼굴 프로세서 상태"""
        if hasattr(metrics, 'face_analysis'):
            return "ACTIVE"
        return "STANDBY"

    def _get_pose_processor_status(self, metrics):
        """포즈 프로세서 상태"""
        if hasattr(metrics, 'pose_analysis'):
            return "ACTIVE"
        return "STANDBY"

    def _get_hand_processor_status(self, metrics):
        """손 프로세서 상태"""
        if hasattr(metrics, 'hand_analysis'):
            return "ACTIVE"
        return "STANDBY"

    def _get_object_processor_status(self, metrics):
        """객체 프로세서 상태"""
        if hasattr(metrics, 'object_analysis'):
            return "ACTIVE"
        return "STANDBY"

    def _draw_expert_system_status(self, frame, title, subtitle, pos, status):
        """전문가 시스템 상태 그리기"""
        x, y = pos
        status_color = self.status_colors.get(status, self.colors["text_silver"])
        
        cv2.putText(frame, f"● {title}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   status_color, 2)
        cv2.putText(frame, f"  {subtitle}", (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   self.colors["text_silver"], 1)

    def _update_data_history(self, metrics):
        """데이터 히스토리 업데이트"""
        current_time = time.time()
        
        # 위험도 점수
        risk_score = getattr(metrics, 'fatigue_risk_score', 0.0) + getattr(metrics, 'distraction_risk_score', 0.0)
        self.data_history["risk_scores"].append(min(1.0, risk_score))
        
        # 주의집중도
        attention_score = getattr(metrics, 'attention_focus_score', 0.0)
        self.data_history["attention_scores"].append(attention_score)
        
        # 심박수
        heart_rate = 0
        if hasattr(metrics, 'face_analysis') and 'rppg' in getattr(metrics, 'face_analysis', {}):
            heart_rate = metrics.face_analysis['rppg'].get('estimated_hr_bpm', 0) / 100.0  # 정규화
        self.data_history["heart_rates"].append(heart_rate)
        
        self.data_history["timestamps"].append(current_time)
        
        # 버퍼 크기 제한
        for key in self.data_history:
            if len(self.data_history[key]) > self.max_history:
                self.data_history[key] = self.data_history[key][-self.max_history:]

    def _draw_line_chart(self, frame, title, data, pos, size, color):
        """라인 차트 그리기"""
        if len(data) < 2:
            return
            
        x, y = pos
        w, h = size
        
        # 차트 배경
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors["bg_dark"], -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        
        # 제목
        cv2.putText(frame, title, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 데이터 정규화 및 그리기
        if data:
            max_val = max(data) if max(data) > 0 else 1.0
            min_val = min(data)
            val_range = max_val - min_val if max_val != min_val else 1.0
            
            points = []
            for i, value in enumerate(data):
                chart_x = x + int((i / len(data)) * w)
                chart_y = y + h - int(((value - min_val) / val_range) * (h - 20))
                points.append((chart_x, chart_y))
            
            if len(points) > 1:
                cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, color, 2)

    def _get_pulse_color(self, heart_rate):
        """심박수에 따른 색상"""
        if heart_rate < 60:
            return self.colors["primary_blue"]
        elif heart_rate < 100:
            return self.colors["success_green"]
        elif heart_rate < 120:
            return self.colors["warning_amber"]
        else:
            return self.colors["danger_red"]

    def _get_emotion_color(self, metrics):
        """감정에 따른 색상"""
        if hasattr(metrics, 'valence_level'):
            if metrics.valence_level > 0.6:
                return self.colors["success_green"]
            elif metrics.valence_level < 0.4:
                return self.colors["danger_red"]
        return self.colors["text_silver"]

    def _get_warning_text(self, state):
        """상태별 경고 텍스트"""
        warning_map = {
            DriverState.FATIGUE_HIGH: "⚠ FATIGUE DETECTED ⚠",
            DriverState.DISTRACTION_DANGER: "⚠ DANGEROUS DISTRACTION ⚠",
            DriverState.PHONE_USAGE: "📱 PHONE USAGE DETECTED 📱",
            DriverState.MICROSLEEP: "😴 MICROSLEEP WARNING 😴",
            DriverState.MULTIPLE_RISK: "🚨 MULTIPLE RISKS 🚨",
            DriverState.EMOTIONAL_STRESS: "😰 EMOTIONAL STRESS 😰",
        }
        return warning_map.get(state)

    def _draw_predictive_warning(self, frame, risk_score):
        """예측적 위험 경고"""
        h, w = frame.shape[:2]
        warning_text = f"PREDICTIVE RISK: {risk_score:.2f}"
        
        # 상단 중앙에 경고 배너
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        x = (w - text_size[0]) // 2
        y = 80
        
        # 배경
        cv2.rectangle(frame, (x - 20, y - 30), (x + text_size[0] + 20, y + 10), 
                     self.colors["warning_amber"], -1)
        cv2.putText(frame, warning_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   self.colors["bg_dark"], 2)

    def _draw_holographic_gaze_tracker(self, frame, metrics, gaze_classifier):
        """홀로그래픽 시선 추적기 - 개선된 3D 시각화"""
        if not gaze_classifier:
            return
            
        h, w = frame.shape[:2]
        center = (w - 150, 200)
        radius = 80
        
        # 3D 구체 시각화 (더 정교한 버전)
        self._draw_3d_gaze_sphere(frame, center, radius, metrics)

    def _draw_3d_gaze_sphere(self, frame, center, radius, metrics):
        """3D 시선 구체 그리기"""
        head_yaw = getattr(metrics, 'head_yaw', 0.0)
        head_pitch = getattr(metrics, 'head_pitch', 0.0)
        
        # 회전 행렬
        yaw_rad = math.radians(-head_yaw)
        pitch_rad = math.radians(head_pitch)
        
        # 구체의 격자선 그리기
        for lat in range(-90, 91, 30):
            points = []
            for lon in range(-180, 181, 15):
                x, y, z = self._sphere_to_screen(lat, lon, radius, yaw_rad, pitch_rad, center)
                if z > 0:  # 앞면만 그리기
                    points.append((int(x), int(y)))
            
            if len(points) > 1:
                cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, 
                             self.colors["primary_blue"], 1)
        
        # 시선 방향 표시
        gaze_x, gaze_y, gaze_z = self._sphere_to_screen(0, 0, radius * 1.2, yaw_rad, pitch_rad, center)
        if gaze_z > 0:
            cv2.circle(frame, (int(gaze_x), int(gaze_y)), 5, self.colors["accent_cyan"], -1)
            cv2.circle(frame, (int(gaze_x), int(gaze_y)), 10, self.colors["accent_cyan"], 2)

    def _sphere_to_screen(self, lat, lon, radius, yaw, pitch, center):
        """구면 좌표를 화면 좌표로 변환"""
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        # 구면 좌표를 직교 좌표로
        x = radius * math.cos(lat_rad) * math.cos(lon_rad)
        y = radius * math.cos(lat_rad) * math.sin(lon_rad)
        z = radius * math.sin(lat_rad)
        
        # 회전 적용
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
        
        # Y축 회전 (yaw)
        x_rot = x * cos_yaw - z * sin_yaw
        z_rot = x * sin_yaw + z * cos_yaw
        
        # X축 회전 (pitch)
        y_rot = y * cos_pitch - z_rot * sin_pitch
        z_final = y * sin_pitch + z_rot * cos_pitch
        
        # 화면 좌표로 변환
        screen_x = center[0] + x_rot
        screen_y = center[1] - y_rot
        
        return screen_x, screen_y, z_final

    def _draw_3d_pose_skeleton(self, frame, results):
        """3D 포즈 스켈레톤 시각화"""
        if not results or "pose" not in results or not results["pose"].pose_landmarks:
            return
        
        # 기존 랜드마크 그리기를 더 스타일리시하게
        self._draw_enhanced_landmarks(frame, results)

    def _draw_neural_network_visualization(self, frame, metrics):
        """신경망 시각화 (S-Class AI 표현)"""
        h, w = frame.shape[:2]
        
        # 우측 중앙에 작은 신경망 시각화
        center_x, center_y = w - 100, h // 2
        
        # 노드들
        nodes = [
            (center_x - 30, center_y - 40),  # 입력층
            (center_x - 30, center_y),
            (center_x - 30, center_y + 40),
            (center_x, center_y - 20),       # 히든층
            (center_x, center_y + 20),
            (center_x + 30, center_y)        # 출력층
        ]
        
        # 연결선 (활성화 상태에 따라 색상 변화)
        connections = [
            (0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4),  # 입력->히든
            (3, 5), (4, 5)  # 히든->출력
        ]
        
        for start, end in connections:
            activation = 0.3 + 0.7 * math.sin(time.time() * 2 + start)
            color_intensity = int(255 * activation)
            line_color = (color_intensity, color_intensity, 255)
            cv2.line(frame, nodes[start], nodes[end], line_color, 2)
        
        # 노드 그리기
        for i, node in enumerate(nodes):
            activation = 0.5 + 0.5 * math.sin(time.time() * 3 + i)
            node_color = (int(100 + 155 * activation), int(100 + 155 * activation), 255)
            cv2.circle(frame, node, 6, node_color, -1)
            cv2.circle(frame, node, 8, self.colors["primary_blue"], 1)

    def _draw_performance_hud(self, frame, perf_stats):
        """성능 HUD - 상단 중앙"""
        h, w = frame.shape[:2]
        
        # FPS 및 시스템 상태
        fps = perf_stats.get('fps', 0.0)
        system_health = perf_stats.get('system_health', 1.0)
        
        # 중앙 상단 위치
        hud_y = 20
        
        # FPS 표시
        fps_color = self.colors["success_green"] if fps > 25 else self.colors["warning_amber"]
        cv2.putText(frame, f"FPS: {fps:.1f}", (w//2 - 100, hud_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        
        # 시스템 헬스
        health_color = self.colors["success_green"] if system_health > 0.8 else self.colors["warning_amber"]
        cv2.putText(frame, f"SYSTEM: {system_health:.1%}", (w//2, hud_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, health_color, 2)

    def _apply_status_effects(self, frame, state, metrics):
        """상태별 특수 효과"""
        if state in [DriverState.FATIGUE_HIGH, DriverState.MICROSLEEP]:
            # 피로 상태 - 화면 가장자리 빨간 글로우
            self._apply_edge_glow(frame, self.colors["danger_red"], 0.3)
        elif state == DriverState.DISTRACTION_DANGER:
            # 주의산만 - 화면 가장자리 황색 글로우
            self._apply_edge_glow(frame, self.colors["warning_amber"], 0.2)
        elif state == DriverState.MULTIPLE_RISK:
            # 다중 위험 - 펄스 효과
            self._apply_pulse_effect(frame, self.colors["critical_magenta"])

    def _apply_edge_glow(self, frame, color, intensity):
        """화면 가장자리 글로우 효과"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # 가장자리 그라데이션
        for i in range(20):
            alpha = intensity * (20 - i) / 20
            thickness = i + 1
            cv2.rectangle(overlay, (i, i), (w - i, h - i), color, thickness)
        
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    def _apply_pulse_effect(self, frame, color):
        """펄스 효과"""
        pulse_intensity = 0.1 + 0.2 * math.sin(self.pulse_phase * 3)
        overlay = frame.copy()
        overlay[:] = color
        cv2.addWeighted(overlay, pulse_intensity, frame, 1 - pulse_intensity, 0, frame)

    def _draw_enhanced_landmarks(self, frame, results):
        """향상된 랜드마크 그리기"""
        # 기존 코드 유지하되 스타일 개선
        invisible_spec = mp_drawing.DrawingSpec(thickness=0, circle_radius=0)
        
        if "pose" in results and results["pose"].pose_landmarks:
            for pose_landmarks in results["pose"].pose_landmarks:
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                    for lm in pose_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    image=frame, 
                    landmark_list=pose_landmarks_proto, 
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=solutions.drawing_styles.get_default_pose_landmarks_style(),
                    connection_drawing_spec=solutions.drawing_styles.get_default_pose_connections_style()
                )
        
        if "hand" in results and results["hand"].hand_landmarks:
            for hand_landmarks in results["hand"].hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                    for lm in hand_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    image=frame, 
                    landmark_list=hand_landmarks_proto, 
                    connections=solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=solutions.drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=solutions.drawing_styles.get_default_hand_connections_style()
                )
        
        if "face" in results and results["face"].face_landmarks:
            for face_landmarks in results["face"].face_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                    for lm in face_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    image=frame, 
                    landmark_list=face_landmarks_proto, 
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=invisible_spec,
                    connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )
                solutions.drawing_utils.draw_landmarks(
                    image=frame, 
                    landmark_list=face_landmarks_proto, 
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=invisible_spec,
                    connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_contours_style()
                )
                solutions.drawing_utils.draw_landmarks(
                    image=frame, 
                    landmark_list=face_landmarks_proto, 
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=invisible_spec,
                    connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
                )


# 하위 호환성을 위한 별칭
EnhancedUIManager = SClassAdvancedUIManager
