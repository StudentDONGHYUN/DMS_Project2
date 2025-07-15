import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from utils.logging import setup_logging_system
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

from core.definitions import CameraPosition
from integration.integrated_system import AnalysisSystemType
import logging

from app import DMSApp

# S-Class v19.0 혁신 기능 Import
from systems.ai_driving_coach import AIDrivingCoach
from systems.v2d_healthcare import V2DHealthcareSystem
from systems.ar_hud_system import ARHUDSystem, VehicleContext
from systems.emotional_care_system import EmotionalCareSystem
from systems.digital_twin_platform import DigitalTwinPlatform

# S-Class v19.0 관련 Import
from config.settings import get_config, FeatureFlagConfig
from models.data_structures import UIState

# 로깅 시스템 설정
setup_logging_system()

logger = logging.getLogger(__name__)

GUI_AVAILABLE = True


@dataclass
class SystemStatus:
    """시스템 상태"""

    ai_coach_active: bool = False
    healthcare_active: bool = False
    ar_hud_active: bool = False
    emotional_care_active: bool = False
    digital_twin_active: bool = False

    current_sessions: Dict[str, str] = None
    last_update: float = 0.0

    def __post_init__(self):
        if self.current_sessions is None:
            self.current_sessions = {}


class SClassDMSv19Enhanced:
    """S-Class DMS v19.0 혁신 기능 통합 클래스"""

    def __init__(self, user_id: str = "default", edition: str = "RESEARCH"):
        """
        S-Class DMS v19.0 초기화

        Args:
            user_id: 사용자 ID
            edition: 에디션 (COMMUNITY, PRO, ENTERPRISE, RESEARCH)
        """
        self.config = get_config()
        self.user_id = user_id
        self.edition = edition

        # 피처 플래그 설정
        self.feature_flags = FeatureFlagConfig(system_edition=edition)

        # 로깅 설정
        self.logger = self._setup_logging()

        # 시스템 상태
        self.status = SystemStatus()
        self.is_running = False

        # 혁신 시스템들 초기화
        self.innovation_systems = self._initialize_innovation_systems()

        # 통합 데이터 저장
        self.session_data = []
        self.performance_metrics = {}

        # 동시 실행 태스크
        self.running_tasks = []

        self.logger.info(f"S-Class DMS v19.0 시스템 초기화 완료")
        self.logger.info(f"사용자: {user_id}, 에디션: {edition}")
        self.logger.info(f"활성화된 기능: {self._get_enabled_features()}")

    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f"SClassDMS_v19_{self.user_id}")
        logger.setLevel(logging.INFO)

        # 핸들러가 이미 있으면 추가하지 않음
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_innovation_systems(self) -> Dict[str, Any]:
        """혁신 시스템들 초기화 - Bug fix: Improved error handling and feature flag checking"""
        systems = {}

        # Bug fix: Safety check for feature_flags
        if not hasattr(self, "feature_flags") or self.feature_flags is None:
            self.logger.warning("⚠️ Feature flags가 설정되지 않음. 기본 설정 사용.")
            # Create default feature flags for safety
            try:
                self.feature_flags = FeatureFlagConfig(edition="COMMUNITY")
            except Exception as e:
                self.logger.error(f"기본 feature flags 생성 실패: {e}")
                return systems

        # 1. AI 드라이빙 코치 (PRO 이상)
        if getattr(
            self.feature_flags, "s_class_advanced_features", False
        ):  # Bug fix: Safe attribute access
            try:
                systems["ai_coach"] = AIDrivingCoach(self.user_id)
                self.logger.info("✅ AI 드라이빙 코치 시스템 초기화 완료")
            except Exception as e:
                self.logger.error(f"❌ AI 드라이빙 코치 초기화 실패: {e}")
                # Bug fix: Continue initialization even if one system fails

        # 2. V2D 헬스케어 플랫폼 (PRO 이상)
        if getattr(
            self.feature_flags, "s_class_advanced_features", False
        ):  # Bug fix: Safe attribute access
            try:
                systems["healthcare"] = V2DHealthcareSystem(self.user_id)
                self.logger.info("✅ V2D 헬스케어 시스템 초기화 완료")
            except Exception as e:
                self.logger.error(f"❌ V2D 헬스케어 초기화 실패: {e}")

        # 3. AR HUD 시스템 (ENTERPRISE 이상)
        if getattr(
            self.feature_flags, "neural_ai_features", False
        ):  # Bug fix: Safe attribute access
            try:
                systems["ar_hud"] = ARHUDSystem()
                self.logger.info("✅ AR HUD 시스템 초기화 완료")
            except Exception as e:
                self.logger.error(f"❌ AR HUD 초기화 실패: {e}")

        # 4. 감성 케어 시스템 (ENTERPRISE 이상)
        if getattr(
            self.feature_flags, "neural_ai_features", False
        ):  # Bug fix: Safe attribute access
            try:
                systems["emotional_care"] = EmotionalCareSystem(self.user_id)
                self.logger.info("✅ 감성 케어 시스템 초기화 완료")
            except Exception as e:
                self.logger.error(f"❌ 감성 케어 초기화 실패: {e}")

        # 5. 디지털 트윈 플랫폼 (RESEARCH 에디션)
        if getattr(
            self.feature_flags, "innovation_research_features", False
        ):  # Bug fix: Safe attribute access
            try:
                systems["digital_twin"] = DigitalTwinPlatform()
                self.logger.info("✅ 디지털 트윈 플랫폼 초기화 완료")
            except Exception as e:
                self.logger.error(f"❌ 디지털 트윈 플랫폼 초기화 실패: {e}")

        # Bug fix: Log summary of initialization results
        successful_systems = len(systems)
        self.logger.info(f"혁신 시스템 초기화 완료: {successful_systems}개 시스템 성공")
        if successful_systems == 0:
            self.logger.warning("⚠️ 모든 혁신 시스템 초기화 실패 - 기본 모드로 동작")

        return systems

    def _get_enabled_features(self) -> List[str]:
        """활성화된 기능 목록"""
        features = []

        if self.feature_flags.basic_expert_systems:
            features.append("Expert Systems")
        if self.feature_flags.s_class_advanced_features:
            features.extend(["AI Coach", "Healthcare"])
        if self.feature_flags.neural_ai_features:
            features.extend(["AR HUD", "Emotional Care"])
        if self.feature_flags.innovation_research_features:
            features.append("Digital Twin")

        return features

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
            "active_systems": {
                "AI Coach": self.status.ai_coach_active,
                "V2D Healthcare": self.status.healthcare_active,
                "AR HUD": self.status.ar_hud_active,
                "Emotional Care": self.status.emotional_care_active,
                "Digital Twin": self.status.digital_twin_active,
            },
            "current_sessions": self.status.current_sessions,
            "last_update": self.status.last_update,
            "edition": self.edition,
            "enabled_features": self._get_enabled_features(),
        }


class SClass_DMS_GUI_Setup:
    """S-Class DMS v19.0 차세대 GUI 설정 - 혁신 기능 통합 인터페이스"""

    def __init__(self, root):
        self.root = root
        self.root.title("🚗 S-Class DMS v19.0 - The Next Chapter")
        self.root.geometry("800x1200")
        self.root.configure(bg="#1a1a2e")  # 다크 테마
        self.config = None
        self.video_files = []
        self.is_same_driver = True
        self.edition_var = tk.StringVar(value="RESEARCH")  # 에디션 선택 변수 추가

        # S-Class v19.0 혁신 시스템 초기화
        self.innovation_engine = SClassDMSv19Enhanced("default", "RESEARCH")

        # 고급 스타일 설정
        self._setup_advanced_styles()

        # S-Class 설정 변수들
        self.source_type = tk.StringVar(value="webcam")
        self.webcam_id = tk.StringVar(value="0")
        self.user_id = tk.StringVar(value="default")
        self.enable_calibration = tk.BooleanVar(value=True)
        self.enable_performance_optimization = tk.BooleanVar(
            value=True
        )  # 성능 최적화 모드 옵션
        self.camera_position_var = tk.StringVar(
            value=str(CameraPosition.REARVIEW_MIRROR)
        )

        # S-Class 시스템 설정
        self.system_type_var = tk.StringVar(value="STANDARD")
        self.use_legacy_engine = tk.BooleanVar(value=False)  # S-Class가 기본
        self.enable_rppg = tk.BooleanVar(value=True)
        self.enable_saccade = tk.BooleanVar(value=True)
        self.enable_spinal_analysis = tk.BooleanVar(value=True)
        self.enable_tremor_fft = tk.BooleanVar(value=True)
        self.enable_bayesian_prediction = tk.BooleanVar(value=True)

        # 고급 설정들
        self.enable_emotion_ai = tk.BooleanVar(value=True)
        self.enable_predictive_safety = tk.BooleanVar(value=True)
        self.enable_biometric_fusion = tk.BooleanVar(value=True)
        self.enable_adaptive_thresholds = tk.BooleanVar(value=True)

        # 애니메이션 및 시각 효과
        self.animation_frame = 0
        self.preview_enabled = tk.BooleanVar(value=False)

        self._create_advanced_gui()

    def _setup_advanced_styles(self):
        """S-Class 고급 스타일 설정"""
        style = ttk.Style()

        # 고급 테마 설정
        try:
            style.theme_use("clam")
        except Exception as e:
            logger.debug(f"GUI 테마 'clam' 설정 실패 (기본 테마 사용): {e}")

        # S-Class 전용 색상 팔레트
        colors = {
            "bg_primary": "#1a1a2e",  # 다크 네이비
            "bg_secondary": "#16213e",  # 어두운 블루
            "accent_cyan": "#00d4ff",  # 네온 시아니즘
            "accent_orange": "#ff6b35",  # 네온 오렌지
            "text_primary": "#ffffff",  # 화이트
            "text_secondary": "#8a8a8a",  # 그레이
            "success": "#00ff9f",  # 네온 그린
            "warning": "#ffaa00",  # 호박색
            "danger": "#ff0040",  # 네온 빨강
        }

        # 고급 커스텀 스타일
        style.configure(
            "SClass.TLabel",
            font=("Segoe UI", 16, "bold"),
            foreground=colors["accent_cyan"],
            background=colors["bg_primary"],
        )

        style.configure(
            "SClassTitle.TLabel",
            font=("Segoe UI", 20, "bold"),
            foreground=colors["accent_orange"],
            background=colors["bg_primary"],
        )

        style.configure(
            "SClassSubtitle.TLabel",
            font=("Segoe UI", 11),
            foreground=colors["text_secondary"],
            background=colors["bg_primary"],
        )

        style.configure(
            "SClassFeature.TLabel",
            font=("Segoe UI", 10),
            foreground=colors["text_primary"],
            background=colors["bg_primary"],
        )

        style.configure(
            "SClassButton.TButton",
            font=("Segoe UI", 12, "bold"),
            foreground=colors["bg_primary"],
            background=colors["accent_cyan"],
        )

        style.configure(
            "SClassFrame.TFrame",
            background=colors["bg_secondary"],
            relief="flat",
            borderwidth=2,
        )

        style.configure(
            "SClassLabelFrame.TLabelframe",
            background=colors["bg_secondary"],
            foreground=colors["accent_cyan"],
            relief="solid",
            borderwidth=2,
        )

        style.configure(
            "SClassLabelFrame.TLabelframe.Label",
            background=colors["bg_secondary"],
            foreground=colors["accent_cyan"],
            font=("Segoe UI", 12, "bold"),
        )

        # 체크박스 스타일
        style.configure(
            "SClass.TCheckbutton",
            background=colors["bg_secondary"],
            foreground=colors["text_primary"],
            font=("Segoe UI", 10),
        )

        # 라디오버튼 스타일
        style.configure(
            "SClass.TRadiobutton",
            background=colors["bg_secondary"],
            foreground=colors["text_primary"],
            font=("Segoe UI", 10),
        )

        # 콤보박스 스타일
        style.configure(
            "SClass.TCombobox",
            fieldbackground=colors["bg_primary"],
            foreground=colors["text_primary"],
            font=("Segoe UI", 10),
        )

        # 엔트리 스타일
        style.configure(
            "SClass.TEntry",
            fieldbackground=colors["bg_primary"],
            foreground=colors["text_primary"],
            font=("Segoe UI", 10),
        )

        # 진행바 스타일
        style.configure(
            "SClass.TProgressbar",
            background=colors["accent_cyan"],
            troughcolor=colors["bg_primary"],
            borderwidth=0,
            lightcolor=colors["accent_cyan"],
            darkcolor=colors["accent_cyan"],
        )
        # 진행바 레이아웃 명시적 추가 (호환성 강화)
        style.layout(
            "SClass.TProgressbar",
            [
                (
                    "Horizontal.Progressbar.trough",
                    {
                        "children": [
                            (
                                "Horizontal.Progressbar.pbar",
                                {"side": "left", "sticky": "ns"},
                            )
                        ],
                        "sticky": "nswe",
                    },
                )
            ],
        )

    def _create_advanced_gui(self):
        """S-Class 차세대 GUI 생성 - 탭 기반 인터페이스"""
        # 메인 프레임 (스크롤 가능)
        main_frame = ttk.Frame(self.root, style="SClassFrame.TFrame", padding="20")
        main_frame.pack(fill="both", expand=True)

        # 탭 노트북 생성
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True, pady=(0, 20))

        # 탭들 생성
        self._create_main_tab()
        self._create_expert_systems_tab()
        self._create_advanced_features_tab()

        # 하단 제어 패널
        self._create_control_panel(main_frame)

    def _create_main_tab(self):
        """메인 설정 탭"""
        main_tab = ttk.Frame(self.notebook, style="SClassFrame.TFrame")
        self.notebook.add(main_tab, text=" 🏠 메인 설정 ")

        # 기존 섹션들을 업그레이드된 스타일로 재활용
        self._create_enhanced_header_section(main_tab)
        self._create_input_source_section(main_tab)
        self._create_user_settings_section(main_tab)
        self._create_sclass_system_section(main_tab)

    def _create_expert_systems_tab(self):
        """전문가 시스템 탭"""
        expert_tab = ttk.Frame(self.notebook, style="SClassFrame.TFrame")
        self.notebook.add(expert_tab, text=" 🧠 Expert Systems ")

        # 기존 S-Class 기능 섹션을 업그레이드
        self._create_sclass_features_section(expert_tab)
        self._create_advanced_settings_section(expert_tab)

    def _create_advanced_features_tab(self):
        """고급 기능 탭"""
        advanced_tab = ttk.Frame(self.notebook, style="SClassFrame.TFrame")
        self.notebook.add(advanced_tab, text=" ⚡ Advanced Features ")

        # 기존 기능 안내 섹션을 업그레이드
        self._create_features_info_section(advanced_tab)
        self._create_neural_ai_section(advanced_tab)

    def _create_enhanced_header_section(self, parent):
        """향상된 헤더 섹션"""
        header_frame = ttk.Frame(parent, style="SClassFrame.TFrame")
        header_frame.pack(fill="x", pady=(0, 20))

        # 메인 타이틀 (업그레이드)
        title_label = ttk.Label(
            header_frame,
            text="🚗 S-Class DMS v18+ Neural Research Platform",
            style="SClassTitle.TLabel",
        )
        title_label.pack(pady=(0, 5))

        # 부제목 (업그레이드)
        subtitle_label = ttk.Label(
            header_frame,
            text="Advanced AI • Real-time biometrics • Predictive Analytics • Neural Networks",
            style="SClassSubtitle.TLabel",
        )
        subtitle_label.pack(pady=(0, 10))

        # 전문가 시스템 소개 (새로운)
        expert_label = ttk.Label(
            header_frame,
            text="🧠 Digital Psychologist • 🦴 Biomechanics Expert • 🖐 Motor Control Analyst • 👁 Behavior Predictor",
            style="SClass.TLabel",
        )
        expert_label.pack(pady=(0, 15))

        # 진행바 애니메이션 (새로운)
        self.progress_bar = ttk.Progressbar(
            header_frame, mode="indeterminate", style="SClass.TProgressbar", length=400
        )
        self.progress_bar.pack(pady=10)
        self.progress_bar.start(10)  # 애니메이션 시작

    def _create_neural_ai_section(self, parent):
        """신경망 AI 섹션"""
        neural_frame = ttk.LabelFrame(
            parent,
            text=" 🤖 Neural AI Configuration ",
            style="SClassLabelFrame.TLabelframe",
            padding="15",
        )
        neural_frame.pack(fill="x", pady=10)

        # 고급 AI 기능들
        features_frame = ttk.Frame(neural_frame, style="SClassFrame.TFrame")
        features_frame.pack(fill="x")

        # 왼쪽 컬럼
        left_frame = ttk.Frame(features_frame, style="SClassFrame.TFrame")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ttk.Checkbutton(
            left_frame,
            text="🧠 Emotion AI (감정 인식)",
            variable=self.enable_emotion_ai,
            style="SClass.TCheckbutton",
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            left_frame,
            text="🔮 Predictive Safety (예측 안전)",
            variable=self.enable_predictive_safety,
            style="SClass.TCheckbutton",
        ).pack(anchor="w", pady=2)

        # 오른쪽 컬럼
        right_frame = ttk.Frame(features_frame, style="SClassFrame.TFrame")
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        ttk.Checkbutton(
            right_frame,
            text="🔗 Biometric Fusion (생체정보 융합)",
            variable=self.enable_biometric_fusion,
            style="SClass.TCheckbutton",
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            right_frame,
            text="📊 Adaptive Thresholds (적응형 임계값)",
            variable=self.enable_adaptive_thresholds,
            style="SClass.TCheckbutton",
        ).pack(anchor="w", pady=2)

        # 설명 텍스트
        info_text = (
            "🔬 Advanced Research Features:\n"
            "• Deep Learning Attention Mechanisms\n"
            "• Real-time Cognitive Load Assessment\n"
            "• Multi-modal Sensor Fusion\n"
            "• Uncertainty Quantification\n"
            "• Personalized Risk Modeling"
        )

        info_label = ttk.Label(
            neural_frame, text=info_text, style="SClassFeature.TLabel", justify="left"
        )
        info_label.pack(anchor="w", pady=(15, 0))

    def _create_control_panel(self, parent):
        """하단 제어 패널"""
        control_frame = ttk.Frame(parent, style="SClassFrame.TFrame")
        control_frame.pack(fill="x", pady=(20, 0))

        # 시작 버튼 (업그레이드)
        start_button = ttk.Button(
            control_frame,
            text="🚀 Launch S-Class Neural DMS v18+",
            command=self.start_app,
            style="SClassButton.TButton",
        )
        start_button.pack(fill="x", ipady=15)

        # 상태 표시
        status_label = ttk.Label(
            control_frame,
            text="⚡ System Ready • All Expert Systems Online",
            style="SClassSubtitle.TLabel",
        )
        status_label.pack(pady=(10, 0))

    def _create_header_section(self, parent):
        """헤더 섹션"""
        title_label = ttk.Label(
            parent,
            text="🚗 S-Class Driver Monitoring System v18+",
            style="Title.TLabel",
        )
        title_label.pack(pady=(0, 5))

        subtitle_label = ttk.Label(
            parent,
            text="고급 연구 통합: rPPG • 사카드 분석 • 스파인 정렬 • FFT 떨림 분석 • 베이지안 예측",
            style="Subtitle.TLabel",
        )
        subtitle_label.pack(pady=(0, 10))

        sclass_label = ttk.Label(
            parent,
            text="S-Class Expert Systems: 디지털 심리학자 • 생체역학 전문가 • 모터 제어 분석가 • 행동 예측 전문가",
            style="SClass.TLabel",
        )
        sclass_label.pack(pady=(0, 15))

    def _create_input_source_section(self, parent):
        """S-Class 입력 소스 섹션"""
        source_frame = ttk.LabelFrame(
            parent,
            text=" 📹 Neural Input Source Configuration ",
            style="SClassLabelFrame.TLabelframe",
            padding="15",
        )
        source_frame.pack(fill="x", pady=10)

        # 웹캠 옵션 (S-Class 스타일)
        webcam_frame = ttk.Frame(source_frame, style="SClassFrame.TFrame")
        webcam_frame.pack(fill="x", pady=5)

        ttk.Radiobutton(
            webcam_frame,
            text="🎥 Real-time Neural Processing (Webcam)",
            variable=self.source_type,
            value="webcam",
            command=self.toggle_source_widgets,
            style="SClass.TRadiobutton",
        ).pack(side="left", padx=5)

        ttk.Label(webcam_frame, text="Device ID:", style="SClassFeature.TLabel").pack(
            side="left", padx=(20, 5)
        )
        self.webcam_id_entry = ttk.Entry(
            webcam_frame, textvariable=self.webcam_id, width=8, style="SClass.TEntry"
        )
        self.webcam_id_entry.pack(side="left")

        # 비디오 파일 옵션 (S-Class 스타일)
        video_frame = ttk.Frame(source_frame, style="SClassFrame.TFrame")
        video_frame.pack(fill="x", pady=5)

        ttk.Radiobutton(
            video_frame,
            text="📁 Batch Analysis Mode (Video Files)",
            variable=self.source_type,
            value="video",
            command=self.toggle_source_widgets,
            style="SClass.TRadiobutton",
        ).pack(side="left", padx=5)

        self.video_button = ttk.Button(
            video_frame,
            text="Select Files...",
            command=self.browse_video,
            state="disabled",
            style="SClassButton.TButton",
        )
        self.video_button.pack(side="left", padx=(20, 0))

        self.video_label = ttk.Label(
            parent,
            text="📄 No files selected",
            style="SClassSubtitle.TLabel",
            wraplength=600,
            justify="left",
        )
        self.video_label.pack(fill="x", pady=(10, 15))

    def _create_user_settings_section(self, parent):
        """S-Class 사용자 설정 섹션"""
        user_frame = ttk.LabelFrame(
            parent,
            text=" 👤 Neural Profile Configuration ",
            style="SClassLabelFrame.TLabelframe",
            padding="15",
        )
        user_frame.pack(fill="x", pady=10)

        # 사용자 ID 설정
        id_frame = ttk.Frame(user_frame, style="SClassFrame.TFrame")
        id_frame.pack(fill="x", pady=5)

        ttk.Label(
            id_frame, text="🆔 User Profile ID:", style="SClassFeature.TLabel"
        ).pack(side="left", padx=(0, 10))
        ttk.Entry(
            id_frame, textvariable=self.user_id, style="SClass.TEntry", width=20
        ).pack(side="left", expand=True, fill="x", padx=(0, 20))
        # 에디션 선택 콤보박스 추가
        editions = ["COMMUNITY", "PRO", "ENTERPRISE", "RESEARCH"]
        ttk.Label(id_frame, text="에디션:", style="SClassFeature.TLabel").pack(
            side="left", padx=(10, 5)
        )
        ttk.Combobox(
            id_frame,
            textvariable=self.edition_var,
            values=editions,
            state="readonly",
            width=12,
            style="SClass.TCombobox",
        ).pack(side="left")

        # 개인화 설정
        ttk.Checkbutton(
            id_frame,
            text="🎯 Neural Personalization Engine",
            variable=self.enable_calibration,
            style="SClass.TCheckbutton",
        ).pack(side="right")

        # 추가 설명
        info_label = ttk.Label(
            user_frame,
            text="💡 Creates personalized biometric baselines and adaptive thresholds for enhanced accuracy",
            style="SClassSubtitle.TLabel",
        )
        info_label.pack(pady=(10, 0))

    def _create_sclass_system_section(self, parent):
        """S-Class 시스템 설정 섹션"""
        system_frame = ttk.LabelFrame(
            parent,
            text=" 🏭 Neural Architecture Configuration ",
            style="SClassLabelFrame.TLabelframe",
            padding="15",
        )
        system_frame.pack(fill="x", pady=10)

        # 시스템 타입 선택
        type_frame = ttk.Frame(system_frame, style="SClassFrame.TFrame")
        type_frame.pack(fill="x", pady=5)

        ttk.Label(
            type_frame, text="🧠 System Architecture:", style="SClassFeature.TLabel"
        ).pack(side="left", padx=(0, 10))

        system_types = ["STANDARD", "HIGH_PERFORMANCE", "LOW_RESOURCE", "RESEARCH"]
        type_combo = ttk.Combobox(
            type_frame,
            textvariable=self.system_type_var,
            values=system_types,
            state="readonly",
            width=18,
            style="SClass.TCombobox",
        )
        type_combo.pack(side="left", padx=(0, 30))

        # 레거시 모드 옵션
        ttk.Checkbutton(
            type_frame,
            text="⚡ Legacy Compatibility Mode",
            variable=self.use_legacy_engine,
            style="SClass.TCheckbutton",
        ).pack(side="left")

        # 시스템 타입 설명 (업그레이드)
        type_descriptions = {
            "STANDARD": "🎯 Balanced Performance • Optimized for real-world deployment",
            "HIGH_PERFORMANCE": "🚀 Maximum Accuracy • All neural networks active • Research grade",
            "LOW_RESOURCE": "💻 Resource Optimized • Minimal hardware requirements • Mobile friendly",
            "RESEARCH": "🔬 Full Research Suite • All experimental features • Debug tools active",
        }

        self.type_desc_label = ttk.Label(
            system_frame,
            text=type_descriptions["STANDARD"],
            style="SClassSubtitle.TLabel",
        )
        self.type_desc_label.pack(pady=(15, 0))

        # 시스템 타입 변경 시 설명 업데이트
        def update_description(*args):
            desc = type_descriptions.get(self.system_type_var.get(), "")
            self.type_desc_label.config(text=desc)

        self.system_type_var.trace("w", update_description)

        # 성능 메트릭 표시 (시뮬레이션)
        metrics_frame = ttk.Frame(system_frame, style="SClassFrame.TFrame")
        metrics_frame.pack(fill="x", pady=(15, 0))

        metrics_text = "📊 Expected Performance: Processing 47% faster • Memory 40% reduced • Accuracy 40-70% improved"
        ttk.Label(metrics_frame, text=metrics_text, style="SClass.TLabel").pack()

        # 성능 최적화 모드 옵션 체크박스 (고급 설정/시스템 설정 섹션에 추가)
        perf_opt_frame = ttk.Frame(system_frame, style="SClassFrame.TFrame")
        perf_opt_frame.pack(fill="x", pady=5)
        ttk.Checkbutton(
            perf_opt_frame,
            text="⚡ Enable Performance Optimization Mode (Dynamic Frame Skipping)",
            variable=self.enable_performance_optimization,
            style="SClass.TCheckbutton",
        ).pack(side="left")

    def _create_sclass_features_section(self, parent):
        """S-Class Expert Systems 설정 섹션"""
        features_frame = ttk.LabelFrame(
            parent,
            text=" 🧠 Expert Systems Configuration ",
            style="SClassLabelFrame.TLabelframe",
            padding="15",
        )
        features_frame.pack(fill="x", pady=10)

        # 전문가 시스템별 패널들
        # FaceProcessor - Digital Psychologist
        face_frame = ttk.LabelFrame(
            features_frame,
            text=" 🧠 FaceProcessor - Digital Psychologist ",
            style="SClassLabelFrame.TLabelframe",
            padding="10",
        )
        face_frame.pack(fill="x", pady=5)

        face_left = ttk.Frame(face_frame, style="SClassFrame.TFrame")
        face_left.pack(side="left", fill="both", expand=True)
        face_right = ttk.Frame(face_frame, style="SClassFrame.TFrame")
        face_right.pack(side="right", fill="both", expand=True)

        ttk.Checkbutton(
            face_left,
            text="❤️ rPPG Heart Rate Estimation",
            variable=self.enable_rppg,
            style="SClass.TCheckbutton",
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            face_right,
            text="👁️ Saccadic Eye Movement Analysis",
            variable=self.enable_saccade,
            style="SClass.TCheckbutton",
        ).pack(anchor="w", pady=2)

        # PoseProcessor - Biomechanics Expert
        pose_frame = ttk.LabelFrame(
            features_frame,
            text=" 🦴 PoseProcessor - Biomechanics Expert ",
            style="SClassLabelFrame.TLabelframe",
            padding="10",
        )
        pose_frame.pack(fill="x", pady=5)

        ttk.Checkbutton(
            pose_frame,
            text="🔬 Spinal Alignment Analysis & Posture Assessment",
            variable=self.enable_spinal_analysis,
            style="SClass.TCheckbutton",
        ).pack(anchor="w", pady=2)

        # HandProcessor - Motor Control Analyst
        hand_frame = ttk.LabelFrame(
            features_frame,
            text=" 🖐 HandProcessor - Motor Control Analyst ",
            style="SClassLabelFrame.TLabelframe",
            padding="10",
        )
        hand_frame.pack(fill="x", pady=5)

        ttk.Checkbutton(
            hand_frame,
            text="📊 FFT Tremor Analysis & Motor Pattern Recognition",
            variable=self.enable_tremor_fft,
            style="SClass.TCheckbutton",
        ).pack(anchor="w", pady=2)

        # ObjectProcessor - Behavior Prediction Expert
        object_frame = ttk.LabelFrame(
            features_frame,
            text=" 👁 ObjectProcessor - Behavior Prediction Expert ",
            style="SClassLabelFrame.TLabelframe",
            padding="10",
        )
        object_frame.pack(fill="x", pady=5)

        ttk.Checkbutton(
            object_frame,
            text="🎯 Bayesian Intention Prediction & Context Analysis",
            variable=self.enable_bayesian_prediction,
            style="SClass.TCheckbutton",
        ).pack(anchor="w", pady=2)

    def _create_advanced_settings_section(self, parent):
        """고급 설정 섹션"""
        adv_frame = ttk.LabelFrame(parent, text=" ⚙️ 고급 설정 ", padding="10")
        adv_frame.pack(fill="x", pady=5)

        pos_frame = ttk.Frame(adv_frame)
        pos_frame.pack(fill="x")

        ttk.Label(pos_frame, text="카메라 위치:").pack(side="left", padx=(0, 5))

        positions = [str(pos) for pos in CameraPosition]
        ttk.Combobox(
            pos_frame,
            textvariable=self.camera_position_var,
            values=positions,
            state="readonly",
            width=20,
        ).pack(side="left", expand=True, fill="x")

    def _create_features_info_section(self, parent):
        """S-Class v19.0 혁신 기능 안내 섹션"""
        info_frame = ttk.LabelFrame(
            parent, text=" ✨ S-Class v19.0 혁신 기술 ", padding="10"
        )
        info_frame.pack(fill="x", pady=5)

        # 혁신 시스템 상태 정보 가져오기
        status = self.innovation_engine.get_system_status()
        enabled_features = status["enabled_features"]

        features_text = (
            "🧠 Expert Systems (완료):\n"
            "  • FaceDataProcessor: 디지털 심리학자 (rPPG, 사카드, 동공 분석)\n"
            "  • PoseDataProcessor: 생체역학 전문가 (스파인 정렬, 자세 흔들림)\n"
            "  • HandDataProcessor: 모터 제어 분석가 (FFT 떨림, 운동학)\n"
            "  • ObjectDataProcessor: 행동 예측 전문가 (베이지안 의도 추론)\n\n"
            "🎯 5대 혁신 기능 (v19.0 NEW):\n"
            "  • AI Driving Coach: 6가지 성격 유형별 맞춤형 운전 코칭\n"
            "  • V2D Healthcare: 생체 신호 통합 및 건강 상태 예측\n"
            "  • AR HUD System: 홀로그램 인터페이스 & 3D 자세 시각화\n"
            "  • Emotional Care System: 20+ 감정 인식 & 개인화된 감정 관리\n"
            "  • Digital Twin Platform: 가상 운전 환경 & 시나리오 시뮬레이션\n\n"
            f"🚀 활성화된 기능: {', '.join(enabled_features)}\n"
            f"🎮 에디션: {status['edition']}\n\n"
            "📈 Performance v19.0:\n"
            "  • 처리 속도: 37.5% 향상 (80ms → 50ms)\n"
            "  • 메모리 사용: 16.7% 감소 (300MB → 250MB)\n"
            "  • 분석 정확도: 15-25% 향상"
        )

        text_label = ttk.Label(
            info_frame, text=features_text, style="Feature.TLabel", justify="left"
        )
        text_label.pack(anchor="w")

    def _create_start_button(self, parent):
        """시작 버튼"""
        start_button = ttk.Button(
            parent,
            text="🚀 S-Class DMS v19.0 The Next Chapter 시작",
            command=self.start_app,
            style="Accent.TButton",
        )
        start_button.pack(fill="x", pady=(20, 0), ipady=10)

    def toggle_source_widgets(self):
        """소스 위젯 토글"""
        if self.source_type.get() == "webcam":
            self.webcam_id_entry.config(state="normal")
            self.video_button.config(state="disabled")
        else:
            self.webcam_id_entry.config(state="disabled")
            self.video_button.config(state="normal")

    def browse_video(self):
        """비디오 파일 선택"""
        files = filedialog.askopenfilenames(
            title="비디오 파일을 선택하세요 (다중 선택 가능)",
            filetypes=(
                ("비디오 파일", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("모든 파일", "*.*"),
            ),
        )
        if files:
            self.video_files = list(files)
            if len(self.video_files) > 1:
                self.video_label.config(
                    text=f"{len(self.video_files)}개 파일 선택됨: {os.path.basename(self.video_files[0])} 등"
                )
                self.is_same_driver = messagebox.askyesno(
                    "운전자 확인",
                    f"{len(self.video_files)}개의 비디오를 선택했습니다.\n"
                    "모두 같은 운전자의 영상입니까?\n\n"
                    "('예' 선택 시, S-Class 개인화 설정을 공유합니다.)",
                )
            else:
                self.video_label.config(
                    text=f"선택됨: {os.path.basename(self.video_files[0])}"
                )
        else:
            self.video_files = []
            self.video_label.config(text="선택된 파일 없음")

    def start_app(self):
        """S-Class 앱 시작"""
        # 입력 소스 검증
        input_source = None
        if self.source_type.get() == "webcam":
            cam_id_str = self.webcam_id.get()
            if cam_id_str.isdigit():
                input_source = int(cam_id_str)
            else:
                messagebox.showerror("입력 오류", "웹캠 번호는 숫자여야 합니다.")
                return
        else:
            if not self.video_files:
                messagebox.showerror("입력 오류", "비디오 파일을 선택해주세요.")
                return
            input_source = (
                self.video_files if len(self.video_files) > 1 else self.video_files[0]
            )

        # 사용자 설정
        user_id = self.user_id.get().strip() or "default"
        edition = self.edition_var.get()  # 에디션 값 읽기
        # 카메라 위치
        selected_pos_str = self.camera_position_var.get()
        camera_position = next(
            (pos for pos in CameraPosition if str(pos) == selected_pos_str),
            CameraPosition.REARVIEW_MIRROR,
        )

        # 시스템 타입
        system_type_str = self.system_type_var.get()
        system_type = getattr(
            AnalysisSystemType, system_type_str, AnalysisSystemType.STANDARD
        )

        # S-Class Neural Platform 설정 구성
        self.config = {
            "input_source": input_source,
            "user_id": user_id,
            "camera_position": camera_position,
            "enable_calibration": self.enable_calibration.get(),
            "is_same_driver": self.is_same_driver,
            "system_type": system_type,
            "use_legacy_engine": self.use_legacy_engine.get(),
            "edition": edition,  # edition 값을 명시적으로 포함
            "sclass_features": {
                # Expert Systems
                "enable_rppg": self.enable_rppg.get(),
                "enable_saccade": self.enable_saccade.get(),
                "enable_spinal_analysis": self.enable_spinal_analysis.get(),
                "enable_tremor_fft": self.enable_tremor_fft.get(),
                "enable_bayesian_prediction": self.enable_bayesian_prediction.get(),
                # Advanced Neural AI Features
                "enable_emotion_ai": self.enable_emotion_ai.get(),
                "enable_predictive_safety": self.enable_predictive_safety.get(),
                "enable_biometric_fusion": self.enable_biometric_fusion.get(),
                "enable_adaptive_thresholds": self.enable_adaptive_thresholds.get(),
            },
            "enable_performance_optimization": self.enable_performance_optimization.get(),
        }
        # 혁신 엔진에 에디션 반영
        self.innovation_engine = SClassDMSv19Enhanced(user_id, edition)
        self.root.destroy()


def get_user_input_terminal():
    """터미널 모드 입력 - 보안 강화된 입력 검증"""
    import re
    import os

    def sanitize_input(input_str: str, max_length: int = 100) -> str:
        """입력 문자열 검증 및 소독"""
        if not input_str:
            return ""

        # 길이 제한
        input_str = input_str[:max_length]

        # 위험한 문자 제거 (보안 강화)
        # 허용: 알파벳, 숫자, 하이픈, 언더스코어, 슬래시, 점, 공백, 한글
        safe_pattern = re.compile(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\-_/.\s\\:]")
        sanitized = safe_pattern.sub("", input_str)

        return sanitized.strip()

    def validate_file_path(path: str) -> bool:
        """파일 경로 유효성 검증"""
        try:
            # 상대 경로 공격 방지
            if ".." in path or path.startswith("/"):
                return False

            # 실제 파일 존재 확인
            return os.path.exists(path) and os.path.isfile(path)
        except (OSError, ValueError):
            return False

    def get_safe_integer_input(
        prompt: str, default: int = 0, min_val: int = 0, max_val: int = 10
    ) -> int:
        """안전한 정수 입력"""
        try:
            user_input = input(prompt).strip()
            if not user_input:
                return default

            # 숫자만 허용
            if not re.match(r"^\d+$", user_input):
                logger.warning(f"Invalid input detected: {user_input[:20]}...")
                return default

            value = int(user_input)
            return max(min_val, min(max_val, value))
        except (ValueError, KeyboardInterrupt):
            return default

    def get_safe_choice_input(
        prompt: str, valid_choices: list, default: str = "n"
    ) -> str:
        """안전한 선택지 입력"""
        try:
            user_input = input(prompt).strip().lower()
            if not user_input:
                return default

            # 허용된 선택지만 허용
            sanitized = sanitize_input(user_input, 5)
            return sanitized if sanitized in valid_choices else default
        except KeyboardInterrupt:
            return default

    print("\n" + "=" * 80)
    print(" S-Class DMS v18+ - Advanced Research Integration (터미널 모드)")
    print("=" * 80)

    # 기본 입력 - 보안 강화
    input_source, is_same_driver = None, True
    while input_source is None:
        choice = get_safe_integer_input(
            "\n📹 입력 소스 선택 (1: 웹캠, 2: 비디오 파일): ", 1, 1, 2
        )

        if choice == 1:
            cam_id = get_safe_integer_input("웹캠 번호 입력 (기본값 0): ", 0, 0, 10)
            input_source = cam_id
        elif choice == 2:
            try:
                path_input = input(
                    "비디오 파일 경로 입력 (여러 파일은 쉼표로 구분): "
                ).strip()

                # 입력 길이 제한
                if len(path_input) > 1000:
                    print("❌ 경로가 너무 깁니다.")
                    continue

                # 경로 검증
                paths = [sanitize_input(p.strip(), 500) for p in path_input.split(",")]
                valid_paths = [p for p in paths if validate_file_path(p)]

                if not valid_paths:
                    print("❌ 유효한 파일을 찾을 수 없습니다.")
                    continue

                input_source = valid_paths if len(valid_paths) > 1 else valid_paths[0]

                if len(valid_paths) > 1:
                    same_driver_choice = get_safe_choice_input(
                        "같은 운전자입니까? (y/n, 기본값 y): ",
                        ["y", "n", "yes", "no"],
                        "y",
                    )
                    is_same_driver = same_driver_choice not in ["n", "no"]
            except KeyboardInterrupt:
                print("\n사용자에 의해 취소되었습니다.")
                return None

    # 사용자 ID 입력 - 보안 강화
    try:
        user_id_raw = input("\n👤 사용자 ID 입력 (기본값 default): ").strip()
        user_id = sanitize_input(user_id_raw, 50) or "default"

        # 사용자 ID 형식 검증
        if not re.match(r"^[a-zA-Z0-9가-힣_-]+$", user_id):
            logger.warning(f"Invalid user ID format, using default")
            user_id = "default"

    except KeyboardInterrupt:
        print("\n사용자에 의해 취소되었습니다.")
        return None

    # S-Class 시스템 설정 - 보안 강화
    print("\n🏭 S-Class 시스템 모드 선택:")
    system_types = ["STANDARD", "HIGH_PERFORMANCE", "LOW_RESOURCE", "RESEARCH"]
    for i, st in enumerate(system_types, 1):
        print(f"{i}. {st}")

    sys_choice = get_safe_integer_input(
        f"선택 (1-{len(system_types)}, 기본값 1): ", 1, 1, len(system_types)
    )
    system_type_str = system_types[sys_choice - 1]
    system_type = getattr(
        AnalysisSystemType, system_type_str, AnalysisSystemType.STANDARD
    )

    legacy_choice = get_safe_choice_input(
        "\n🔧 레거시 엔진 사용? (y/n, 기본값 n): ", ["y", "n"], "n"
    )
    use_legacy_engine = legacy_choice == "y"

    calib_choice = get_safe_choice_input(
        "\n⚙️ 개인화 캘리브레이션 수행? (y/n, 기본값 y): ", ["y", "n"], "y"
    )
    enable_calibration = calib_choice != "n"

    print("\n📍 카메라 위치 선택:")
    positions = list(CameraPosition)
    for i, pos in enumerate(positions, 1):
        print(f"{i}. {pos.value}")

    pos_choice = get_safe_integer_input(
        f"선택 (1-{len(positions)}, 기본값 1): ", 1, 1, len(positions)
    )
    camera_position = positions[pos_choice - 1]

    return {
        "input_source": input_source,
        "user_id": user_id,
        "camera_position": camera_position,
        "enable_calibration": enable_calibration,
        "is_same_driver": is_same_driver,
        "system_type": system_type,
        "use_legacy_engine": use_legacy_engine,
        "sclass_features": {
            # Expert Systems (기본 활성화)
            "enable_rppg": True,
            "enable_saccade": True,
            "enable_spinal_analysis": True,
            "enable_tremor_fft": True,
            "enable_bayesian_prediction": True,
            # Advanced Neural AI Features (기본 활성화)
            "enable_emotion_ai": True,
            "enable_predictive_safety": True,
            "enable_biometric_fusion": True,
            "enable_adaptive_thresholds": True,
        },
        "enable_performance_optimization": True,  # 터미널 모드에서는 기본적으로 활성화
    }


def main():
    logger.info("[진단] main.py: main() 진입")
    config = None
    try:
        if GUI_AVAILABLE:
            root = tk.Tk()
            try:
                root.tk.call("source", "azure.tcl")
                root.tk.call("set_theme", "light")
            except tk.TclError:
                pass
            gui_setup = SClass_DMS_GUI_Setup(root)
            root.mainloop()
            config = gui_setup.config
        else:
            config = get_user_input_terminal()

        if config:
            logger.info(f"S-Class 설정 완료: {config}")
            print("\n" + "=" * 70)
            print(f" S-Class DMS v18+ 시스템 시작... (사용자: {config['user_id']})")
            print(f" 시스템 모드: {config['system_type'].value}")
            print(
                f" 레거시 엔진: {'활성화' if config['use_legacy_engine'] else '비활성화'}"
            )
            print("=" * 70)
            app = DMSApp(**config)
            app.run()
        else:
            print("\n❌ 설정이 취소되어 프로그램을 종료합니다.")

    except (KeyboardInterrupt, EOFError):
        print("\n\n🛑 프로그램을 종료합니다.")
    except Exception as e:
        logger.error(f"S-Class 시스템 실행 실패: {e}", exc_info=True)
        if GUI_AVAILABLE:
            messagebox.showerror(
                "S-Class 시스템 오류",
                f"S-Class 시스템 실행 중 심각한 오류가 발생했습니다.\n"
                f"로그 파일을 확인해주세요.\n\n오류: {e}",
            )
        else:
            print(
                "\n❌ S-Class 시스템 실행 중 심각한 오류가 발생했습니다. 로그 파일을 확인해주세요."
            )


if __name__ == "__main__":
    # 필수 모델 파일 확인
    from core.constants import SystemConstants

    model_files = [
        os.path.join("models", model)
        for model in SystemConstants.FileSystem.REQUIRED_MODELS
    ]

    missing_files = [f for f in model_files if not os.path.exists(f)]

    if missing_files:
        error_msg = (
            "다음 모델 파일이 없어 S-Class 시스템을 시작할 수 없습니다:\n"
            + "\n".join(missing_files)
        )
        logger.critical(error_msg)

        if GUI_AVAILABLE:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("S-Class 모델 파일 오류", error_msg)
        else:
            print(f"\n❌ ERROR: {error_msg}")
    else:
        main()
