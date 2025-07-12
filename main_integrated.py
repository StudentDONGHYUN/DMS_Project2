"""
S-Class DMS Integrated Main System
통합된 운전자 모니터링 시스템 메인 진입점
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import os
import sys
import asyncio
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# Core imports
from core.definitions import CameraPosition
from config.settings import get_config, FeatureFlagConfig
from models.data_structures import UIState
from utils.logging_v2 import setup_logging_system

# System imports
from systems.dms_system import DMSSystem
from systems.mediapipe_manager import MediaPipeManager
from systems.performance import PerformanceOptimizer
from systems.personalization import PersonalizationEngine

# IO Handler imports
from io_handler.video_input_v2 import VideoInputManager
from io_handler.ui_handler import UIHandler

# Innovation Systems (v19 features)
from systems.ai_driving_coach import AIDrivingCoach
from systems.v2d_healthcare import V2DHealthcareSystem
from systems.ar_hud_system import ARHUDSystem
from systems.emotional_care_system import EmotionalCareSystem
from systems.digital_twin_platform import DigitalTwinPlatform

# Setup logging
setup_logging_system()
logger = logging.getLogger(__name__)

GUI_AVAILABLE = True


class DMSSystemGUI:
    """DMS 시스템 GUI 설정 인터페이스"""

    def __init__(self, root):
        self.root = root
        self.root.title("🚗 S-Class DMS - Integrated Driver Monitoring System")
        self.root.geometry("900x1200")
        self.root.configure(bg='#1a1a2e')
        
        # Configuration
        self.config = get_config()
        self.video_files = []
        self.is_same_driver = True
        
        # System variables
        self.source_type = tk.StringVar(value="webcam")
        self.webcam_id = tk.StringVar(value="0")
        self.user_id = tk.StringVar(value="default")
        self.enable_calibration = tk.BooleanVar(value=True)
        self.camera_position_var = tk.StringVar(value=str(CameraPosition.REARVIEW_MIRROR))
        
        # System configuration
        self.system_edition = tk.StringVar(value="RESEARCH")
        self.enable_advanced_features = tk.BooleanVar(value=True)
        self.enable_neural_ai = tk.BooleanVar(value=True)
        self.enable_innovation_features = tk.BooleanVar(value=True)
        
        # Feature toggles
        self.enable_rppg = tk.BooleanVar(value=True)
        self.enable_saccade = tk.BooleanVar(value=True)
        self.enable_spinal_analysis = tk.BooleanVar(value=True)
        self.enable_tremor_fft = tk.BooleanVar(value=True)
        self.enable_bayesian_prediction = tk.BooleanVar(value=True)
        self.enable_emotion_ai = tk.BooleanVar(value=True)
        self.enable_predictive_safety = tk.BooleanVar(value=True)
        self.enable_biometric_fusion = tk.BooleanVar(value=True)
        self.enable_adaptive_thresholds = tk.BooleanVar(value=True)
        
        # UI state
        self.preview_enabled = tk.BooleanVar(value=False)
        
        # Setup GUI
        self._setup_styles()
        self._create_gui()

    def _setup_styles(self):
        """GUI 스타일 설정"""
        style = ttk.Style()
        
        try:
            style.theme_use('clam')
        except Exception as e:
            logger.debug(f"GUI theme 'clam' setup failed: {e}")
        
        # Color palette
        colors = {
            'bg_primary': '#1a1a2e',
            'bg_secondary': '#16213e',
            'accent_cyan': '#00d4ff',
            'accent_orange': '#ff6b35',
            'text_primary': '#ffffff',
            'text_secondary': '#8a8a8a',
            'success': '#00ff9f',
            'warning': '#ffaa00',
            'danger': '#ff0040'
        }
        
        # Configure styles
        style.configure("Title.TLabel", 
                       font=("Segoe UI", 20, "bold"),
                       foreground=colors['accent_orange'],
                       background=colors['bg_primary'])
        
        style.configure("Subtitle.TLabel", 
                       font=("Segoe UI", 11),
                       foreground=colors['text_secondary'],
                       background=colors['bg_primary'])
        
        style.configure("Feature.TLabel", 
                       font=("Segoe UI", 10),
                       foreground=colors['text_primary'],
                       background=colors['bg_primary'])
        
        style.configure("Button.TButton", 
                       font=("Segoe UI", 12, "bold"),
                       foreground=colors['bg_primary'],
                       background=colors['accent_cyan'])
        
        style.configure("Frame.TFrame",
                       background=colors['bg_secondary'],
                       relief='flat',
                       borderwidth=2)
        
        style.configure("LabelFrame.TLabelframe",
                       background=colors['bg_secondary'],
                       foreground=colors['accent_cyan'],
                       relief='solid',
                       borderwidth=2)
        
        style.configure("LabelFrame.TLabelframe.Label",
                       background=colors['bg_secondary'],
                       foreground=colors['accent_cyan'],
                       font=("Segoe UI", 12, "bold"))

    def _create_gui(self):
        """GUI 생성"""
        # Main frame
        main_frame = ttk.Frame(self.root, style="Frame.TFrame", padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True, pady=(0, 20))
        
        # Create tabs
        self._create_main_tab()
        self._create_system_tab()
        self._create_features_tab()
        self._create_advanced_tab()
        
        # Control panel
        self._create_control_panel(main_frame)

    def _create_main_tab(self):
        """메인 설정 탭"""
        main_tab = ttk.Frame(self.notebook, style="Frame.TFrame")
        self.notebook.add(main_tab, text=" 🏠 Main Settings ")
        
        # Header
        header_frame = ttk.Frame(main_tab, style="Frame.TFrame")
        header_frame.pack(fill="x", pady=(0, 20))
        
        ttk.Label(header_frame, text="S-Class DMS Integrated System", 
                 style="Title.TLabel").pack()
        ttk.Label(header_frame, text="Advanced Driver Monitoring with AI Innovation", 
                 style="Subtitle.TLabel").pack()
        
        # Input source section
        self._create_input_source_section(main_tab)
        
        # User settings section
        self._create_user_settings_section(main_tab)
        
        # System edition section
        self._create_system_edition_section(main_tab)

    def _create_system_tab(self):
        """시스템 설정 탭"""
        system_tab = ttk.Frame(self.notebook, style="Frame.TFrame")
        self.notebook.add(system_tab, text=" ⚙️ System Configuration ")
        
        # System features
        self._create_system_features_section(system_tab)
        
        # Performance settings
        self._create_performance_section(system_tab)

    def _create_features_tab(self):
        """기능 설정 탭"""
        features_tab = ttk.Frame(self.notebook, style="Frame.TFrame")
        self.notebook.add(features_tab, text=" 🧠 Advanced Features ")
        
        # Core features
        self._create_core_features_section(features_tab)
        
        # Innovation features
        self._create_innovation_features_section(features_tab)

    def _create_advanced_tab(self):
        """고급 설정 탭"""
        advanced_tab = ttk.Frame(self.notebook, style="Frame.TFrame")
        self.notebook.add(advanced_tab, text=" 🔬 Research & Development ")
        
        # Research features
        self._create_research_features_section(advanced_tab)
        
        # Debug options
        self._create_debug_section(advanced_tab)

    def _create_input_source_section(self, parent):
        """입력 소스 설정 섹션"""
        frame = ttk.LabelFrame(parent, text="Input Source Configuration", 
                              style="LabelFrame.TLabelframe")
        frame.pack(fill="x", pady=(0, 10))
        
        # Source type selection
        source_frame = ttk.Frame(frame)
        source_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(source_frame, text="Source Type:", style="Feature.TLabel").pack(side="left")
        
        source_options = ttk.Frame(source_frame)
        source_options.pack(side="right")
        
        ttk.Radiobutton(source_options, text="Webcam", variable=self.source_type, 
                       value="webcam", style="Feature.TLabel").pack(side="left")
        ttk.Radiobutton(source_options, text="Video File", variable=self.source_type, 
                       value="video", style="Feature.TLabel").pack(side="left")
        
        # Webcam ID
        webcam_frame = ttk.Frame(frame)
        webcam_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(webcam_frame, text="Webcam ID:", style="Feature.TLabel").pack(side="left")
        ttk.Entry(webcam_frame, textvariable=self.webcam_id, width=10).pack(side="right")
        
        # Video file selection
        video_frame = ttk.Frame(frame)
        video_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(video_frame, text="Browse Video Files", 
                  command=self.browse_video, style="Button.TButton").pack(side="right")
        
        # Camera position
        position_frame = ttk.Frame(frame)
        position_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(position_frame, text="Camera Position:", style="Feature.TLabel").pack(side="left")
        position_combo = ttk.Combobox(position_frame, textvariable=self.camera_position_var, 
                                     values=[str(pos) for pos in CameraPosition], state="readonly")
        position_combo.pack(side="right")

    def _create_user_settings_section(self, parent):
        """사용자 설정 섹션"""
        frame = ttk.LabelFrame(parent, text="User Configuration", 
                              style="LabelFrame.TLabelframe")
        frame.pack(fill="x", pady=(0, 10))
        
        # User ID
        user_frame = ttk.Frame(frame)
        user_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(user_frame, text="User ID:", style="Feature.TLabel").pack(side="left")
        ttk.Entry(user_frame, textvariable=self.user_id, width=20).pack(side="right")
        
        # Calibration
        calib_frame = ttk.Frame(frame)
        calib_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Checkbutton(calib_frame, text="Enable Calibration", 
                       variable=self.enable_calibration, style="Feature.TLabel").pack(side="left")
        
        # Same driver
        driver_frame = ttk.Frame(frame)
        driver_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Checkbutton(driver_frame, text="Same Driver Session", 
                       variable=self.is_same_driver, style="Feature.TLabel").pack(side="left")

    def _create_system_edition_section(self, parent):
        """시스템 에디션 설정 섹션"""
        frame = ttk.LabelFrame(parent, text="System Edition", 
                              style="LabelFrame.TLabelframe")
        frame.pack(fill="x", pady=(0, 10))
        
        edition_frame = ttk.Frame(frame)
        edition_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(edition_frame, text="Edition:", style="Feature.TLabel").pack(side="left")
        edition_combo = ttk.Combobox(edition_frame, textvariable=self.system_edition,
                                   values=["COMMUNITY", "PRO", "ENTERPRISE", "RESEARCH"], 
                                   state="readonly")
        edition_combo.pack(side="right")
        
        # Edition description
        desc_frame = ttk.Frame(frame)
        desc_frame.pack(fill="x", padx=10, pady=5)
        
        self.edition_description = ttk.Label(desc_frame, text="", style="Subtitle.TLabel")
        self.edition_description.pack()
        
        # Update description when edition changes
        edition_combo.bind('<<ComboboxSelected>>', self._update_edition_description)
        self._update_edition_description()

    def _create_system_features_section(self, parent):
        """시스템 기능 설정 섹션"""
        frame = ttk.LabelFrame(parent, text="System Features", 
                              style="LabelFrame.TLabelframe")
        frame.pack(fill="x", pady=(0, 10))
        
        # Feature toggles
        features_frame = ttk.Frame(frame)
        features_frame.pack(fill="x", padx=10, pady=5)
        
        # Column 1
        col1 = ttk.Frame(features_frame)
        col1.pack(side="left", fill="x", expand=True)
        
        ttk.Checkbutton(col1, text="Advanced Features", 
                       variable=self.enable_advanced_features, style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(col1, text="Neural AI", 
                       variable=self.enable_neural_ai, style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(col1, text="Innovation Features", 
                       variable=self.enable_innovation_features, style="Feature.TLabel").pack(anchor="w")
        
        # Column 2
        col2 = ttk.Frame(features_frame)
        col2.pack(side="right", fill="x", expand=True)
        
        ttk.Checkbutton(col2, text="Adaptive Thresholds", 
                       variable=self.enable_adaptive_thresholds, style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(col2, text="Performance Optimization", 
                       variable=tk.BooleanVar(value=True), style="Feature.TLabel").pack(anchor="w")

    def _create_performance_section(self, parent):
        """성능 설정 섹션"""
        frame = ttk.LabelFrame(parent, text="Performance Settings", 
                              style="LabelFrame.TLabelframe")
        frame.pack(fill="x", pady=(0, 10))
        
        # Target FPS
        fps_frame = ttk.Frame(frame)
        fps_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(fps_frame, text="Target FPS:", style="Feature.TLabel").pack(side="left")
        fps_var = tk.StringVar(value="30")
        ttk.Entry(fps_frame, textvariable=fps_var, width=10).pack(side="right")
        
        # Processing mode
        mode_frame = ttk.Frame(frame)
        mode_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(mode_frame, text="Processing Mode:", style="Feature.TLabel").pack(side="left")
        mode_var = tk.StringVar(value="BALANCED")
        mode_combo = ttk.Combobox(mode_frame, textvariable=mode_var,
                                 values=["LOW_RESOURCE", "BALANCED", "HIGH_PERFORMANCE"], 
                                 state="readonly")
        mode_combo.pack(side="right")

    def _create_core_features_section(self, parent):
        """핵심 기능 설정 섹션"""
        frame = ttk.LabelFrame(parent, text="Core Analysis Features", 
                              style="LabelFrame.TLabelframe")
        frame.pack(fill="x", pady=(0, 10))
        
        # Core features
        features_frame = ttk.Frame(frame)
        features_frame.pack(fill="x", padx=10, pady=5)
        
        # Column 1
        col1 = ttk.Frame(features_frame)
        col1.pack(side="left", fill="x", expand=True)
        
        ttk.Checkbutton(col1, text="rPPG Heart Rate", 
                       variable=self.enable_rppg, style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(col1, text="Saccade Analysis", 
                       variable=self.enable_saccade, style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(col1, text="Spinal Alignment", 
                       variable=self.enable_spinal_analysis, style="Feature.TLabel").pack(anchor="w")
        
        # Column 2
        col2 = ttk.Frame(features_frame)
        col2.pack(side="right", fill="x", expand=True)
        
        ttk.Checkbutton(col2, text="Tremor FFT Analysis", 
                       variable=self.enable_tremor_fft, style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(col2, text="Bayesian Prediction", 
                       variable=self.enable_bayesian_prediction, style="Feature.TLabel").pack(anchor="w")

    def _create_innovation_features_section(self, parent):
        """혁신 기능 설정 섹션"""
        frame = ttk.LabelFrame(parent, text="Innovation Features", 
                              style="LabelFrame.TLabelframe")
        frame.pack(fill="x", pady=(0, 10))
        
        # Innovation features
        features_frame = ttk.Frame(frame)
        features_frame.pack(fill="x", padx=10, pady=5)
        
        # Column 1
        col1 = ttk.Frame(features_frame)
        col1.pack(side="left", fill="x", expand=True)
        
        ttk.Checkbutton(col1, text="Emotion AI", 
                       variable=self.enable_emotion_ai, style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(col1, text="Predictive Safety", 
                       variable=self.enable_predictive_safety, style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(col1, text="Biometric Fusion", 
                       variable=self.enable_biometric_fusion, style="Feature.TLabel").pack(anchor="w")
        
        # Column 2
        col2 = ttk.Frame(features_frame)
        col2.pack(side="right", fill="x", expand=True)
        
        ttk.Checkbutton(col2, text="AI Driving Coach", 
                       variable=tk.BooleanVar(value=True), style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(col2, text="V2D Healthcare", 
                       variable=tk.BooleanVar(value=True), style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(col2, text="AR HUD System", 
                       variable=tk.BooleanVar(value=False), style="Feature.TLabel").pack(anchor="w")

    def _create_research_features_section(self, parent):
        """연구 기능 설정 섹션"""
        frame = ttk.LabelFrame(parent, text="Research & Development", 
                              style="LabelFrame.TLabelframe")
        frame.pack(fill="x", pady=(0, 10))
        
        # Research features
        features_frame = ttk.Frame(frame)
        features_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Checkbutton(features_frame, text="Digital Twin Platform", 
                       variable=tk.BooleanVar(value=True), style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(features_frame, text="Advanced Cognitive Modeling", 
                       variable=tk.BooleanVar(value=True), style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(features_frame, text="Experimental Algorithms", 
                       variable=tk.BooleanVar(value=True), style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(features_frame, text="Detailed Logging", 
                       variable=tk.BooleanVar(value=True), style="Feature.TLabel").pack(anchor="w")

    def _create_debug_section(self, parent):
        """디버그 설정 섹션"""
        frame = ttk.LabelFrame(parent, text="Debug Options", 
                              style="LabelFrame.TLabelframe")
        frame.pack(fill="x", pady=(0, 10))
        
        # Debug options
        debug_frame = ttk.Frame(frame)
        debug_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Checkbutton(debug_frame, text="Enable Preview Mode", 
                       variable=self.preview_enabled, style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(debug_frame, text="Verbose Logging", 
                       variable=tk.BooleanVar(value=False), style="Feature.TLabel").pack(anchor="w")
        ttk.Checkbutton(debug_frame, text="Performance Monitoring", 
                       variable=tk.BooleanVar(value=True), style="Feature.TLabel").pack(anchor="w")

    def _create_control_panel(self, parent):
        """제어 패널"""
        control_frame = ttk.Frame(parent, style="Frame.TFrame")
        control_frame.pack(fill="x", pady=(20, 0))
        
        # Start button
        start_button = ttk.Button(control_frame, text="🚀 Start DMS System", 
                                 command=self.start_system, style="Button.TButton")
        start_button.pack(pady=10)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready to start", 
                                     style="Subtitle.TLabel")
        self.status_label.pack()

    def _update_edition_description(self, event=None):
        """에디션 설명 업데이트"""
        descriptions = {
            "COMMUNITY": "Basic features for community users",
            "PRO": "Advanced features for professional use",
            "ENTERPRISE": "Enterprise-grade features with AI",
            "RESEARCH": "Full research capabilities with innovation features"
        }
        
        edition = self.system_edition.get()
        description = descriptions.get(edition, "Unknown edition")
        self.edition_description.config(text=description)

    def browse_video(self):
        """비디오 파일 선택"""
        files = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if files:
            self.video_files = list(files)
            messagebox.showinfo("Files Selected", f"Selected {len(self.video_files)} video files")

    def start_system(self):
        """시스템 시작"""
        try:
            # Get configuration
            config = self._get_system_config()
            
            # Update status
            self.status_label.config(text="Starting system...")
            self.root.update()
            
            # Start system asynchronously
            asyncio.run(self._start_system_async(config))
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            messagebox.showerror("Error", f"Failed to start system: {e}")
            self.status_label.config(text="Failed to start")

    async def _start_system_async(self, config: Dict[str, Any]):
        """비동기 시스템 시작"""
        try:
            # Create DMS system
            dms_system = DMSSystem(
                user_id=config['user_id'],
                camera_position=config['camera_position'],
                enable_calibration=config['enable_calibration'],
                system_edition=config['system_edition'],
                feature_flags=config['feature_flags']
            )
            
            # Initialize system
            await dms_system.initialize()
            
            # Start system
            await dms_system.start()
            
            # Update status
            self.status_label.config(text="System running")
            
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            raise

    def _get_system_config(self) -> Dict[str, Any]:
        """시스템 설정 생성"""
        # Create feature flags
        feature_flags = FeatureFlagConfig(edition=self.system_edition.get())
        
        # Update feature flags based on GUI selections
        feature_flags.enable_rppg_heart_rate = self.enable_rppg.get()
        feature_flags.enable_saccade_analysis = self.enable_saccade.get()
        feature_flags.enable_spinal_alignment = self.enable_spinal_analysis.get()
        feature_flags.enable_fft_tremor_analysis = self.enable_tremor_fft.get()
        feature_flags.enable_bayesian_prediction = self.enable_bayesian_prediction.get()
        feature_flags.enable_emotion_ai = self.enable_emotion_ai.get()
        feature_flags.enable_predictive_safety = self.enable_predictive_safety.get()
        feature_flags.enable_biometric_fusion = self.enable_biometric_fusion.get()
        
        return {
            'user_id': self.user_id.get(),
            'camera_position': CameraPosition(self.camera_position_var.get()),
            'enable_calibration': self.enable_calibration.get(),
            'is_same_driver': self.is_same_driver,
            'system_edition': self.system_edition.get(),
            'source_type': self.source_type.get(),
            'webcam_id': int(self.webcam_id.get()) if self.webcam_id.get().isdigit() else 0,
            'video_files': self.video_files,
            'feature_flags': feature_flags,
            'preview_enabled': self.preview_enabled.get()
        }


def main():
    """메인 함수"""
    try:
        # Create root window
        root = tk.Tk()
        
        # Create GUI
        gui = DMSSystemGUI(root)
        
        # Start GUI
        root.mainloop()
        
    except Exception as e:
        logger.error(f"GUI startup failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()