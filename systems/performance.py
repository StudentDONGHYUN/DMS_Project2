import numpy as np
from collections import deque
import logging
import time
import csv
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """실시간 성능 최적화 시스템"""

    def __init__(self, enable_optimization=True):
        self.performance_history = deque(maxlen=300)
        self.optimization_active = False
        self.target_fps = 30
        self.min_fps = 15
        self.total_frame_count = 0
        self._last_log_time = 0
        self._log_interval = 0.03
        self.frame_sampling_ratio = 1  # 1: 모든 프레임 처리, 2: 2프레임 중 1프레임만 처리 등
        self._sampling_counter = 0
        self.enable_optimization = enable_optimization  # 사용자가 GUI에서 켜고 끌 수 있음
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"performance_v6_{timestamp}.csv"
        self._init_csv()
        logger.info("PerformanceOptimizer 초기화 완료")

    def _init_csv(self):
        Path("performance_logs").mkdir(exist_ok=True)
        csv_file = Path("performance_logs") / self.csv_path
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'processing_time_ms', 'fps', 'optimization_active'])

    def _log_to_csv(self, processing_time, fps):
        csv_file = Path("performance_logs") / self.csv_path
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), processing_time, fps, self.optimization_active])

    def log_performance(self, processing_time, fps):
        current_time = time.time()
        if current_time - self._last_log_time < self._log_interval:
            return
        self._last_log_time = current_time
        self.total_frame_count += 1
        self.performance_history.append({"timestamp": time.time(), "processing_time": processing_time, "fps": fps})
        if len(self.performance_history) >= 30:
            self._check_performance_issues()
        self._log_to_csv(processing_time, fps)

    def _check_performance_issues(self):
        if not self.enable_optimization:
            if self.optimization_active:
                self._deactivate_optimization()
            return
        recent_performance = list(self.performance_history)[-30:]
        avg_fps = np.mean([p["fps"] for p in recent_performance])
        avg_processing_time = np.mean([p["processing_time"] for p in recent_performance])
        if avg_fps < self.min_fps or avg_processing_time > 200:
            if not self.optimization_active:
                self._activate_optimization()
        elif avg_fps > self.min_fps * 1.2 and avg_processing_time < 100:
            if self.optimization_active:
                self._deactivate_optimization()

    def should_process_frame(self):
        """
        동적 프레임 샘플링: 최적화 모드일 때 일정 비율로 프레임을 스킵
        예: frame_sampling_ratio=2이면 2프레임 중 1프레임만 처리
        """
        if self.frame_sampling_ratio <= 1:
            return True
        self._sampling_counter = (self._sampling_counter + 1) % self.frame_sampling_ratio
        return self._sampling_counter == 0

    def _activate_optimization(self):
        self.optimization_active = True
        self.frame_sampling_ratio = min(self.frame_sampling_ratio + 1, 4)  # 최대 1/4만 처리
        logger.warning(f"성능 최적화 모드 활성화 (프레임 샘플링 비율: {self.frame_sampling_ratio})")
        strategies = ["프레임 스키핑 증가", "분석 해상도 감소", "일부 분석 기능 비활성화", "메모리 정리 강화"]
        for strategy in strategies:
            logger.info(f"적용 전략: {strategy}")

    def _deactivate_optimization(self):
        self.optimization_active = False
        self.frame_sampling_ratio = 1
        logger.info("성능 최적화 모드 비활성화 - 정상 성능 복구 (프레임 샘플링 비율: 1)")

    def get_optimization_status(self):
        if len(self.performance_history) < 10:
            return {"active": False, "avg_fps": 0, "avg_processing_time": 0}
        recent = list(self.performance_history)[-10:]
        return {
            "active": self.optimization_active,
            "avg_fps": np.mean([p["fps"] for p in recent]),
            "avg_processing_time": np.mean([p["processing_time"] for p in recent]),
            "performance_score": self._calculate_performance_score(recent),
        }

    def save_session_summary(self):
        if not self.performance_history:
            return
        summary_file = Path("performance_logs") / f"summary_{self.csv_path.replace('.csv', '.json')}"
        recent_data = list(self.performance_history)
        summary = {
            "session_duration": recent_data[-1]["timestamp"] - recent_data[0]["timestamp"] if recent_data else 0,
            "total_frames": self.total_frame_count,
            "avg_processing_time": np.mean([p["processing_time"] for p in recent_data]),
            "avg_fps": np.mean([p["fps"] for p in recent_data]),
            "optimization_activations": sum(1 for p in recent_data if self.optimization_active)
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"세션 요약 저장: {summary_file}")

    def _calculate_performance_score(self, performance_data):
        avg_fps = np.mean([p["fps"] for p in performance_data])
        avg_time = np.mean([p["processing_time"] for p in performance_data])
        fps_score = min(1.0, avg_fps / self.target_fps)
        time_score = max(0.0, 1.0 - avg_time / 200.0)
        return (fps_score + time_score) / 2.0
