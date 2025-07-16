# DMS 모델 경로 문제 해결 및 점진적 위험도 증가 시스템 구현

## 🔧 해결된 문제점

### 1. 모델 파일 경로 매칭 문제
**문제**: `holistic_landmarker.task` 파일이 없는데 코드에서 사용하려고 함
**해결**: 파일 존재 여부 확인 후 조건부 설정

```python
# systems/mediapipe_manager.py
# 수정 전:
self.task_configs[TaskType.HOLISTIC_LANDMARKER] = TaskConfig(
    task_type=TaskType.HOLISTIC_LANDMARKER,
    model_path=str(self.model_base_path / "holistic_landmarker.task"),
    # ...
)

# 수정 후:
holistic_model_path = self.model_base_path / "holistic_landmarker.task"
if holistic_model_path.exists():
    self.task_configs[TaskType.HOLISTIC_LANDMARKER] = TaskConfig(
        task_type=TaskType.HOLISTIC_LANDMARKER,
        model_path=str(holistic_model_path),
        # ...
    )
else:
    logger.warning(f"Holistic Landmarker 모델 파일을 찾을 수 없습니다: {holistic_model_path}")
```

### 2. 모델 파일 검증 강화
**개선**: 모델 파일 존재 확인 시 더 상세한 디버깅 정보 제공

```python
# systems/mediapipe_manager.py
# 수정 전:
if not Path(config.model_path).exists():
    logger.warning(f"모델 파일 없음: {config.model_path}")
    return False

# 수정 후:
if not Path(config.model_path).exists():
    logger.error(f"모델 파일 없음: {config.model_path}")
    logger.error(f"현재 작업 디렉토리: {Path.cwd()}")
    logger.error(f"models 디렉토리 내용: {list(self.model_base_path.glob('*'))}")
    return False

# 모델 파일 크기 확인
model_size = Path(config.model_path).stat().st_size
logger.info(f"모델 파일 로드 중: {config.model_path} ({model_size / (1024*1024):.2f}MB)")
```

## 🎯 점진적 위험도 증가 시스템 구현

### 1. 손 감지 실패 점진적 위험도 증가
**문제**: 손 감지 실패 시 즉시 `risk_score: 1.0` 설정
**해결**: 10프레임에 걸쳐 점진적 위험도 증가

```python
# analysis/processors/hand_processor.py
class HandDataProcessor:
    def __init__(self, metrics_updater: IMetricsUpdater):
        # ... 기존 코드 ...
        
        # 연속 감지 실패 추적 (점진적 위험도 증가용)
        self.consecutive_failures = 0
        self.max_failures_for_max_risk = 10  # 10프레임 후 최대 위험도

    async def _get_default_hand_analysis(self) -> Dict[str, Any]:
        """Default analysis data to return when no hands are detected"""
        # 연속 감지 실패 횟수 증가
        self.consecutive_failures += 1
        
        # 점진적 위험도 증가 (0.0 -> 1.0)
        risk_score = min(self.consecutive_failures / self.max_failures_for_max_risk, 1.0)
        
        return {
            'hands_detected_count': 0,
            'steering_skill': {'skill_score': 0.0, 'feedback': 'No hands detected', 'components': {}},
            'distraction_behaviors': {'risk_score': risk_score, 'behaviors': ['No hands detected'], 'phone_detected': False},
            'driving_technique': {'technique_rating': 'unknown', 'score': 0.0},
            'overall_hand_safety': 0.0,
            'hand_positions': []
        }

    async def process_data(self, result, timestamp):
        if not result or not hasattr(result, 'hand_landmarks') or not result.hand_landmarks:
            return await self._handle_no_hands_detected()

        # ... 손 처리 로직 ...
        
        # 손 감지 성공 시 연속 실패 카운터 초기화
        self.consecutive_failures = 0
        
        # ... 나머지 코드 ...
```

### 2. 얼굴 감지 실패 점진적 위험도 증가
**개선**: 얼굴 감지 실패 시 졸음 confidence 점진적 증가

```python
# analysis/processors/face_processor.py
class FaceDataProcessor:
    def __init__(self, ...):
        # ... 기존 코드 ...
        
        # 연속 감지 실패 추적 (점진적 위험도 증가용)
        self.consecutive_failures = 0
        self.max_failures_for_max_risk = 10  # 10프레임 후 최대 위험도

    async def _handle_no_face_detected(self) -> Dict[str, Any]:
        """Return default values when no face is detected"""
        logger.warning("No face detected - setting all facial metrics to default values")
        
        # 연속 감지 실패 횟수 증가
        self.consecutive_failures += 1
        
        # 점진적 위험도 증가 (0.0 -> 1.0)
        risk_score = min(self.consecutive_failures / self.max_failures_for_max_risk, 1.0)
        
        default_gaze = self._get_default_pose_gaze_data()['gaze']
        return {
            'face_detected': False,
            'drowsiness': {'status': 'no_face', 'confidence': risk_score},
            'emotion': {'state': None, 'confidence': 0.0},
            'gaze': default_gaze,
            'driver': {'identity': 'unknown', 'confidence': 0.0},
            # ... 나머지 기본값들 ...
        }

    async def process_data(self, data: Any, image: np.ndarray, timestamp: float) -> Dict[str, Any]:
        if not data or not data.face_landmarks:
            return await self._handle_no_face_detected()

        landmarks = data.face_landmarks[0]
        results = {'face_detected': True}
        
        # 얼굴 감지 성공 시 연속 실패 카운터 초기화
        self.consecutive_failures = 0
        
        # ... 나머지 처리 로직 ...
```

### 3. 포즈 감지 실패 점진적 위험도 증가
**개선**: 포즈 감지 실패 시 운전 적합도 점진적 감소

```python
# analysis/processors/pose_processor.py
class PoseDataProcessor:
    def __init__(self, metrics_updater: IMetricsUpdater):
        # ... 기존 코드 ...
        
        # 연속 감지 실패 추적 (점진적 위험도 증가용)
        self.consecutive_failures = 0
        self.max_failures_for_max_risk = 10  # 10프레임 후 최대 위험도

    async def _handle_no_pose_detected(self) -> Dict[str, Any]:
        """Handle no pose detected scenario"""
        logger.warning("No pose detected - backup mode or sensor recalibration needed")
        
        # 연속 감지 실패 횟수 증가
        self.consecutive_failures += 1
        
        # 점진적 위험도 증가 (0.0 -> 1.0)
        risk_score = min(self.consecutive_failures / self.max_failures_for_max_risk, 1.0)
        
        return { 
            'pose_detected': False, 
            'pose_analysis': {
                'driving_suitability': 1.0 - risk_score,  # 위험도가 높을수록 운전 적합도 낮음
                'biomechanical_health': {'overall_score': 1.0 - risk_score, 'risk_factors': ['Pose detection failed'], 'recommendations': ['Adjust camera position']}
            }
        }

    async def process_data(self, result, timestamp):
        if not result or not result.pose_landmarks:
            return await self._handle_no_pose_detected()
        
        pose_result = result
        results = {}
        
        # 포즈 감지 성공 시 연속 실패 카운터 초기화
        self.consecutive_failures = 0
        
        # ... 나머지 처리 로직 ...
```

## 📊 시스템 개선 효과

### 변경 전 (문제 상황)
- **손 감지 실패**: 즉시 `risk_score: 1.0` → `distraction: critical`
- **얼굴 감지 실패**: `drowsiness.confidence: 0.0` (무의미)
- **포즈 감지 실패**: `driving_suitability: 0.0` (과도한 경고)
- **모델 파일 누락**: 초기화 실패 → 전체 시스템 오류

### 변경 후 (개선된 상황)
- **손 감지 실패**: 10프레임에 걸쳐 `risk_score: 0.0 → 1.0` 점진적 증가
- **얼굴 감지 실패**: 점진적 `drowsiness.confidence` 증가
- **포즈 감지 실패**: 점진적 `driving_suitability` 감소
- **모델 파일 누락**: 파일 존재 여부 확인 후 조건부 초기화

## 🔄 점진적 위험도 증가 시스템 작동 방식

### 시나리오 1: 일시적 감지 실패
```
프레임 1: 감지 실패 → risk_score: 0.1 (경고 없음)
프레임 2: 감지 실패 → risk_score: 0.2 (경고 없음)
프레임 3: 감지 성공 → risk_score: 0.0 (정상화)
```

### 시나리오 2: 지속적 감지 실패
```
프레임 1-5: 감지 실패 → risk_score: 0.1-0.5 (warning 수준)
프레임 6-8: 감지 실패 → risk_score: 0.6-0.8 (danger 수준)
프레임 9-10: 감지 실패 → risk_score: 0.9-1.0 (critical 수준)
프레임 11+: 감지 실패 → risk_score: 1.0 (최대 위험도 유지)
```

### 시나리오 3: 간헐적 감지 실패
```
프레임 1-3: 감지 실패 → risk_score: 0.1-0.3
프레임 4: 감지 성공 → risk_score: 0.0 (초기화)
프레임 5-7: 감지 실패 → risk_score: 0.1-0.3 (다시 증가)
```

## 🛠️ 추가 개선 사항

### 1. 설정 가능한 매개변수
```python
# config/settings.py에 추가 권장
@dataclass
class DetectionFailureConfig:
    max_failures_for_max_risk: int = 10  # 최대 위험도까지 프레임 수
    risk_increase_curve: str = "linear"  # "linear", "exponential", "logarithmic"
    reset_on_success: bool = True  # 성공 시 즉시 초기화 여부
```

### 2. 감지 품질 기반 조정
```python
# 감지 품질에 따른 동적 임계값 조정
def adjust_failure_threshold(detection_quality: float) -> int:
    """감지 품질에 따른 실패 임계값 조정"""
    if detection_quality > 0.8:
        return 15  # 높은 품질일 때 더 관대하게
    elif detection_quality > 0.5:
        return 10  # 기본값
    else:
        return 5   # 낮은 품질일 때 더 엄격하게
```

### 3. 환경 적응형 시스템
```python
# 조명 조건, 카메라 각도 등에 따른 동적 조정
def adaptive_risk_calculation(consecutive_failures: int, environment_score: float) -> float:
    """환경 조건을 고려한 위험도 계산"""
    base_risk = consecutive_failures / 10.0
    environment_factor = max(0.5, environment_score)  # 환경이 나쁠수록 더 관대하게
    return min(base_risk / environment_factor, 1.0)
```

## 📈 예상 효과

### 1. 사용자 경험 개선
- 일시적 감지 실패로 인한 false positive 경고 감소
- 점진적 경고 증가로 자연스러운 사용자 반응 유도
- 카메라 위치 조정이나 조명 변화 시 시스템 안정성 향상

### 2. 시스템 안정성 향상
- 모델 파일 누락으로 인한 초기화 실패 방지
- 감지 실패 시 시스템 크래시 방지
- 다양한 환경 조건에서의 강건성 증가

### 3. 진단 및 디버깅 개선
- 상세한 모델 파일 로딩 정보 제공
- 감지 실패 패턴 추적 가능
- 시스템 상태 모니터링 향상

---

**결론**: 모델 경로 매칭 문제를 해결하고 점진적 위험도 증가 시스템을 구현함으로써, DMS 시스템의 안정성과 사용자 경험을 크게 개선했습니다. 이제 일시적 감지 실패에 대해 과도한 경고가 발생하지 않고, 실제 위험 상황에서는 점진적으로 경고 수준이 증가하여 더 자연스러운 사용자 경험을 제공합니다.