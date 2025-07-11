# DMS 프로젝트 디버깅/자동패치/문제해결 히스토리

> **이 파일은 시스템 자동 패치, 구조적 진단, 반복 방지 목적의 모든 변경/시도 내역을 기록합니다.**

---

## [2025-07-10] 자동 패치/문제 해결 내역

### 1. S-Class 프로세서 비동기화
- **적용 파일:**
  - analysis/processors/face_processor_s_class.py
  - analysis/processors/pose_processor_s_class.py
  - analysis/processors/hand_processor_s_class.py
  - analysis/processors/object_processor_s_class.py
- **주요 변경점:**
  - 모든 S-Class 프로세서의 process_data를 async def로 통일
  - 내부 await/비동기 호출 일관성 확보
- **목적:**
  - 병렬/비동기 분석 성능 극대화
  - 호출부 await 오류/TypeError 방지

---

### 2. process_data 인자/await/분기 일치화
- **적용 파일:**
  - analysis/orchestrator/orchestrator_advanced.py
  - app.py
- **주요 변경점:**
  - face 프로세서 호출 시 image 인자 포함 3개 인자(data, image, timestamp)로 통일
  - object 프로세서 async 여부 분기 처리
  - await None 오류 방지
- **목적:**
  - S-Class/일반 프로세서 혼용 환경 완전 호환
  - 인자 누락/await 오류 반복 방지

---

### 3. PersonalizationEngine/DynamicAnalysisEngine 비동기 initialize 구현
- **적용 파일:**
  - systems/personalization.py
  - systems/dynamic.py
- **주요 변경점:**
  - async def initialize(self) 실제 동작 구현
  - 프로필 비동기 저장(aiofiles), 상태 리셋 등
- **목적:**
  - 비동기 컴포넌트 초기화 오류 방지
  - 실시간 환경에서 블로킹 최소화

---

### 4. MediaPipeManager 콜백 비동기 처리
- **적용 파일:**
  - systems/mediapipe_manager.py
- **주요 변경점:**
  - on_face_result 등 콜백이 async면 asyncio.create_task로 실행
- **목적:**
  - coroutine was never awaited 경고/분석 파이프라인 미동작 방지

---

### 5. 입력/MediaPipe/프로세서 데이터 흐름 상세 로깅 (디버깅 자동화)
- **적용 파일:**
  - systems/mediapipe_manager.py
  - analysis/processors/face_processor_s_class.py
  - analysis/processors/pose_processor_s_class.py
  - analysis/processors/hand_processor_s_class.py
  - analysis/processors/object_processor_s_class.py
- **주요 변경점:**
  - 입력 프레임 shape/dtype/min/max, MediaPipe result/landmarks, 프로세서 입력값 등 logger.debug로 모두 출력
- **목적:**
  - 어디서 데이터가 끊기는지(입력/MediaPipe/프로세서) 자동 추적
  - 반복적 감지 실패/원인 미상 오류 근본 진단

---

### 6. 실행 멈춤/루프 진입 실패 진단용 진입/프레임 획득 로깅 자동 패치
- **적용 파일:**
  - main.py
  - app.py
  - io_handler/video_input.py
- **주요 변경점:**
  - main(), DMSApp.run(), main_async_loop, 프레임 루프 시작, VideoInputManager.get_frame() 등 주요 진입점에 logger.info/print 추가
  - get_frame에서 프레임이 None이면 logger.error("프레임 획득 실패") 출력
- **목적:**
  - 어디서 멈추는지, 프레임 루프/분석 루프 진입 여부, 입력 획득 실패 여부를 명확히 추적
  - 구조적 deadlock/루프 미진입/프레임 획득 실패 근본 진단

---

## [향후 액션]
- 로그/DEBUG 출력에서 데이터 흐름 이상 지점 발견 시, 해당 부분 구조적 패치
- 모든 자동 패치/수정/시도 내역은 이 파일에 반드시 기록
- 반복/중복 시도 방지, 히스토리 기반 진단/개선 지속

---

> **이 파일은 시스템이 자동으로 갱신하며, 모든 디버깅/수정/패치 내역을 투명하게 남깁니다.** 

### 2025-07-11 비디오 재생 즉시 종료 문제 개선 패치

- io_handler/video_input.py
  - 여러 개의 비디오 파일 중 하나라도 열리지 않으면 다음 파일로 자동 전환하도록 개선
  - VideoCapture 객체가 열리지 않거나 스레드가 예외로 중단될 경우, 명확한 에러 메시지(init_error_message) 저장
  - 프레임 획득 실패가 반복될 때, 일정 횟수 초과 시 사용자 안내 메시지 제공

- app.py
  - main_async_loop에서 error_count가 10회 초과 시, 무조건 종료하지 않고 사용자에게 명확한 안내 메시지 출력
  - VideoInputManager의 에러 메시지를 활용해 프레임 획득 실패/예외 발생 시 원인 메시지를 GUI/터미널에 안내

---
이 패치로 비디오 입력/프레임 획득 실패, 스레드 예외, 환경 문제 등 복합적 원인에 대해 사용자가 명확히 인지할 수 있도록 개선됨. 

- (2024-07-11) 콜백 기반 단일 경로로 시각화 구조를 복구(분석 결과가 모두 준비된 시점에만 draw_enhanced_results+cv2.imshow를 호출)하고, main_async_loop에서 분석/시각화 직접 호출을 제거하여 최적화 구조를 유지하면서 시각화 문제를 완전히 해결하는 패치를 적용함. 

- (2024-07-11) MediaPipeManager.run_tasks, detect_async, 콜백 등록/호출 등 모든 경로에 초강력 진단 로그를 추가하여 분석 파이프라인 단절의 근본 원인을 추적하는 패치 적용. 

## 2025-07-11 app.py 전체 초강력 진단 로그 일괄 삽입

- 모든 함수 진입/종료, 주요 분기(if/elif/else/continue/break/return), 예외, 루프 반복, 데이터 전달부에 logger.info/logger.error로 상세 진단 로그 추가
- 로그 메시지: [진단] app.py: 함수명/분기/조건/예외/데이터 상태 등 최대한 상세하게 출력
- 목적: 실행 경로, 데이터 흐름, 예외, 조건 분기, 루프 반복 등 모든 코드 흐름을 실시간으로 로그에 기록하여, 분석/시각화 파이프라인 단절 및 비정상 경로를 100% 추적하기 위함
- 예시: logger.info("[진단] app.py: main_async_loop 루프 반복"), logger.error("[진단] app.py: initialize 예외: {e}") 등
- 적용 범위: app.py 전체 