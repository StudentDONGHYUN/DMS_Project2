import cv2
import threading
import time
import os
import logging
import contextlib
from typing import Optional, Dict

# 전역 안전 모드 플래그
safe_mode = False

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def video_capture_context(source):
    """VideoCapture 객체를 안전하게 관리하는 Context Manager"""
    cap = None
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"VideoCapture 열기 실패: {source}")
        yield cap
    except Exception as e:
        logger.error(f"VideoCapture Context Manager 오류: {e}")
        raise
    finally:
        if cap is not None:
            try:
                cap.release()
                logger.debug("VideoCapture 리소스 정리 완료")
            except Exception as e:
                logger.warning(f"VideoCapture 정리 중 오류: {e}")


@contextlib.asynccontextmanager
async def async_video_capture_context(source):
    """비동기 VideoCapture Context Manager"""
    cap = None
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Async VideoCapture 열기 실패: {source}")
        yield cap
    except Exception as e:
        logger.error(f"Async VideoCapture Context Manager 오류: {e}")
        raise
    finally:
        if cap is not None:
            try:
                cap.release()
                logger.debug("Async VideoCapture 리소스 정리 완료")
            except Exception as e:
                logger.warning(f"Async VideoCapture 정리 중 오류: {e}")


class MultiVideoCalibrationManager:
    """다중 비디오 캘리브레이션 관리"""

    def __init__(self, user_id: str):
        self.is_same_driver = True
        self.shared_calibration_data = None

    def set_driver_continuity(self, is_same: bool):
        self.is_same_driver = is_same
        logger.info(
            f"운전자 연속성 설정됨: {'동일 운전자' if is_same else '다른 운전자'}"
        )

    def should_skip_calibration(self) -> bool:
        return self.is_same_driver and self.shared_calibration_data is not None

    def save_calibration_data(self, data: Dict):
        if self.is_same_driver:
            self.shared_calibration_data = data.copy()

    def get_shared_calibration_data(self) -> Optional[Dict]:
        return self.shared_calibration_data


class VideoInputManager:
    """비동기 입력 관리자"""

    def __init__(self, input_source):
        self.input_source = input_source
        self.cap = None
        self.is_video_mode = isinstance(input_source, (str, list, tuple))
        self.video_playlist = []
        self.current_video_index = -1
        self.playback_speed = 1.0
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.stopped = True
        self.video_changed_flag = False
        self.init_error_message = None  # 에러 메시지 저장용
        self.fps = 30  # 프레임 속도 저장용

    def initialize(self) -> bool:
        """비디오 입력 초기화 (강화된 오류 처리 및 타이밍 제어)"""
        import logging

        logger = logging.getLogger(__name__)
        logger.info("[진단] video_input.initialize 진입")
        try:
            logger.info(
                f"[진단] video_input.initialize - self.input_source={self.input_source}"
            )
            self.init_error_message = None  # 에러 메시지 저장용
            if self.is_video_mode:
                self.video_playlist = (
                    self.input_source
                    if isinstance(self.input_source, list)
                    else [self.input_source]
                )
                if not self.video_playlist:
                    logger.error("비어있는 비디오 플레이리스트")
                    self.init_error_message = "비어있는 비디오 플레이리스트"
                    return False
                logger.info(f"비디오 플레이리스트: {len(self.video_playlist)}개 파일")
                for i, video_file in enumerate(self.video_playlist):
                    logger.info(f"  {i + 1}. {os.path.basename(video_file)}")
                # 여러 파일 중 열리는 첫 파일을 찾음
                self.current_video_index = 0
                cap_found = False
                for idx, video_file in enumerate(self.video_playlist):
                    cap = self._create_optimized_capture(video_file)
                    if cap and cap.isOpened():
                        self.cap = cap
                        self.current_video_index = idx
                        cap_found = True
                        # backend 정보, 속성, 코덱 등 진단 로그
                        backend = (
                            self.cap.getBackendName()
                            if hasattr(self.cap, "getBackendName")
                            else "unknown"
                        )
                        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                        codec = "".join(
                            [chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]
                        )
                        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        logger.info(
                            f"[진단] VideoCapture backend: {backend}, codec: {codec}, 총 프레임: {total_frames}"
                        )
                        break
                    else:
                        logger.error(f"비디오 파일 열기 실패: {video_file}")
                if not cap_found:
                    logger.error("모든 비디오 파일 열기 실패")
                    self.init_error_message = "모든 비디오 파일 열기 실패"
                    return False
            else:
                logger.info(f"웹캠 모드 - 디바이스 {self.input_source}")
                self.cap = self._create_optimized_capture(int(self.input_source))
                if self.cap and self.cap.isOpened():
                    pass
            if not self.cap or not self.cap.isOpened():
                logger.error(f"입력 소스 열기 실패: {self.input_source}")
                self.init_error_message = f"입력 소스 열기 실패: {self.input_source}"
                return False
            logger.info("VideoCapture 객체 성공적으로 생성됨")
            # backend 정보, 속성, 코덱 등 진단 로그 (웹캠 포함)
            backend = (
                self.cap.getBackendName()
                if hasattr(self.cap, "getBackendName")
                else "unknown"
            )
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(
                f"[진단] VideoCapture backend: {backend}, codec: {codec}, 총 프레임: {total_frames}"
            )
            # 리더 스레드 시작
            self.stopped = False
            self.capture_thread = threading.Thread(
                target=self._reader_thread, daemon=True
            )
            self.capture_thread.start()
            # 리더 스레드가 시작될 때까지 잠시 대기
            logger.info("리더 스레드 시작 대기 중...")
            time.sleep(0.5)  # 스레드 안정화 대기
            # 첫 번째 프레임 대기 (최대 5초)
            first_frame_timeout = 5.0
            start_time = time.time()
            logger.info(f"첫 번째 프레임 대기 중 (최대 {first_frame_timeout}초)...")

            # Initialize thread health check variables
            consecutive_failures = 0
            max_consecutive_failures = 3
            last_health_check = time.time()
            health_check_interval = 0.5  # Check every 0.5 seconds

            while time.time() - start_time < first_frame_timeout:
                current_time = time.time()

                # Check for first frame in thread-safe manner
                frame_received = False
                thread_alive = False
                stopped_flag = False

                # Lock-protected frame check
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame_received = True

                # Check thread status with proper timing
                if current_time - last_health_check >= health_check_interval:
                    if self.capture_thread:
                        thread_alive = self.capture_thread.is_alive()
                    stopped_flag = self.stopped
                    last_health_check = current_time

                    # Thread health monitoring
                    if not thread_alive and not stopped_flag:
                        consecutive_failures += 1
                        logger.warning(
                            f"리더 스레드 비활성 감지 ({consecutive_failures}/{max_consecutive_failures})"
                        )

                        if consecutive_failures >= max_consecutive_failures:
                            logger.error("리더 스레드가 반복적으로 실패함")
                            self.init_error_message = "리더 스레드가 반복적으로 실패함"
                            return False
                    else:
                        consecutive_failures = 0  # Reset counter if thread is alive

                if frame_received:
                    logger.info("✅ 첫 번째 프레임 수신 성공")
                    logger.info("✅ 입력 소스 초기화 및 스레드 시작 완료")
                    return True

                if stopped_flag:
                    logger.error("리더 스레드가 예상치 못하게 중단됨")
                    self.init_error_message = "리더 스레드가 예상치 못하게 중단됨"
                    return False

                time.sleep(0.1)

            # 타임아웃 발생 - 최종 상태 검사
            logger.warning(f"첫 번째 프레임 대기 타임아웃 ({first_frame_timeout}초)")

            # Final comprehensive state check
            final_frame_check = False
            final_thread_alive = False
            final_stopped_flag = False

            # One final frame check
            with self.frame_lock:
                if self.current_frame is not None:
                    final_frame_check = True

            # Thread status check
            if self.capture_thread:
                final_thread_alive = self.capture_thread.is_alive()
            final_stopped_flag = self.stopped

            if final_frame_check:
                logger.info("타임아웃 후 프레임 발견됨 - 정상 진행")
                return True
            elif not final_stopped_flag and final_thread_alive:
                logger.info("리더 스레드가 실행 중이므로 계속 진행")
                return True
            else:
                logger.error("리더 스레드가 중단되었거나 시작되지 않음")
                self.init_error_message = "리더 스레드가 중단되었거나 시작되지 않음"
                return False
        except Exception as e:
            logger.error(f"VideoInputManager 초기화 실패: {e}", exc_info=True)
            self.init_error_message = f"VideoInputManager 초기화 실패: {e}"
            return False

    def _reader_thread(self):
        """비디오 프레임 읽기 스레드 (강화된 오류 처리 및 진단)"""
        import logging

        logger = logging.getLogger(__name__)
        logger.info("[진단] video_input._reader_thread 진입")
        try:
            while not self.stopped and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.frame_lock:
                        self.current_frame = frame
                        # logger.info(
                        #     f"[진단] video_input._reader_thread self.current_frame 갱신: shape={getattr(frame, 'shape', None)}"
                        # )
                else:
                    logger.warning(
                        f"[진단] video_input._reader_thread: 프레임 획득 실패 (ret={ret})"
                    )
                    # 프레임 획득 실패 시 비디오 다음으로 넘어가기 시도
                    if self.is_video_mode and not self._try_next_video():
                        logger.info("모든 비디오 재생 완료")
                        break

                # 프레임 레이트 제어 (웹캠은 지연 없음, 비디오 파일만 제어)
                if self.is_video_mode:
                    # 비디오 파일: 원본 FPS에 맞춰 재생
                    actual_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30
                    if actual_fps > 0:
                        time.sleep(1.0 / actual_fps)
                    else:
                        time.sleep(0.033)  # 30fps 기본값
                else:
                    # 웹캠: 지연 없음 (하드웨어가 자연스럽게 제어)
                    time.sleep(0.001)  # 최소 지연
            logger.info(
                f"[진단] video_input._reader_thread 루프 종료: stopped={self.stopped}, cap_opened={self.cap.isOpened() if self.cap else None}"
            )
        except Exception as e:
            logger.error(f"[진단] video_input._reader_thread 예외: {e}", exc_info=True)
        logger.info("[진단] video_input._reader_thread 함수 종료")

    def _create_optimized_capture(self, source):
        """플랫폼별 최적화된 VideoCapture 생성 (환경 적응형 처리)"""
        import platform

        # 환경 감지
        is_windows = platform.system() == "Windows"
        is_rb2_platform = os.path.exists("/dev/video0") and not is_windows

        try:
            if is_rb2_platform:
                # RB2 플랫폼: GStreamer 하드웨어 가속 시도
                return self._create_rb2_optimized_capture(source)
            else:
                # Windows/개발 환경: 표준 OpenCV 사용
                return self._create_standard_capture(source)

        except Exception as e:
            logger.error(f"최적화된 VideoCapture 생성 실패: {e}, 기본 방식으로 폴백")
            return self._create_standard_capture(source)

    def _create_rb2_optimized_capture(self, source):
        """RB2 플랫폼용 GStreamer 하드웨어 가속 캡처"""
        try:
            if isinstance(source, int):  # 웹캠 (V4L2 디바이스)
                gstreamer_pipeline = (
                    f"v4l2src device=/dev/video{source} ! "
                    "video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 ! "
                    "videoconvert ! "
                    "appsink max-buffers=1 drop=true"
                )
                logger.info(
                    f"RB2 최적화: V4L2 GStreamer 파이프라인 사용 (디바이스: /dev/video{source})"
                )
                cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

                if not cap.isOpened():
                    logger.warning("GStreamer V4L2 파이프라인 실패, 기본 방식으로 폴백")
                    cap = cv2.VideoCapture(source)

                return cap

            elif isinstance(source, str):  # 비디오 파일
                gstreamer_pipeline = (
                    f"filesrc location={source} ! "
                    "decodebin ! "
                    "videoconvert ! "
                    "video/x-raw,format=BGR ! "
                    "appsink max-buffers=1 drop=true"
                )
                logger.info(
                    f"RB2 최적화: 하드웨어 디코더 GStreamer 파이프라인 사용 (파일: {os.path.basename(source)})"
                )
                cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

                if not cap.isOpened():
                    logger.warning("GStreamer 파일 파이프라인 실패, 기본 방식으로 폴백")
                    cap = cv2.VideoCapture(source)

                return cap

            else:
                return cv2.VideoCapture(source)

        except Exception as e:
            logger.warning(f"RB2 GStreamer 최적화 실패: {e}, 표준 방식으로 폴백")
            return self._create_standard_capture(source)

    def _create_standard_capture(self, source):
        """표준 OpenCV VideoCapture 생성 (Windows/개발 환경용) - 강화된 진단 기능"""
        try:
            if isinstance(source, str):
                # 파일 존재 및 속성 확인
                if not os.path.exists(source):
                    logger.error(f"비디오 파일이 존재하지 않습니다: {source}")
                    return None

                # 파일 크기 및 확장자 확인
                file_size = os.path.getsize(source) / (1024 * 1024)  # MB
                file_ext = os.path.splitext(source)[1].lower()
                logger.info(
                    f"비디오 파일 정보: {os.path.basename(source)} ({file_size:.2f}MB, {file_ext})"
                )

                if file_size < 0.1:
                    logger.warning(f"비디오 파일이 너무 작습니다: {file_size:.2f}MB")

                logger.info(f"Windows/개발 환경: 표준 파일 디코더 사용")

                # 다양한 백엔드로 시도
                backends_to_try = [
                    (cv2.CAP_FFMPEG, "FFMPEG"),
                    (cv2.CAP_DSHOW, "DirectShow"),
                    (cv2.CAP_MSMF, "MediaFoundation"),
                ]

                cap = None
                for backend_id, backend_name in backends_to_try:
                    try:
                        logger.info(f"{backend_name} 백엔드로 시도 중...")
                        cap = cv2.VideoCapture(source, backend_id)

                        if cap.isOpened():
                            # 테스트 프레임 읽기
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None:
                                logger.info(
                                    f"✅ {backend_name} 백엔드로 성공 (테스트 프레임: {test_frame.shape})"
                                )
                                # 처음으로 되돌리기
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                break
                            else:
                                logger.warning(
                                    f"{backend_name} 백엔드는 열렸지만 프레임 읽기 실패"
                                )
                                cap.release()
                                cap = None
                        else:
                            logger.warning(f"{backend_name} 백엔드로 열기 실패")
                            if cap:
                                cap.release()
                            cap = None
                    except Exception as e:
                        logger.warning(f"{backend_name} 백엔드 시도 중 오류: {e}")
                        if cap:
                            cap.release()
                        cap = None

                # 모든 백엔드 실패 시 기본 방식 시도
                if cap is None:
                    logger.warning("모든 전용 백엔드 실패, 기본 VideoCapture 시도")
                    cap = cv2.VideoCapture(source)

                # 최적화 설정 적용
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # 비디오 정보 로깅
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0

                    logger.info(
                        f"비디오 속성: {width}x{height}, {fps:.2f}fps, {frame_count}프레임, {duration:.1f}초"
                    )

                    # 코덱 정보 (가능한 경우)
                    try:
                        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                        codec = "".join(
                            [chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]
                        )
                        logger.info(f"비디오 코덱: {codec}")
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.debug(f"코덱 정보 추출 실패: {e}")
                    except Exception as e:
                        logger.warning(f"코덱 정보 추출 중 예상치 못한 오류: {e}")

                return cap

            elif isinstance(source, int):
                logger.info(
                    f"Windows/개발 환경: 표준 웹캠 캡처 사용 (디바이스: {source})"
                )
                cap = cv2.VideoCapture(source)

                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)

                    # 웹캠 테스트 프레임 읽기
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"웹캠 테스트 성공: {test_frame.shape}")
                    else:
                        logger.warning("웹캠이 열렸지만 프레임 읽기 실패")

                return cap

            else:
                logger.info(f"기타 소스 타입: {type(source)}")
                return cv2.VideoCapture(source)

        except Exception as e:
            logger.error(f"표준 VideoCapture 생성 실패: {e}", exc_info=True)
            return None

    def _try_next_video(self):
        if self.current_video_index >= len(self.video_playlist) - 1:
            return False
        self.current_video_index += 1
        if self.cap:
            self.cap.release()
        self.cap = self._create_optimized_capture(
            self.video_playlist[self.current_video_index]
        )
        if self.cap.isOpened():
            logger.info(
                f"다음 비디오 로드 (RB2 최적화): {os.path.basename(self.video_playlist[self.current_video_index])}"
            )
            self.video_changed_flag = True
            return True
        return False

    def get_frame(self):
        """리더 스레드에서 저장한 최신 프레임 반환 (스레드 안전, 항상 numpy)"""
        try:
            with self.frame_lock:
                if self.current_frame is not None:
                    return self.current_frame.copy()
                else:
                    return None
        except Exception as e:
            logger.error(f"get_frame 오류: {e}")
            return None

    def has_video_changed(self):
        if self.video_changed_flag:
            self.video_changed_flag = False
            return True
        return False

    def set_playback_speed(self, speed: float):
        self.playback_speed = max(0.1, speed)

    def get_playback_info(self):
        info = {"mode": "video" if self.is_video_mode else "webcam"}
        if self.is_video_mode and self.video_playlist:
            info.update(
                {
                    "current_file": os.path.basename(
                        self.video_playlist[self.current_video_index]
                    ),
                    "total_videos": len(self.video_playlist),
                    "current_video": self.current_video_index + 1,
                }
            )
        return info

    def is_running(self):
        return not self.stopped

    def release(self):
        """리소스 정리 (예외 안전성 보장)"""
        global safe_mode
        error_count = 0
        try:
            self.stopped = True
        except Exception as e:
            error_count += 1
        try:
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1)
                if self.capture_thread.is_alive():
                    error_count += 1
        except Exception as e:
            error_count += 1
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except Exception as e:
            error_count += 1
        finally:
            if error_count >= 2:
                safe_mode = True  # 시스템 전체 안전 모드 진입

    def __enter__(self):
        """Context Manager 진입"""
        if not self.initialize():
            raise RuntimeError("VideoInputManager 초기화 실패")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager 종료 (예외 발생 시에도 리소스 정리 보장)"""
        global safe_mode
        try:
            self.release()
        except (OSError, RuntimeError) as e:
            error_count = getattr(self, '_exit_error_count', 0) + 1
            self._exit_error_count = error_count
            if error_count >= 2:
                safe_mode = True  # 시스템 전체 안전 모드 진입
        except Exception as e:
            error_count = getattr(self, '_exit_error_count', 0) + 1
            self._exit_error_count = error_count
            if error_count >= 2:
                safe_mode = True  # 시스템 전체 안전 모드 진입
        return False
