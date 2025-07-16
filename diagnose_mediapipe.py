#!/usr/bin/env python3
"""
MediaPipe 진단 도구
DMS 시스템의 MediaPipe 관련 문제를 진단하고 해결방안을 제시합니다.
"""

import sys
import os
from pathlib import Path
import importlib.util


def check_python_version():
    """Python 버전 확인"""
    print(f"🐍 Python 버전: {sys.version}")
    version_info = sys.version_info

    if version_info.major < 3:
        print("❌ Python 3.x 이상이 필요합니다.")
        return False
    elif version_info.minor < 8:
        print("⚠️  Python 3.8 이상 권장 (MediaPipe 호환성)")
        return True
    else:
        print("✅ Python 버전 적합")
        return True


def check_mediapipe_installation():
    """MediaPipe 설치 상태 확인"""
    print("\n📦 MediaPipe 설치 상태 확인:")

    try:
        import mediapipe as mp

        print(f"✅ MediaPipe 설치됨 - 버전: {mp.__version__}")
        return True
    except ImportError as e:
        print(f"❌ MediaPipe 설치되지 않음: {e}")
        print("💡 해결방법:")
        print("   pip install mediapipe>=0.10.9")
        print("   또는")
        print("   pip install --upgrade mediapipe")
        return False


def check_mediapipe_tasks():
    """MediaPipe Tasks API 확인"""
    print("\n🔧 MediaPipe Tasks API 확인:")

    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            PoseLandmarker,
            HandLandmarker,
            GestureRecognizer,
            ObjectDetector,
        )

        print("✅ MediaPipe Tasks API 모든 모듈 사용 가능")
        return True
    except ImportError as e:
        print(f"❌ MediaPipe Tasks API 모듈 누락: {e}")
        print("💡 해결방법:")
        print("   pip install --upgrade mediapipe")
        return False


def check_opencv():
    """OpenCV 설치 상태 확인"""
    print("\n🎥 OpenCV 설치 상태 확인:")

    try:
        import cv2

        print(f"✅ OpenCV 설치됨 - 버전: {cv2.__version__}")
        return True
    except ImportError as e:
        print(f"❌ OpenCV 설치되지 않음: {e}")
        print("💡 해결방법:")
        print("   pip install opencv-python")
        return False


def check_model_files():
    """모델 파일 존재 확인"""
    print("\n📁 모델 파일 존재 확인:")

    models_dir = Path("models")
    if not models_dir.exists():
        print(f"❌ models 디렉토리 없음: {models_dir.absolute()}")
        return False

    print(f"📂 models 디렉토리: {models_dir.absolute()}")

    required_models = [
        "face_landmarker.task",
        "pose_landmarker_heavy.task",
        "hand_landmarker.task",
        "gesture_recognizer.task",
    ]

    optional_models = [
        "pose_landmarker_full.task",
        "pose_landmarker_lite.task",
        "holistic_landmarker.task",
        "efficientdet_lite0.tflite",
    ]

    all_good = True

    # 필수 모델 파일 확인
    print("\n🔴 필수 모델 파일:")
    for model in required_models:
        model_path = models_dir / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {model} ({size_mb:.1f}MB)")
        else:
            print(f"  ❌ {model} - 파일 없음")
            all_good = False

    # 선택적 모델 파일 확인
    print("\n🟡 선택적 모델 파일:")
    for model in optional_models:
        model_path = models_dir / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {model} ({size_mb:.1f}MB)")
        else:
            print(f"  ⚠️  {model} - 파일 없음 (선택사항)")

    # 디렉토리 내 모든 파일 나열
    print("\n📋 models 디렉토리 전체 파일:")
    all_files = list(models_dir.glob("*"))
    if all_files:
        for file in sorted(all_files):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  📄 {file.name} ({size_mb:.1f}MB)")
    else:
        print("  (빈 디렉토리)")

    return all_good


def check_system_resources():
    """시스템 리소스 확인"""
    print("\n💻 시스템 리소스 확인:")

    try:
        import psutil

        # CPU 정보
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        print(f"🔧 CPU: {cpu_count}코어, 사용률: {cpu_percent}%")

        # 메모리 정보
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_used_percent = memory.percent
        print(f"🧠 메모리: {memory_gb:.1f}GB, 사용률: {memory_used_percent}%")

        # GPU 정보 (선택사항)
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                for gpu in gpus:
                    print(f"🎮 GPU: {gpu.name}, 메모리: {gpu.memoryTotal}MB")
            else:
                print("🎮 GPU: 감지되지 않음")
        except ImportError:
            print("🎮 GPU: GPUtil 미설치 (선택사항)")

        return True
    except ImportError:
        print("❌ psutil 미설치 - 시스템 리소스 확인 불가")
        return False


def test_mediapipe_basic():
    """MediaPipe 기본 기능 테스트"""
    print("\n🧪 MediaPipe 기본 기능 테스트:")

    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        # 모델 파일 경로
        models_dir = Path("models")
        face_model = models_dir / "face_landmarker.task"

        if not face_model.exists():
            print("❌ face_landmarker.task 파일이 없어 테스트 불가")
            return False

        # Face Landmarker 초기화 테스트
        print("🔄 Face Landmarker 초기화 테스트...")
        base_options = BaseOptions(model_asset_path=str(face_model))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        face_landmarker = vision.FaceLandmarker.create_from_options(options)
        print("✅ Face Landmarker 초기화 성공")

        # 정리
        face_landmarker.close()
        print("✅ MediaPipe 기본 기능 테스트 완료")
        return True

    except Exception as e:
        print(f"❌ MediaPipe 기본 기능 테스트 실패: {e}")
        return False


def provide_solutions():
    """해결방안 제시"""
    print("\n🛠️ 문제 해결 가이드:")

    print("\n1. MediaPipe 설치 문제:")
    print("   pip install --upgrade mediapipe>=0.10.9")
    print("   pip install opencv-python")
    print("   pip install numpy scipy")

    print("\n2. Python 버전 문제:")
    print("   Python 3.8 이상 사용 권장")
    print("   가상환경 사용 권장: python -m venv venv")

    print("\n3. 모델 파일 문제:")
    print("   필수 모델 파일들을 models/ 디렉토리에 다운로드")
    print(
        "   MediaPipe 공식 모델: https://developers.google.com/mediapipe/solutions/vision/face_landmarker"
    )

    print("\n4. 시스템 리소스 문제:")
    print("   최소 4GB RAM 권장")
    print("   CPU 최적화 모드 사용 고려")

    print("\n5. 권한 문제:")
    print("   카메라 접근 권한 확인")
    print("   파일 읽기/쓰기 권한 확인")


def main():
    """메인 진단 함수"""
    print("🔍 DMS MediaPipe 진단 도구")
    print("=" * 50)

    all_checks = []

    # 1. Python 버전 확인
    all_checks.append(check_python_version())

    # 2. MediaPipe 설치 확인
    all_checks.append(check_mediapipe_installation())

    # 3. MediaPipe Tasks API 확인
    all_checks.append(check_mediapipe_tasks())

    # 4. OpenCV 확인
    all_checks.append(check_opencv())

    # 5. 모델 파일 확인
    all_checks.append(check_model_files())

    # 6. 시스템 리소스 확인
    all_checks.append(check_system_resources())

    # 7. MediaPipe 기본 기능 테스트
    all_checks.append(test_mediapipe_basic())

    # 결과 요약
    print("\n📊 진단 결과 요약:")
    print("=" * 50)

    passed = sum(all_checks)
    total = len(all_checks)

    print(f"✅ 통과: {passed}/{total}")
    print(f"❌ 실패: {total - passed}/{total}")

    if all(all_checks):
        print("\n🎉 모든 진단 통과! DMS 시스템 실행 준비 완료")
    else:
        print("\n⚠️  일부 문제 발견됨")
        provide_solutions()

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
