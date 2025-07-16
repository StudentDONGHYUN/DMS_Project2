#!/usr/bin/env python3
"""
MediaPipe 0.10.21 API 간단 테스트
DMS 시스템의 MediaPipe 초기화 과정을 단순화해서 테스트합니다.
"""

import cv2
import numpy as np
from pathlib import Path
import sys


def test_face_landmarker():
    """Face Landmarker 테스트"""
    print("🔄 Face Landmarker 테스트 시작...")

    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        # 모델 파일 경로
        model_path = Path("models/face_landmarker.task")
        if not model_path.exists():
            print(f"❌ 모델 파일 없음: {model_path}")
            return False

        # BaseOptions 생성
        base_options = BaseOptions(model_asset_path=str(model_path))

        # FaceLandmarkerOptions 생성 (MediaPipe 0.10.21 API)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.IMAGE,
        )

        # FaceLandmarker 생성
        face_landmarker = vision.FaceLandmarker.create_from_options(options)
        print("✅ Face Landmarker 초기화 성공")

        # 정리
        face_landmarker.close()
        return True

    except Exception as e:
        print(f"❌ Face Landmarker 테스트 실패: {e}")
        return False


def test_hand_landmarker():
    """Hand Landmarker 테스트"""
    print("🔄 Hand Landmarker 테스트 시작...")

    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        # 모델 파일 경로
        model_path = Path("models/hand_landmarker.task")
        if not model_path.exists():
            print(f"❌ 모델 파일 없음: {model_path}")
            return False

        # BaseOptions 생성
        base_options = BaseOptions(model_asset_path=str(model_path))

        # HandLandmarkerOptions 생성 (MediaPipe 0.10.21 API)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE,
        )

        # HandLandmarker 생성
        hand_landmarker = vision.HandLandmarker.create_from_options(options)
        print("✅ Hand Landmarker 초기화 성공")

        # 정리
        hand_landmarker.close()
        return True

    except Exception as e:
        print(f"❌ Hand Landmarker 테스트 실패: {e}")
        return False


def test_pose_landmarker():
    """Pose Landmarker 테스트"""
    print("🔄 Pose Landmarker 테스트 시작...")

    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        # 모델 파일 경로
        model_path = Path("models/pose_landmarker_heavy.task")
        if not model_path.exists():
            print(f"❌ 모델 파일 없음: {model_path}")
            return False

        # BaseOptions 생성
        base_options = BaseOptions(model_asset_path=str(model_path))

        # PoseLandmarkerOptions 생성 (MediaPipe 0.10.21 API)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            output_segmentation_masks=True,
            running_mode=vision.RunningMode.IMAGE,
        )

        # PoseLandmarker 생성
        pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        print("✅ Pose Landmarker 초기화 성공")

        # 정리
        pose_landmarker.close()
        return True

    except Exception as e:
        print(f"❌ Pose Landmarker 테스트 실패: {e}")
        return False


def test_gesture_recognizer():
    """Gesture Recognizer 테스트"""
    print("🔄 Gesture Recognizer 테스트 시작...")

    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        # 모델 파일 경로
        model_path = Path("models/gesture_recognizer.task")
        if not model_path.exists():
            print(f"❌ 모델 파일 없음: {model_path}")
            return False

        # BaseOptions 생성
        base_options = BaseOptions(model_asset_path=str(model_path))

        # GestureRecognizerOptions 생성 (MediaPipe 0.10.21 API)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE,
        )

        # GestureRecognizer 생성
        gesture_recognizer = vision.GestureRecognizer.create_from_options(options)
        print("✅ Gesture Recognizer 초기화 성공")

        # 정리
        gesture_recognizer.close()
        return True

    except Exception as e:
        print(f"❌ Gesture Recognizer 테스트 실패: {e}")
        return False


def test_with_sample_image():
    """샘플 이미지로 실제 감지 테스트"""
    print("🔄 샘플 이미지로 실제 감지 테스트...")

    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        import mediapipe as mp

        # 흰색 더미 이미지 생성 (480x640)
        dummy_image = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # MediaPipe Image로 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_image)

        # Face Landmarker로 테스트
        model_path = Path("models/face_landmarker.task")
        if model_path.exists():
            base_options = BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                running_mode=vision.RunningMode.IMAGE,
            )

            face_landmarker = vision.FaceLandmarker.create_from_options(options)

            # 이미지 처리
            result = face_landmarker.detect(mp_image)
            print(
                f"✅ Face detection 결과: {len(result.face_landmarks) if result.face_landmarks else 0}개 얼굴 감지"
            )

            face_landmarker.close()

        return True

    except Exception as e:
        print(f"❌ 샘플 이미지 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🧪 MediaPipe 0.10.21 API 간단 테스트")
    print("=" * 50)

    # MediaPipe 버전 확인
    try:
        import mediapipe as mp

        print(f"📦 MediaPipe 버전: {mp.__version__}")
    except ImportError as e:
        print(f"❌ MediaPipe 가져오기 실패: {e}")
        return

    # 각 모델 테스트
    tests = [
        ("Face Landmarker", test_face_landmarker),
        ("Hand Landmarker", test_hand_landmarker),
        ("Pose Landmarker", test_pose_landmarker),
        ("Gesture Recognizer", test_gesture_recognizer),
        ("Sample Image Test", test_with_sample_image),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            print(f"{'✅' if result else '❌'} {name}: {'성공' if result else '실패'}")
        except Exception as e:
            print(f"❌ {name}: 예외 발생 - {e}")
            results.append(False)
        print("-" * 30)

    # 결과 요약
    print("\n📊 테스트 결과:")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"✅ 성공: {passed}/{total}")
    print(f"❌ 실패: {total - passed}/{total}")

    if all(results):
        print("\n🎉 모든 테스트 통과! MediaPipe 0.10.21 API 호환성 확인 완료")
    else:
        print("\n⚠️  일부 테스트 실패. DMS 시스템 실행 전 문제 해결 필요")


if __name__ == "__main__":
    main()
