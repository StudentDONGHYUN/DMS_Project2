"""
최신 MediaPipe Tasks API를 사용한 랜드마크 시각화 유틸리티
S-Class DMS v19+ 호환
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# 최신 MediaPipe Tasks API Drawing 상수
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

# 최신 색상 팔레트 (S-Class 디자인)
class DrawingColors:
    """S-Class DMS용 색상 팔레트"""
    FACE_MESH = (192, 192, 192)          # 연한 회색
    FACE_CONTOURS = (255, 255, 255)      # 흰색
    FACE_IRISES = (0, 255, 255)          # 시아니즘
    POSE_LANDMARKS = (0, 255, 0)         # 초록색
    POSE_CONNECTIONS = (255, 255, 0)     # 노란색
    HAND_LANDMARKS = (255, 0, 0)         # 빨간색
    HAND_CONNECTIONS = (0, 0, 255)       # 파란색
    LEFT_HAND = (0, 255, 0)              # 초록색
    RIGHT_HAND = (255, 0, 0)             # 빨간색

# 최신 MediaPipe Tasks 연결 상수
class TasksConnections:
    """최신 MediaPipe Tasks API 연결 상수"""
    
    # Face Mesh 연결 (468개 랜드마크)
    FACE_OVAL = [
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
        (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
        (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
        (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
        (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
        (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)
    ]
    
    # Pose 연결 (33개 랜드마크)
    POSE_CONNECTIONS = [
        # 몸통
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        # 팔
        (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        # 다리
        (23, 24), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
        # 얼굴
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10)
    ]
    
    # Hand 연결 (21개 랜드마크)
    HAND_CONNECTIONS = [
        # 엄지
        (0, 1), (1, 2), (2, 3), (3, 4),
        # 검지
        (0, 5), (5, 6), (6, 7), (7, 8),
        # 중지
        (0, 9), (9, 10), (10, 11), (11, 12),
        # 약지
        (0, 13), (13, 14), (14, 15), (15, 16),
        # 새끼
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

def draw_landmarks_on_image(
    image: np.ndarray,
    landmarks: List,
    connections: List[Tuple[int, int]] = None,
    landmark_color: Tuple[int, int, int] = (0, 255, 0),
    connection_color: Tuple[int, int, int] = (255, 255, 255),
    landmark_radius: int = 3,
    connection_thickness: int = 2
) -> np.ndarray:
    """
    최신 MediaPipe Tasks API용 범용 랜드마크 그리기 함수
    """
    if not landmarks:
        return image
    
    annotated_image = image.copy()
    height, width = image.shape[:2]
    
    # 랜드마크 점들 그리기
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(annotated_image, (x, y), landmark_radius, landmark_color, -1)
    
    # 연결선 그리기
    if connections:
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_landmark = landmarks[start_idx]
                end_landmark = landmarks[end_idx]
                
                start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
                end_point = (int(end_landmark.x * width), int(end_landmark.y * height))
                
                cv2.line(annotated_image, start_point, end_point, connection_color, connection_thickness)
    
    return annotated_image

def draw_face_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """
    최신 MediaPipe Tasks API를 사용한 얼굴 랜드마크 그리기
    """
    if not detection_result or not hasattr(detection_result, 'face_landmarks') or not detection_result.face_landmarks:
        return rgb_image
    
    annotated_image = rgb_image.copy()
    
    try:
        for face_landmarks in detection_result.face_landmarks:
            # Face mesh tesselation (가장 세밀한 연결)
            annotated_image = draw_landmarks_on_image(
                annotated_image,
                face_landmarks,
                connections=None,  # 너무 복잡하므로 점만 표시
                landmark_color=DrawingColors.FACE_MESH,
                landmark_radius=1
            )
            
            # Face contours (윤곽선)
            annotated_image = draw_landmarks_on_image(
                annotated_image,
                face_landmarks,
                connections=TasksConnections.FACE_OVAL,
                landmark_color=DrawingColors.FACE_CONTOURS,
                connection_color=DrawingColors.FACE_CONTOURS,
                landmark_radius=2,
                connection_thickness=2
            )
            
            # 눈 영역 강조 (iris landmarks)
            if len(face_landmarks) >= 468:  # 전체 face mesh
                # 왼쪽 눈 (landmarks 468-477)
                left_iris_center = face_landmarks[468] if len(face_landmarks) > 468 else None
                right_iris_center = face_landmarks[473] if len(face_landmarks) > 473 else None
                
                height, width = annotated_image.shape[:2]
                
                if left_iris_center:
                    left_center = (int(left_iris_center.x * width), int(left_iris_center.y * height))
                    cv2.circle(annotated_image, left_center, 5, DrawingColors.FACE_IRISES, -1)
                
                if right_iris_center:
                    right_center = (int(right_iris_center.x * width), int(right_iris_center.y * height))
                    cv2.circle(annotated_image, right_center, 5, DrawingColors.FACE_IRISES, -1)
                    
    except Exception as e:
        logger.warning(f"Face landmark 그리기 중 오류 (무시하고 계속): {e}")
    
    return annotated_image

def draw_pose_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """
    최신 MediaPipe Tasks API를 사용한 포즈 랜드마크 그리기
    """
    if not detection_result or not hasattr(detection_result, 'pose_landmarks') or not detection_result.pose_landmarks:
        return rgb_image
    
    annotated_image = rgb_image.copy()
    
    try:
        for pose_landmarks in detection_result.pose_landmarks:
            # Pose landmarks와 connections 그리기
            annotated_image = draw_landmarks_on_image(
                annotated_image,
                pose_landmarks,
                connections=TasksConnections.POSE_CONNECTIONS,
                landmark_color=DrawingColors.POSE_LANDMARKS,
                connection_color=DrawingColors.POSE_CONNECTIONS,
                landmark_radius=4,
                connection_thickness=3
            )
            
    except Exception as e:
        logger.warning(f"Pose landmark 그리기 중 오류 (무시하고 계속): {e}")
    
    return annotated_image

def draw_hand_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """
    최신 MediaPipe Tasks API를 사용한 손 랜드마크 그리기
    """
    if not detection_result or not hasattr(detection_result, 'hand_landmarks') or not detection_result.hand_landmarks:
        return rgb_image
    
    annotated_image = rgb_image.copy()
    
    try:
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = getattr(detection_result, 'handedness', [])
        
        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            # 손이 왼손인지 오른손인지 구분
            if idx < len(handedness_list) and handedness_list[idx]:
                handedness_category = handedness_list[idx][0].category_name
                hand_color = DrawingColors.LEFT_HAND if handedness_category == "Left" else DrawingColors.RIGHT_HAND
            else:
                hand_color = DrawingColors.HAND_LANDMARKS
            
            # Hand landmarks와 connections 그리기
            annotated_image = draw_landmarks_on_image(
                annotated_image,
                hand_landmarks,
                connections=TasksConnections.HAND_CONNECTIONS,
                landmark_color=hand_color,
                connection_color=hand_color,
                landmark_radius=3,
                connection_thickness=2
            )
            
            # 손 라벨 표시
            if handedness_list and idx < len(handedness_list):
                height, width = annotated_image.shape[:2]
                
                # 손목 위치 (landmark 0)를 기준으로 라벨 위치 결정
                wrist = hand_landmarks[0]
                text_x = int(wrist.x * width)
                text_y = int(wrist.y * height) - MARGIN
                
                # 화면 경계 체크
                text_y = max(MARGIN, text_y)
                
                handedness_category = handedness_list[idx][0].category_name
                cv2.putText(
                    annotated_image, 
                    handedness_category,
                    (text_x, text_y), 
                    cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, 
                    HANDEDNESS_TEXT_COLOR, 
                    FONT_THICKNESS, 
                    cv2.LINE_AA
                )
                
    except Exception as e:
        logger.warning(f"Hand landmark 그리기 중 오류 (무시하고 계속): {e}")
    
    return annotated_image

def draw_detection_boxes(
    image: np.ndarray, 
    detections,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    객체 탐지 결과의 바운딩 박스를 그리기
    """
    if not detections:
        return image
    
    annotated_image = image.copy()
    height, width = image.shape[:2]
    
    try:
        for detection in detections:
            # Bounding box 좌표 (normalized coordinates)
            bbox = detection.bounding_box
            
            # 픽셀 좌표로 변환
            left = int(bbox.origin_x * width)
            top = int(bbox.origin_y * height)
            right = int((bbox.origin_x + bbox.width) * width)
            bottom = int((bbox.origin_y + bbox.height) * height)
            
            # 바운딩 박스 그리기
            cv2.rectangle(annotated_image, (left, top), (right, bottom), color, thickness)
            
            # 카테고리와 신뢰도 표시
            if hasattr(detection, 'categories') and detection.categories:
                category = detection.categories[0]
                label = f"{category.category_name}: {category.score:.2f}"
                
                # 라벨 배경
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(
                    annotated_image, 
                    (left, top - label_size[1] - 10), 
                    (left + label_size[0], top), 
                    color, 
                    -1
                )
                
                # 라벨 텍스트
                cv2.putText(
                    annotated_image, 
                    label, 
                    (left, top - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    1
                )
                
    except Exception as e:
        logger.warning(f"Detection box 그리기 중 오류 (무시하고 계속): {e}")
    
    return annotated_image

def create_comprehensive_visualization(
    image: np.ndarray,
    face_result=None,
    pose_result=None,
    hand_result=None,
    object_result=None
) -> np.ndarray:
    """
    모든 MediaPipe Tasks 결과를 종합적으로 시각화
    """
    annotated_image = image.copy()
    
    # 순서대로 그리기 (겹치는 부분 고려)
    if object_result:
        annotated_image = draw_detection_boxes(annotated_image, object_result.detections)
    
    if pose_result:
        annotated_image = draw_pose_landmarks_on_image(annotated_image, pose_result)
    
    if face_result:
        annotated_image = draw_face_landmarks_on_image(annotated_image, face_result)
    
    if hand_result:
        annotated_image = draw_hand_landmarks_on_image(annotated_image, hand_result)
    
    return annotated_image

# 레거시 호환성 함수들 (기존 코드와의 호환성 유지)
def draw_landmarks_legacy_compatible(image, landmarks, connections=None):
    """레거시 코드 호환성을 위한 래퍼 함수"""
    logger.warning("레거시 drawing 함수 사용됨. 최신 API로 업데이트를 권장합니다.")
    return draw_landmarks_on_image(image, landmarks, connections)
