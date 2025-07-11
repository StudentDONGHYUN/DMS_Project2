"""
최신 MediaPipe Tasks API + 기존 Drawing Solutions API 조합
S-Class DMS v19+ 호환
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from typing import Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# MediaPipe Solutions (Drawing용)
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_face_mesh = solutions.face_mesh
mp_pose = solutions.pose
mp_hands = solutions.hands

# Drawing 상수
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

def draw_face_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """
    최신 MediaPipe Tasks API 결과를 기존 Drawing API로 시각화
    """
    if not detection_result or not hasattr(detection_result, 'face_landmarks') or not detection_result.face_landmarks:
        return rgb_image
    
    annotated_image = rgb_image.copy()
    
    try:
        for face_landmarks in detection_result.face_landmarks:
            # Tasks API 결과를 Drawing API 형식으로 변환
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in face_landmarks
            ])

            # 기존 Solutions Drawing API 사용
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
            
    except Exception as e:
        logger.warning(f"Face landmark 그리기 중 오류 (무시하고 계속): {e}")
    
    return annotated_image

def draw_pose_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """
    최신 MediaPipe Tasks API 결과를 기존 Drawing API로 시각화
    """
    if not detection_result or not hasattr(detection_result, 'pose_landmarks') or not detection_result.pose_landmarks:
        return rgb_image
    
    annotated_image = rgb_image.copy()
    
    try:
        for pose_landmarks in detection_result.pose_landmarks:
            # Tasks API 결과를 Drawing API 형식으로 변환
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in pose_landmarks
            ])
            
            # 기존 Solutions Drawing API 사용
            mp_drawing.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_pose_connections_style()
            )
            
    except Exception as e:
        logger.warning(f"Pose landmark 그리기 중 오류 (무시하고 계속): {e}")
    
    return annotated_image

def draw_hand_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """
    최신 MediaPipe Tasks API 결과를 기존 Drawing API로 시각화
    """
    if not detection_result or not hasattr(detection_result, 'hand_landmarks') or not detection_result.hand_landmarks:
        return rgb_image
    
    annotated_image = rgb_image.copy()
    
    try:
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = getattr(detection_result, 'handedness', [])
        
        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            # Tasks API 결과를 Drawing API 형식으로 변환
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in hand_landmarks
            ])
            
            # 기존 Solutions Drawing API 사용
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
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
    객체 탐지 결과의 바운딩 박스를 그리기 (Tasks API 전용)
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
    Tasks API 결과 + Solutions Drawing API 조합
    """
    annotated_image = image.copy()
    
    # 순서대로 그리기 (겹치는 부분 고려)
    if object_result and hasattr(object_result, 'detections'):
        annotated_image = draw_detection_boxes(annotated_image, object_result.detections)
    
    if pose_result:
        annotated_image = draw_pose_landmarks_on_image(annotated_image, pose_result)
    
    if face_result:
        annotated_image = draw_face_landmarks_on_image(annotated_image, face_result)
    
    if hand_result:
        annotated_image = draw_hand_landmarks_on_image(annotated_image, hand_result)
    
    return annotated_image

# 추가 유틸리티 함수들
def draw_gesture_annotations(image: np.ndarray, gesture_result) -> np.ndarray:
    """제스처 인식 결과 표시"""
    if not gesture_result or not hasattr(gesture_result, 'gestures'):
        return image
    
    annotated_image = image.copy()
    height, width = image.shape[:2]
    
    try:
        gestures = gesture_result.gestures
        hand_landmarks = getattr(gesture_result, 'hand_landmarks', [])
        
        for idx, gesture_list in enumerate(gestures):
            if gesture_list and idx < len(hand_landmarks):
                # 가장 높은 신뢰도의 제스처
                top_gesture = gesture_list[0]
                gesture_name = top_gesture.category_name
                confidence = top_gesture.score
                
                # 손목 위치에 제스처 라벨 표시
                if hand_landmarks[idx]:
                    wrist = hand_landmarks[idx][0]  # 손목 (landmark 0)
                    text_x = int(wrist.x * width)
                    text_y = int(wrist.y * height) - 50
                    
                    # 제스처 라벨
                    label = f"{gesture_name}: {confidence:.2f}"
                    cv2.putText(
                        annotated_image,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),  # 노란색
                        2,
                        cv2.LINE_AA
                    )
                    
    except Exception as e:
        logger.warning(f"Gesture annotation 중 오류: {e}")
    
    return annotated_image

# 레거시 호환성 함수들
def draw_landmarks_legacy_compatible(image, landmarks, connections=None):
    """레거시 코드 호환성을 위한 래퍼 함수"""
    logger.info("레거시 호환성 함수 사용 중 - 기존 Solutions API 활용")
    
    if connections:
        mp_drawing.draw_landmarks(
            image, landmarks, connections,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_pose_connections_style()
        )
    else:
        mp_drawing.draw_landmarks(image, landmarks)
    
    return image
