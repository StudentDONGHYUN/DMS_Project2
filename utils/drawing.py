"""
Drawing Utilities - Enhanced System
통합 시스템과 호환되는 그리기 유틸리티
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

def draw_face_landmarks_on_image(
    image: np.ndarray,
    landmarks: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1
) -> np.ndarray:
    """
    얼굴 랜드마크를 이미지에 그리기
    
    Args:
        image: 입력 이미지
        landmarks: 랜드마크 좌표 리스트 [(x, y), ...]
        color: 색상 (B, G, R)
        thickness: 선 두께
        
    Returns:
        랜드마크가 그려진 이미지
    """
    try:
        if not landmarks:
            return image
        
        # Convert landmarks to integer coordinates
        points = []
        for landmark in landmarks:
            if len(landmark) >= 2:
                x, y = int(landmark[0]), int(landmark[1])
                points.append((x, y))
        
        # Draw landmarks
        for point in points:
            cv2.circle(image, point, 2, color, thickness)
        
        return image
        
    except Exception as e:
        print(f"Error drawing face landmarks: {e}")
        return image

def draw_pose_landmarks_on_image(
    image: np.ndarray,
    landmarks: List[Tuple[float, float]],
    connections: Optional[List[Tuple[int, int]]] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    포즈 랜드마크를 이미지에 그리기
    
    Args:
        image: 입력 이미지
        landmarks: 랜드마크 좌표 리스트
        connections: 연결선 정보 [(start_idx, end_idx), ...]
        color: 색상 (B, G, R)
        thickness: 선 두께
        
    Returns:
        랜드마크가 그려진 이미지
    """
    try:
        if not landmarks:
            return image
        
        # Convert landmarks to integer coordinates
        points = []
        for landmark in landmarks:
            if len(landmark) >= 2:
                x, y = int(landmark[0]), int(landmark[1])
                points.append((x, y))
        
        # Draw connections if provided
        if connections and len(points) > 1:
            for start_idx, end_idx in connections:
                if 0 <= start_idx < len(points) and 0 <= end_idx < len(points):
                    cv2.line(image, points[start_idx], points[end_idx], color, thickness)
        
        # Draw landmarks
        for point in points:
            cv2.circle(image, point, 3, color, -1)
        
        return image
        
    except Exception as e:
        print(f"Error drawing pose landmarks: {e}")
        return image

def draw_hand_landmarks_on_image(
    image: np.ndarray,
    landmarks: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    손 랜드마크를 이미지에 그리기
    
    Args:
        image: 입력 이미지
        landmarks: 랜드마크 좌표 리스트
        color: 색상 (B, G, R)
        thickness: 선 두께
        
    Returns:
        랜드마크가 그려진 이미지
    """
    try:
        if not landmarks:
            return image
        
        # Convert landmarks to integer coordinates
        points = []
        for landmark in landmarks:
            if len(landmark) >= 2:
                x, y = int(landmark[0]), int(landmark[1])
                points.append((x, y))
        
        # Hand connections (MediaPipe hand landmark structure)
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(image, points[start_idx], points[end_idx], color, thickness)
        
        # Draw landmarks
        for point in points:
            cv2.circle(image, point, 2, color, -1)
        
        return image
        
    except Exception as e:
        print(f"Error drawing hand landmarks: {e}")
        return image

def draw_bounding_box(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],  # (x, y, width, height)
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    바운딩 박스를 이미지에 그리기
    
    Args:
        image: 입력 이미지
        bbox: 바운딩 박스 (x, y, width, height)
        label: 라벨 텍스트
        color: 색상 (B, G, R)
        thickness: 선 두께
        
    Returns:
        바운딩 박스가 그려진 이미지
    """
    try:
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label if provided
        if label:
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            
            # Draw background for text
            cv2.rectangle(image, (x, y - text_height - 10), 
                         (x + text_width, y), color, -1)
            
            # Draw text
            cv2.putText(image, label, (x, y - 5), font, font_scale, 
                       (255, 255, 255), text_thickness)
        
        return image
        
    except Exception as e:
        print(f"Error drawing bounding box: {e}")
        return image

def draw_text_on_image(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.7,
    thickness: int = 2,
    background_color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    이미지에 텍스트 그리기
    
    Args:
        image: 입력 이미지
        text: 그릴 텍스트
        position: 텍스트 위치 (x, y)
        color: 텍스트 색상 (B, G, R)
        font_scale: 폰트 크기 스케일
        thickness: 선 두께
        background_color: 배경 색상 (옵션)
        
    Returns:
        텍스트가 그려진 이미지
    """
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = position
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background if specified
        if background_color:
            cv2.rectangle(image, (x, y - text_height - baseline), 
                         (x + text_width, y + baseline), background_color, -1)
        
        # Draw text
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
        
        return image
        
    except Exception as e:
        print(f"Error drawing text: {e}")
        return image

def draw_progress_bar(
    image: np.ndarray,
    value: float,
    position: Tuple[int, int],
    size: Tuple[int, int] = (200, 20),
    color: Tuple[int, int, int] = (0, 255, 0),
    background_color: Tuple[int, int, int] = (50, 50, 50)
) -> np.ndarray:
    """
    진행률 바를 이미지에 그리기
    
    Args:
        image: 입력 이미지
        value: 진행률 (0.0 ~ 1.0)
        position: 바 위치 (x, y)
        size: 바 크기 (width, height)
        color: 진행률 색상 (B, G, R)
        background_color: 배경 색상 (B, G, R)
        
    Returns:
        진행률 바가 그려진 이미지
    """
    try:
        x, y = position
        width, height = size
        
        # Clamp value to 0-1 range
        value = max(0.0, min(1.0, value))
        
        # Draw background
        cv2.rectangle(image, (x, y), (x + width, y + height), background_color, -1)
        
        # Draw progress
        progress_width = int(width * value)
        if progress_width > 0:
            cv2.rectangle(image, (x, y), (x + progress_width, y + height), color, -1)
        
        # Draw border
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 1)
        
        return image
        
    except Exception as e:
        print(f"Error drawing progress bar: {e}")
        return image

def draw_landmarks_with_confidence(
    image: np.ndarray,
    landmarks: List[Tuple[float, float]],
    confidences: List[float],
    color: Tuple[int, int, int] = (0, 255, 0),
    max_radius: int = 5
) -> np.ndarray:
    """
    신뢰도에 따라 크기가 다른 랜드마크 그리기
    
    Args:
        image: 입력 이미지
        landmarks: 랜드마크 좌표 리스트
        confidences: 신뢰도 리스트 (0.0 ~ 1.0)
        color: 색상 (B, G, R)
        max_radius: 최대 반지름
        
    Returns:
        랜드마크가 그려진 이미지
    """
    try:
        if not landmarks or len(landmarks) != len(confidences):
            return image
        
        for landmark, confidence in zip(landmarks, confidences):
            if len(landmark) >= 2:
                x, y = int(landmark[0]), int(landmark[1])
                radius = max(1, int(max_radius * confidence))
                cv2.circle(image, (x, y), radius, color, -1)
        
        return image
        
    except Exception as e:
        print(f"Error drawing landmarks with confidence: {e}")
        return image

# Legacy compatibility functions (keeping existing API)
def draw_face_landmarks(image, landmarks, color=(0, 255, 0), thickness=1):
    """Legacy function for backward compatibility"""
    return draw_face_landmarks_on_image(image, landmarks, color, thickness)

def draw_pose_landmarks(image, landmarks, connections=None, color=(255, 0, 0), thickness=2):
    """Legacy function for backward compatibility"""
    return draw_pose_landmarks_on_image(image, landmarks, connections, color, thickness)

def draw_hand_landmarks(image, landmarks, color=(0, 0, 255), thickness=2):
    """Legacy function for backward compatibility"""
    return draw_hand_landmarks_on_image(image, landmarks, color, thickness)
