"""
Drawing Utilities v2
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
        landmarks: 랜드마크 좌표 리스트 [(x, y), ...]
        connections: 연결할 랜드마크 인덱스 쌍 [(start_idx, end_idx), ...]
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
        if connections:
            for connection in connections:
                if (connection[0] < len(points) and 
                    connection[1] < len(points)):
                    start_point = points[connection[0]]
                    end_point = points[connection[1]]
                    cv2.line(image, start_point, end_point, color, thickness)
        
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
        
        # Draw hand connections (basic hand skeleton)
        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm connections
        ]
        
        # Draw connections
        for connection in hand_connections:
            if (connection[0] < len(points) and 
                connection[1] < len(points)):
                start_point = points[connection[0]]
                end_point = points[connection[1]]
                cv2.line(image, start_point, end_point, color, thickness)
        
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
    바운딩 박스 그리기
    
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
            # Get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, text_thickness
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (x, y - text_height - baseline - 5),
                (x + text_width, y),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (x, y - baseline - 5),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness
            )
        
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
        text: 텍스트
        position: 위치 (x, y)
        color: 텍스트 색상 (B, G, R)
        font_scale: 폰트 크기
        thickness: 선 두께
        background_color: 배경 색상 (None이면 배경 없음)
        
    Returns:
        텍스트가 그려진 이미지
    """
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        x, y = position
        
        # Draw background if specified
        if background_color:
            cv2.rectangle(
                image,
                (x, y - text_height - baseline - 5),
                (x + text_width, y + baseline + 5),
                background_color,
                -1
            )
        
        # Draw text
        cv2.putText(
            image,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness
        )
        
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
    진행률 바 그리기
    
    Args:
        image: 입력 이미지
        value: 진행률 값 (0.0 ~ 1.0)
        position: 위치 (x, y)
        size: 크기 (width, height)
        color: 진행률 바 색상 (B, G, R)
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
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            background_color,
            -1
        )
        
        # Draw progress
        progress_width = int(width * value)
        if progress_width > 0:
            cv2.rectangle(
                image,
                (x, y),
                (x + progress_width, y + height),
                color,
                -1
            )
        
        # Draw border
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            (255, 255, 255),
            1
        )
        
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
    신뢰도와 함께 랜드마크 그리기
    
    Args:
        image: 입력 이미지
        landmarks: 랜드마크 좌표 리스트 [(x, y), ...]
        confidences: 신뢰도 리스트 [0.0 ~ 1.0, ...]
        color: 기본 색상 (B, G, R)
        max_radius: 최대 반지름
        
    Returns:
        랜드마크가 그려진 이미지
    """
    try:
        if not landmarks or not confidences:
            return image
        
        for i, (landmark, confidence) in enumerate(zip(landmarks, confidences)):
            if len(landmark) >= 2:
                x, y = int(landmark[0]), int(landmark[1])
                
                # Adjust color based on confidence
                alpha = max(0.3, confidence)
                adjusted_color = tuple(int(c * alpha) for c in color)
                
                # Adjust radius based on confidence
                radius = max(1, int(max_radius * confidence))
                
                cv2.circle(image, (x, y), radius, adjusted_color, -1)
        
        return image
        
    except Exception as e:
        print(f"Error drawing landmarks with confidence: {e}")
        return image