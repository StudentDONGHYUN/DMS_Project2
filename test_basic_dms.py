# test_basic_dms.py - 기본 DMS 기능 테스트 (MediaPipe 기반)

import cv2
import mediapipe as mp
import time
import logging
import argparse
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicDMSTest:
    """기본 DMS 기능 테스트 클래스"""
    
    def __init__(self, confidence=0.5, max_faces=1, max_hands=2):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # MediaPipe 초기화
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )
        
        # 통계
        self.frame_count = 0
        self.face_detection_count = 0
        self.hand_detection_count = 0
        self.pose_detection_count = 0
        
        logger.info("기본 DMS 테스트 클래스 초기화 완료")
    
    def calculate_eye_aspect_ratio(self, face_landmarks, eye_indices):
        """눈 종횡비(EAR) 계산 - 기본 피로도 감지"""
        try:
            landmarks = face_landmarks.landmark
            
            # 눈의 주요 점들 추출
            eye_points = []
            for idx in eye_indices:
                point = landmarks[idx]
                eye_points.append([point.x, point.y])
            
            eye_points = np.array(eye_points)
            
            # 수직 거리들
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            
            # 수평 거리
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            
            # EAR 계산
            ear = (A + B) / (2.0 * C)
            return ear
            
        except Exception as e:
            logger.warning(f"EAR 계산 실패: {e}")
            return 0.3  # 기본값
    
    def detect_head_pose(self, face_landmarks, image_shape):
        """기본 머리 포즈 감지"""
        try:
            h, w = image_shape[:2]
            
            # 주요 얼굴 랜드마크
            landmarks = face_landmarks.landmark
            nose_tip = landmarks[1]
            chin = landmarks[175]
            left_eye = landmarks[33]
            right_eye = landmarks[362]
            
            # 화면 좌표로 변환
            nose_tip_2d = np.array([nose_tip.x * w, nose_tip.y * h])
            chin_2d = np.array([chin.x * w, chin.y * h])
            left_eye_2d = np.array([left_eye.x * w, left_eye.y * h])
            right_eye_2d = np.array([right_eye.x * w, right_eye.y * h])
            
            # 기본 각도 계산
            face_center_x = (left_eye_2d[0] + right_eye_2d[0]) / 2
            face_center_y = (left_eye_2d[1] + right_eye_2d[1]) / 2
            
            # 수평/수직 편향
            horizontal_deviation = abs(face_center_x - w/2) / w
            vertical_deviation = abs(face_center_y - h/2) / h
            
            return {
                'horizontal_deviation': horizontal_deviation,
                'vertical_deviation': vertical_deviation,
                'looking_away': horizontal_deviation > 0.3 or vertical_deviation > 0.3
            }
            
        except Exception as e:
            logger.warning(f"머리 포즈 감지 실패: {e}")
            return {'horizontal_deviation': 0, 'vertical_deviation': 0, 'looking_away': False}
    
    def detect_hand_on_wheel(self, hand_results, image_shape):
        """핸들 위치 손 감지 (기본 구현)"""
        h, w = image_shape[:2]
        hands_in_driving_zone = 0
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # 손목 위치
                wrist = hand_landmarks.landmark[0]
                wrist_x, wrist_y = wrist.x * w, wrist.y * h
                
                # 운전 영역 정의 (화면 중앙 하단 영역)
                driving_zone_x_min = w * 0.2
                driving_zone_x_max = w * 0.8
                driving_zone_y_min = h * 0.4
                driving_zone_y_max = h * 0.9
                
                if (driving_zone_x_min <= wrist_x <= driving_zone_x_max and
                    driving_zone_y_min <= wrist_y <= driving_zone_y_max):
                    hands_in_driving_zone += 1
        
        return hands_in_driving_zone
    
    def process_frame(self, frame):
        """프레임 처리 및 분석"""
        self.frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe 처리
        face_results = self.face_mesh.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        
        # 분석 결과 초기화
        analysis = {
            'face_detected': False,
            'hands_detected': 0,
            'pose_detected': False,
            'eye_aspect_ratio': {'left': 0.3, 'right': 0.3},
            'head_pose': {'horizontal_deviation': 0, 'vertical_deviation': 0, 'looking_away': False},
            'hands_in_driving_zone': 0,
            'fatigue_risk': 'LOW',
            'distraction_risk': 'LOW'
        }
        
        # 얼굴 분석
        if face_results.multi_face_landmarks:
            self.face_detection_count += 1
            analysis['face_detected'] = True
            
            for face_landmarks in face_results.multi_face_landmarks:
                # 눈 종횡비 계산 (기본 피로도)
                left_eye_indices = [33, 7, 163, 144, 145, 153]  # 왼쪽 눈 주요 점들
                right_eye_indices = [362, 382, 381, 380, 374, 373]  # 오른쪽 눈 주요 점들
                
                analysis['eye_aspect_ratio']['left'] = self.calculate_eye_aspect_ratio(face_landmarks, left_eye_indices)
                analysis['eye_aspect_ratio']['right'] = self.calculate_eye_aspect_ratio(face_landmarks, right_eye_indices)
                
                # 머리 포즈 분석
                analysis['head_pose'] = self.detect_head_pose(face_landmarks, frame.shape)
                
                # 얼굴 랜드마크 그리기
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        
        # 손 분석
        if hand_results.multi_hand_landmarks:
            self.hand_detection_count += 1
            analysis['hands_detected'] = len(hand_results.multi_hand_landmarks)
            analysis['hands_in_driving_zone'] = self.detect_hand_on_wheel(hand_results, frame.shape)
            
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # 손 랜드마크 그리기
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # 포즈 분석
        if pose_results.pose_landmarks:
            self.pose_detection_count += 1
            analysis['pose_detected'] = True
            
            # 포즈 랜드마크 그리기
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # 기본 위험도 평가
        avg_ear = (analysis['eye_aspect_ratio']['left'] + analysis['eye_aspect_ratio']['right']) / 2
        if avg_ear < 0.25:
            analysis['fatigue_risk'] = 'HIGH'
        elif avg_ear < 0.3:
            analysis['fatigue_risk'] = 'MEDIUM'
        
        if analysis['head_pose']['looking_away'] or analysis['hands_in_driving_zone'] < 1:
            analysis['distraction_risk'] = 'HIGH'
        elif analysis['hands_in_driving_zone'] < 2:
            analysis['distraction_risk'] = 'MEDIUM'
        
        return frame, analysis
    
    def draw_analysis_overlay(self, frame, analysis):
        """분석 결과 오버레이"""
        h, w = frame.shape[:2]
        
        # 기본 정보
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 감지 상태
        face_color = (0, 255, 0) if analysis['face_detected'] else (0, 0, 255)
        cv2.putText(frame, f"Face: {'YES' if analysis['face_detected'] else 'NO'}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        
        cv2.putText(frame, f"Hands: {analysis['hands_detected']} ({analysis['hands_in_driving_zone']} on wheel)", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        pose_color = (0, 255, 0) if analysis['pose_detected'] else (0, 0, 255)
        cv2.putText(frame, f"Pose: {'YES' if analysis['pose_detected'] else 'NO'}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
        
        # 피로도 분석
        if analysis['face_detected']:
            ear_left = analysis['eye_aspect_ratio']['left']
            ear_right = analysis['eye_aspect_ratio']['right']
            cv2.putText(frame, f"Eye AR: L={ear_left:.3f} R={ear_right:.3f}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 위험도 표시
        fatigue_color = self.get_risk_color(analysis['fatigue_risk'])
        distraction_color = self.get_risk_color(analysis['distraction_risk'])
        
        cv2.putText(frame, f"Fatigue Risk: {analysis['fatigue_risk']}", 
                   (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fatigue_color, 2)
        cv2.putText(frame, f"Distraction Risk: {analysis['distraction_risk']}", 
                   (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, distraction_color, 2)
        
        # 통계 표시
        if self.frame_count > 0:
            face_rate = (self.face_detection_count / self.frame_count) * 100
            cv2.putText(frame, f"Detection Rate: Face {face_rate:.1f}%", 
                       (w-250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def get_risk_color(self, risk_level):
        """위험도별 색상 반환"""
        if risk_level == 'HIGH':
            return (0, 0, 255)  # 빨간색
        elif risk_level == 'MEDIUM':
            return (0, 255, 255)  # 노란색
        else:
            return (0, 255, 0)  # 녹색

def test_basic_dms(device_id=0, confidence=0.5):
    """기본 DMS 테스트 실행"""
    logger.info(f"=== 기본 DMS 테스트 시작 (카메라: {device_id}) ===")
    
    try:
        # DMS 테스트 객체 생성
        dms_test = BasicDMSTest(confidence=confidence)
        
        # 웹캠 열기
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            logger.error(f"카메라를 열 수 없음: {device_id}")
            return False
        
        # 카메라 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("기본 DMS 테스트 실행 중... 'q'를 눌러 종료")
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("프레임을 읽을 수 없음")
                break
            
            # DMS 처리
            processed_frame, analysis = dms_test.process_frame(frame)
            
            # 결과 오버레이
            dms_test.draw_analysis_overlay(processed_frame, analysis)
            
            # FPS 계산 및 표시
            elapsed = time.time() - start_time
            fps = dms_test.frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (processed_frame.shape[1]-100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 화면 표시
            cv2.imshow("기본 DMS 테스트", processed_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 스크린샷 저장
                filename = f"dms_test_frame_{dms_test.frame_count}.jpg"
                cv2.imwrite(filename, processed_frame)
                logger.info(f"스크린샷 저장: {filename}")
        
        # 최종 통계
        total_time = time.time() - start_time
        avg_fps = dms_test.frame_count / total_time
        
        logger.info("=== 기본 DMS 테스트 결과 ===")
        logger.info(f"총 처리 프레임: {dms_test.frame_count}")
        logger.info(f"평균 FPS: {avg_fps:.2f}")
        logger.info(f"얼굴 감지율: {(dms_test.face_detection_count/dms_test.frame_count)*100:.1f}%")
        logger.info(f"손 감지율: {(dms_test.hand_detection_count/dms_test.frame_count)*100:.1f}%")
        logger.info(f"포즈 감지율: {(dms_test.pose_detection_count/dms_test.frame_count)*100:.1f}%")
        
        cap.release()
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        logger.error(f"기본 DMS 테스트 중 오류: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="기본 DMS 기능 테스트")
    parser.add_argument("--camera", "-c", type=int, default=0, help="카메라 장치 ID")
    parser.add_argument("--confidence", type=float, default=0.5, help="MediaPipe 신뢰도 임계값")
    
    args = parser.parse_args()
    
    logger.info("DMS 기본 기능 테스트 시작")
    
    success = test_basic_dms(args.camera, args.confidence)
    
    if success:
        logger.info("✅ 기본 DMS 테스트 완료")
    else:
        logger.error("❌ 기본 DMS 테스트 실패")