import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# ---------------------
# Mediapipe 초기화
# ---------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ---------------------
# 학습된 모델 로드
# ---------------------

model = load_model('movement_classification_model.h5')

# ---------------------
# 비디오에서 관절 데이터 추출 및 모델 예측
# ---------------------

def predict_movement(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    sampling_rate = 5  # 매 5번째 프레임만 처리
    movement_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % sampling_rate != 0:
            continue  # 샘플링 비율에 따라 프레임 건너뜀

        # Mediapipe로 관절 인식
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # 관절 좌표 가져오기
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            current_landmarks = np.array([(landmark.x, landmark.y) for landmark in landmarks])
            movement_data.append(current_landmarks.flatten())  # 좌표를 평탄화하여 1D 배열로 저장

    # 비디오 캡처 종료
    cap.release()

    # 모델을 사용하여 예측 실행
    if movement_data:
        movement_data = np.array(movement_data)
        predictions = model.predict(movement_data)

        # 예측 결과 출력
        for i, prediction in enumerate(predictions):
            label = "정상" if prediction < 0.5 else "비정상"
            print(f"Frame {i + 1}: {label}")
    else:
        print("No movement data found in video.")

    # Mediapipe 종료
    pose.close()

# ---------------------
# 비디오 파일 경로 설정 및 예측 실행
# ---------------------

test_video_path = './SODA_751468842.mov'  # 테스트할 비디오 파일 경로
predict_movement(test_video_path)
