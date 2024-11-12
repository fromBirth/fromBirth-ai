import cv2
import mediapipe as mp
import numpy as np
import os
import glob

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 데이터 저장 경로 설정
output_dir = './movement_data/'
os.makedirs(output_dir, exist_ok=True)

# 비디오 파일들이 있는 폴더 경로 설정
video_folder = './video_files/'  # 여기에 비디오 파일들을 넣어둠

# 여러 확장자 패턴으로 비디오 파일 목록 가져오기
video_paths = []
for ext in ('*.mp4', '*.mov', '*.avi', '*.mkv'):  # 필요한 확장자를 추가
    video_paths.extend(glob.glob(os.path.join(video_folder, ext)))

# 각 비디오 파일에 대해 처리
for idx, video_path in enumerate(video_paths):
    # 비디오 캡처 열기
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    sampling_rate = 5  # 매 5번째 프레임만 처리

    # 움직임 데이터를 저장할 리스트 초기화
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

            # 관절 좌표를 movement_data 리스트에 추가
            movement_data.append(current_landmarks.flatten())  # 좌표를 평탄화하여 1D 배열로 저장

    # 비디오 캡처 해제
    cap.release()

    # movement_data를 numpy 배열로 변환하여 저장
    movement_data = np.array(movement_data)
    np.save(os.path.join(output_dir, f'normal_movement_data_{idx + 1}.npy'), movement_data)  # 비디오마다 다른 이름으로 저장

# Mediapipe 해제
pose.close()
print("모든 비디오 파일 처리 완료")
