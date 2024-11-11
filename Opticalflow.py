#파이썬 버전문제로 라이브러리 추가는아직안함 버전맞추고 readme대로하면될듯 일단은 이버전으로 영상움직임판단하면될듯
import cv2
import mediapipe as mp
import numpy as np


def getvideoresult(video_path1):
    # Mediapipe pose 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # 비디오 파일 경로 여기에 동영상넣음
    video_path = video_path1

    # 비디오 캡처 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다. 파일 경로를 확인해 주세요.")
        exit()

    # 첫 번째 프레임 읽기
    ret, prev_frame = cap.read()
    if not ret:
        print("비디오의 첫 번째 프레임을 읽을 수 없습니다.")
        cap.release()
        exit()

    # 비디오 해상도 줄이기 (예: 50%)
    prev_frame = cv2.resize(prev_frame, (prev_frame.shape[1] // 2, prev_frame.shape[0] // 2))

    # 첫 번째 프레임을 그레이스케일로 변환
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Optical Flow 기반 움직임 탐지 (프레임 샘플링)
    frame_count = 0
    sampling_rate = 5  # 매 5번째 프레임만 처리
    movement_threshold = 0.5  # 움직임을 판단할 임계값
    pixel_movement_threshold = 0.4  # 움직임이 임계값을 넘는 픽셀 비율 기준
    movement_detected_count = 0  # "움직임 감지" 프레임 카운트
    movement_not_detected_count = 0  # "움직임 없음" 프레임 카운트

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 비디오 해상도 줄이기
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        frame_count += 1
        if frame_count % sampling_rate != 0:
            continue  # 샘플링 비율에 따라 프레임 건너뜀

        # Mediapipe로 관절 인식
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # 관절 좌표 가져오기
            landmarks = results.pose_landmarks.landmark

            # 현재 프레임을 그레이스케일로 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Optical Flow 계산
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Optical Flow의 magnitude 계산
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # 움직임이 임계값을 넘는 픽셀들의 비율 계산
            motion_pixels = magnitude > movement_threshold
            motion_ratio = np.sum(motion_pixels) / motion_pixels.size

            # 관절을 연결하여 신체 범위 설정
            h, w, _ = frame.shape
            points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks]

            # 신체 영역을 Bounding Box로 설정
            min_x = min(p[0] for p in points)
            max_x = max(p[0] for p in points)
            min_y = min(p[1] for p in points)
            max_y = max(p[1] for p in points)

            # 신체 범위 내에서만 Optical Flow 적용
            if motion_ratio > pixel_movement_threshold:
                # 움직임이 감지되었음을 기록
                movement_detected_count += 1
                print(f"Frame {frame_count}: 움직임 감지됨 (움직임 픽셀 비율: {motion_ratio:.2f})")

                # 신체 범위 내에서만 움직임 강조 (빨간색)
                mask = magnitude > movement_threshold
                flow_mask = np.zeros_like(mask)
                flow_mask[min_y:max_y, min_x:max_x] = mask[min_y:max_y, min_x:max_x]
                frame[flow_mask] = [0, 0, 255]  # 빨간색으로 강조
            else:
                # 움직임 없음 카운트 증가
                movement_not_detected_count += 1
                print(f"Frame {frame_count}: 움직임 없음")

            # Mediapipe로 인식한 관절과 연결선을 그리기 (시각적으로 보기 위해)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 이전 프레임 업데이트
            prev_gray = gray

    # 비디오 캡처 종료
    cap.release()

    # 최종 판단: 움직임 감지 카운트와 움직임 없음 카운트를 비교
    if movement_detected_count > movement_not_detected_count:
        return 1
        #여기에 리턴값 주면됨 움직일때리턴값
    else:
        #여기에 리턴값 주면됨 비정상움직임일때 리턴값
        return 0