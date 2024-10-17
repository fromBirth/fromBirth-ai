import cv2
import mediapipe as mp
import os

def get_video_filenames(directory):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')  # 처리할 영상 파일 확장자들
    video_files = [f for f in os.listdir(directory) if f.endswith(video_extensions)]
    return video_files

# MediaPipe 설정
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 영상에서 10프레임마다 이미지 추출
def extract_frames(video_path, output_folder, frame_interval=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 비디오 파일 열기
    video_name = video_path.split('/')[1].split('.')[0]
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_count = 0

    makeDir(output_folder, video_name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 지정한 간격마다 추출
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder,video_name, f"frame_{extracted_count}.jpg")

            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"{video_name} Extracted {extracted_count} frames.")

# 이미지를 MediaPipe로 분석하고 랜드마크를 텍스트 파일로 저장
def process_images_and_save_landmarks(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_name = image_folder.split('/')[1]

    makeDir(output_folder, video_name)

    # Holistic 모델 초기화
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        for image_file in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)

            if image is None:
                continue

            # MediaPipe에서 랜드마크 감지
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # results = pose.process(image_rgb)

            # 랜드마크 데이터를 텍스트 파일로 저장
            landmark_file = os.path.join(output_folder,video_name, f"{os.path.splitext(image_file)[0]}.txt")
            with open(landmark_file, 'w') as f:
                # 포즈 랜드마크 저장
                if results.pose_landmarks:
                    for landmark in results.pose_landmarks.landmark:
                        f.write(f"Pose,{landmark.x},{landmark.y},{landmark.z}\n")

    print("Landmarks saved.")

def makeDir(output_folder, video_name):
    if not os.path.exists(output_folder + '/' + video_name) :
        os.makedirs(output_folder + '/' + video_name)