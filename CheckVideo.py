import os

from fastapi import UploadFile

from VideoHandler import extract_frames, process_images_and_save_landmarks

image_folder = "check_extracted_frames"
landmark_folder = "check_landmarks"

def check_video(video_path):
    video_name = video_path.split("/")[1].split(".")[0]

    # 1. 영상에서 프레임 추출
    if not os.path.exists(image_folder + '/' + video_name):
        extract_frames(video_path, image_folder, frame_interval=10)

    # 2. 추출된 프레임을 분석하여 랜드마크를 텍스트 파일로 저장
    process_images_and_save_landmarks(image_folder + '/' + video_name, landmark_folder)
