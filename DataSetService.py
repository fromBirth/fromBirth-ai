import os
import numpy as np


# 랜드마크 데이터 파일에서 좌표를 읽어 리스트로 변환
def parse_landmark_file(landmark_file):
    data = []
    with open(landmark_file, 'r') as f:
        for line in f:
            category, x, y, z = line.strip().split(',')
            data.append([float(x), float(y), float(z)])
    return np.array(data).flatten()  # 1D 벡터로 변환


# 데이터셋 구축: 모든 랜드마크 파일을 읽고 하나의 데이터셋으로 결합
def create_dataset(landmark_folder):
    dataset = []
    labels = []  # 레이블 데이터를 위한 리스트 (지도 학습용)
    video_name = landmark_folder.split('/')[1]

    for file_name in os.listdir(landmark_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(landmark_folder, file_name)

            # 랜드마크 데이터를 벡터로 변환
            landmark_vector = parse_landmark_file(file_path)

            if len(landmark_vector) == 0:
                continue

            # 벡터 데이터 저장 (여기에 레이블 추가 필요)
            dataset.append(landmark_vector)

    # 파일명 또는 메타데이터에서 레이블 추출 (예: 동작 클래스)
    label = video_name.split('_')[1]  # 가정: 파일명에 레이블 정보 포함
    labels.append(label)

    # Numpy 배열로 변환
    dataset = np.array(dataset)
    labels = np.array(labels)

    # 데이터 정규화 (0~1 범위로 스케일링)
    dataset = dataset / np.max(dataset)  # 예시: 정규화

    # dataset 저장 폴더 만들기
    dataset_folder = 'datasets'
    label_folder = 'labels'
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    # 학습 데이터셋 저장 (CSV 또는 NumPy 파일로 저장)
    np.save(dataset_folder + '/' + video_name + '_' + 'landmark_dataset.npy', dataset)  # 데이터 저장
    np.save(label_folder + '/' + video_name + '_' + 'landmark_labels.npy', labels)  # 레이블 저장

    print("Dataset and labels saved.")

    return dataset, labels
