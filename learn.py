import os.path

import numpy as np
from DataSetService import create_dataset
from VideoHandler import extract_frames, process_images_and_save_landmarks, get_video_filenames
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

video_folder = 'learn_video/'

video_names = get_video_filenames(video_folder)
image_folder = 'extracted_frames'  # 프레임 저장 폴더
landmark_folder = 'landmarks'  # 랜드마크 저장 폴더
dataset_folder = 'datasets'

def learn():
    get_mark()
    X_train, X_test, y_train, y_test = preprocess()
    model = learn_model(X_train, y_train)
    model_test(X_test, y_test, model)

def get_mark():
    for video_full_name in video_names:
        video_name = video_full_name.split('.')[0]

        # 1. 영상에서 프레임 추출
        if not os.path.exists(image_folder + '/' + video_name):
            extract_frames(video_folder + video_full_name, image_folder, frame_interval=10)

        # 2. 추출된 프레임을 분석하여 랜드마크를 텍스트 파일로 저장
        process_images_and_save_landmarks(image_folder + '/' + video_name, landmark_folder)

        # 데이터셋 생성
        create_dataset(landmark_folder + '/' + video_name)

# 데이터셋 로드 함수
def load_datasets(landmark_dir, label_dir):
    # 모든 데이터셋을 담을 리스트
    all_landmarks = []
    all_labels = []

    # 디렉토리 내 모든 랜드마크 및 레이블 파일을 처리
    for landmark_file in os.listdir(landmark_dir):
        if landmark_file.endswith('.npy'):
            # 랜드마크 파일의 경로
            landmark_path = os.path.join(landmark_dir, landmark_file)

            # 대응하는 레이블 파일명 추출 (파일 이름 규칙에 맞게)
            label_file = landmark_file.replace('dataset', 'labels')  # 파일 이름에 맞춰 규칙 적용
            label_path = os.path.join(label_dir, label_file)

            # 파일이 모두 존재하는지 확인
            if os.path.exists(landmark_path) and os.path.exists(label_path):
                # 랜드마크 데이터 로드
                landmarks = np.load(landmark_path)
                all_landmarks.append(landmarks)

                # 레이블 데이터 로드
                label = np.load(label_path)
                all_labels.append(label)
            else:
                print(f"파일을 찾을 수 없습니다: {landmark_file} 또는 {label_file}")

    # NumPy 배열로 변환
    X = np.array(all_landmarks)
    y = np.array(all_labels)
    return X, y

def preprocess():
    # 여러 데이터셋을 저장한 폴더 경로
    landmark_dir = 'datasets'  # 랜드마크 데이터 폴더
    label_dir = 'labels'  # 레이블 데이터 폴더

    # 데이터 로드
    X, y = load_datasets(landmark_dir, label_dir)

    # 데이터 전처리 (정규화: 표준화 스케일링)
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(len(X), -1)).reshape(X.shape)

    # 학습/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"학습 데이터 크기: {X_train.shape}, 테스트 데이터 크기: {X_test.shape}")

    return X_train, X_test, y_train, y_test

# 딥러닝 모델 설계
def create_model(input_shape):
    model = Sequential()

    # 입력층 (입력 데이터 크기에 맞게 설정)
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3))  # 과적합 방지용 Dropout

    # 은닉층
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    # 출력층 (이진 분류 문제이므로 뉴런 1개와 sigmoid 활성화 함수 사용)
    model.add(Dense(1, activation='sigmoid'))

    # 모델 컴파일 (이진 분류에 적합한 binary_crossentropy 손실 함수와 Adam 옵티마이저)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def learn_model(X_train, y_train):
    # 모델 생성
    input_shape = (X_train.shape[1],)
    model = create_model(input_shape)

    # 모델 구조 확인
    model.summary()

    # 모델 학습
    history = model.fit(X_train, y_train,
                        epochs=50,  # 에포크 수
                        batch_size=32,  # 배치 사이즈
                        validation_split=0.2,  # 검증 데이터 비율
                        verbose=1)

    return model

def model_test(X_test, y_test, model):
    # 테스트 데이터로 모델 성능 평가
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"테스트 정확도: {test_accuracy * 100:.2f}%")

    # 예측 수행
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # 0.5 기준으로 클래스 분류

    # Confusion Matrix 출력
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report 출력
    report = classification_report(y_test, y_pred)
    print(report)

def main():
    learn()

if __name__ == '__main__':
    main()