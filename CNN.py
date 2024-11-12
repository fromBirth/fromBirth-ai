from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import glob

# 모델 파일 경로 설정
model_path = './movement_classification_model.h5'  # .h5 형식으로 저장

# 학습 데이터 파일 경로 설정
normal_data_files = glob.glob('./movement_data/normal_movement_data_*.npy')  # 모든 정상 데이터 파일
abnormal_data_files = glob.glob('./movement_data/abnormal_movement_data_*.npy')  # 모든 비정상 데이터 파일

# 정상 데이터 파일을 모두 로드하여 결합
if normal_data_files:
    normal_data = [np.load(f) for f in normal_data_files]
    normal_data = np.concatenate(normal_data, axis=0)  # 모든 정상 데이터를 하나의 배열로 결합
else:
    raise FileNotFoundError("정상 데이터 파일이 존재하지 않습니다.")

# 비정상 데이터 파일을 모두 로드하여 결합
if abnormal_data_files:
    abnormal_data = [np.load(f) for f in abnormal_data_files]
    abnormal_data = np.concatenate(abnormal_data, axis=0)
else:
    raise FileNotFoundError("비정상 데이터 파일이 존재하지 않습니다.")

# 레이블 설정 (정상: 0, 비정상: 1)
normal_labels = np.zeros(len(normal_data))
abnormal_labels = np.ones(len(abnormal_data))

# 데이터 결합
X_train = np.concatenate([normal_data, abnormal_data], axis=0)
y_train = np.concatenate([normal_labels, abnormal_labels], axis=0)

# 기존 모델이 있으면 불러오기, 없으면 새로 생성
if os.path.exists(model_path):
    # 모델 불러오기
    model = load_model(model_path)
    print("기존 모델을 불러왔습니다.")
else:
    # 새 모델 생성
    model = Sequential([
        Flatten(input_shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # 이진 분류를 위한 sigmoid 활성화 함수
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    print("새 모델을 생성했습니다.")

# 모델 추가 학습
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 모델 저장 경로 폴더가 없으면 생성
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# 학습 완료 후 모델 저장
model.save(model_path)  # .h5 형식으로 저장
print("모델이 저장되었습니다.")
