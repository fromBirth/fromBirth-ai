import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 이미지 불러오기
image = cv2.imread('test2.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

print(results.pose_landmarks)

# 관절에 점 찍기
if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# 결과 이미지 보기
cv2.imshow('Pose Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
