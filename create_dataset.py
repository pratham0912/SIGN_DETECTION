import os
import pickle
import mediapipe as mp
import cv2

DATA_DIR = './data'

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,           # ⭐ IMPORTANT
    model_complexity=0,
    min_detection_confidence=0.7
)

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):

    class_path = os.path.join(DATA_DIR, dir_)

    for img_path in os.listdir(class_path):

        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(class_path, img_path))

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        # ⭐ Use ONLY first hand
        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # ⭐ Ensure correct feature size
            if len(data_aux) == 42:

                data.append(data_aux)
                labels.append(int(dir_))   # convert to int

print("Total samples:", len(data))

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
