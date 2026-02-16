import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
from collections import deque

# ---------- Load trained model ----------
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# ---------- Initialize camera ----------
cap = cv2.VideoCapture(0)

# ---------- MediaPipe setup ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------- Label mapping ----------
labels_dict = {
    0:'A', 1:'B', 2:'C', 3:'G', 4:'Y',
    5:'1', 6:'2', 7:'3', 8:'4', 9:'5'
}

# ---------- Gesture → Voice commands ----------
voice_commands = {
    'A': "hello ",
    'B': "my",
    'C': "name ",
    'G': "GG",
    'Y': "is",
    '1': " one",
    '2': " two",
    '3': " three",
    '4': " four",
    '5': " CHUL CHUL CHUL"
}

# ---------- Text-to-Speech ----------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

last_spoken = None

# ---------- Cooldown control ----------
last_time = 0
cooldown = 1.0   # seconds between commands

# ---------- Prediction smoothing ----------
prediction_history = deque(maxlen=10)

# ---------- Main loop ----------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        data_aux = []
        x_ = []
        y_ = []

        # Collect coordinates
        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        # Normalize coordinates
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        # Ensure correct feature size
        if len(data_aux) == 42:

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # ---------- Prediction ----------
            prediction = model.predict([np.asarray(data_aux)])
            predicted_class = int(prediction[0])

            # ---------- Temporal smoothing ----------
            prediction_history.append(predicted_class)

            stable_class = max(
                set(prediction_history),
                key=prediction_history.count
            )

            predicted_character = labels_dict.get(stable_class, '?')

            # ---------- Voice command ----------
            command = voice_commands.get(predicted_character, "")
            current_time = time.time()

            if (command and
                command != last_spoken and
                current_time - last_time > cooldown):

                threading.Thread(
                    target=speak,
                    args=(command,),
                    daemon=True
                ).start()

                last_spoken = command
                last_time = current_time

            # ---------- Draw bounding box ----------
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 0, 0), 4)

            cv2.putText(frame,
                        predicted_character,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # ---------- Show frame ----------
    cv2.imshow('Gesture Voice Control', frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------- Cleanup ----------
cap.release()
cv2.destroyAllWindows()
