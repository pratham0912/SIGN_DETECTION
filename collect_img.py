import os
import cv2
import mediapipe as mp

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 10
dataset_size = 500   # ⭐ increase

labels_map = {
    0:'A', 1:'B', 2:'C', 3:'G', 4:'Y',
    5:'1', 6:'2', 7:'3', 8:'4', 9:'5'
}

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

for j in range(number_of_classes):

    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j} ({labels_map[j]})')

    # ----- Wait for user -----
    while True:
        ret, frame = cap.read()

        cv2.putText(frame,
                    f'Ready for {labels_map[j]} ? Press Q',
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    counter = 0

    while counter < dataset_size:

        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        # ⭐ Save ONLY if hand detected
        if results.multi_hand_landmarks:

            cv2.imwrite(
                os.path.join(class_dir, f'{counter}.jpg'),
                frame
            )

            counter += 1

        cv2.imshow('frame', frame)
        cv2.waitKey(60)   # ⭐ slow capture for variation

cap.release()
cv2.destroyAllWindows()
