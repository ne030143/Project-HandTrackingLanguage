import cv2 
import mediapipe as mp 
import numpy as np 
import pickle
import pandas as pd

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True,
max_num_hands=1, min_detection_confidence=0.7)

with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)
  

def image_processed(hand_img):
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)
    output = hands.process(img_flip)
    
    try:
        data = output.multi_hand_landmarks[0]
        print(data)
        data = str(data)
        data = data.strip().split('\n')
        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']
        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)
        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0])
    

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks is not None:
        data = image_processed(img)
        data = np.array(data)
        y_pred = svm.predict(data.reshape(-1,63))
        print(y_pred)

        font = cv2.FONT_HERSHEY_PLAIN
        org = (50, 100)
        fontScale = 7
        color = (0, 0, 255)
        thickness = 5
        img = cv2.putText(img, str(y_pred[0]), org, font, 
                          fontScale, color, thickness, cv2.LINE_AA)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: 
            mpDraw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)        

    cv2.imshow('frame', img)
    if cv2.waitKey(1)  == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




