import numpy as np
import cv2 
from pathlib import Path

def get_image():
    Class = 'R'
    Path('DATASET/'+Class).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    i = 0    
    while True:
       
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame")
            break
        # frame = cv.flip(frame,1)
        i += 1
        if i % 5==0:
            cv2.imwrite('DATASET/'+Class+'/'+str(i)+'.png',frame)
      
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q') or i > 500:
            break
  
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
   get_image()
  