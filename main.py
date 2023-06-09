from cvzone.HandTrackingModule import HandDetector
import cv2
import functions as f
import numpy as np
from collections import Counter
import tensorflow as tf
from spello.model import SpellCorrectionModel


offset = 100
size_img = 64,64
signs = []
word = ""
sp_word = ""
message = ""

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
sp = SpellCorrectionModel(language='en')
sp.load("./model.pkl")

while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw=False) 

    if hands:
        hand = hands[0]
        bbox = hand["bbox"] 
        x, y, w, h = hand['bbox']  
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        hsv_sign = f.get_hsv_img(imgCrop)
        temp = f.prepare_img(imgCrop)
        # if type(temp) != np.ndarray:
        #     continue
        sign = f.get_sign(temp)
        signs.append(sign)
        cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
        cv2.putText(img, sign, (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)   
        if type(hsv_sign) == np.ndarray:
            cv2.imshow("ImgCrop", hsv_sign)
                
        if len(signs) == 30:
            count_signs = Counter(signs)
            keys = [k for k, v in sorted(count_signs.items(),
                                              key=lambda item: item[1], reverse=True)]
            tmp = count_signs[keys[0]]
            probability = tmp/len(signs)
            print(f"Prediction:{probability}")
            signs = []
            if probability > 0.7 and keys[0] != None:
                if keys[0] == 'space':
                    message += f" {sp_word}"
                    word = ""
                else:
                    word += keys[0]
                sp_word = sp.spell_correct(word)["spell_corrected_text"]

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cv2.putText(img, str(fps), (8, 80), cv2.FONT_HERSHEY_PLAIN,
                                3, (255, 0, 255), 4)  
    cv2.putText(img, message + sp_word, (8, 450), cv2.FONT_HERSHEY_PLAIN,
                                3, (255, 0, 255), 4)  
    cv2.imshow("Window", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()