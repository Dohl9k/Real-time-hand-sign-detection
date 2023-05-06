from cvzone.HandTrackingModule import HandDetector
import cv2
import functions as f
import tensorflow as tf
import numpy as np
from collections import Counter

offset = 100
size_img = 64,64

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
classifier = tf.keras.models.load_model('asl_predictor.h5')
gestures = []


while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands = detector.findHands(img, draw=False)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        hand = hands[0]
        bbox = hand["bbox"]  # Bounding box info x,y,w,h
        x, y, w, h = hand['bbox']  
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        try:
            temp = cv2.resize(imgCrop, size_img)
        except:
            continue
        # images = []
        # images.append(temp)
        # images = np.array(images)
        # images = images.astype('float32')/255.0
        # image = images[0]
        temp = np.array(temp)
        temp = temp.astype('float32')/255.0
        prediction = classifier.predict(temp.reshape(1,64,64,3))[0]
        prediction = np.argmax(prediction)
        gest = f.get_label(prediction)
        gestures.append(gest)
        cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
        cv2.putText(img, gest, (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)      
        
        cv2.imshow("ImageCrop", imgCrop)
        
        if len(gestures) == 30:
            count_gestures = Counter(gestures)
            print(count_gestures)
            # gest_max = list(count_gestures.keys())
            # print(gest_max)
            # tmp = count_gestures[gest_max[0]]
            # print(tmp)
            gest_max = [k for k, v in sorted(count_gestures.items(),
                                              key=lambda item: item[1], reverse=True)]
            print(f"List {gest_max}")
            tmp = count_gestures[gest_max[0]]
            print(f"Answer {tmp}")
            print(f"Prediction:{tmp/len(gestures)}")
            break

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cv2.putText(img, str(fps), (8, 80), cv2.FONT_HERSHEY_PLAIN,
                                3, (255, 0, 255), 4)  
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()