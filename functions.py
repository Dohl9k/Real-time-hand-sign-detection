import cv2
import numpy as np
import tensorflow as tf

labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
                   'Z':25, 'space':26}

classifier = tf.keras.models.load_model('asl_predictor.h5')


def get_label(prediction): 
    for ins in labels_dict:
        if prediction == labels_dict[ins]:
            return ins


def prepare_img(img):
    size_img = 64,64
    try:
        img = cv2.resize(img, size_img)
    except:
        return None
    img = np.array(img)
    img = img.astype('float32')/255.0
    return img

def get_sign(img):
    try:
        prediction = classifier.predict(img.reshape(1,64,64,3), verbose=0)[0]
        prediction = np.argmax(prediction)
    except:
        return None
    sign = get_label(prediction)
    return sign


def get_hsv_img(img):
    try:
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except cv2.error:
        return None
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)
    return YCrCb_result