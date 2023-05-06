import cv2
import numpy as np
import tensorflow as tf

labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
                   'Z':25}

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
