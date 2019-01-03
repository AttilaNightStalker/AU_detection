"""
This module is the main module in this package. It loads emotion recognition model from a file,
shows a webcam image, recognizes face and it's emotion and draw emotion on the image.
"""

#from image_commons import draw_with_alpha
#from model import emotions
#from landmark_face import get_facelandmark, get_feature
#from draw_dataset import _load_emoticons
from cv2 import WINDOW_NORMAL
import sys
import cv2
import glob 
import time
import math
from landmark_face import get_facelandmark
from sklearn.externals import joblib
from sklearn import svm
import numpy as np
from PIL import Image
import threading
from sklearn.decomposition import FastICA
import xml.dom.minidom

#emoticons = _load_emoticons(emotions)
#print "[INFO] Load Emotion Icons"
resize = 350
faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#clf_rbf_AU1 = joblib.load("FullAUFullDataModel/clf_rbf_AU1.m")
#clf_rbf_AU2 = joblib.load("FullAUFullDataModel/clf_rbf_AU2.m")
clf_rbf_AU4 = joblib.load("/home/attila/AU_Detection/NNmethod/Model/model004")
#clf_rbf_AU5 = joblib.load("FullAUFullDataModel/clf_rbf_AU5.m")
#clf_rbf_AU6 = joblib.load("FullAUFullDataModel/clf_rbf_AU6.m")
#clf_rbf_AU7 = joblib.load("FullAUFullDataModel/clf_rbf_AU7.m")
#clf_rbf_AU9 = joblib.load("FullAUFullDataModel/clf_rbf_AU9.m")
#clf_rbf_AU10 = joblib.load("FullAUFullDataModel/clf_rbf_AU10.m")
#clf_rbf_AU12 = joblib.load("FullAUFullDataModel/clf_rbf_AU12.m")
#clf_rbf_AU15 = joblib.load("FullAUFullDataModel/clf_rbf_AU15.m")
#clf_rbf_AU20 = joblib.load("FullAUFullDataModel/clf_rbf_AU20.m")
#clf_rbf_AU24 = joblib.load("FullAUFullDataModel/clf_rbf_AU24.m")
#clf_rbf_AU25 = joblib.load("FullAUFullDataModel/clf_rbf_AU25.m")
#clf_rbf_AU26 = joblib.load("FullAUFullDataModel/clf_rbf_AU26.m")
#clf_rbf_AU27 = joblib.load("FullAUFullDataModel/clf_rbf_AU27.m")
print ("[INFO] Load Classify Model")
Path = "/home/attila/AU_Detection/S001-080.avi"

def find_faces(image):
    faces_coordinates = _locate_faces(image)
    cutted_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in faces_coordinates]
    #loop over the cutted faces and draw a rectangle surrounding each
    for (x, y, w, h) in faces_coordinates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    normalized_faces = [_normalize_face(face) for face in cutted_faces]
    return zip(normalized_faces, faces_coordinates)

def _normalize_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (resize, resize))
    return face;

def _locate_faces(image):
    faces = faceDet.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )
    return faces  # list of (x, y, w, h)

def predict(Image1, Image2):
    model = clf_rbf_AU4
    Features = []
    for FirstNormFace, (Fx,Fy,Fw,Fh) in find_faces(Image1):
        FirstFeatureList = get_facelandmark(FirstNormFace)

        if FirstFeatureList is None:
            continue

        FXs = FirstFeatureList[::2]
        FYs = FirstFeatureList[1::2]
        F_coordi_x = []
        F_coordi_y = []
        for i in [1,4,5,8,19,21,22,23,25,26,28,30,13,14,18,31,37,34,40]:
            tx = FXs[i]
            ty = FYs[i]
            F_coordi_x.append(tx)
            F_coordi_y.append(ty)

        for normalized_face, (x, y, w, h) in find_faces(Image2):
            featureList = get_facelandmark(normalized_face)
            if featureList is None:
                continue
            #draw_face_landmark(featureList, (x,y,w,h),Image2)

            Xs = featureList[::2]
            Ys = featureList[1::2]
            IndexList = [1,4,5,8,19,21,22,23,25,26,28,30,13,14,18,31,37,34,40]
            for s in range(19):
                i = IndexList[s]
                Xxi = Xs[i]
                Yyi = Ys[i]
                Features.append(Yyi - F_coordi_y[s])
                Features.append(Xxi - F_coordi_x[s])

                for t in range(19):
                    j = IndexList[t]
                    Xxj = Xs[j]
                    Yyj = Ys[j]
                    Features.append(math.hypot(Xxi-Xxj,Yyi-Yyj))
                    Features.append(math.hypot(Xxi-Xxj,Yyi-Yyj) - 
                        math.hypot(F_coordi_x[s]-F_coordi_x[t], F_coordi_y[s]-F_coordi_y[t]))
        if len(Features) != 760:
            return 0
        #featureArray = np.array(Features).reshape(1,-1)
        #featureArray = Features
        pred = model.predict(Features) # do prediction
        print ("returning pred result")
        return pred[0]
    print ("face or ldmk not found")
    return 0

def show_webcam_and_run(window_size=None, window_name="shit", update_time=10):
    FrameDiff = 5

    cv2.namedWindow(window_name, WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if window_size:
        width, height = window_size
    cv2.resizeWindow(window_name, width, height)

    #vc = cv2.VideoCapture(0)
    vc = cv2.VideoCapture(Path)

    if vc.isOpened():
        read_value, webcam_image = vc.read()
    else:
        print("webcam not found")
        return
    cv2.putText(webcam_image, "Press 'Esc' to quit",
                        (10, 460), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    
    #accumulate some data
    ImageSet = []
    i = 0
    while i < FrameDiff:
        if len(find_faces(webcam_image)):
            ImageSet.append(webcam_image)
            i = i+1
            print "hell face", i
        read_value, webcam_image = vc.read()

    while read_value:
        if len(find_faces(webcam_image)):
            result = predict(ImageSet[0],webcam_image)
            print "result", result

            for i in range(FrameDiff-1):
                ImageSet[i] = ImageSet[i+1]
            ImageSet[FrameDiff-1] = webcam_image

            if result == 1:
                cv2.putText(webcam_image, "AU4 detected", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))
                print "AU4 detected"
            else:
                cv2.putText(webcam_image, "NONE", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))    
        
        cv2.imshow(window_name, webcam_image)
        read_value, webcam_image = vc.read()

        key = cv2.waitKey(20)
        if key == 27:
            break;
    
    cv2.destroyWindow(window_name)




if __name__ == '__main__':
    # use learnt model
    cam = cv2.VideoCapture(0)
    print(cam.isOpened())
    window_name = 'AU Detection(press ESC to exit)'
    show_webcam_and_run(window_size=(1200, 1600), window_name=window_name, update_time=8)


