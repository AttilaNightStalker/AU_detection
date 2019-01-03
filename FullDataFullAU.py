#!/usr/bin/env python
#coding=utf-8

"""
This module is the main module in this package. It loads emotion recognition model from a file,
shows a webcam image, recognizes face and it's emotion and draw emotion on the image.
"""

from cv2 import WINDOW_NORMAL
import sys
import cv2
import glob 
import time
import os
import math
from landmark_face import get_facelandmark
from sklearn.externals import joblib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import svm
import numpy as np
from PIL import Image
import threading
from sklearn.decomposition import FastICA
import xml.dom.minidom

resize = 350
faceDet = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

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

def draw_face_landmark(featureList, (x, y, w, h), img):
    Xs = featureList[::2]
    Ys = featureList[1::2]
    x_nose = Xs[31 - 18] * w / resize
    y_nose = Ys[31 - 18] * h / resize
    #cv2.circle(img, (x, y), 5, (255, 0, 0), 0)
    #cv2.circle(img, (x + x_nose, y + y_nose), 5, (255, 0, 0), 0)
    #print len(Xs)
    IndexList = [1,4,5,8,19,21,22,23,25,26,28,30,13,14,18,31,37,34,40]
    for i in range(len(Xs)):
        xx = Xs[i] * w / resize
        yy = Ys[i] * h / resize
        if i in IndexList:
            cv2.circle(img, (x + xx, y + yy), 3, (0, 0, 255), 0)
    return img

def ProduceDataPerSubject():
    """
    This function is used to computes the 'X','Y' for svm,and store them in a standard format

    """
    ReadPath1 = "/home/romer/Desktop/FAC_DataBase/Cohn-Kanade database/CK+/extended-cohn-kanade-images/cohn-kanade-images/S"
    SubIndex = 0
    SessionIndex = 0
    PhotoIndex = 0       
    ReadPath2 = ".png"
    TotalX = []
    TotalY = []
    for SubIndex in range(1,6):

        FoldPath = ReadPath1+str(SubIndex).zfill(3)+"/"
        LabelPath = "/home/romer/Desktop/FAC_DataBase/Cohn-Kanade database/CK+/FACS_labels/FACS/S"+str(SubIndex).zfill(3)+"/"
        if os.path.isdir(FoldPath):
            print ("SubIndex", SubIndex)
            for SessionIndex in range(1,20):
                
                SessionPath = FoldPath+str(SessionIndex).zfill(3)+"/"
                if (os.path.isdir(SessionPath)):
                    LabelSessionPath = LabelPath+str(SessionIndex).zfill(3)+"/"
                    TxtName = os.listdir(LabelSessionPath)[0]
                    pfile = open(LabelSessionPath+TxtName)
                    tMMPy = []
                    TxtData = pfile.readlines()
                    for line in TxtData:
                        msg = line.split()
                        #print "msg", msg
                        tMMPy.append(int(float(msg[0]))) #ty contains the facs labels of the session accordingly
                    print "ty:", tMMPy

                    tMMPSy = 0
                    for ttt in tMMPy:
                        if AUnum in tMMPy:
                            tMMPSy = 1
            
                

                    PhotoIndex = 1
                    ReadPath3 = SessionPath+"S"+str(SubIndex).zfill(3)+"_"+str(SessionIndex).zfill(3)+"_"
                    PhotoPath = ReadPath3+str(PhotoIndex).zfill(8)+ReadPath2
                    img = cv2.imread(PhotoPath)

                    #save the features points of the first frame
                    FirstNormFace, (Fx,Fy,Fw,Fh) = find_faces(img)[0]
                    FirstFeatureList = get_facelandmark(FirstNormFace)
                
                    FXs = FirstFeatureList[::2]
                    FYs = FirstFeatureList[1::2]
                    F_coordi_x = []
                    F_coordi_y = []
                    for i in [1,4,5,8,19,21,22,23,25,26,28,30,13,14,18,31,37,34,40]:
                        tx = FXs[i] * Fw / resize
                        ty = FYs[i] * Fh / resize
                        F_coordi_x.append(tx)
                        F_coordi_y.append(ty)

                    while os.path.isfile(PhotoPath):
                        #FeatureIndex = 0
                        img = cv2.imread(PhotoPath)
                        X = []
                        for normalized_face, (x, y, w, h) in find_faces(img):
                            featureList = get_facelandmark(normalized_face)
                            if featureList is None:
                                continue

                            Xs = featureList[::2]
                            Ys = featureList[1::2]
                            IndexList = [1,4,5,8,19,21,22,23,25,26,28,30,13,14,18,31,37,34,40]
                            for s in range(19):
                                i = IndexList[s]
                                Xxi = Xs[i]*w/resize
                                Yyi = Ys[i]*w/resize
                                X.append(Yyi - F_coordi_y[s])
                                X.append(Xxi - F_coordi_x[s])

                                for t in range(19):
                                    j = IndexList[t]
                                    Xxj = Xs[j]*w/resize
                                    Yyj = Ys[j]*w/resize
                                    X.append(math.hypot(Xxi-Xxj,Yyi-Yyj))
                                    X.append(math.hypot(Xxi-Xxj,Yyi-Yyj) - 
                                        math.hypot(F_coordi_x[s]-F_coordi_x[t], F_coordi_y[s]-F_coordi_y[t]))
                        #print X    
                        TotalX.append(X)
                        TotalY.append(tMMPSy)
                        #print tMMPy
                        #print TotalY
                        PhotoIndex += 1
                        PhotoPath = ReadPath3+str(PhotoIndex).zfill(8)+ReadPath2

                    print ("Session ",SessionIndex," done")

            #print "done", SubIndex
            #print "TotalY", TotalY
    TrainX = np.array(TotalX)
    print TrainX[0]
    TrainY = np.array([0,1,1,1,1,1,0,0,0,0,1])
    print"TrainY:", TrainY
    clf_rbf = svm.SVC().fit(TrainX,TrainY)
    print "train done"
    joblib.dump(TrainX, "DataX")
    #joblib.dump(TrainY, "DataY")

def train():
    X = joblib.load("DataX")
    Y = joblib.load("DataY")

    #traim
    clf_linear  = svm.SVC(kernel = 'linear').fit(X,Y)
    print 'linear done'
    clf_poly    = svm.SVC(kernel = 'poly', degree = 3).fit(X,Y)
    print 'poly done'
    clf_rbf     = svm.SVC().fit(X,Y)
    print 'rbf done'
    clf_sigmoid = svm.SVC(kernel = 'sigmoid').fit(X,Y)
    print 'sigmoid done'

    joblib.dump(clf_linear, "models/clf_linear.m")
    joblib.dump(clf_poly, "models/clf_poly.m")
    joblib.dump(clf_rbf, "models/clf_rbf.m")
    joblib.dump(clf_sigmoid, "models/clf_sigmoid.m")

def test(Session_Num):

    clf_linear = joblib.load("models/clf_linear.m")
    clf_poly = joblib.load("models/clf_poly.m")
    clf_rbf = joblib.load("models/clf_rbf.m")
    clf_sigmoid = joblib.load("models/clf_sigmoid.m")

    window_name = 'test'
    # cv2.namedWindow(window_name, WINDOW_NORMAL)
   # window_size=(1600, 1200)
    #if window_size:
    #    width, height = window_size
    #    cv2.resizeWindow(window_name, width, height)

    ReadPath1 = "/home/nigel/桌面/FAC_DataBase/MMI database/mmi-facial-expression-database_subject1/Sessions/"
    ReadPath2 = "/S001-"
    ReadPath3 = ".avi"
    #Session_Num = 79
    num = str(Session_Num).zfill(3)
    Path = ReadPath1+str(Session_Num)+ReadPath2+num+ReadPath3
    vc = cv2.VideoCapture(Path)
    read_value, webcam_image = vc.read()
	

    FirstNormFace, (Fx,Fy,Fw,Fh) = find_faces(webcam_image)[0]
    FirstFeatureList = get_facelandmark(FirstNormFace)
    FXs = FirstFeatureList[::2]
    FYs = FirstFeatureList[1::2]
    F_coordi_x = []
    F_coordi_y = []
    for i in [1,4,5,8,19,21,22,23,25,26,28,30,13,14,18,31,37,34,40]:
        tx = FXs[i] * Fw / resize
        ty = FYs[i] * Fh / resize
        F_coordi_x.append(tx)
        F_coordi_y.append(ty)


    FrameNum = 0
    Features = []
    resultY = []
    for i in range(760):
        Features.append([])
    while read_value:
        FeatureIndex = 0
        for normalized_face, (x, y, w, h) in find_faces(webcam_image):
            featureList = get_facelandmark(normalized_face)
            if featureList is None:
                continue
            #webcam_image = draw_face_landmark(featureList, (x,y,w,h), webcam_image)
            Xs = featureList[::2]
            Ys = featureList[1::2]
            IndexList = [1,4,5,8,19,21,22,23,25,26,28,30,13,14,18,31,37,34,40]
            for s in range(19):
                i = IndexList[s]
                Xxi = Xs[i]*w/resize
                Yyi = Ys[i]*w/resize
                Features[FeatureIndex].append(Yyi - F_coordi_y[s])
                    #Features[FeatureIndex].append(1)
                FeatureIndex+=1
                Features[FeatureIndex].append(Xxi - F_coordi_x[s])
                FeatureIndex+=1

                for t in range(19):
                    j = IndexList[t]
                    Xxj = Xs[j]*w/resize
                    Yyj = Ys[j]*w/resize
                    Features[FeatureIndex].append(math.hypot(Xxi-Xxj,Yyi-Yyj))
                    FeatureIndex+=1
                    Features[FeatureIndex].append(math.hypot(Xxi-Xxj,Yyi-Yyj) - math.hypot(F_coordi_x[s]-F_coordi_x[t], F_coordi_y[s]-F_coordi_y[t]))
                    FeatureIndex+=1
            #print FeatureIndex
        X = []
        for i in range(FeatureIndex):
            t = []
            f = derivative(Features[i],t)
            print "****************"
            print i
            print len(f)
            X.append(f[FrameNum])
        if not X == []:
            # print X
            Y = clf_linear.predict(X)
	    resultY.append(Y)
	    print '$$$$$$'
            print Y
	    print '$$$$$$'
            FrameNum += 1
        print FrameNum
        # cv2.imshow(window_name, webcam_image)
        read_value, webcam_image = vc.read()
    print resultY

    vc = cv2.VideoCapture(Path)
    read_value, webcam_image = vc.read()

    window_name = 'test'
    window_size = (1200, 1600)
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    width, height = window_size
    cv2.resizeWindow(window_name, width, height)
    while read_value:
	for normalized_face, (x, y, w, h) in find_faces(webcam_image):
            featureList = get_facelandmark(normalized_face)
            if featureList is None:
                continue
            webcam_image = draw_face_landmark(featureList, (x,y,w,h), webcam_image)

        if resultY[30] == [1]:
	    cv2.putText(webcam_image, "YES", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))
	else:
	    cv2.putText(webcam_image, "NO", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))
	cv2.imshow(window_name, webcam_image)
	cv2.waitKey(20)
	read_value, webcam_image = vc.read()

    
    cv2.destroyWindow(window_name)

def TrainCrossValidation(AUnum):

    #read data process
    ReadPath1 = "/home/attila/AU_Detection/cohn-kanade-images/S"
    SubIndex = 0
    SessionIndex = 0
    print "AUnum ", AUnum
    TotalX, TotalY = [],[]
    for SubIndex in range(1,1000):
        
        FoldPath = ReadPath1+str(SubIndex).zfill(3)+"/"
        #LabelPath = "/home/romer/Desktop/FAC_DataBase/Cohn-Kanade database/CK+/FACS_labels/FACS/S"+str(SubIndex).zfill(3)+"/"
        if os.path.isdir(FoldPath):
            print "SubIndex", SubIndex
            #for SessionIndex in range(1,20):
                
            #    SessionPath = FoldPath+str(SessionIndex).zfill(3)+"/"
            #    if (os.path.isdir(SessionPath)):
                    #LabelSessionPath = LabelPath+str(SessionIndex).zfill(3)+"/"
                    #TxtName = os.listdir(LabelSessionPath)[0]
                    #pfile = open(LabelSessionPath+TxtName)
                    #ty = []
                    #TxtData = pfile.readlines()
                    #for line in TxtData:
                    #    msg = line.split()
                        #print "msg", msg
                    #    ty.append(int(float(msg[0]))) #ty contains the facs labels of the session accordingly
            SX = joblib.load(FoldPath+"DataX")
            SY = joblib.load(FoldPath+"DataY")

            SX = SX.tolist()
            SY = SY.tolist()
            for i in range(len(SY)):
                if AUnum in SY[i]:
                    SY[i] = 1
                else:
                    SY[i] = 0

            #print "len(SX):",len(SX)
            #print "len(SY):", len(SY)
            DeleteIndex = []
            for i in range(len(SX)):
                if len(SX[i]) != 760:
                    DeleteIndex.append(i)
            for i in range(len(DeleteIndex)):
                DeleteIndex[i] -= i

            for i in DeleteIndex:
                del SX[i]
                del SY[i]

            #print "len(SX):",len(SX)
            #print "len(SY):", len(SY)

            TotalX = TotalX + SX
            TotalY = TotalY + SY
            #print SY

            #print "Sub ",SubIndex," loaded"

    #Leave One Subject Out Cross Validation process
    #TotalX1 = joblib.load("DataX")
    #print TotalX1[0]
    #TotalY1 = joblib.load("DataY")
    #print "len TotalX",len(TotalX)
    #print "len TotalY",len(TotalY)
    print ("data loaded")
    '''
    logo = LeaveOneGroupOut()
    Scores = []
    TestNum = 0
    #print "len TotalX1: ", len(TotalX1)
    for train, test in logo.split(TotalX, TotalY, groups = TotalL):
        TestNum += 1
        TrainX, TrainY, TestX,TestY = [],[],[],[]

        print train
        for i in train:
            TrainX.append(TotalX[i])
            TrainY.append(TotalY[i])

        print test
        for i in test:
            TestX.append(TotalX[i])
            TestY.append(TotalY[i])
    '''
    TotalX = np.array(TotalX)
    TotalY = np.array(TotalY)


    for i in TotalX:
        if len(i) != 760:
            print "shit"

        #print "TestX:", TestX[0]
        #print "TestY:", TestY
        #clf_linear  = svm.SVC(kernel = 'linear').fit(TrainX,TrainY)
        #print 'linear done'
        #clf_poly    = svm.SVC(kernel = 'poly', degree = 3).fit(TrainX,TrainY)
        #print 'poly done'

    clf_rbf = svm.SVC().fit(TotalX,TotalY)
    pathtt = "FullAUFullDataModel_x/clf_rbf_AU"+str(AUnum)+".m"
    joblib.dump(clf_rbf,pathtt)
    print 'rbf done'
        #clf_sigmoid = svm.SVC(kernel = 'sigmoid').fit(TrainX,TrainY)
        #print 'sigmoid done'
    pScore = clf_rbf.score(TotalX,TotalY)
    try:
        fobj = open("FullAUFullDataModel_x/result.txt",'a')
    except IOError:
        print "shit"
    else:
        fobj.write("AUnum "+str(AUnum)+" score: "+str(pScore)+'\n')
        fobj.close()
    print "AU ", AUnum,":", pScore, "done"
    





if __name__ == '__main__':
    #train( window_size=(1600, 1200), window_name=window_name, update_time=8)
    #ComputeFeatures()
    #train()
    #X = joblib.load("DataX")
    #Y = joblib.load("DataY")
    #ProduceDataPerSubject()
    # clf_linear = joblib.load("models/clf_linear.m")
    # print 'linear', clf_linear.score(X,Y)
    # clf_poly = joblib.load("models/clf_poly.m")
    # print 'poly', clf_poly.score(X,Y)
    # clf_rbf = joblib.load("models/clf_rbf.m")
    # print 'rbf', clf_rbf.score(X,Y)
    # clf_sigmoid = joblib.load("models/clf_sigmoid.m")
    # print 'sigmoid', clf_sigmoid.score(X,Y)
    #num = sys.argv[1]
    #test(int(num))
    for ssss in [1,2,4,5,6,7,9,10,12,15,20,24,25,26,27]:
        TrainCrossValidation(ssss)

