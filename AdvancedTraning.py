from cv2 import WINDOW_NORMAL
import sys
import cv2
import glob 
import time
import os
import math
import string
from landmark_face import get_facelandmark
from sklearn.externals import joblib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import svm
import numpy as np
from PIL import Image
import threading
from sklearn.decomposition import FastICA
import xml.dom.minidom

def TrainEveryDim(AUnum):

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
	print "data loaded"
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
	TotalY = np.array(TotalY)


	for i in range(760):
		tempX = []
		for k in range(len(TotalX)):
			tempX.append([TotalX[k][i]])

		tempX = np.array(tempX)

		clf_rbf = svm.SVC().fit(tempX,TotalY)
		print "Dimension ", i, "Done"
		#clf_sigmoid = svm.SVC(kernel = 'sigmoid').fit(TrainX,TrainY)
		#print 'sigmoid done'
		pScore = clf_rbf.score(tempX,TotalY)
		try:
			fobj = open("DimsResult/result.txt",'a')
		except IOError:
			print "shit"
		else:
			fobj.write("AUnum "+str(AUnum)+" Dim "+str(i)+" score: "+str(pScore)+'\n')
			fobj.close()
		print "Dim", i, ": ",  pScore, 
		print "done"

	print AUnum, " Traning Done"


def SelectAndTrain (AUnum):

	offset = "75"

	file = open("DimsResult/result.txt", "r")
	results = []
	for i in range(760):
		line = file.readline()

		line = line.strip('\n')
		line = line.split(" ")
		results.append([i, string.atof(line[5])])

	results.sort(key = lambda x : x[1], reverse = True)
	
	selected = []

	for dim in results:
		if dim[1] >= 0.75:
			selected.append(dim[0])

	try:
		idxres = open("DimsResult/ArrayIndex_"+offset+".txt",'a')
	except IOError:
		print "shit"

	print "length: ", len(selected)
	for idx in selected:
		idxres.write(str(idx))
		idxres.write('\n')

	ReadPath1 = "/home/attila/AU_Detection/cohn-kanade-images/S"
	SubIndex = 0
	SessionIndex = 0
	print "AUnum ", AUnum
	TotalX, TotalY = [],[]
	for SubIndex in range(1,1000):
		
		FoldPath = ReadPath1+str(SubIndex).zfill(3)+"/"

		if os.path.isdir(FoldPath):
			print "SubIndex", SubIndex

			SX = joblib.load(FoldPath+"DataX")
			SY = joblib.load(FoldPath+"DataY")

			SX = SX.tolist()
			SY = SY.tolist()
			for i in range(len(SY)):
				if AUnum in SY[i]:
					SY[i] = 1
				else:
					SY[i] = 0

			DeleteIndex = []
			for i in range(len(SX)):
				if len(SX[i]) != 760:
					DeleteIndex.append(i)
			for i in range(len(DeleteIndex)):
				DeleteIndex[i] -= i

			for i in DeleteIndex:
				del SX[i]
				del SY[i]			

			TotalX = TotalX + SX
			TotalY = TotalY + SY

	print "data loaded"

	TotalY = np.array(TotalY)

	tempX = []

	#print selected

	for temp in TotalX:
		tempX.append(SelectArrayDim(selected, temp))

	tempX = np.array(tempX)

	clf_rbf = svm.SVC().fit(tempX,TotalY)
	pScore = clf_rbf.score(tempX,TotalY)

	pathtt = "DimsResult/AU"+str(AUnum)+"_"+"75"+".m"
	joblib.dump(clf_rbf,pathtt)
	print 'rbf done'
	pScore = clf_rbf.score(tempX,TotalY)

	print "score: ", pScore


def SelectArrayDim (select, ori):
	result = []
	for i in select:
		result.append(ori[i])
	return result


if __name__ == '__main__':
	#TrainEveryDim(4)
	SelectAndTrain(4)