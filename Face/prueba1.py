# import the necessary packages
from imutils import paths
from funciones import detectFaceOpenCVHaar, detectFaceOpenCVDnn, detectFaceDlibHog
import face_recognition  #paquete de python
import argparse
import pickle
import cv2
import os
import time
import urllib.request
import sys
import dlib

import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
#https://medium.com/@nilk35/how-to-use-ipwebcam-for-opencv-project-using-python-3-5-e8a3183e6517

url="http://192.168.1.139:8080/shot.jpg"
#url1="http://admin:admin@192.168.1.102:8080/video"
url1=0
 
# OpenCV HAAR
faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

#OpenCV DNN supports 2 networks.
# 1. FP16 version of the original caffe implementation ( 5.4 MB )
# 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
DNN = "TF"
#DNN = "CAFFE"
if DNN=="CAFFE":
    modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "models/opencv_face_detector_uint8.pb"
    configFile = "models/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

conf_threshold = 0.7

# DLIB HoG
hogFaceDetector = dlib.get_frontal_face_detector()



source = 0
if len(sys.argv) > 1:
    source = sys.argv[1]


cap = cv2.VideoCapture(url1)


hasFrame, frame = cap.read()
cv2.namedWindow("Face Detection Comparison", cv2.WINDOW_NORMAL)

frame_count = 0
tt_opencvHaar = 0
tt_opencvDnn = 0
tt_dlibHog = 0
tt_dlibMmod = 0

# Read until video is completed
#while(1):
while(cap.isOpened()):
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
    frame_count += 1

    #imgResp=urllib.request.urlopen(url)
    #imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    #frame=cv2.imdecode(imgNp,-1)

    t = time.time()
    outOpencvHaar, bboxes = detectFaceOpenCVHaar(faceCascade, frame)
    tt_opencvHaar += time.time() - t
    fpsOpencvHaar = frame_count / tt_opencvHaar
    label = "OpenCV Haar ; FPS : {:.2f}".format(fpsOpencvHaar)
    cv2.putText(outOpencvHaar, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

    t = time.time()
    outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,frame, conf_threshold)
    tt_opencvDnn += time.time() - t
    fpsOpencvDnn = frame_count / tt_opencvDnn
    label = "OpenCV DNN ; FPS : {:.2f}".format(fpsOpencvDnn)
    cv2.putText(outOpencvDnn, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

    t = time.time()
    outDlibHog, bboxes = detectFaceDlibHog(hogFaceDetector,frame)
    tt_dlibHog += time.time() - t
    fpsDlibHog = frame_count / tt_dlibHog
    label = "DLIB HoG ; ; FPS : {:.2f}".format(fpsDlibHog)
    cv2.putText(outDlibHog, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

    #top = np.hstack([outOpencvHaar, outOpencvDnn])
    #bottom = np.hstack([outDlibHog, outDlibMMOD])
    #combined = np.vstack([top, bottom])
    cv2.imshow("Face Detection Comparison", outDlibHog)

    if frame_count == 1:
        tt_opencvHaar = 0
        tt_opencvDnn = 0
        tt_dlibHog = 0
        tt_dlibMmod = 0

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()