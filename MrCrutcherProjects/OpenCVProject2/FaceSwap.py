from __future__ import print_function
import cv2 as cv
import argparse
def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)

    if(len(faces) == 2):
        x1, y1, w1, h1 = faces[0]
        x2, y2, w2, h2 = faces[1]

        faceROI1 = frame[y1 : y1 + h1, x1 : x1 + w1]
        faceROI2 = frame[y2 : y2 + h2, x2 : x2 + w2]

        ROI1shape = faceROI1.shape[0]
        ROI2shape = faceROI2.shape[0]
        
        faceROI1 = cv.resize(faceROI1, (ROI2shape, ROI2shape))
        faceROI2 = cv.resize(faceROI2, (ROI1shape, ROI1shape))

        # if(faceROI1.shape[0] > faceROI2.shape[0]):
        #     diff = faceROI1.shape[0] - faceROI2.shape[0]
        #     if(diff % 2 == 1):
        #         faceROI1 = faceROI1[0 : faceROI1.shape[0] - 1, 0 : faceROI1.shape[0] - 1]
        #     faceROI1 = faceROI1[int(diff / 2) : faceROI1.shape[0] - int(diff / 2), int(diff / 2) : faceROI1.shape[0] - int(diff / 2)]
        # elif (faceROI1.shape[0] < faceROI2.shape[0]):
        #     diff = faceROI2.shape[0] - faceROI1.shape[0]
        #     if(diff % 2 == 1):
        #         faceROI2 = faceROI2[0 : faceROI2.shape[0] - 1, 0 : faceROI2.shape[0] - 1]
        #     faceROI2 = faceROI2[int(diff / 2) : faceROI2.shape[0] - int(diff / 2), int(diff / 2) : faceROI2.shape[0] - int(diff / 2)]

        frame[y1 : y1 + faceROI2.shape[0], x1 :x1 + faceROI2.shape[1]] = faceROI2
        frame[y2 : y2 + faceROI1.shape[0], x2 :x2 + faceROI1.shape[1]] = faceROI1
        
        cv.imshow("face1", faceROI1)
        cv.imshow("face2", faceROI2)

    cv.imshow('Capture - Face detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/OpenCVProject2/haarcascade_frontalface_alt.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
face_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break