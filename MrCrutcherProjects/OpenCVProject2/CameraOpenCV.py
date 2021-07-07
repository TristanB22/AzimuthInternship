from __future__ import print_function
import cv2 as cv
import argparse
def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)

    for (x,y,w,h) in faces:
        # center = (x + w//2, y + h//2)
        # frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        frame = cv.putText(frame, "Tristan", (y, y), cv.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2)
        frame = cv.line(frame, (x, y), (y, y), (0, 0, 255), 1)
        faceROI = frame[y:y+h,x:x+w]
        faceROI = cv.GaussianBlur(faceROI, (51, 51), 30)
        frame[y:y+faceROI.shape[0], x:x+faceROI.shape[1]] = faceROI

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