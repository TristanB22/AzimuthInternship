import cv2 as cv

while True:
	cap = cv.VideoCapture(0)
	ret, frame = cap.read()
	cv.imshow("Video", frame)
	if cv.waitKey(3) == ord('q'):
		break
