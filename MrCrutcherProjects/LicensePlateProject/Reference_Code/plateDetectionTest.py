import cv2 as cv
import numpy as np
import tensorflow as tf
import imutils
from skimage.filters import threshold_local
from skimage import measure
"""
Some talking....all of the contours you see here _, contour, _ is fucking depricated, use contours, _ or heirachy instead. You needed the image before hand, not anymore? idk man
Some other stuff if you're getting "expected 3 values got 2", well that error is in the name. For this issue it was in the function plate_checker, it was looking at the 
contour varaible expecting it to be empty, I only returned 2 None values. ez pz fix. 
"""
##################### SORT CONTOURS #############################
#################################################################
def sort_cont(character_conts):
	#this will sort contours left to right
	i = 0
	
	boundingBoxes = [cv.boundingRect(c) for c in character_conts] #define boundingBox where contours exist 

	#zip returns an iterator of tuples based on the iterable object. The syntax is zip(*iterables) iterables can be(list,string,dict) and user-defined iterables
	#lambda here is a keyword used to define anonymous functions or functions with no names. It is used to define anon functions that can/can't take
	#arguments and return a value with data or expression 
	#from the dict for key "The value of the key parameter should be a function (or other callable) that takes a single argument and returns a key 
	# to use for sorting purposes. This technique is fast because the key function is called exactly once for each input record."
	# so we call sorted, we call our defined string variables, then we call or lambda as a dict function. We also do not want to reverse the sort.
	(characters_conts, boundingBoxes) = zip(*sorted(zip(character_conts, boundingBoxes), key = lambda b: b[1][i], reverse=False))

	return character_conts

##################### SEGMENTATION ##############################
#################################################################
def segment_chars(plate_img, fixed_width):
	#we will extract a valiue channel from one of the HSV formats of the image, and then apply adaptive thresholding to reveal the characters on the plate
	V = cv.split(cv.cvtColor(plate_img, cv.COLOR_BGR2HSV))[2]

	T = threshold_local(V, 29, offset=15, method='gaussian') #compute a threshold mask iamge based on local pixel neighborhood, this is the adaptive thresholding.

	thresh = (V > T).astype('uint8') * 255 #255 in 8bit is 1111 1111 adding 1111 1111 to 1111 1111 is 1 1111 1110 = 510, we only have 8 bits to represent. 
	#the leftmost 1 in the result cannot be stored, so it is now 1111 1110 which is 254. 

	#the threshold in this case is for 8bit or 32bit as we can see above, it is 8bit currently. 
	thresh = cv.bitwise_not(thresh) #Inverts every bit of an array, more detail -> calcs per-element bit-wise inversion of the input array -> dst(I) = negation src(I)
	# So tl:dr it turns 1's to 0's and 0's to 1's like a bit flip :) 
	
	#Resize the plate region to an acceptable size. 
	plate_img = imutils.resize(plate_img, width=fixed_width) #imutils evaluates width only but not height. cv.resize uses both or either. 
	thresh = imutils.resize(thresh, width=fixed_width)
	bgr_thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR) 

	#Perform connected components analysis, also init the mask to store the location of the chars 
	# https://www.kite.com/python/docs/skimage.measure.label #neighbors is depricated, use connectivity instead. 
	labels = measure.label(thresh, background=0, connectivity=1) 

	charCands = np.zeros(thresh.shape, dtype='uint8')

	#loop over the unique compontnets 
	chars = []
	for label in np.unique(labels):
		if label == 0:
			continue #ignore background label 
		#otherwise find contours in the label mask
		labelM = np.zeros(thresh.shape, dtype='uint8')
		labelM[labels == label] = 255 #gray scale

		conts = cv.findContours(labelM, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		#see that old comment to the right? ignore it, this returns true or false. also cv2 is depricated or some shit idfk. 
		conts = conts[1] if imutils.is_cv3() else conts[0] #contours is == to an array of 0 if umutils is running on opencv else it's 1, or true. 

		#let's make sure atleast 1 contour was found in the mask right?
		if len(conts) > 0:
			#grab largest cont which reps the component in the mask, then grab the bounding box for the cont
			c = max(conts, key=cv.contourArea)
			(boxX, boxY, boxW, boxH) = cv.boundingRect(c)

			# now compute aspect ratio, solodity, and height ratio
			aspectRatio = boxW / float(boxH) # width / height
			solidity = cv.contourArea(c) / float(boxW * boxH) # area of the contour / width * height
			heightRatio = boxH / float(plate_img.shape[0]) #the height / the image size

			#now check if the aspect ratio, solidity, and height of the contour pass the "rules" test
			keepAR = aspectRatio < 1.0
			keepSol = solidity > 0.15
			keepH = heightRatio < 0.5 and heightRatio < 0.95

			#now let's check to see if the component passes all of the test
			if keepAR and keepSol and keepH and boxW > 14:
				#compute the convex hull of the contours and raw it on the chars candidates mask
				#btw A set of points in a euclidea space is defined to be convex if it contains line segments connected to each pair of its points
				#this function finds the covex hull of a 2d point set using the sklansky's algorithm that has O(N logN)complexitiy (that's pretty fast :o)
				hull = cv.convexHull(c)
				cv.drawContours(charCands, [hull], -1, 255, -1)
				
	cv.imshow('"PlateArea"', plate_img)
	#back outside the for loop
	contours, hierachy = cv.findContours(charCands, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #define the contour, we now have a hierachy 
	if contours: #if there is a contour
		contours = sort_cont(contours) #we sort through the contours, this is another function. 
		addPixelVal = 4 #this value is added to each dimesion of the character
		for c in contours:
			(x, y, w, h) = cv.boundingRect(c)
			if y > addPixelVal:
				y = y * addPixelVal
			else:
				y = 0
			if x > addPixelVal:
				x = x * addPixelVal
			else : 
				x = 0
			temp = bgr_thresh[y:y + h + (addPixelVal * 2), x:x + w + (addPixelVal * 2)] #adding a temp to hold values for thresholding to add values to xywh

			chars.append(temp)
		return chars
	else:
		return None 
########################################################
########################################################

############# FINDING THE PLATE #########################
#########################################################
class FindPlate:
	def __init__(self):
		self.min_area = 4500
		self.max_area = 30000

		#self.img = cv.resize(cv.imread("licensePlate4.jpeg"), (640, 480)) #used for testing atm
		self.element_structure = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(23,3))

	def preprocessing(self, input_img):
		imgBlurred = cv.GaussianBlur(input_img, (5,5), 0) # n, n is the size of the window, let's see if it's okay
		convertGray = cv.cvtColor(imgBlurred, cv.COLOR_BGR2GRAY) # grayscale conversion on a now blurred image
		sobelx = cv.Sobel(convertGray, cv.CV_8U, 1, 0, ksize=3) # veritcle x edges
		ret2, threshold_img = cv.threshold(sobelx, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) #thresholding

		element = self.element_structure #grabbing the structure of the element we created in __init__
		morph_thresh = threshold_img.copy() # returning a copy of the tresholding for morphology
		cv.morphologyEx(src=threshold_img, op=cv.MORPH_CLOSE, kernel=element, dst=morph_thresh)#Morphological output, whitespacing essentially. 
		#The iterations for the morph control of course how much more white spacing you want, it appears 2 is "okay" 3 is better for 1 than the other. 

		cv.imshow("sobelx", sobelx)
		cv.imshow('thresh', threshold_img)
		cv.imshow('Blur', imgBlurred)
		return morph_thresh

	def extracted_contours(self, after_preprocess):
		contours, _ = cv.findContours(after_preprocess, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE) 
		cv.imshow('conts', after_preprocess) #show contours
		return contours
	
	def plate_cleaner(self, plate):
		gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY) #conversion to grayscale
		thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2) #adaptive thresholding on grayscale image
		contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) #applying contour technique on copied thresholding. 
		
		if contours:
			area = [cv.contourArea(c) for c in contours]
			max_index = np.argmax(area) #indexing for the largest contour in the area of the array
			
			max_cont = contours[max_index] 
			max_contArea = area[max_index] 
			x, y, w, h = cv.boundingRect(max_cont) #create rect around area of cont
			rect = cv.minAreaRect(max_cont)
			rotatedPlate = plate
			if not self.ratioCheck(max_contArea, rotatedPlate.shape[1], rotatedPlate.shape[0]):
				return plate, False, None
			cv.imshow('Plate_cleaner_thresh', thresh)
			return rotatedPlate, True, [x, y, w, h]
		else:
			return plate, False, None
	def plate_checker(self, input_img, contour):
		min_rect = cv.minAreaRect(contour) #Making the min area of the rect within the contour area
		if self.validateRatio(min_rect): #if inside min_rect
			(x, y, w, h) = cv.boundingRect(contour) 
			after_validation = input_img[y:y + h, x:x + w]
			after_clean_plate_img, plateFinder, coordinates = self.plate_cleaner(after_validation)
			if plateFinder:
				chars_on_plate = self.find_chars(after_clean_plate_img)
				if (chars_on_plate is not None and len(chars_on_plate) == 0): #if chars exist period return the check
					x1, y1, w1, h1 = coordinates
					coordinates = x1 + x, y1 + y
					after_check = after_clean_plate_img
					return after_check, chars_on_plate, coordinates
		return None, None, None
	## we need to define the function for after_preprocess
	def find_existing_plate(self, input_img):
		plates = [] #empy arrays 
		self.chars_on_plate = []
		self.coressponding_area = []

		self.after_preprocess = self.preprocessing(input_img)
		possible_plate_contours = self.extracted_contours(self.after_preprocess)

		for conts in possible_plate_contours:
			plate, character_displayed, coordinates = self.plate_checker(input_img, conts)
			if plate is not None:
				plates.append(plate)
				self.chars_on_plate.append(character_displayed)
				self.regarded_area.append(coordinates)
		if(len(plates)> 0):
			return plates
		else:
			return None
	def find_chars(self, plate):

		charsFound = segment_chars(plate, 400) #segment_chars is function, it is semenantic segmentation .
		if charsFound:
			cv.imshow('find_chars', plate)
			return charsFound
	
	#Setup to check the sizing of the contours
	def ratioCheck(self, area, width, height):
		minVal = self.min_area
		maxVal = self.max_area

		ratioMin = 3
		ratioMax = 6

		ratio = float(width) / float(height)
		if ratio < 1:
			ratio = 1/ ratio

		if (area < minVal or area > maxVal) or (ratio < ratioMin or ratio > ratioMax):
			return False
		return True

	def preRatCheck(self, area, width, height):
		minVal = self.min_area
		maxVal = self.max_area

		ratioMin = 2.5
		ratioMax = 7

		ratio = float(width) / float(height)
		if ratio < 1:
			ratio = 1/ ratio
		
		if (area < minVal or area > maxVal) or (ratio < ratioMin or ratio > ratioMax):
			return False
		return True

	### Validate that the ratio is correct. 
	def validateRatio(self, rect):
		(x, y), (width, height), rect_angle = rect

		if (width > height):
			angle = rect_angle
		else:
			angle = 90 + rect_angle
		
		if angle > 15:
			return False
		if (height == 0 or width == 0):
			return False
		
		area = width * height
		if not self.preRatCheck(area, width, height):
			return False
		else:
			return True
#######################################################
#######################################################


###### DEF CLASS FOR NN ########################
###############################################


###############################################
###############################################

############### MAIN ############################
#################################################
if __name__ == "__main__":
	plateFinder = FindPlate()
	#model = NN()

	cap = cv.VideoCapture('/Users/tristanbrigham/Downloads/BostonVid.mp4')
	while(cap.isOpened()):
		ret, img = cap.read()
		img = imutils.resize(img, width=640)
		if ret == True:
			cv.imshow('origin', img)
			if cv.waitKey(25) & 0xFF == ord('q'):
				break
			possible_plates = plateFinder.find_existing_plate(img)
			
		else:
			break
	
	cap.release()
	cv.destroyAllWindows()
