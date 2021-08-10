print("\n\nLOADING PROGRAM\n\n")

import cv2 as cv
print("Loaded CV")
import numpy as np
print("Loaded NP")
import tensorflow as tf
print("Loaded TF")
import imutils
print("Loaded IMUTILS")
# from tensorflow import keras
# print("Loaded KERAS")


            ### GLOBAL VARIABLES FOR THE PROGRAM ###
            
collect_data = False        #if true, asks the user for data on what letter is detected. input nothing if image is not a letter or contains more than one letter
get_chars = False           #if true, applies the algorithm model to the characters that are detected to get what the plate says
optimize = True             #checks to see whether the user only wants the program to analyze the bottom portion of the vid/image
start_frame_number = 0      #where does the user want the video to start?
frames_skipped = 20         #how many frames pass before the frame is analyzed (for instance, analyze every 20th frame if this value is 20)

video_path = "/Users/tristanbrigham/Downloads/BostonVid.mp4"
folder_path = "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/"
training_data_path = "/Users/tristanbrigham/GithubProjects/AI_Training_Data/LicensePlateProject/"

letter_dict = {}
model = tf.keras.models.load_model(folder_path + "model.h5")

    ########################################################################################
    #################################### GENERAL SETUP #####################################
    ########################################################################################

def skip_forward():
    frame_count = cap.get(cv.CAP_PROP_POS_FRAMES)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_count + 2000)

def setup_dictionary():
    alphabet = open(folder_path + "alphabet.txt")
    for count, line in enumerate(alphabet.readlines()):
        letter_dict[count] = line[0]
    print(letter_dict)


### Class for plate detection

    ########################################################################################
    #################################### SETUP PROGRAM! ####################################
    ########################################################################################

class FindPlate:

    #should maybe make the parameters global variables or controlled by the command line
    # Have to adjust so that the min and max are larger when analyzing the images and smaller when looking at the vids
    def __init__(self, counter, check_wait = False, imgAddress = None, img = None):

        self.check_wait = check_wait    #initializing whether we need to wait between drawing contours for debugging

        if imgAddress is None and img is not None:  #getting the image from the video
            self.img = img
        elif imgAddress is not None and img is None:
            self.img = cv.resize(cv.imread(imgAddress), (860, 480))
        else:
            print("-----------------------ERROR FINDING IMAGE-----------------------")
            exit(0)

        # imutils.resize(self.img, height = 300, inter=cv.INTER_CUBIC)

        if(counter == 0):
            self.setup_exec() #execute the program
        else:
            self.show_images()                          #Show the images
        self.check_keys()
    

    def setup_exec(self):

        self.settings_init()

        if optimize:        #Currently, optimize makes it so that only the bottom portion of the screen is analyzed
            self.offset = int(self.img.shape[0] * (self.divideArea - 1) / self.divideArea)  #How many pixels in the y direction are not analyzed from the top
            self.top_img = self.img[ : self.offset ]    #Taking the top potion of the image and saving it for later
            self.img = self.img[self.offset : ]         #reassigning the image to the portion being analyed
        
        self.x = self.img.shape[1]                      #getting the width of the image

        self.img_copy = self.img.copy()                 #getting copies for analysis/blurring
        self.img_rects = self.img.copy()                #the copy that will be used for bounding rectangles
        self.Canny = None                               #Initializing variable to hold Canny image
        self.run()

        if optimize:                                    #After being run, rejoin the images if optimize is on
            self.img = np.append(self.top_img, self.img, axis=0)
            self.img_rects = np.append(self.top_img, self.img_rects, axis=0)
        
        self.show_images_exec()



    def settings_init(self):
        self.divideArea = 2.5   #This is the denominator for how much of the screen is analyzed (analyzes the [1/(variable)] portion of the image/vid)
                                #For example, if the bottom half should be analyzed, put in '2' 

        self.amt_digits = 6     #defining the amount of characters and digits that should be found on the license plate 

        self.ratio_max = 3      #This is the maximum width to height ratio that a license plate can be in the program (for the US, about 4 is good while for EU plates, about 6-7 is good)
        self.ratio_min = 1.5      #This is the minimum width to height ratio 

        self.angle_min = 84     #After the angle of the cv.areaMinRect has a modulo 90 applied to it, the angle either needs to be close to upright (this value or above)
        self.angle_max = 6      # or close to horizontal (this value or below) in degrees

        self.img_size = self.img.shape[0] * self.img.shape[1]

        #current size: about 240,000 pixels
        self.area_min = int(self.img_size / 1380)      #minimum area of the accepted bounding boxes -- it recognizes plates with smaller values but there is no way that characters can be picked out. No use to have smaller
        self.area_max = int(self.img_size / 600)     #max area of the accepted bounding boxes

        self.lower_canny = 110    #upper value for canny thresholding
        self.upper_canny = 120   #Lower value for canny thresholding

        #ASPECT variables are not used:
        self.aspect_max = 1      #the max amount of area that the license plate can cover within a bounding box to be considered
        self.aspect_min = 0.3    #the minimum amount of area that a license plate can cover within a bounding box to be considered

        self.img_dilate = 40    #specifying the value that the pixels which are being brightened will be increased by
        self.blur = (5, 5)    #initializing the size component of the gaussian blur that is applied to the image
        self.offset = 0         #initializing the variable which keeps track of how far from the top of the image the program begins to analyze
        self.top_img = None     #initializing the variable which may hold the top part of the image for later
        self.element_structure = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(5, 5)) #basic elem structure for blurring
        self.letter_contour_min = 2000 #The minimum size a contour has to be for it to be considered for letter analysis

        self.roi_array = []      #array for holding the ROI's


    def run(self):                                      #master run function for the program
        _ = self.contour_manipulation(self.preprocess_canny_contours())



    ########################################################################################
    ################################### ANALYZING IMAGES ###################################
    ########################################################################################



    def preprocess_canny_contours(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY) #get grayscale image
        gray = cv.GaussianBlur(gray, self.blur, 0)         #Apply a blur
        edged = cv.Canny(gray, self.lower_canny, self.upper_canny)  #Getting the canny contours
        self.Canny = edged                              #assign the variable
        contours = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)    #Get the contours of the Canny image [remember that this will return more contours than we need
                                                                                            #Because Canny just returns lines]
        contours = imutils.grab_contours(contours)      #Get the contours using imutils
        
        # cv.drawContours(self.img_copy, contours, -1, (0, 0, 255), thickness=2)   #Draw the contours onto the copied image
        return contours


    def analyze_image(self):        # getting an array of the potential letters from each license plate ROI
        letterArrays = []

        #SHOWING THE ROI's
        for count, regionOfInterest in enumerate(self.roi_array):
            data = self.process_ROI(regionOfInterest, count)
            if data is not None:
                letterArrays.append(data)
        
        return letterArrays


    def process_ROI(self, roi, counter):
        regionOfInterest = roi[int(roi.shape[0] / 4) : roi.shape[0] - int(roi.shape[0] / 5), int(roi.shape[1] / 18) : roi.shape[1] - int(roi.shape[1] / 18)] #
        name = "ROI {}".format(counter)                           #format the name
        regionOfInterest = cv.cvtColor(regionOfInterest, cv.COLOR_BGR2GRAY)

        regionOfInterest[: int(regionOfInterest.shape[0] / 6), :] += self.img_dilate    #Increasing the brightness of the top of the image (BREAKS WITH HIGH VALUES because of overflow)

        regionOfInterest = imutils.resize(regionOfInterest, height=200, inter=cv.INTER_AREA)
        image = cv.GaussianBlur(regionOfInterest, (0, 0), 3)
        image = cv.addWeighted(image, 1.5, regionOfInterest, -0.5, 0)

        ret, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        thresh = cv.bitwise_not(thresh)
        thresh = cv.erode(thresh, (81, 61), iterations = 15)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = self.sort_contours_left(contours)
        cv.imshow("GRAY {}".format(counter), imutils.resize(thresh, height=200))

        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        letters = []
        for contour in contours:
            if cv.contourArea(contour) > self.letter_contour_min:
                x, y, w, h = cv.boundingRect(contour)
                letterInterest = thresh[0 : y + h, x : x + w]
                # cv.imshow("Letter {}".format(count), imutils.resize(letterInterest, height = 100))
                # cv.moveWindow("Letter {}".format(count), 600, 110 * count)
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))
                letterImage = cv.resize(letterInterest, (60, 80))
                letters.append(letterImage)

        cv.imshow(name, image)   #showing and resizing image
        cv.moveWindow(name, 0, 110 * counter - 50)                #Moving the ROI windows into the right spot on the screen
        
        if len(letters) > 4:
            if collect_data:
                NeuralNetwork.label_letter(letters)
            return letters
        else: return None



    ########################################################################################
    #################################### CONTOUR SORTING ###################################
    ########################################################################################


    def sort_contours_left(self, contours):                   #This sorts the contours based on how far the contours are from the middle of the screen (only looks at the x-pos)
        retContourMapping = []
        for i, contour in enumerate(contours):                     #for every contour, first get the middle part of the bouding box x-pos wise
            x, _, _, _ = cv.boundingRect(contour)                           #then we take the abs. value of that and sort those values in increasing fashion
            retContourMapping.append((contour, x, i))

        retContourMapping.sort(key=lambda tup: tup[1])  # sorts in place by distance from vertical horizontal line

        contours = []
        for contour, _, _ in retContourMapping:
            contours.append(contour)
        return contours



    def contour_manipulation(self, contours):
        checkIndividual = False                         #contours which are in the middle of the screen x-wise. It can also be changed to look at contours close to 
        ret = []                                        #a certain y-val
        self.roi_array = []

        for c in contours:
            boolRect = self.check_min_rect(c)

            if boolRect:
                ret.append(c)
            
            if checkIndividual:                         #if the check individual option is on, then go through the contours one-by-one, write them to the image, and show the image
                print("\n\nCONTOUR: {}".format(cv.contourArea(c)))
                cv.imshow("Bounding Rects", self.img_rects)
                if cv.waitKey(0) & 0xFF == ord('c'):    #This cycles through to the next contour
                    continue
                elif cv.waitKey(0) & 0xFF == ord('f'):  #This makes it so that the rest of the contours are drawn in an instant
                    checkIndividual = False
                elif cv.waitKey(0) & 0xFF == ord('q'):  #quits the program
                    exit()
        return ret 



    ############# NOT BEING USED #############
    def sort_contours_middle(self, contours):           #This sorts the contours based on how far the contours are from the middle of the screen (only looks at the x-pos)
        rects = []
        for c in contours:
            rects.append((cv.boundingRect(c), c))       #Creating a tuple array with bouding rects and the contours
        retContourMapping = []

        for i in range(len(rects)):                     #for every contour, first get the middle part of the bouding box x-pos wise
            rect, contour = rects[i]                    #Then we are going to subtract that value from the middle of the screen 
            x, _, w, _ = rect                           #then we take the abs. value of that and sort those values in increasing fashion
            x = int(self.x / 2) - x                     #If necessary, this would allow us to put a cap on processing and only look at contours in the middle of the screen
            x = abs(x + int(w/2))
            retContourMapping.append((i, x, rects[i], contour))

        retContourMapping.sort(key=lambda tup: tup[1])  # sorts in place by distance from vertical horizontal line

        keys = []
        for index, _, _, _ in retContourMapping:
            keys.append(index)
        return keys



    ########################################################################################
    ################################### CHECKING CONTOURS ##################################
    ########################################################################################



    def check_min_rect(self, contour):                    #function for getting the min-area rectangle and validating whether it is ok
        rect = cv.minAreaRect(contour)                  #get the min area rect
        box = cv.boxPoints(rect)                       
        box = np.int0(box)                              #for drawing the min area rectangles
        rx, ry, rw, rh = cv.boundingRect(contour)
        if self.validateRatio(rect, rw, rh):
            cv.drawContours(self.img_rects,[box], 0, (0, 255, 0), 4)
            brect = self.img[ry : ry + rh, rx : rx + rw]
            self.roi_array.append(brect)
            return True                                 #if everything is right, then return the contour and true to show that it is valid
        else:
            # cv.drawContours(self.img_rects,[box], 0, (0, 0, 255), 1)
            return False                          #else, return the contour and false



    def rat_check(self, width, height):         
        ratio = float(width) / float(height)            #check whether the width to height ratio is wrong
        if ratio < 1:
            ratio = 1 / ratio
        
        return not (ratio < self.ratio_min or ratio > self.ratio_max) #if the area is not in range or the ratio is off, return false



    def validateRatio(self, rect, rw, rh):                   #more checking that the contour could be a license-plate
        (x, y), (width, height), angle = rect           #get all of the data about the minarea bounding rectangle
        if width == 0 or height == 0:
            return False

        angle = angle % 90
        area = width * height                           
        
        if not ((angle < self.angle_max or angle > self.angle_min) and (area > self.area_min and area < self.area_max)):
            return False

        if rw < rh:
            return False

        return self.rat_check(width, height)




    ########################################################################################
    #################################### SHOWING IMAGES ####################################
    ########################################################################################



    def show_images(self, height = 300):      #showing the images and putting them in the right place on the screen
        cv.imshow("Original", imutils.resize(self.img, height = 200))
        # self.check_keys()                                           #kind of inefficient to be checking the keys every time, but otherwise the program is unresponsive


    def show_images_exec(self, height = 300):
        # cv.imshow("Contours", imutils.resize(self.img_copy, height = height))
        cv.imshow("Bounding Rects", self.img_rects)
        cv.imshow("Canny", imutils.resize(self.Canny, height = height))

        data = self.analyze_image()
        if len(data) > 0 and get_chars:
            print("")
            neuralNet = NeuralNetwork()
            for image_arr in data:
                print(neuralNet.get_chars_array(image_arr))
        self.show_images()
        self.check_keys()
        



    def check_keys(self):
        if self.check_wait:                                   #if going through the contours, check if q is pressed
            key = cv.waitKey(0) & 0xFF
            print("NEXT IMAGE")
            if key == ord('q'):
                exit(0)
        else:
            key = cv.waitKey(1)
            if key & 0xFF == ord('q'):                  #exit button
                exit(0)
            elif key == ord('s'):
                skip_forward()
            elif key & 0xFF == ord('p'):                # this creates a pause button for the video, in essence
                print("VIDEO PAUSED")
                while True:
                    key = cv.waitKey(25) & 0xFF
                    if key == ord('p'):                 #unpause
                        break
                    elif key == ord('q'):               #quit the program button
                        exit(0)
                    elif key == ord('s'):
                        skip_forward()




    ########################################################################################
    #################################### NEURAL NETWORK ####################################
    ########################################################################################

imageNumber = 0
training_file_keys = training_data_path + "training_data.txt"

class NeuralNetwork:
    
    def __init__(self):
        self.model = model
        self.plate_ret = ""

    ################ TRAINING THE MODEL ################ 

    def label_letter(imagearr):
        for image in imagearr:
            print("FRAME COUNT: {}".format(cap.get(cv.CAP_PROP_POS_FRAMES)))
            
            global imageNumber 
            global training_file_keys
            
            cv.imshow("POSSIBLE LETTER", image)
            cv.waitKey(1)
            
            imageNumber = imageNumber + 1
            letter = input("Please input the letter: ").upper()
            hexval = ":".join("{:02x}".format(ord(c)) for c in letter)
            
            if len(letter) < 1 or hexval == "0c":
                letter = '_'
            else:
                letter = letter[0]
            
            file = open(training_data_path + str(imageNumber) + ".txt", "w")
            
            for row in image:
                np.savetxt(file, row)
           
            print("Letter passed: " + letter)
            training_file = open(training_file_keys, "a")
            training_file.write("\n" + str(letter))


    ################ PREDICTING WITH THE MODEL ################

    def get_chars_array(self, array):
        ret = ""
        for image in array:
            ret += self.predict_char(image)
        return ret

    def predict_char(self, image):
        image_formatted = self.setup_array(image)
        pred = model.predict(image_formatted)
        return letter_dict[int(np.argmax(pred))]

    def setup_array(self, image):
        number_array = np.zeros((1, 80, 60, 1), dtype="float32")
        number_array[0] = image.reshape(80, 60, 1)
        return number_array


    ################ MODEL FUNCTIONS ################

    def network_summary(self):
        return self.model.summary()


    ########################################################################################
    ############################### VID PROCESSING AND SETUP ###############################
    ########################################################################################


if __name__ == "__main__":

    setup_dictionary()

    #addresses for testing still images on my machine:
    imageAddresses = [
        "licensePlate.jpeg",
        "licensePlate2.jpeg",
        "licensePlate3.jpeg",
        "licensePlate4.jpeg",
        "licensePlate5.jpeg"
    ]

    imageAddresses2 = [
        "license_plate_letter_0",
        "license_plate_letter_1",
        "license_plate_letter_2",
        "license_plate_letter_3"
    ]

    print("\n\nWelcome\nPlease press q to quit the program\nPlease press p to pause and unpause during the video\nPlease press anything else to continue through the images")
    print("\nOnce you have looked at all of the still images, the video will begin\n\n")
    print("Green boxes signify possible license plate regions \nwhile red ones show other ROI's which were picked up and discarded")
    
    cap = cv.VideoCapture(video_path)
    print("Starting Video @ frame " + str(start_frame_number))
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame_number) #setting the starting frame number to the correct number

    if collect_data:
        file_keys = open(training_file_keys, "r")
        imageNumber = int(file_keys.readline().rstrip())
        print("INDEX: " + str(imageNumber))

    counter = 0

    while(cap.isOpened()):              #reading and analyzing the video as it runs
        counter = counter + 1
        counter = counter % frames_skipped
        ret, img = cap.read()
        if ret == True:
            FindPlate(counter=counter, img = img)
        else:
            break
    
    cap.release()
    cv.destroyAllWindows()

