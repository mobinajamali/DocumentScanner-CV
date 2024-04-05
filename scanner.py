import numpy as np
import cv2 as cv
from utils import stackImages

# 1. CAPTURE WEBCAM
cap = cv.VideoCapture(0)

# 1. SET CAPTURE PARAMETERS
FRAME_WIDTH = 480  
FRAME_HEIGHT = 640
cap.set(10, 0)


# Define the codec and create VideoWriter object
out = cv.VideoWriter('output2.avi',
                     cv.VideoWriter_fourcc(*'XVID'), 
                     20.0, 
                     (1152, 768)) # final stacked image size


# 2. PREPROCESS FUNCTION
def preProcessing(img):

    # GRAY SCALE
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # BLUR
    blur = cv.GaussianBlur(gray, (5,5), 1)
    # CANNY EDGE DETECTION
    canny = cv.Canny(blur, 200, 200)
    # USE DILATION TO MAKE THE EDGES THICKER AND THEN EROSION TO MAKE THEM THINNER AGAIN
    kernel = np.ones((5,5))
    dial = cv.dilate(canny, kernel, iterations=2)
    # FINAL IMG
    imgThresh = cv.erode(dial, kernel, iterations=1)

    return imgThresh, gray
    

# 3. CONTOUR DETECTION (THE BIGGEST ONE AVAILABLE)
def getContours(img):

    biggest = np.array([])
    maxArea = 0
    contours, hierachy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour_display = cv.drawContours(result, contours, -1, (0, 255, 0), 10)
    #print("Number of contours:", len(contours))
    for cnt in contours:
        # FIND THE AREA OF THE CONTOUR
        area = cv.contourArea(cnt)
        # GIVE MIN THRESHOLD TO PREVENT CAPTURING NOISES
        if area > 5000:
            # CALCULATE CURVE LENGTH (TO APPROXIMATE THE CORNERS OF THE EDGES)
            peri = cv.arcLength(cnt, True)
            # APPROX NUM CORNER POINTS FOR EACH SHAPE
            approx = cv.approxPolyDP(cnt, 0.02*peri, True)
            # GET THE BIGGEST CONTOUR
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    
    # DRAW IT OUT
    cv.drawContours(imgContour, biggest, -1, (0, 255, 0), 20)
    return biggest, contour_display


# NOTE: the ordering of the points has to be like [[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]
# but this changes by the camera angle and we need to make sure to reorder them to this structure
# 5. REORDERING THE POINT STRUCTURE (origin > lowest some, diagonal contour point > highest sum)
def reorder(myPoints):

    myPoints = myPoints.reshape((4,2)) # initially biggest has the shape (4,2,1) and 1 is redundant (4 points (x,y for each))
    myPointsNew = np.zeros((4,1,2), np.int32) # same shape as original one
    add = myPoints.sum(1) # add axis 1 for each two points
    myPointsNew[0] = myPoints[np.argmin(add)]  # smallest add at the first point
    myPointsNew[3] = myPoints[np.argmax(add)]  # largest one at the last point

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print("NewPoints",myPointsNew)

    return myPointsNew



# NOTE: once we have the biggest contour we need use its corner 
# points to warp the img and get the birds eye view

# 4. GET THE BIRD EYE VIEW (need two points, create metrics and get the prespective)
def getWarp(img, biggest):

    biggest = reorder(biggest)
    # [[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [FRAME_WIDTH, 0], [0, FRAME_HEIGHT], [FRAME_WIDTH, FRAME_HEIGHT]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv.warpPerspective(img, matrix, (FRAME_WIDTH, FRAME_HEIGHT))
    
    # 6. after warping the image reduce some pixels (remove 20 pixels from each side)
    imgCropped = imgOutput[10:imgOutput.shape[0]-10, 10:imgOutput.shape[1]-10]
    # RESIZE THE IMG TO THE PREVIOUS SIZE
    imgCropped = cv.resize(imgCropped, (FRAME_WIDTH, FRAME_HEIGHT))

    return imgCropped


# 1. DISPLAY THE FRAME
frames = []
img_count = 0
while True:
    success, frame = cap.read()

    # RESIZE THE IMG
    frame = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    # 3.
    result = frame.copy()
    imgContour = frame.copy()
    blank = np.zeros(frame.shape[:2], dtype='uint8')
    # 2, 3, 4.
    imgThresh, gray = preProcessing(frame)
    biggest, result = getContours(imgThresh)

    # 7. TO STACK UP THE RESULTS
    if biggest.size != 0:
        warp = getWarp(frame,biggest)
        # create adaptive threshold of the result img
        warp_gray = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)
        adaptive_thresh = cv.adaptiveThreshold(warp_gray, 
                                           255, 
                                           cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv.THRESH_BINARY, 
                                           11, 
                                           2)
        #warp2 = getWarp(frame,adaptive_thresh)
        imageArray = ([frame, gray, imgThresh, result],
                    [imgContour, warp, warp_gray, adaptive_thresh])
        #cv.imshow('Image Warped', warp)
    else:
        imageArray = ([frame, gray, imgThresh, result],
                    [imgContour, blank, blank, blank])
    
    names = [["Original", "Gray", "Threshold", "Contours"], ["Biggest Contour", "Warp", "Warp Gray", "Adaptive Thresh"]]
    stackedImages = stackImages(0.6, imageArray, names)

    # Write the frame to the output video
    out.write(stackedImages)
    cv.imshow('Result', stackedImages)
    print("Dimensions of stackedImages:", stackedImages.shape)


    if cv.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()