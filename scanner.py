import numpy as np
import cv2 as cv

# 1. CAPTURE WEBCAM
cap = cv.VideoCapture(0)

# 1. SET CAPTURE PARAMETERS
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
cap.set(10, 150)

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

    return imgThresh
    

# 3. CONTOUR DETECTION (THE BIGGEST ONE AVAILABLE)
def getContours(img):
    biggest = []
    maxArea = 0
    contours, hierachy = cv.findContours(img, cv.RETR_EXTERNAL, cv. CHAIN_APPROX_NONE)
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
    cv.drawContours(imgContour, biggest, -1, (255, 0, 0), 15)
    return biggest


# NOTE: once we have the biggest contour we need use its corner 
# points to warp the img and get the birds eye view
# 4. GET THE BIRD EYE VIEW
def getWarp(img, biggest):
    pass


# 1. DISPLAY THE FRAME
while True:
    success, frame = cap.read()
    # RESIZE THE IMG
    frame = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    # 3.
    imgContour = frame.copy()
    # 2, 3, 4.
    imgThresh = preProcessing(frame)
    biggest = getContours(imgThresh)
    print(biggest)
    getWarp(frame, biggest)

    cv.imshow('Result', imgContour)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break