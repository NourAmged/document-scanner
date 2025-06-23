import cv2 as cv
import numpy as np

def nothing(x):
    pass

def initialize_track_bar():
    cv.namedWindow("Track_bar")
    cv.resizeWindow("Track_bar", 360, 240)
    cv.createTrackbar("Threshold1", "Track_bar", 200, 255, nothing)
    cv.createTrackbar("Threshold2", "Track_bar", 200, 255, nothing)
    
def values_track_bar():
    threshold1 = cv.getTrackbarPos("Threshold1", "Track_bar")
    threshold2 = cv.getTrackbarPos("Threshold2", "Track_bar")
    
    return threshold1, threshold2

def biggest_contour(contours):
    biggest = np.array([])
    
    biggest_area = 0
    
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 5000:
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * peri, True)
            if area > biggest_area and len(approx) == 4:
                biggest = approx
                biggest_area = area
                
    return biggest, biggest_area


def reorder(points):
    points = points.reshape((4,2))
    
    new_points = np.zeros((4, 1, 2), dtype = np.int32)

    add = points.sum(axis = 1)
    
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    
    diff = np.diff(points, axis = 1)
    
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    
    return new_points
    