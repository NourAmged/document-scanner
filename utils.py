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

def draw_rectangle(img, biggest, thickness):
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img


import cv2 as cv
import numpy as np

def stacked_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    
    # Ensure all images have 3 channels (convert grayscale to BGR)
    for i in range(rows):
        for j in range(cols):
            img = img_array[i][j]
            if img is None:
                continue
            if len(img.shape) == 2:  # grayscale
                img_array[i][j] = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # Determine reference size based on first image
    width = int(img_array[0][0].shape[1] * scale)
    height = int(img_array[0][0].shape[0] * scale)

    # Create blank image placeholder
    image_blank = np.zeros((height, width, 3), dtype=np.uint8)

    # Resize all images to the same dimensions and fill None with blank
    for i in range(rows):
        for j in range(cols):
            if img_array[i][j] is None:
                img_array[i][j] = image_blank
            else:
                img_array[i][j] = cv.resize(img_array[i][j], (width, height), interpolation=cv.INTER_AREA)

    # Stack horizontally row by row
    hor_images = [np.hstack(img_array[i]) for i in range(rows)]
    
    # Stack all rows vertically
    stacked_image = np.vstack(hor_images)

    return stacked_image



