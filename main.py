import numpy as np
import utils
import cv2 as cv

web_cam_feed = True
path_image = "path/to/image.jpg" 
cap = None
if web_cam_feed:
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_BRIGHTNESS, 160)  
else:
    img = cv.imread(path_image)

height_image = 640
width_image = 480

utils.initialize_track_bar() 

while True:
    if web_cam_feed:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break
    else:
        if img is None:
            print(f"Failed to load image from {path_image}")
            break


    img = cv.resize(img, (width_image, height_image))
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 1)
    
    thresh = utils.values_track_bar()
    img_thresh_hold = cv.Canny(img_blur, thresh[0], thresh[1])
    
    kernel = np.ones((5, 5))
    
    img_dial = cv.dilate(img_thresh_hold, kernel, iterations = 2)
    img_thresh_hold = cv.erode(img_dial, kernel, iterations = 1)
    
    img_contours = img.copy()
    img_points_contours = img.copy()
    
    contours, hierarchy = cv.findContours(img_thresh_hold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_contours, contours, -1, (0, 255, 0), 10)
    cv.imshow("Result-1", img_contours)
    
    biggest, biggest_area = utils.biggest_contour(contours)
    
    if biggest.size != 0:
        biggest = utils.reorder(biggest)
        cv.drawContours(img_points_contours, biggest, -1, (0, 255, 0), 20)
    
    cv.imshow("Result-2", img_points_contours)


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

if cap:
    cap.release()
cv.destroyAllWindows()