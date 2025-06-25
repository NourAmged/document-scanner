from datetime import datetime
import numpy as np
import utils
import cv2 as cv



web_cam_feed = False
path_image = "Untitled.jpg" 
cap = None



if web_cam_feed:
    cap = cv.VideoCapture("http://192.168.1.2:8080/video")
else:
    img = cv.imread(path_image)

height_image = 720
width_image = 1080

utils.initialize_track_bar() 

while True:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
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
    img_warped_colored = img.copy()
    img_adaptive_thresh = img.copy()
    
    contours, hierarchy = cv.findContours(img_thresh_hold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_contours, contours, -1, (0, 255, 0), 10)
    
    biggest, biggest_area = utils.biggest_contour(contours)
    
    if biggest.size != 0:
        
        biggest = utils.reorder(biggest)
        cv.drawContours(img_points_contours, biggest, -1, (0, 255, 0), 20)
        img_points_contours = utils.draw_rectangle(img_points_contours, biggest, 2)
        
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [width_image, 0], [0, height_image], [width_image, height_image]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        
        img_warped_colored = cv.warpPerspective(img, matrix, (width_image, height_image))
        
        img_warped_colored = img_warped_colored[10:img_warped_colored.shape[0] - 10, 10:img_warped_colored.shape[1] - 10]
        img_warped_colored = cv.resize(img_warped_colored, (width_image, height_image))
        
        img_warped_gray = cv.cvtColor(img_warped_colored, cv.COLOR_BGR2GRAY)
        img_adaptive_thresh = cv.adaptiveThreshold(img_warped_gray, 255, 1, 1, 7, 2)
        img_adaptive_thresh = cv.bitwise_not(img_adaptive_thresh)
        img_adaptive_thresh = cv.medianBlur(img_adaptive_thresh, 3)
        
       
    image_array = ([img_contours, img_points_contours],
                      [img_warped_colored, img_adaptive_thresh])
    
    labels = [["Contours", "Biggest Contour"],
              "Warp Perspective", "Adaptive Threshold"] 
            
    stacked_image = utils.stacked_images(0.5, image_array)

    cv.imshow("Stacked Images", stacked_image)

    key = cv.waitKey(1) & 0xFF
    
    if key == ord('s'):
        cv.imwrite("Img" + timestamp + ".jpg", img_warped_colored)
        cv.imwrite("ImgThresh" + timestamp + ".jpg", img_adaptive_thresh)
    if key == ord('q'):
        break

if cap:
    cap.release()
cv.destroyAllWindows()