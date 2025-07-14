import cv2
import numpy as np
import matplotlib.pyplot as plt
# importing os module for environment variables
import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file
load_dotenv()



#canny is an old algorithm used in edge detection in images
#canny edge detection implementation:
def canny(img):
    #check if video is still running(just to avoid any error in the program)
    if img is None:
        img.release()
        cv2.destroyAllWindows()
        exit()
        
    #convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5 #any object of size k*k will be erased

    #blur image to reduce noise
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)


    #use canny edge detection
        
    canny = cv2.Canny(blur, 50,150)

    return canny

    '''
    Canny detects all edges in the image, but we are interested in road lanes only
    Region of interest implementation: '''
def region_of_interest(canny):
	height = canny.shape[0]
	width = canny.shape[1]
	#third element gives the chnnel size, but it is out of our interest
	mask = np.zeros_like(canny)
	triangle = np.array([[ #give the three points of the triangle
	(250, 600), ####### We might need to change this according to the file we work on############
	(885, 205),    ####### We might need to change this according to the file we work on############
	(1550, 550), ]],  ####### We might need to change this according to the file we work on############
	np.int32)
	
	#make a mask for everything except the portion defined by the triangle
	cv2.fillPoly(mask, triangle, 255)
	#to apply the mask, use bitwise and
	masked_image = cv2.bitwise_and(canny, mask)
    
    
	return masked_image

'''
	Computers can't understand if the image has lines or not, the image is just some colored pixels.
	to make computer understand that this is a line we use hough lines
	a probability of where the lines are: 
	'''
def hough_lines(cropped_canny):
	lines_coordinates = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
	#we should know roughly the size of lane line in image in pixels, any other size the hough transformer will consider it not a lane
	return lines_coordinates # array with all the locations of the coordinates of the lines
	
		#still need to display them on image

#Make the lines out of the coordinates 
def make_lines(img, lines):
    height, width = img.shape[:2]
    line_img = np.zeros_like(img) #??
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Print the coordinates to debug
                print(f"Original Coordinates: {(x1, y1, x2, y2)}")
                
                # Ensure the points are integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Clamp the coordinates to be within the image bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                
                print(f"Clamped Coordinates: {(x1, y1, x2, y2)}")
                cv2.line(img, (x1,y1), (x2,y2), 
                                (0,0,255), #color
                                20) #thickness
    return img
		
    '''
    at this point, any lane line shape ( dashed or continous ) is detected
    but, we want to make a continous line that edges the lane
    
 
       '''

#take slope and intercept to create coordinat points
def make_points(img, lineSI):
    height = img.shape[0]
    try:
        slope, intercept = lineSI
    except TypeError:
        slope, intercept = 0.1,0
    y1 = int(height)
    y2 = int(y1*1.0/5) #the height of the end of the lane, can be adjusted to your image
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1,y1,x2,y2]]

#Merge all lines into one line
def avg_slope_intercept(img, lines):
	left_fit = []
	right_fit = []
	
	#call each line to know whether it belongs to left or right lan
	for line in lines:
		#each line is consisting of two points
		for x1, x2, y1, y2 in line:
			#get the slope and intercept of a line: y = mx + c
			fit = np.polyfit((x1,y1), (x2,y2), 1) #two points and degree
			slope = fit[0]
			intercept = fit[1]
			# to know if a line is left lane or right lane: by slope
			if slope <-0.1:
				left_fit.append((slope, intercept))
			elif slope > 0.1:
				right_fit.append((slope, intercept))
	left_fit_avg = np.average(left_fit, axis = 0) #avg slope and intercept
	right_fit_avg = np.average(right_fit, axis = 0)
	'''
	Avgs are no use, transform them into x, y coordinate form
	use make points function
	'''
	left_lane = make_points(img, left_fit_avg)
	right_lane = make_points(img, right_fit_avg)
      
	lanes = [left_lane, right_lane]
	return lanes


import numpy as np

def avg_slope_intercept_video(img, lines):
    left_fit = []
    right_fit = []
    
    # Ensure lines are not empty
    if lines is None:
        return None  # No lines detected, return None
    
    # Classify each line as either left lane or right lane
    for line in lines:
        # Each line consists of two points
        for x1, y1, x2, y2 in line:
            # Get the slope and intercept of the line: y = mx + c
            fit = np.polyfit((x1, x2), (y1, y2), 1)  # two points and degree=1
            slope = fit[0]
            intercept = fit[1]
            
            # Classify line based on the slope value
            if slope < -0.3:
                left_fit.append((slope, intercept))
            elif slope > 0.3:
                right_fit.append((slope, intercept))

    # Check if left_fit and right_fit are empty
    if left_fit:
        left_fit_avg = np.average(left_fit, axis=0)  # Average slope and intercept
        left_lane = make_points(img, left_fit_avg)
    else:
        left_lane = None  # No left lane detected

    if right_fit:
        right_fit_avg = np.average(right_fit, axis=0)
        right_lane = make_points(img, right_fit_avg)
    else:
        right_lane = None  # No right lane detected
    
    if left_lane is not None and right_lane is not None:
        lanes = [left_lane, right_lane]
    elif left_lane is not None:
        lanes = [left_lane]
    elif right_lane is not None:
        lanes = [right_lane]
    else:
        return None  # No lanes detected 

    return lanes

def weighted_img(img, initial_img, a = 0.8, b = 1., c = 0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOte: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a,img, b, c)


# Open the video file
image = cv2.VideoCapture(r"E:\Sama\Courses\Udacity\Self driving car\(61) Enrico Piu _ Formula Gloria _ E2SS 1000 _ Alghero - Scala Piccada 2018 - YouTube and 6 more pages - School - Microsoft​ Edge 2024-10-04 00-01-23.mp4")

# Check if video was opened successfully
if not image.isOpened():
    print("Error: Could not open video file")
else:
    ret, frame = image.read()  # Read the first frame
    if ret:
        # Convert the BGR image to RGB for displaying with matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the image using matplotlib
        plt.imshow(frame_rgb)
        plt.title("Image before")
        plt.show()
    else:
        print("Error: Could not read the frame")

# Assuming the following functions are already defined: canny, region_of_interest, hough_lines, avg_slope_intercept, make_lines

# Apply the Canny edge detection on the frame
canny_image = canny(frame)
plt.imshow(canny_image)
plt.title("canny image")
#plt.show()

# Apply region of interest mask
region = region_of_interest(canny_image)
plt.imshow(region)
plt.title("region image")
#plt.show()

# Detect lines using Hough Transform
lines = hough_lines(region)
plt.imshow(region)
plt.title("hough lines")
#plt.show()

# Average the slopes and intercepts to get a smooth lane
line_img = avg_slope_intercept(frame, lines)
plt.imshow(line_img)
plt.title("avg slope intercept")
#plt.show()

# Overlay the detected lanes on the original frame
lane_detected_image = make_lines(frame, line_img)
# Check the type to ensure it's a NumPy array (image format)
print(type(lane_detected_image))  # Should output <class 'numpy.ndarray'>

# Convert the BGR image to RGB for displaying with matplotlib (since OpenCV uses BGR format by default)
lane_detected_image_rgb = cv2.cvtColor(lane_detected_image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(lane_detected_image_rgb)
plt.title("Lane Detected Image")
plt.axis("off")  # Hide axis
#plt.show()

# Open the video file
#image = cv2.VideoCapture(r"E:\Sama\Courses\Udacity\Self driving car\(61) G.Giametta - 52° Salita Monti Iblei 2021- C.I.V.M. - Formula Gloria - YouTube and 6 more pages - School - Microsoft​ Edge 2024-10-03 22-40-59.mp4")

# Initialize the previous_lane variable
previous_lane = None

# Loop through each frame of the video
while image.isOpened():
    ret, frame = image.read()  # Read a frame
    
    # Break the loop if no frame is returned
    if not ret:
        print("Video processing completed.")
        break
    
    # Apply Canny edge detection on the frame
    canny_image = canny(frame)
    
    # Apply region of interest mask
    cropped_canny = region_of_interest(canny_image)

    # Detect lines using Hough Transform
    lines = hough_lines(cropped_canny)
    
    # Average the slopes and intercepts to get a smooth lane
    avg_line_img = avg_slope_intercept_video(frame, lines)
    
    # Overlay the detected lanes on the original frame
    lane_detected_image = make_lines(frame, avg_line_img)
    
    # Display the frame with detected lanes using OpenCV
    # cv2.imshow("cropped canny", cropped_canny)

    # cv2.imshow("Lane Detected Video", lane_detected_image)
    frame1 = cv2.cvtColor(cropped_canny, cv2.COLOR_GRAY2BGR)
    frame1 = cv2.resize(frame1, (950, 540))
    frame2 = cv2.resize(lane_detected_image, (950, 540))

    combined_frame = np.hstack((frame1, frame2))
    
    # Display the combined frame
    cv2.imshow('Side by Side Videos', combined_frame)
    
    # Wait for a short period and allow breaking the loop with 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
image.release()
cv2.destroyAllWindows()
