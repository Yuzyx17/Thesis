import cv2
from utilities.const import *
from utilities.util import *
from pre.norm import *

def segment_leaf(image):
    border_size = 10  # Adjust the border size as needed
    image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
    test = useWhiteBalance(image)
    test = cv2.medianBlur(test, ksize=3)

    hsv = cv2.cvtColor(test, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)
    _, g, _ = cv2.split(test)

    v = cv2.equalizeHist(v)
    g = cv2.equalizeHist(g)
    
    v = cv2.convertScaleAbs(v, alpha=1.25)
    g = cv2.convertScaleAbs(g, alpha=1.25)

    _, v = cv2.threshold(v, 195, 255, cv2.THRESH_BINARY)
    _, g = cv2.threshold(g, 195, 255, cv2.THRESH_BINARY)

    # Define the shape of the kernel (e.g., a square)
    kernel_shape = cv2.MORPH_RECT  # You can use cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, or cv2.MORPH_CROSS

    # Define the size of the kernel (width and height)
    kernel_size = (5, 5)  # Adjust the size as needed

    # Create the kernel
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    
    leaf = cv2.bitwise_xor(v, s)
    leaf = cv2.bitwise_xor(leaf, g)
    leaf = cv2.dilate(leaf, kernel, iterations=2)
    leaf = cv2.bitwise_or(s, v)
    mask = leaf

    # dise = cv2.bitwise_or(a, B)
    # dise = cv2.bitwise_or(dise, h)
    # mask = cv2.bitwise_or(dise, leaf)
    # # mask = cv2.bitwise_and(mask, z)
    # mask = cv2.bitwise_or(a, s)
    # mask = cv2.bitwise_and(mask, v)
    # mask = cv2.convertScaleAbs(mask, alpha=1.25)

    # THresholding
    _, mask = cv2.threshold(mask, 196, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(image, image, mask=mask)

    # With Canny
    test = cv2.Canny(mask, 100, 400)
    test = cv2.dilate(test, kernel=np.ones((3, 3), np.uint8), iterations=1)
    # test = cv2.erode(test, kernel=np.ones((3, 3), np.uint8), iterations=1)
    tmask = np.zeros_like(test)
    contours, _ = cv2.findContours(test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(tmask, [largest_contour], 0, 255, -1)
    tmask = cv2.bitwise_and(image, image, mask=tmask)
    tmask = cv2.resize(tmask, (FEAT_W, FEAT_H))
    mask = cv2.resize(mask, (FEAT_W, FEAT_H))
    return mask

def combine_leaf_and_disease(leaf, disease):
    # Create a mask for the disease region
    disease_mask = np.zeros_like(leaf)
    disease_mask[disease > 0] = (255, 255, 255)

    # Combine the leaf and disease regions using bitwise OR operation
    combined = cv2.bitwise_or(leaf, disease_mask)

    return combined

def process_image(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([70, 255, 255])

    # Create a mask for green regions
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    # Convert the result to grayscale
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    # Create an empty mask to store the filtered contours
    mask = np.zeros_like(thresh)

    # Draw the filtered contours on the mask
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Bitwise-AND the mask with the original image
    final_image = cv2.bitwise_and(image, image, mask=mask)

    return final_image
