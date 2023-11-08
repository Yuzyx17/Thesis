import cv2
from utilities.const import *
from utilities.util import *
from pre.norm import *
from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage import morphology


def useGlobalThreshold(img):
    return cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

def useAdapativeThreshold(img):
    return cv2.adaptiveThreshold(rgbAsGray(img), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)

def useCustomThreshold(img):
     # Define the lower and upper green color thresholds add to constant
    lower_green = np.array([0, 0, 8])
    upper_green = np.array([255, 255, 24])

    # Create a mask for the green color range
    mask = cv2.inRange(useSaturation(img), lower_green, upper_green)
    mask = cv2.bitwise_not(mask)

    result = useMask(img, mask)

    gray = rgbAsGray(img)
    _, mask = useGlobalThreshold(gray)

    return useMask(result, mask)

def useKClusterSegment(img):
    # Convert the image data type to float32 for KMeans
    pixels = np.float32(useShape(img))

    # Define the number of clusters (K)
    k = 5

    # Apply K-means clustering
    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]

    # Reshape the segmented image to the original shape
    return segmented_image.reshape(img.shape)

def useCombined(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Histogram Equalization for image enhancement
    gray = cv2.equalizeHist(gray)

    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Define a kernel for the morphological operations
    kernel = np.ones((3,3), np.uint8)

    # Perform opening to remove noise and separate objects
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

    # Perform closing to fill in holes
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations = 2)
    closing = cv2.bitwise_not(closing)
    closing = cv2.bitwise_and(image, image, mask=closing)

    return closing

def useContiguous(image):
    # Blur the image
    gray = rgbAsGray(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Thresholding
    _, binary = cv2.threshold(blurred, 32, 224, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Create an empty image to store the result
    result = np.zeros_like(gray)

    # Filter blocks
    for i in range(1, num_labels):
        # You can replace this condition with your own criteria
        if stats[i, cv2.CC_STAT_AREA] > 0:
            result[labels == i] = 255
    
    result = cv2.bitwise_and(image, image, mask=result)

    return result

def useColorful(image):
    # 1. Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)

    # 2. Calculate the variance of the Laplacian (measure of blurriness)
    laplacian = cv2.Laplacian(sharpened, cv2.CV_64F).var()

    # 3. Remove blurry parts
    blur_threshold = 100
    mask_blur = np.where(laplacian > blur_threshold, 1, 0).astype('uint8')

    # 3a. Thresholding to maintain green parts
    # Read the grayscale image
    gray_image = rgbAsGray(image)

    # Threshold the lightness value
    ret, lightness_binary_mask = cv2.threshold(gray_image, 16, 248, cv2.THRESH_BINARY)
    # Convert the image to the BGR color space
    bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract the green channel
    green_channel = bgr_image[:,:,1]

    # Apply thresholding to the green channel
    green_threshold_value = 32
    ret, green_binary_mask = cv2.threshold(green_channel, green_threshold_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine the binary masks using logical OR
    final_binary_mask = cv2.bitwise_or(lightness_binary_mask, green_binary_mask)

    # 3b. Edge detection (Canny) to maintain the leaf part
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # 3c. Morphological methods to remove tiny parts
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=3)
    filtered = morphology.remove_small_objects(eroded.astype(bool), min_size=100)

    # Create a mask based on the filtered components
    mask_leaf = filtered.astype(np.uint8)

    # Combine the masks
    combined_mask = cv2.bitwise_and(mask_blur, final_binary_mask, mask_leaf)

    # Apply the combined mask to the original image
    segmented_image = cv2.bitwise_and(sharpened, sharpened, mask=combined_mask)

    return segmented_image

def useContours(image, cont):
    # Find contours
    contours, _ = cv2.findContours(cont, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    mask = np.zeros_like(cont)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(image, image, mask=mask)

def useMorph(image):
    # Convert the image to grayscale
     # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to segment the leaf from the background
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the leaf
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the leaf)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the leaf
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)

    # Apply morphological operations to refine the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply the mask to the original image to remove unwanted parts
    processed_image = cv2.bitwise_and(image, image, mask=mask)

    return processed_image

def extract_disease(image):

    # Split the LAB image into individual channels
    r_channel, g_channel, b_channel = cv2.split(image)

    # Threshold the A and B channels to remove green leaf
    # Adjust the threshold values
    threshold = cv2.threshold(r_channel, 28, 250, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    threshold = cv2.threshold(g_channel, 128, 172, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    threshold = cv2.threshold(b_channel, 28, 250, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # b_threshold = cv2.threshold(b_channel, 155, 255, cv2.THRESH_BINARY)[1]

    # Combine the A and B thresholds
    # combined_threshold = cv2.bitwise_and(a_threshold, b_threshold)

    # Invert the combined threshold to retain the brown disease part
    # disease_mask = cv2.bitwise_not(combined_threshold)

    # Apply the disease mask to the original image to remove the background
    disease_only = cv2.bitwise_and(image, image, mask=threshold)

    return disease_only

def extract_leaf(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255, 255])  # Adjust the upper bound

    # Create a mask for green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Extract green regions
    leaf_green = cv2.bitwise_and(image, image, mask=mask)

    return leaf_green

def segment_leaf(image):
    border_size = 10  # Adjust the border size as needed
    image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
    # image = cv2.resize(image, (HEIGHT, WIDTH))
    # kernel = np.ones((5, 5), np.uint8)
    test = cv2.medianBlur(image, ksize=3)

    lab = cv2.cvtColor(test, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(test, cv2.COLOR_RGB2HSV)
    l, a, B = cv2.split(lab)
    h, s, v = cv2.split(hsv)
    r, g, b = cv2.split(test)

    l = cv2.equalizeHist(l)
    a = cv2.equalizeHist(a)
    v = cv2.equalizeHist(v)
    g = cv2.equalizeHist(g)
    B = cv2.equalizeHist(B)
    h = cv2.equalizeHist(h)
    
    # a = 255-a
    B = 255-B

    l = cv2.convertScaleAbs(l, alpha=1.25)
    a = cv2.convertScaleAbs(a, alpha=1.25)
    v = cv2.convertScaleAbs(v, alpha=1.25)
    B = cv2.convertScaleAbs(B, alpha=1.25)
    g = cv2.convertScaleAbs(g, alpha=1.25)
    h = cv2.convertScaleAbs(h, alpha=1.25)

    _, l = cv2.threshold(l, 195, 255, cv2.THRESH_BINARY)
    _, a = cv2.threshold(a, 195, 255, cv2.THRESH_BINARY)
    _, v = cv2.threshold(v, 195, 255, cv2.THRESH_BINARY)
    _, B = cv2.threshold(B, 195, 255, cv2.THRESH_BINARY)
    _, g = cv2.threshold(g, 195, 255, cv2.THRESH_BINARY)
    _, h = cv2.threshold(h, 195, 255, cv2.THRESH_BINARY)

    # Define the shape of the kernel (e.g., a square)
    kernel_shape = cv2.MORPH_RECT  # You can use cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, or cv2.MORPH_CROSS

    # Define the size of the kernel (width and height)
    kernel_size = (5, 5)  # Adjust the size as needed

    # Create the kernel
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    
    leaf = cv2.bitwise_xor(v, s)
    leaf = cv2.bitwise_xor(leaf, g)
    leaf = cv2.dilate(leaf, kernel, iterations=2)

    dise = cv2.bitwise_or(a, B)
    dise = cv2.bitwise_or(dise, h)
    leaf = cv2.bitwise_and(v, v)
    mask = cv2.bitwise_or(dise, leaf)
    # # mask = cv2.bitwise_and(mask, z)
    # mask = cv2.bitwise_or(a, s)
    # mask = cv2.bitwise_and(mask, v)
    mask = cv2.convertScaleAbs(mask, alpha=1.25)

    # Without Canny
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
    tmask = cv2.resize(tmask, (TESTH, TESTW))
    mask = cv2.resize(mask, (TESTH, TESTW))
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
