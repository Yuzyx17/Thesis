from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv
from skimage.feature import graycomatrix
from skimage import img_as_ubyte
from skimage.feature import hog

from utilities.util import *

def getGLCMFeatures(image):
    gray_image = img_as_ubyte(rgbAsGray(image))
    glcm = graycomatrix(gray_image, distances=[5], angles=[0], symmetric=True, normed=True)
    # Extract features from GLCM
    contrast = np.mean(glcm[0, 0, :, :])
    dissimilarity = np.mean(glcm[0, 0, :, :])
    homogeneity = np.mean(glcm[0, 0, :, :])
    energy = np.mean(glcm[0, 0, :, :])
    correlation = np.mean(glcm[0, 0, :, :])

    return [contrast, dissimilarity, homogeneity, energy, correlation]

def getShapeFeatures(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate shape features (example: aspect ratio and circularity)
    _, binary_threshold = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)
        aspect_ratio = area / (perimeter ** 2)
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    else:
        aspect_ratio = 0.0
        circularity = 0.0

    return [aspect_ratio, circularity]

def getHOGFeatures(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True):
    
    image = rgbAsGray(image)
    # Display the original image, LBP, and HOG features
    fd, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block, visualize=visualize)
    
    return hog_image.flatten()

def getLBPFeatures(image, radius=1, points=8):

    image = rgbAsGray(image)
    n_points = points * radius
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
    feature_vector_lbp, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    return feature_vector_lbp

def getHSVFeatures(image):
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract features from HSV
    h_mean = np.mean(hsv_image[:,:,0])
    s_mean = np.mean(hsv_image[:,:,1])
    v_mean = np.mean(hsv_image[:,:,2])

    h_std = np.std(hsv_image[:,:,0])
    s_std = np.std(hsv_image[:,:,1])
    v_std = np.std(hsv_image[:,:,2])

    return h_mean, s_mean, v_mean, h_std, s_std, v_std

def getLABFeatures(image):
    # Convert the image to HSV
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract features from HSV
    l_mean = np.mean(lab_image[:,:,0])
    a_mean = np.mean(lab_image[:,:,1])
    b_mean = np.mean(lab_image[:,:,2])

    l_std = np.std(lab_image[:,:,0])
    a_std = np.std(lab_image[:,:,1])
    b_std = np.std(lab_image[:,:,2])

    return l_mean, a_mean, b_mean, l_std, a_std, b_std

def getColorFeatures(image):
    # Extract features from HSV
    r_mean = np.mean(image[:,:,0])
    g_mean = np.mean(image[:,:,1])
    b_mean = np.mean(image[:,:,2])

    r_std = np.std(image[:,:,0])
    g_std = np.std(image[:,:,1])
    b_std = np.std(image[:,:,2])

    return r_mean, g_mean, b_mean, r_std, g_std, b_std

def getStatFeatures(image):
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))
    return mean, std

def getCoCoFeatures(image):
    # Apply bilateral filter
    coherence = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to grayscale
    coherence_gray = cv2.cvtColor(coherence, cv2.COLOR_BGR2GRAY)

    # Extract features from color coherence
    coherence_mean = np.mean(coherence_gray)
    coherence_std = np.std(coherence_gray)

    return coherence_mean, coherence_std

def getHistFeatures(image, bins=256):
    # Convert the image to grayscale
    gray_image = rgbAsGray(image)

    # Compute histogram
    hist, _ = np.histogram(gray_image.flatten(), bins=bins, range=[0, 256])

    # Add a small constant to avoid division by zero or invalid multiplication
    epsilon = 1e-10
    hist_normalized = (hist / np.sum(hist)) + epsilon

    # Extract features from histogram
    hist_mean = np.mean(hist_normalized)
    hist_std = np.std(hist_normalized)
    hist_entropy = -np.sum(hist_normalized * np.log2(hist_normalized))

    return [hist_mean, hist_std, hist_entropy]

def getColHistFeatures(image, num_bins=256, channels=(0, 1, 2), ranges=(0, 256)):
    hist_features = []

    for channel in channels:
        channel_hist = cv2.calcHist([image], [channel], None, [num_bins], ranges=ranges)
        channel_hist = channel_hist.flatten()
        hist_features.extend(channel_hist)

    return hist_features

def getFeatures(image):
    return np.hstack((
                    # getHOGFeatures(image),
                    getGLCMFeatures(image),
                    getHistFeatures(image),
                    getLBPFeatures(image),
                    getHSVFeatures(image),
                    getLABFeatures(image),
                    getColorFeatures(image),
                    getColHistFeatures(image),
                    # getShapeFeatures(image),
                    getCoCoFeatures(image)
                    ))