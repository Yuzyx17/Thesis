from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from skimage.feature import hog

import numpy as np
import cv2

def getGLCMFeatures(image, distance=5, angles=0, levels=256):
    gray_image = img_as_ubyte(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    glcm = graycomatrix(gray_image, distances=[distance], angles=[angles], levels=levels, symmetric=True, normed=True)

    # Calculate the GLCM features
    energy = graycoprops(glcm, 'energy').mean()
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()

    return [contrast, dissimilarity, homogeneity, energy, correlation]

def getHOGFeatures(image, orientations, pixels_per_cell, cells_per_block):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Display the original image, LBP, and HOG features
    hog_features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block, feature_vector=True)

    return hog_features

def getHOGImageFeatures(image, orientations, pixels_per_cell, cells_per_block):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block, visualize=True)
    hog_image = hog_image.flatten()

    return hog_image

def getLBPFeatures(image, radius=1, points=8):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    n_points = points * radius
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
    feature_vector_lbp, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    
    return feature_vector_lbp

def getHSVFeatures(image):
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    # Extract features from HSV
    h_mean = np.mean(h)
    s_mean = np.mean(s)
    v_mean = np.mean(v)

    h_std = np.std(h)
    s_std = np.std(s)
    v_std = np.std(v)

    return [h_mean, s_mean, v_mean, h_std, s_std, v_std]

def getLABFeatures(image):
    # Convert the image to HSV
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)

    # Extract features from HSV
    l_mean = np.mean(l)
    a_mean = np.mean(a)
    b_mean = np.mean(b)

    l_std = np.std(l)
    a_std = np.std(a)
    b_std = np.std(b)

    return [l_mean, a_mean, b_mean, l_std, a_std, b_std]

def getRGBFeatures(image):
    # Extract features from HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(image)

    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    r_std = np.std(r)
    g_std = np.std(g)
    b_std = np.std(b)

    return [r_mean, g_mean, b_mean, r_std, g_std, b_std]

def getStatFeatures(image):
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))
    stat_features = np.concatenate((mean, std))

    return stat_features

def getColorFeatures(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Convert image to LAB color space

    # Split the LAB image into its components
    L, A, B = cv2.split(image)

    # Create 1D arrays for A and B channels
    A = A.flatten()
    B = B.flatten()

    # Combine A and B channels into a single 1D array
    ab = np.concatenate((A, B))
    return ab

def getCCVFeatures(image, num_bins=8):
    # Convert the image to the HSV color space----
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # # Split the HSV image into individual channels
    # h, s, v = cv2.split(hsv_image)

    # # Calculate the color histograms for each channel
    # hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    # hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    # hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])

    # # Normalize the histograms
    # hist_h /= hist_h.sum()
    # hist_s /= hist_s.sum()
    # hist_v /= hist_v.sum()

    # # Calculate the color coherence features
    # color_coherence = np.concatenate((hist_h, hist_s, hist_v), axis=None)

    # # Calculate the mean and standard deviation
    # mean = np.mean(color_coherence)
    # std_dev = np.std(color_coherence)

    # return [mean, std_dev]
    
    # Convert image to LAB color space--------------
    # lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # # Initialize CCV bins for each channel (L, A, B)
    # ccv_bins = [8, 8, 8]  # You can adjust the number of bins as needed

    # # Split LAB channels
    # channels = cv2.split(lab_image)

    # # Initialize CCV histogram
    # ccv_hist = np.zeros((ccv_bins[0], ccv_bins[1], ccv_bins[2]))

    # height, width = image.shape[:2]

    # for i in range(height):
    #     for j in range(width):
    #         # Get pixel values in LAB
    #         pixel = lab_image[i, j]

    #         # Determine bin indices for each channel
    #         l_bin = int(pixel[0] * ccv_bins[0] / 256)
    #         a_bin = int(pixel[1] * ccv_bins[1] / 256)
    #         b_bin = int(pixel[2] * ccv_bins[2] / 256)

    #         # Increment the corresponding bin in CCV histogram
    #         ccv_hist[l_bin, a_bin, b_bin] += 1

    # # Normalize the CCV histogram
    # ccv_hist /= (height * width)

    # # Flatten the 3D histogram into a 1D feature vector
    # ccv_feature = ccv_hist.flatten()

    # return ccv_feature  

    # Convert the image to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Initialize the CCV histogram
    ccv_hist = np.zeros((num_bins, num_bins, num_bins))

    # Define the size of the regions
    region_width = lab_image.shape[1] // num_bins
    region_height = lab_image.shape[0] // num_bins

    # Calculate CCV for each region
    for i in range(num_bins):
        for j in range(num_bins):
            # Get the current region
            region = lab_image[j * region_height:(j + 1) * region_height,
                              i * region_width:(i + 1) * region_width]

            # Calculate the histogram for the region
            hist = cv2.calcHist([region], [0, 1, 2], None, [num_bins, num_bins, num_bins],
                                [0, 256, 0, 256, 0, 256])

            # Normalize the histogram and store it in the CCV
            hist /= np.sum(hist)
            ccv_hist += hist

    # Flatten the CCV histogram
    ccv_vector = ccv_hist.flatten()
        
    return ccv_vector

def getHistFeatures(image, bins):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

def getColHistFeatures(image, num_bins, channels, ranges):
    hist_features = []

    for channel in channels:
        channel_hist = cv2.calcHist([image], [channel], None, [num_bins], ranges=ranges)
        channel_hist = channel_hist.flatten()
        hist_features.extend(channel_hist)
    
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    for channel in channels:
        channel_hist = cv2.calcHist([lab_image], [channel], None, [num_bins], ranges=ranges)
        channel_hist = channel_hist.flatten()
        hist_features.extend(channel_hist)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for channel in channels:
        channel_hist = cv2.calcHist([hsv_image], [channel], None, [num_bins], ranges=ranges)
        channel_hist = channel_hist.flatten()
        hist_features.extend(channel_hist)
        
    return hist_features