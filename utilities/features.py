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
    hog_features, _ = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block, visualize=True)
    # hog_image = hog_image.flatten()
    return hog_features
    return np.concatenate((hog_features, hog_image))

def getHOGImageFeatures(image, orientations, pixels_per_cell, cells_per_block):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block, visualize=True)
    hog_image = hog_image.flatten()

    return hog_image

def getLBPFeatures(image, radius, points):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_image = local_binary_pattern(image, points, radius, method='uniform')
    feature_vector_lbp, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
    
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

def getCCVFeatures(image, num_bins):
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

# def getColHistFeatures(image):
#     # Convert the image to RGB, HSV, and LAB color spaces
#     rgb_image = image
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

#     # Initialize histograms
#     rgb_hist = []
#     hsv_hist = []
#     lab_hist = []

#     # Compute histograms for each channel in each color space
#     for i in range(3):  # Loop over channels (R, G, B)
#         rgb_hist_channel, _ = np.histogram(rgb_image[:, :, i].ravel(), bins=256, range=(0, 256), density=True)
#         hsv_hist_channel, _ = np.histogram(hsv_image[:, :, i].ravel(), bins=256, range=(0, 1), density=True)
#         lab_hist_channel, _ = np.histogram(lab_image[:, :, i].ravel(), bins=256, range=(-128, 128), density=True)

#         rgb_hist.extend(rgb_hist_channel)
#         hsv_hist.extend(hsv_hist_channel)
#         lab_hist.extend(lab_hist_channel)

#     return np.concatenate((rgb_hist, hsv_hist, lab_hist))

# def getHistFeatures(image, bins):
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Compute histogram
#     hist, _ = np.histogram(gray_image.flatten(), bins=bins, range=[0, 256])

#     # Add a small constant to avoid division by zero or invalid multiplication
#     epsilon = 1e-10
#     hist_normalized = (hist / np.sum(hist)) + epsilon

#     # Extract features from histogram
#     hist_mean = np.mean(hist_normalized)
#     hist_std = np.std(hist_normalized)
#     hist_entropy = -np.sum(hist_normalized * np.log2(hist_normalized))

#     return [hist_mean, hist_std, hist_entropy]

# def getCCVFeatures(image, num_bins=8):
    # # Convert the image to the HSV color space----
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

def getColHistFeatures(image, channels=(0, 1, 2), num_bins=256):
    hist_features = []

    for channel in channels:
        channel_hist = cv2.calcHist([image], [channel], None, [num_bins], ranges=(0, 256))
        channel_hist = channel_hist.flatten()
        hist_features.extend(channel_hist)
    
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    for channel in channels:
        channel_hist = cv2.calcHist([lab_image], [channel], None, [num_bins], ranges=(-128, 128))
        channel_hist = channel_hist.flatten()
        hist_features.extend(channel_hist)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for channel in channels:
        channel_hist = cv2.calcHist([hsv_image], [channel], None, [num_bins], ranges=(0, 1))
        channel_hist = channel_hist.flatten()
        hist_features.extend(channel_hist)
        
    return hist_features