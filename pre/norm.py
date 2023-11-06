import cv2
from utilities.const import *
from utilities.util import *
from skimage import exposure

def useNormBright(img):
    gray = rgbAsGray(img)
    mean = gray.mean()
    scaling_factor = LTHRESHOLD / mean
    normalized_image = cv2.convertScaleAbs(img, alpha=scaling_factor, beta=0)
    return normalized_image

def useGDenoise(img):
    return cv2.GaussianBlur(img, DENOISE_KERNEL, DENOISE_SIGMA)  # Adjust kernel size and sigmaX use needed

def useBDenoise(img):
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def useSaturation(img):
    image = rgbAsHsv(img)
    image[:, :, 1] = image[:, :, 1] * 1.5 #Add to Constant
    
    return image

def useSharpening(img):
    sharpened = cv2.addWeighted(img, 1.5, img, -0.5, 0)
    return sharpened

def useFSharpening(img):

    kernel = np.array([[-1,-1,-1],[-1,10,-1],[-1,-1,-1]])  # Sharpening kernel
    return cv2.filter2D(img, -1, kernel)


def useWWhiteBalance(img):
    # Convert the image to float32
    image_float = img.astype(float)

    # Compute the average color of the image
    avg_color = image_float.mean(axis=(0, 1))

    # Perform white balancing using the Gray World Assumption
    white_balanced_image = (image_float / avg_color) * [128, 128, 128]
    return np.clip(white_balanced_image, 0, 255).astype(np.uint8)

def useCLAHE(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_channels = [clahe.apply(img[:, :, i]) for i in range(3)]
    return np.stack(enhanced_channels, axis=-1)

def useScaleAbs(img, alpha=1.0, beta=0.0):
    adjusted_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted_image

def useContrast(img):
    # Convert the image to grayscale
    gray_image = rgbAsGray(img)

    # Scale the pixel values for contrast enhancement
    alpha = 1.5  # Contrast control (1.0 means no change)
    beta = 10    # Brightness control (0 means no change)
    enhanced_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)

    # Convert the enhanced image back to BGR (if needed)
    enhanced_image_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    return enhanced_image_bgr

def useResize(img):
    return cv2.resize(img, (WIDTH, HEIGHT))

def removeBlur(image):
        # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute gradient magnitude
    gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    gradient = cv2.convertScaleAbs(gradient)

    # Thresholding
    threshold = 50
    _, binary_mask = cv2.threshold(gradient, threshold, 255, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Apply mask to the original image
    result = cv2.bitwise_and(image, image, mask=binary_mask)
    return result

def eqHist(image):
    enhanced_image = cv2.equalizeHist(image)
    return enhanced_image
