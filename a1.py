import os
import cv2 as cv
import numpy as np
import argparse

from skimage import feature, data, exposure
import matplotlib.pyplot as plt

def handle_config(color_format, img_path):
    """
        Extract colors panels of image from specific path based on HSV or YCrCb

        Args:
            color_format (string): Either HSV or YCrCb
            img_path     (string): path to the extracted image
        Returns: void
    """
    img = cv.imread(os.path.join(img_path))
    cv.imshow("picture", img)
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    model_option = ""
    if color_format == 'HSV':
        model_option = cv.COLOR_BGR2HSV
    elif color_format == 'YCrCb':
        model_option = cv.COLOR_BGR2YCR_CB
    else:
        raise Exception("No Such Color Model")

    img_hsv = cv.cvtColor(img, model_option)
    a, b, c = cv.split(img_hsv)
    blank = np.full((img_height * 2, img_width * 2, 3), 255, dtype='uint8')
    
    blank[0:img_height, 0:img_width] = img

    blank[0:img_height, img_width:img_width * 2] = cv.merge([a, a, a])

    blank[img_height:img_height * 2, img_width:img_width * 2] = cv.merge([c,c, c])

    blank[img_height:img_height * 2, 0:img_width] = cv.merge([b, b, b])

    cv.imshow(color_format + "_" + img_path, blank)
    cv.waitKey(0)
    cv.destroyAllWindows()

def sift_detect(img):
    """
        Detects the sift keypoints and descryptors of an image

        Args:
            img (np.array): pixel array represent image
        Returns:
            nm.array: pixel array of input image with marked keypoints
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    print("Length of features is ", len(descriptors) * len(descriptors[0]))
    print(descriptors)
    img_with_keypoints = cv.drawKeypoints(img, keypoints, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_with_keypoints

def hog_transform(img):
    """
        Transform a specific image into HOG
        
        Args:
            img (np.array): input array of pixels
        Returns:
            np.array: HOG
    """

    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Compute HOG features
    features, hog_image = feature.hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm='L2'
    )

    print(features)
    print("len of the features ", len(features))
    
    # Normalize for display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image_rescaled

def handle_foption(method, file_path):
    """
        Analyse keypoints and descriptors
        Args:
            method (string): method for analysis
            file_path (string): path to the image file in the system
        Returns:
            void
    """
    img = cv.imread(os.path.join(file_path))
    blank = np.zeros((img.shape[0], img.shape[1] * 2, 3), dtype='uint8')
    blank[0:img.shape[0], 0:img.shape[1]] = img
    if method == 'SIFT': 
        img_with_keypoints = sift_detect(img)
        blank[0:img.shape[0], img.shape[1]:img.shape[1] * 2] = img_with_keypoints
    elif method == 'HOG':
        h, w = img.shape[:2]
        hog_img = hog_transform(img)

        if hog_img.shape[:2] != (h, w):
            hog_img = cv.resize(hog_img, (w, h), interpolation=cv.INTER_LINEAR)
        if hog_img.dtype != np.uint8:
            hog_img = (np.clip(hog_img, 0, 1) * 255).astype(np.uint8)
        hog_bgr = cv.cvtColor(hog_img, cv.COLOR_GRAY2BGR)
        blank[0:img.shape[0], img.shape[1]:img.shape[1] * 2] = hog_bgr
    cv.imshow(method + "_" + file_path, blank)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Assignment 1")
    parser.add_argument(
        "-c", "--coption",
        nargs=2,
        help="Config"
    )
    parser.add_argument(
        "-f", "--foption",
        nargs=2,
        help="Feature Detection"
    )
    parser.add_argument(
        "-r", "--run",
        nargs=1,
        help="Run Classification"
    )

    args = parser.parse_args()
    if args.coption:
        color_format = args.coption[0]
        img_path = args.coption[1]
        print("color format ", color_format)
        print("file path ", img_path)
        handle_config(color_format, img_path)
    if args.foption:
        method = args.foption[0]
        file_path = args.foption[1]
        print("method ", method)
        print("file path ", file_path)
        handle_foption(method, file_path)
    if args.run:
        folder_path = args.run[0]
        print("folder ", folder_path)
main()