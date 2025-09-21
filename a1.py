import os
import cv2 as cv
import numpy as np
import argparse

from skimage import feature, data, exposure
import matplotlib.pyplot as plt

# ===Classification===

def sift_descriptors_extract(image_path):
    """
        Extract sift descriptors of image from a path
    """
    gray = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

def list_images_by_class(root_dir):
    """
    Returns: classes (sorted list), dict[class_name] -> list of image paths
    """
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    classes.sort()
    paths_by_class = {}
    for cls in classes:
        cdir = os.path.join(root_dir, cls)
        files = [os.path.join(cdir, f) for f in os.listdir(cdir)
                if os.path.isfile(os.path.join(cdir, f)) and f.lower().endswith(".jpg")]
        files.sort()
        paths_by_class[cls] = files
    return classes, paths_by_class

# Split folder into train and test
def split_train_test(classes, paths_by_class, train_ratio=0.8):
    """
        Split the images dateset randomly 80/20 train/test
    """
    X_tr, X_t, y_tr, y_t = [], [], [], []
    for ci, cls in enumerate(classes):
        indices = np.arange(len(paths_by_class[cls]))
        np.random.shuffle(indices)
        for i in range(len(indices)):
            if i + 1 <= train_ratio * len(indices):
                X_tr.append(paths_by_class[cls][indices[i]])
                y_tr.append(ci)
            else:
                X_t.append(paths_by_class[cls][indices[i]])
                y_t.append(ci)
    return X_tr, np.array(y_tr, np.int32), X_t, np.array(y_t, np.int32)

def build_vocabulary(image_paths, n_clusters=400, mx_des_per_image=300):
    """
        Build the vocab words for training images
    """
    paths = np.array(image_paths)
    descs = []
    for path in paths:
        d = sift_descriptors_extract(path)
        indices = np.arange(len(d))
        keep = np.random.choice(indices, min(indices.size, mx_des_per_image), replace=False)
        d = d[keep]
        if d is None:
            continue
        descs.append(d)
    
    all_descriptors = np.vstack(descs).astype(np.float32)

    _, _, centers = cv.kmeans(
        all_descriptors,
        n_clusters,
        None,
        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3),
        10,
        cv.KMEANS_PP_CENTERS
    )
    return centers

def build_histogram(descriptors, vocabulary):
    """
        Building histogram features for set of descriptors and certain vocab
    """
    K = vocabulary.shape[0]

    d_norm2 = np.sum(descriptors*descriptors, axis=1, keepdims=True)
    c_norm2 = np.sum(vocabulary*vocabulary, axis=1)[None]
    dot = descriptors @ vocabulary.T

    # NxK matrix representing distance of N descriptor each to K vocab
    dist2 = d_norm2 + c_norm2 - 2.0 * dot

    # Find the closest vocab to each descriptor
    assign = np.argmin(dist2, axis=1)

    hist = np.zeros(K)
    for a in assign:
        if a < K:
            hist[a] += 1
    
    # Normalization
    n = np.linalg.norm(hist)
    if n > 0:
        hist /= n
    return hist

def train_linear_svm(Xtr, ytr, C=2.0):
    """
        Train a linear multi-class SVM using OpenCV (one-vs-one internally).
    """

    print("C is ", C)
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setC(float(C))
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 2000, 1e-6))
    Xtr32 = Xtr.astype(np.float32)
    ytr32 = ytr.astype(np.int32).reshape(-1, 1)
    svm.train(Xtr32, cv.ml.ROW_SAMPLE, ytr32)
    return svm

def svm_predict(svm, X):
    """
        Predict test set X with certain svm
    """
    X32 = X.astype(np.float32)
    _, yhat = svm.predict(X32)
    return yhat.ravel().astype(np.int32)


def run_sift_svm(X_train, X_test, y_train, y_test, n_clusters=200, mx_des_per_image=300, C=2.0):
    """
        One run of SIFT-BoVW + Linear SVM on a given split.
    """
    vocab = build_vocabulary(X_train, n_clusters=n_clusters, mx_des_per_image=mx_des_per_image)
    if vocab is None:
        return 0.0  # no descriptors found anywhere

    # Featurize train
    features_train = [build_histogram(sift_descriptors_extract(p), vocab) for p in X_train]
    Xtr = np.vstack(features_train).astype(np.float32)

    # Featurize test
    features_test = [build_histogram(sift_descriptors_extract(p), vocab) for p in X_test]
    Xte = np.vstack(features_test).astype(np.float32)

    # Train and evaluate
    svm = train_linear_svm(Xtr, y_train, C=C)
    yhat = svm_predict(svm, Xte)
    acc = float((yhat == y_test).mean())
    return acc

def handle_run(folder_path, n_run = 5):
    """
        handle -r, run the sift+svm predict 5 times and record mean and deviation of accuracy
    """
    classes, paths = list_images_by_class(folder_path)
    print("\n--- Running SIFT+SVM Classification ---")
    accs = []
    for i in range(n_run):
        X_tr, y_tr, X_t, y_t = split_train_test(classes, paths)
        acc = run_sift_svm(X_tr, X_t, y_tr, y_t)
        accs.append(acc)
        print(f"[SIFT+SVM Run {i}] accuracy = {acc:.4f}")
    accs = np.array(accs, dtype=np.float64)
    mean_acc = accs.mean()
    std_dev = accs.std(ddof=1) if len(accs) > 1 else 0.0
    print(f"SIFT+SVM Mean accuracy: {mean_acc:.4f}")
    print(f"SIFT+SVM Std deviation: {std_dev:.4f}")

# ===Visualization===

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

def handle_fsave(method, file_path, save_path):
    img = cv.imread(os.path.join(file_path))
    blank = np.zeros((img.shape[0], img.shape[1] * 2, 3), dtype='uint8')
    blank[0:img.shape[0], 0:img.shape[1]] = img
    paths = file_path.split('/')
    out_name= paths[len(paths) - 2] + paths[len(paths) - 1]
    if method == 'SIFT': 
        img_with_keypoints = sift_detect(img)
        blank[0:img.shape[0], img.shape[1]:img.shape[1] * 2] = img_with_keypoints
        cv.imwrite(save_path + "/" + "SIFT_" + out_name, blank, params=None)
    elif method == 'HOG':
        h, w = img.shape[:2]
        hog_img = hog_transform(img)

        if hog_img.shape[:2] != (h, w):
            hog_img = cv.resize(hog_img, (w, h), interpolation=cv.INTER_LINEAR)
        if hog_img.dtype != np.uint8:
            hog_img = (np.clip(hog_img, 0, 1) * 255).astype(np.uint8)
        hog_bgr = cv.cvtColor(hog_img, cv.COLOR_GRAY2BGR)
        blank[0:img.shape[0], img.shape[1]:img.shape[1] * 2] = hog_bgr
        cv.imwrite(save_path + "/" + "HOG_" + out_name, blank, params=None)

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

    parser.add_argument(
        "-fs", "--fsave",
        nargs = 3,
        help="Run and save features"
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
        # print("folder ", folder_path)
        handle_run(folder_path)
    if args.fsave:
        method = args.fsave[0]
        open_path = args.fsave[1]
        save_path = args.fsave[2]
        handle_fsave(method, open_path, save_path)
main()