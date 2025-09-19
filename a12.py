import os
import cv2 as cv
import numpy as np
import argparse

from skimage import feature, exposure
import matplotlib.pyplot as plt

# =========================
# =====  CLASSIFIERS  =====
# =========================

def _train_linear_svm(Xtr, ytr, C=1.0):
    """
    Train a linear multi-class SVM using OpenCV (one-vs-one internally).
    """
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setC(float(C))
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 2000, 1e-6))
    Xtr32 = Xtr.astype(np.float32)
    ytr32 = ytr.astype(np.int32).reshape(-1, 1)
    svm.train(Xtr32, cv.ml.ROW_SAMPLE, ytr32)
    return svm

def _svm_predict(svm, X):
    X32 = X.astype(np.float32)
    _, yhat = svm.predict(X32)
    return yhat.ravel().astype(np.int32)

def run_classification_hog_svm(
    root_dir,
    *,
    n_runs=5,
    train_ratio=0.8,
    C=1.0,
    hog_size=(128, 128),
    hog_orientations=9,
    hog_ppc=(16, 16),
    hog_cpb=(2, 2)
):
    """
    Runs the HOG+SVM classification with tuned parameters.
    Returns mean accuracy and std deviation.
    """
    hog_params = dict(
        size=hog_size,
        orientations=hog_orientations,
        pixels_per_cell=hog_ppc,
        cells_per_block=hog_cpb,
        block_norm="L2-Hys",
        transform_sqrt=True
    )
    print("\n--- Running HOG+SVM Classification ---")
    print(f"[HOG] size={hog_size} ori={hog_orientations} ppc={hog_ppc} cpb={hog_cpb}")
    print(f"[SVM] Linear C={C}")

    classes, feats_by_cls = _precompute_hog_features(root_dir, hog_params)

    rng_master = np.random.default_rng()
    accs = []
    for run_id in range(1, n_runs + 1):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        Xtr, ytr, Xte, yte = _make_split(feats_by_cls, classes, train_ratio, rng)
        mu, sg = _fit_standardizer(Xtr)
        Xtr = _apply_standardizer(Xtr, mu, sg)
        Xte = _apply_standardizer(Xte, mu, sg)
        svm = _train_linear_svm(Xtr, ytr, C=C)
        yhat = _svm_predict(svm, Xte)
        acc = float((yhat == yte).mean())
        accs.append(acc)
        print(f"[HOG+SVM Run {run_id}] accuracy = {acc:.4f}")

    accs = np.array(accs, dtype=np.float64)
    mean_acc = accs.mean()
    std_dev = accs.std(ddof=1) if len(accs) > 1 else 0.0
    print(f"HOG+SVM Mean accuracy: {mean_acc:.4f}")
    print(f"HOG+SVM Std deviation: {std_dev:.4f}")
    return mean_acc, std_dev

def _get_sift_descriptors(img_path):
    """Extract SIFT descriptors from a single image."""
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    sift = cv.SIFT_create()
    _, descriptors = sift.detectAndCompute(img, None)
    return descriptors

def _build_visual_vocabulary(image_paths, n_clusters=200):
    """Builds a visual vocabulary using K-Means clustering on SIFT descriptors."""
    descriptors_list = []
    np.random.shuffle(image_paths)
    for p in image_paths[:50]:
        des = _get_sift_descriptors(p)
        if des is not None and des.size > 0:
            descriptors_list.append(des)
    
    if not descriptors_list:
        return None
        
    all_descriptors = np.vstack(descriptors_list).astype(np.float32)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_PP_CENTERS
    compactness, labels, centers = cv.kmeans(
        all_descriptors, n_clusters, None, criteria, 10, flags
    )
    return centers

def _create_bow_hist(descriptors, vocabulary):
    """Creates a Bag-of-Visual-Words histogram for an image."""
    if descriptors is None:
        return np.zeros(vocabulary.shape[0], dtype=np.float32)

    flann = cv.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
    matches = flann.knnMatch(descriptors.astype(np.float32), vocabulary, k=1)
    
    histogram = np.zeros(vocabulary.shape[0], dtype=np.float32)
    for m in matches:
        if m:
            best_match = m[0]
            visual_word_idx = best_match.trainIdx
            histogram[visual_word_idx] += 1
            
    norm = np.linalg.norm(histogram)
    if norm > 0:
        histogram /= norm
        
    return histogram

def run_classification_sift_svm(
    root_dir,
    *,
    n_runs=5,
    train_ratio=0.8,
    C=1.0,
    n_clusters=200
):
    """
    Runs the SIFT+BoVW+SVM classification with tuned parameters.
    Returns mean accuracy and std deviation.
    """
    print("\n--- Running SIFT+SVM Classification ---")
    print(f"[SIFT] Bag-of-Visual-Words with {n_clusters} clusters")
    print(f"[SVM] Linear C={C}")

    classes, paths_by_class = _list_images_by_class(root_dir)
    all_paths = [p for cls_paths in paths_by_class.values() for p in cls_paths]
    vocabulary = _build_visual_vocabulary(all_paths, n_clusters=n_clusters)
    if vocabulary is None:
        print("Error: Could not build visual vocabulary.")
        return 0.0, 0.0

    feats_by_cls = {}
    for cls in classes:
        feats = []
        for p in paths_by_class[cls]:
            des = _get_sift_descriptors(p)
            hist = _create_bow_hist(des, vocabulary)
            feats.append(hist)
        feats_by_cls[cls] = np.array(feats, dtype=np.float32)

    rng_master = np.random.default_rng()
    accs = []
    for run_id in range(1, n_runs + 1):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        Xtr, ytr, Xte, yte = _make_split(feats_by_cls, classes, train_ratio, rng)
        mu, sg = _fit_standardizer(Xtr)
        Xtr = _apply_standardizer(Xtr, mu, sg)
        Xte = _apply_standardizer(Xte, mu, sg)
        svm = _train_linear_svm(Xtr, ytr, C=C)
        yhat = _svm_predict(svm, Xte)
        acc = float((yhat == yte).mean())
        accs.append(acc)
        print(f"[SIFT+SVM Run {run_id}] accuracy = {acc:.4f}")

    accs = np.array(accs, dtype=np.float64)
    mean_acc = accs.mean()
    std_dev = accs.std(ddof=1) if len(accs) > 1 else 0.0
    print(f"SIFT+SVM Mean accuracy: {mean_acc:.4f}")
    print(f"SIFT+SVM Std deviation: {std_dev:.4f}")
    return mean_acc, std_dev

def compare_features_performance(root_dir):
    """
    Compares the classification performance of HOG vs. SIFT features using SVM.
    """
    print("--- Comparing HOG and SIFT Features ---")
    
    # Run HOG+SVM
    hog_mean, hog_std = run_classification_hog_svm(root_dir)
    
    # Run SIFT+SVM
    sift_mean, sift_std = run_classification_sift_svm(root_dir)
    
    print("\n--- Comparison Summary ---")
    print(f"HOG+SVM: Mean Accuracy = {hog_mean:.4f} (Std Dev = {hog_std:.4f})")
    print(f"SIFT+SVM: Mean Accuracy = {sift_mean:.4f} (Std Dev = {sift_std:.4f})")
    
    if hog_mean > sift_mean:
        print("\nHOG features appear to yield better results.")
    elif sift_mean > hog_mean:
        print("\nSIFT features appear to yield better results.")
    else:
        print("\nBoth HOG and SIFT features yield similar results.")

# =========================
# ===== SHARED UTILS  =====
# =========================

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def _list_images_by_class(root_dir):
    """
    Returns: classes (sorted list), dict[class_name] -> list of image paths
    """
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    classes.sort()
    paths_by_class = {}
    for cls in classes:
        cdir = os.path.join(root_dir, cls)
        files = [os.path.join(cdir, f) for f in os.listdir(cdir)
                 if os.path.isfile(os.path.join(cdir, f)) and f.lower().endswith(_IMG_EXTS)]
        files.sort()
        paths_by_class[cls] = files
    return classes, paths_by_class

def _precompute_hog_features(root_dir, hog_params):
    """Precomputes HOG features for all images once."""
    classes, paths_by_class = _list_images_by_class(root_dir)
    feats_by_cls = {}
    for cls in classes:
        feats = []
        for p in paths_by_class[cls]:
            img = cv.imread(p)
            if img is None:
                continue
            f = extract_hog_feature_for_classifier(img, **hog_params)
            feats.append(f)
        if len(feats) == 0:
            feats_by_cls[cls] = np.zeros((0, 1), dtype="float32")
        else:
            feats_by_cls[cls] = np.vstack(feats).astype("float32")
    return classes, feats_by_cls

def _prep_for_hog_cls(img_bgr, crop_border=0.1, size=(128, 128), use_clahe=True):
    """
    Light preproc: central crop, grayscale/Y, optional CLAHE, resize.
    """
    h, w = img_bgr.shape[:2]
    if crop_border > 0:
        dx = int(w * crop_border / 2)
        dy = int(h * crop_border / 2)
        if dx > 0 and dy > 0:
            img_bgr = img_bgr[dy:h - dy, dx:w - dx]
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    gray = cv.resize(gray, size, interpolation=cv.INTER_AREA)
    return gray

def extract_hog_feature_for_classifier(
    img_bgr,
    *,
    size=(128, 128),
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True
):
    """
    Returns a 1D float32 HOG descriptor for classification.
    """
    g = _prep_for_hog_cls(img_bgr, size=size, use_clahe=True)
    feats = feature.hog(
        g,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        transform_sqrt=transform_sqrt,
        visualize=False,
        feature_vector=True
    )
    return feats.astype("float32")

def _split_indices_per_class(n, train_ratio, rng):
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tr = max(1, int(round(n * train_ratio)))
    n_tr = min(n_tr, n - 1)
    return idx[:n_tr], idx[n_tr:]

def _make_split(feats_by_cls, classes, train_ratio, rng):
    Xtr, ytr, Xte, yte = [], [], [], []
    for ci, cls in enumerate(classes):
        X = feats_by_cls[cls]
        n = X.shape[0]
        if n < 2:
            continue
        tr_idx, te_idx = _split_indices_per_class(n, train_ratio, rng)
        Xtr.append(X[tr_idx]); ytr.append(np.full(tr_idx.shape[0], ci, dtype=np.int32))
        Xte.append(X[te_idx]); yte.append(np.full(te_idx.shape[0], ci, dtype=np.int32))
    if not Xtr:
        raise RuntimeError("No usable data found in the dataset.")
    Xtr = np.vstack(Xtr).astype("float32")
    Xte = np.vstack(Xte).astype("float32")
    ytr = np.concatenate(ytr); yte = np.concatenate(yte)
    return Xtr, ytr, Xte, yte

def _fit_standardizer(X):
    mu = X.mean(axis=0, dtype=np.float64)
    sigma = X.std(axis=0, dtype=np.float64)
    sigma[sigma == 0] = 1.0
    return mu.astype("float32"), sigma.astype("float32")

def _apply_standardizer(X, mu, sigma):
    return (X - mu) / sigma

# --- Visualization and main functions ---

def handle_config(color_format, img_path):
    """
    Extract colors panels of image from specific path based on HSV or YCrCb
    """
    img = cv.imread(os.path.join(img_path))
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return

    img_height, img_width = img.shape[:2]
    
    model_option = ""
    if color_format == 'HSV':
        model_option = cv.COLOR_BGR2HSV
    elif color_format == 'YCrCb':
        model_option = cv.COLOR_BGR2YCR_CB
    else:
        print("Error: Invalid color format. Use HSV or YCrCb.")
        return

    img_converted = cv.cvtColor(img, model_option)
    a, b, c = cv.split(img_converted)

    canvas_height = img_height * 2
    canvas_width = img_width * 2
    blank = np.full((canvas_height, canvas_width, 3), 255, dtype='uint8')

    blank[0:img_height, 0:img_width] = img
    blank[0:img_height, img_width:canvas_width] = cv.merge([a, a, a])
    blank[img_height:canvas_height, img_width:canvas_width] = cv.merge([b, b, b])
    blank[img_height:canvas_height, 0:img_width] = cv.merge([c, c, c])

    cv.imshow(f"{color_format} Channel Breakdown: {os.path.basename(img_path)}", blank)
    cv.waitKey(0)
    cv.destroyAllWindows()

def sift_detect(img):
    """
    Detects and visualizes SIFT keypoints and descriptors of an image.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    if descriptors is not None:
        print(f"Number of SIFT keypoints detected: {len(keypoints)}")
    else:
        print("No SIFT keypoints detected.")

    img_with_keypoints = cv.drawKeypoints(
        img, 
        keypoints, 
        None, 
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return img_with_keypoints

def hog_transform(img):
    """
    Transform an image into HOG features and return the visualization.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    resized_gray = cv.resize(gray, (int(w * 0.75), int(h * 0.75)), interpolation=cv.INTER_AREA)

    features, hog_image = feature.hog(
        resized_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm='L2-Hys'
    )
    
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_img = cv.resize(hog_image_rescaled, (w, h), interpolation=cv.INTER_LINEAR)
    hog_bgr = cv.cvtColor((hog_img * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)

    return features, hog_bgr

def handle_foption(method, file_path):
    """
    Analyze keypoints and descriptors
    """
    img = cv.imread(os.path.join(file_path))
    if img is None:
        print(f"Error: Could not read image at {file_path}")
        return

    h, w = img.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype='uint8')
    canvas[0:h, 0:w] = img

    if method == 'SIFT': 
        img_with_keypoints = sift_detect(img)
        canvas[0:h, w:w*2] = img_with_keypoints
        cv.imshow(f"SIFT Keypoint Analysis: {os.path.basename(file_path)}", canvas)
    elif method == 'HOG':
        _, hog_bgr = hog_transform(img)
        canvas[0:h, w:w*2] = hog_bgr
        cv.imshow(f"HOG Feature Analysis: {os.path.basename(file_path)}", canvas)
    else:
        print("Error: Invalid feature method. Use SIFT or HOG.")
        return

    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Assignment 1")
    parser.add_argument("-c", "--coption", nargs=2, help="Config")
    parser.add_argument("-f", "--foption", nargs=2, help="Feature Detection")
    parser.add_argument("-r", "--run", nargs=1, help="Run Classification")
    parser.add_argument("--compare", nargs=1, help="Compare HOG vs SIFT classification performance")

    args = parser.parse_args()
    if args.coption:
        color_format = args.coption[0]
        img_path = args.coption[1]
        print(f"Color format: {color_format}")
        print(f"File path: {img_path}")
        handle_config(color_format, img_path)
    elif args.foption:
        method = args.foption[0]
        file_path = args.foption[1]
        print(f"Method: {method}")
        print(f"File path: {file_path}")
        handle_foption(method, file_path)
    elif args.run:
        folder_path = args.run[0]
        print(f"Running HOG+SVM Classification on folder: {folder_path}")
        run_classification_sift_svm(folder_path)
    elif args.compare:
        folder_path = args.compare[0]
        compare_features_performance(folder_path)
    else:
        parser.print_help()
        
main()