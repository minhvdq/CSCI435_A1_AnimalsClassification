import os, glob, random
import numpy as np
import cv2
from skimage.feature import hog as sk_hog

# ---------- low-level diagnostics ----------

def _sift_keypoint_focus(img_bgr, center_frac=0.6):
    """
    Fraction of SIFT keypoints that fall inside the central box covering
    `center_frac` of width and height. If no keypoints, returns 0.0.
    """
    h, w = img_bgr.shape[:2]
    cx0 = int((1 - center_frac) * 0.5 * w)
    cy0 = int((1 - center_frac) * 0.5 * h)
    cx1 = w - cx0
    cy1 = h - cy0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kps = sift.detect(gray, None)

    if not kps:
        return 0.0

    inside = 0
    for kp in kps:
        x, y = kp.pt
        if (cx0 <= x < cx1) and (cy0 <= y < cy1):
            inside += 1
    return inside / max(1, len(kps))


def _hog_objectness_ratio(img_bgr, center_frac=0.6):
    """
    Proxy for 'HOG will work': Sobel gradient energy in the center region
    divided by energy in the border. >1 means object-like edges dominate center.
    """
    h, w = img_bgr.shape[:2]
    cx0 = int((1 - center_frac) * 0.5 * w)
    cy0 = int((1 - center_frac) * 0.5 * h)
    cx1 = w - cx0
    cy1 = h - cy0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    center_energy = mag[cy0:cy1, cx0:cx1].sum()
    border_energy = mag.sum() - center_energy
    # avoid divide-by-zero; add tiny epsilon
    return float(center_energy / (border_energy + 1e-6))


# ---------- single-image decision ----------

def prefer_hog_for_image(img_bgr, *,
                         center_frac=0.6,
                         t_sift_focus=0.60,
                         t_hog_ratio=1.50):
    """
    Returns (prefer_hog: bool, metrics: dict). Rule-of-thumb:
      - If central gradient energy dominates (hog_ratio >= t_hog_ratio) and
        SIFT keypoints are NOT strongly centered (sift_focus < t_sift_focus) -> HOG
      - If SIFT keypoints ARE strongly centered and hog_ratio is weak -> SIFT
      - Otherwise: tie-break toward HOG (simpler pipeline for this assignment).
    """
    sf = _sift_keypoint_focus(img_bgr, center_frac=center_frac)
    hr = _hog_objectness_ratio(img_bgr, center_frac=center_frac)

    # hard decisions first
    if (hr >= t_hog_ratio) and (sf < t_sift_focus):
        prefer_hog = True
    elif (hr < t_hog_ratio) and (sf >= t_sift_focus):
        prefer_hog = False
    else:
        # ambiguous zone -> slight bias to HOG for simplicity/robustness
        # you can flip this if your Task 1 clearly favored SIFT
        prefer_hog = True

    return prefer_hog, {"sift_focus": sf, "hog_ratio": hr,
                        "t_sift_focus": t_sift_focus, "t_hog_ratio": t_hog_ratio}


# ---------- dataset-level decision (sample a few images per class) ----------

def prefer_hog_for_dataset(root_dir,
                           class_names=None,
                           samples_per_class=5,
                           seed=42,
                           **kwargs):
    """
    Scans subfolders (class_names or immediate subdirs), samples up to
    `samples_per_class` images per class, averages diagnostics, and decides.
    Returns (prefer_hog: bool, summary: dict).
    """
    rng = random.Random(seed)
    if class_names is None:
        # take immediate subdirectories as classes
        class_names = [d for d in os.listdir(root_dir)
                       if os.path.isdir(os.path.join(root_dir, d))]
        class_names.sort()

    all_metrics = []
    for cls in class_names:
        pattern = os.path.join(root_dir, cls, "*")
        imgs = [p for p in glob.glob(pattern)
                if os.path.isfile(p) and p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]
        if not imgs:
            continue
        rng.shuffle(imgs)
        for p in imgs[:samples_per_class]:
            img = cv2.imread(p)
            if img is None:
                continue
            prefer_hog_img, m = prefer_hog_for_image(img, **kwargs)
            m["class"] = cls
            m["path"]  = p
            m["prefer_hog_image"] = prefer_hog_img
            all_metrics.append(m)

    if not all_metrics:
        # default to HOG if nothing could be read
        return True, {"note": "No images found/readable; defaulting to HOG."}

    # Aggregate
    sf_vals = np.array([m["sift_focus"] for m in all_metrics], dtype=np.float32)
    hr_vals = np.array([m["hog_ratio"] for m in all_metrics], dtype=np.float32)
    img_votes = np.array([m["prefer_hog_image"] for m in all_metrics], dtype=bool)

    # Majority vote across sampled images; tie-break to HOG
    prefer_hog_majority = bool(img_votes.mean() >= 0.5)

    summary = {
        "n_samples": int(len(all_metrics)),
        "mean_sift_focus": float(sf_vals.mean()),
        "mean_hog_ratio": float(hr_vals.mean()),
        "median_sift_focus": float(np.median(sf_vals)),
        "median_hog_ratio": float(np.median(hr_vals)),
        "image_level_hog_vote_fraction": float(img_votes.mean()),
        "t_sift_focus": kwargs.get("t_sift_focus", 0.60),
        "t_hog_ratio": kwargs.get("t_hog_ratio", 1.50),
        "class_names": class_names,
    }
    return prefer_hog_majority, summary

# Decide from a few representative images
img = cv2.imread("./Animals10/Bird/1.jpg")
prefer_hog, metrics = prefer_hog_for_image(img)
print("Prefer HOG?", prefer_hog, metrics)

# Or decide from the dataset root (samples a handful per class)
prefer_hog_ds, summary = prefer_hog_for_dataset("./Animals10",
                                                class_names=["Bird","Cat","Deer","Dog","Duck","Frog","Goat","Horse","Lion","Tiger"],
                                                samples_per_class=5,
                                                t_sift_focus=0.60,
                                                t_hog_ratio=1.50)
print("Prefer HOG (dataset)?", prefer_hog_ds, summary)
