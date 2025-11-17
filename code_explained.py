"""
code_explained.py

A BEGINNERFRIENDLY, LINEBYLINE EXPLANATION OF THE NOTEBOOK `cv_assignment.ipynb`.

HOW TO USE THIS FILE:
1. You can run this script directly (python code_explained.py) if you have the required
   dependencies installed and the dataset placed correctly.
2. Every block of original notebook code has been rewritten here with VERY DETAILED
   comments so that even someone new to Python + Computer Vision can follow.
3. Read the comments above each section first; then look at the code.

LEGEND FOR COMMENT STYLES:
 # >>> Explanation of WHY something is done
 # (Inline) Quick meaning of that specific line
 Long multiline headers give highlevel context.

NOTE: Some long computations (like clustering for Bag of Words) are kept concise so the
focus stays on understanding, not just raw performance.
"""

# 0. IMPORTS AND BASIC SETUP

# We import all the Python libraries we need. If you are new:
#  numpy: powerful library for numerical arrays and math.
#  os, time, math, random: standard Python utilities for file paths, timing, math functions, and randomness.
#  matplotlib.pyplot: for plotting and showing images.
#  collections.Counter: for counting things (like labels) easily.
#  pathlib.Path: modern, convenient way to work with filesystem paths.
#  PIL.Image: from Pillow library, used to open image files.
# If any import fails, install the library using pip (e.g., pip install numpy pillow matplotlib scikitlearn).

import numpy as np, os, time, math, random
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from PIL import Image

print("[INFO] Imports loaded. Ready to go!\n")

# Set a random seed for reproducibility (so results are the same every run)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# 1. INTEGRAL IMAGE IMPLEMENTATION

# GOAL: Transform a normal image into an integral image (a.k.a. summedarea table).
# WHY: Once we have an integral image, we can compute the SUM of any rectangular
#      region of the original image using ONLY 4 array lookups (O(1) time), instead
#      of summing all the pixels inside that rectangle (which would be slower).
# BASIC FORMULA (for position x,y):
#   I(x,y) = image(x,y) + I(x1,y) + I(x,y1)  I(x1,y1)
# IMPORTANT: We must handle edges carefully (when x1 or y1 goes out of bounds).

# >>> This function builds the integral image using explicit loops (educational approach)
# instead of a terse numpy.cumsum oneliner, so you can see exactly how it works.
def compute_integral_image(img):
    img = np.asarray(img)              # Make sure input is a NumPy array
    h, w = img.shape                   # Height and width of the image
    integ = np.zeros((h, w), dtype=np.int64)  # Create output array (use 64bit to avoid overflow)
    # We loop over every row i and column j
    for i in range(h):
        row_sum = 0                    # Running sum for the current row so far
        for j in range(w):
            row_sum += int(img[i, j])  # Add current pixel value to row running total
            # Add the value above (integ[i1, j]) unless we're at the first row.
            # This effectively builds the 2D cumulative total.
            integ[i, j] = row_sum + (integ[i  1, j] if i > 0 else 0)
    return integ                       # Return the integral image

# >>> This helper returns the sum of pixels inside a rectangle.
# Coordinates are inclusive: (top, left) is topleft; (bottom, right) is bottomright.
# We use the integral image and inclusion–exclusion principle.
def rect_sum(integ, top, left, bottom, right):
    if top > bottom or left > right:
        return 0  # invalid rectangle => sum is zero
    # a = bottomright corner
    a = integ[bottom, right]
    # b = value just above the rectangle
    b = integ[top  1, right] if top > 0 else 0
    # c = value just left of the rectangle
    c = integ[bottom, left  1] if left > 0 else 0
    # d = value diagonally aboveleft (counted twice, so we add it back)
    d = integ[top  1, left  1] if (top > 0 and left > 0) else 0
    return a  b  c + d

# >>> Quick probabilistic test: randomly sample rectangles and compare fast result
# with a bruteforce sum over the original image.
def verify_integral(img, trials=20):
    integ = compute_integral_image(img)
    h, w = img.shape
    for _ in range(trials):
        # Pick a random valid rectangle
        t = np.random.randint(0, h)
        b = np.random.randint(t, h)
        l = np.random.randint(0, w)
        r = np.random.randint(l, w)
        fast = rect_sum(integ, t, l, b, r)        # O(1) method
        brute = img[t:b + 1, l:r + 1].sum()       # Direct summation
        if fast != brute:
            print("[ERROR] Mismatch found!", (t, l, b, r), fast, brute)
            return False
    return True

# >>> Small test matrix so we can easily check the correctness.
_test = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])  # Sum should be 45 total
_integ_test = compute_integral_image(_test)
print("[CHECK] Integral image verification passed?", verify_integral(_test, 30))
print("[CHECK] Bottomright of integral should be 45 >", _integ_test[1, 1], "\n")


# 2. HAARLIKE FEATURES (EDGE / LINE PATTERNS)

# HAAR FEATURES = simple rectangular patterns where we subtract sums of dark/light regions.
# WHY: They were famously used in the Viola–Jones face detector because they are very fast
#      to compute with integral images.
# We'll implement four types:
#   1. Vertical Edge:   Left half minus Right half
#   2. Horizontal Edge: Top half minus Bottom half
#   3. Horizontal Line: (Top band + Bottom band)  Middle band
#   4. Vertical Line:   (Left band + Right band)  Middle band
# NOTE: These patterns emphasize contrast changes and ignore uniform illumination.

# >>> Load a standard test image named cameraman.* (PNG/JPG). Put it in the SAME folder.
def load_cameraman():
    for name in ['cameraman.png', 'cameraman.jpg']:
        if Path(name).exists():
            return np.array(Image.open(name).convert('L'))  # Convert to grayscale
    raise FileNotFoundError("Please place 'cameraman.png' or 'cameraman.jpg' in this directory.")

# Load the cameraman image and build its integral image
cameraman_image = load_cameraman()
integral_cameraman = compute_integral_image(cameraman_image)
H, W = cameraman_image.shape

# We'll analyze a 50x50 region in the center just as a focus area for scanning features.
center_size = 50
region_top = H // 2  center_size // 2
region_left = W // 2  center_size // 2

# >>> Feature 1: vertical edge (split rectangle into left/right). Width must be even.
def haar_vertical_edge(int_img, top, left, height, width):
    if width % 2:  # If width not divisible by 2, pattern invalid → return 0 response
        return 0
    mid = left + width // 2  1        # Column separating left and right halves
    left_sum = rect_sum(int_img, top, left, top + height  1, mid)
    right_sum = rect_sum(int_img, top, mid + 1, top + height  1, left + width  1)
    return left_sum  right_sum        # Positive if left brighter, negative if right brighter

# >>> Feature 2: horizontal edge (split rectangle into top/bottom). Height must be even.
def haar_horizontal_edge(int_img, top, left, height, width):
    if height % 2:
        return 0
    mid = top + height // 2  1        # Row separating top and bottom halves
    top_sum = rect_sum(int_img, top, left, mid, left + width  1)
    bottom_sum = rect_sum(int_img, mid + 1, left, top + height  1, left + width  1)
    return top_sum  bottom_sum

# >>> Feature 3: horizontal line (three horizontal bands). Height must be divisible by 3.
def haar_horizontal_line(int_img, top, left, height, width):
    if height % 3:
        return 0
    band = height // 3                 # Height of each band
    y1 = top + band  1                # End row of first band
    y2 = top + 2 * band  1            # End row of second band
    white1 = rect_sum(int_img, top, left, y1, left + width  1)
    black_mid = rect_sum(int_img, y1 + 1, left, y2, left + width  1)
    white2 = rect_sum(int_img, y2 + 1, left, top + height  1, left + width  1)
    return white1  black_mid + white2 # White  black + white pattern

# >>> Feature 4: vertical line (three vertical bands). Width must be divisible by 3.
def haar_vertical_line(int_img, top, left, height, width):
    if width % 3:
        return 0
    band = width // 3
    x1 = left + band  1               # End column of first band
    x2 = left + 2 * band  1           # End column of second band
    white1 = rect_sum(int_img, top, left, top + height  1, x1)
    black_mid = rect_sum(int_img, top, x1 + 1, top + height  1, x2)
    white2 = rect_sum(int_img, top, x2 + 1, top + height  1, left + width  1)
    return white1  black_mid + white2

# >>> Slide (scan) a feature function across a region to find the strongest response.
# We test all topleft positions where the feature of size (fh, fw) fits inside the region.
def slide_feature(int_img, feature_fn, region_top, region_left, region_size, fh, fw):
    values = []  # store all responses
    locs = []    # store corresponding (row, col) positions
    for r in range(region_top, region_top + region_size  fh + 1):
        for c in range(region_left, region_left + region_size  fw + 1):
            val = feature_fn(int_img, r, c, fh, fw)
            values.append(val)
            locs.append((r, c))
    if not values:
        return None, None, None  # nothing scanned
    arr = np.array(values)
    # We use absolute value to ignore sign (brightdark vs darkbright both interesting)
    best_index = np.argmax(np.abs(arr))
    return arr[best_index], locs[best_index], arr

# Define which features to test
feature_list = [
    ("Vertical Edge", haar_vertical_edge),
    ("Horizontal Edge", haar_horizontal_edge),
    ("Horizontal Line", haar_horizontal_line),
    ("Vertical Line", haar_vertical_line),
]

# We'll use a 24x24 feature window (must satisfy divisibility for some patterns)
filter_height = filter_width = 24
strongest = {}
for name, fn in feature_list:
    val, loc, all_values = slide_feature(integral_cameraman, fn,
                                         region_top, region_left,
                                         center_size, filter_height, filter_width)
    if loc is not None:
        strongest[name] = (val, loc)

print("[INFO] Strongest Haarlike feature responses (value, (row,col)):")
for k, v in strongest.items():
    print("   ", k, v)
print()

# >>> Visualize the strongest location for each feature (draw a yellow box).
plt.figure(figsize=(14, 4))
for i, (name, (val, (r, c))) in enumerate(strongest.items()):
    ax = plt.subplot(1, len(strongest), i + 1)
    ax.imshow(cameraman_image, cmap='gray')
    ax.add_patch(plt.Rectangle((c, r), filter_width, filter_height,
                               ec='yellow', fc='none', lw=2))
    ax.set_title(name)
    ax.axis('off')
plt.tight_layout()
plt.show()


# 3. DATASET LOADING (KTHTIPS TEXTURE DATA)

# GOAL: Load many texture images from folders, convert to grayscale, resize to a fixed size.
# WHY RESIZE: Machine learning models need consistent feature lengths.
# We choose 64x64 = 4096 pixels for a balance between detail and speed.
# DATASET EXPECTATION: A folder 'kthdataset' containing 10 subfolders (one per texture class).

# Path to dataset root (change if needed)
dataset_dir = Path('kthdataset')
if not dataset_dir.exists():
    raise RuntimeError("Dataset folder 'kthdataset' was not found. Place it next to this script.")

# Valid image file extensions we will accept
valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

# Gather all image paths recursively (rglob walks through subfolders)
image_paths = sorted([p for p in dataset_dir.rglob('*') if p.suffix.lower() in valid_exts])
if not image_paths:
    raise RuntimeError("No images found inside 'kthdataset'.")

# Discover unique class names from parent folder names
class_names = sorted({p.parent.name for p in image_paths})
if len(class_names) < 10:
    print("[WARN] Fewer than 10 classes detected (maybe dataset incomplete):", len(class_names))
class_to_idx = {name: i for i, name in enumerate(class_names)}  # Map class name → numeric label

# >>> Simple grayscale loader using Pillow.
def load_gray(path):
    return np.array(Image.open(path).convert('L'))

# Manual nearestneighbor resize to avoid extra dependencies.
# This shows you exactly how resizing works at a basic level.
def resize_nearest(img, size=(64, 64)):
    h, w = img.shape
    H, W = size
    out = np.zeros(size, dtype=img.dtype)
    for i in range(H):
        src_i = min(int(i * h / H), h  1)   # Map new row i to original row
        row = img[src_i]
        for j in range(W):
            src_j = min(int(j * w / W), w  1)  # Map new col j to original col
            out[i, j] = row[src_j]
    return out

# Load and process every image.
all_images = []
all_labels = []
for p in image_paths:
    g = load_gray(p)
    g = resize_nearest(g, (64, 64))
    all_images.append(g)
    all_labels.append(class_to_idx[p.parent.name])

# Convert lists to arrays for machine learning usage.
X = np.stack(all_images)          # Shape: (num_samples, 64, 64)
y = np.array(all_labels)          # Shape: (num_samples,)
print(f"[INFO] Loaded dataset: X.shape={X.shape}, num_classes={len(class_names)}\n")


# 4. STRATIFIED 70/30 TRAIN/TEST SPLIT (EXACT)

# GOAL: Split the data so exactly 70% of ALL samples go to training, 30% to test,
#       while preserving each class's proportion as closely as possible.
# STEPS:
#   1. Count how many samples each class has.
#   2. Compute desired_train = 0.7 * count for each class.
#   3. Take floors (integer part) first; record fractional leftovers.
#   4. Distribute leftover slots to classes with biggest fractional parts until we hit the total target.
#   5. Ensure test set isn't empty for a class (if possible).

rng = np.random.default_rng(RANDOM_SEED)   # NumPy's Generator for reproducible shuffling
all_indices = np.arange(len(y))            # Indices [0, 1, 2, ..., N1]
train_indices = []
test_indices = []

# Target absolute number of training examples
target_train_total = int(round(0.7 * len(y)))
unique_classes = np.unique(y)

# Count samples per class
class_counts = {cls: (y == cls).sum() for cls in unique_classes}
# Desired (floating) number of train samples per class
desired_train = {cls: class_counts[cls] * 0.7 for cls in unique_classes}
# Floor allocation (base)
base_train = {cls: int(math.floor(desired_train[cls])) for cls in unique_classes}
# Fractional remainders
fractions = {cls: desired_train[cls]  base_train[cls] for cls in unique_classes}
# Current total allocated to training so far
current_total = sum(base_train.values())
# Difference from target (positive => need to add some; negative => remove some)
diff = target_train_total  current_total

# Adjust counts according to largest fractional remainders if we need more
if diff > 0:
    # Sort classes by descending fractional remainder
    for cls, _ in sorted(fractions.items(), key=lambda x: x[1], reverse=True):
        if diff == 0:
            break
        # Only increase if it won't leave zero test samples (except class size ==1)
        if base_train[cls] < class_counts[cls]  1 or class_counts[cls] == 1:
            base_train[cls] += 1
            diff = 1
elif diff < 0:
    # Need to remove some allocations: take from smallest fractional remainders first
    for cls, _ in sorted(fractions.items(), key=lambda x: x[1]):
        if diff == 0:
            break
        if base_train[cls] > 0:
            base_train[cls] = 1
            diff += 1

# Make sure each class with size > 1 has at least 1 test sample
for cls in unique_classes:
    if class_counts[cls] > 1 and base_train[cls] == class_counts[cls]:
        base_train[cls] = 1

# Now actually pick which indices go to train vs test
for cls in unique_classes:
    cls_inds = all_indices[y == cls]
    rng.shuffle(cls_inds)          # Randomize order
    k_train = base_train[cls]
    train_part = cls_inds[:k_train]
    test_part = cls_inds[k_train:]
    train_indices.extend(train_part.tolist())
    test_indices.extend(test_part.tolist())

# Convert to NumPy arrays and sort (sorting = deterministic order, helpful for reproducibility)
train_indices = np.array(train_indices); train_indices.sort()
test_indices = np.array(test_indices); test_indices.sort()

# Create final train/test splits
X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

actual_train_ratio = X_train.shape[0] / X.shape[0]
actual_test_ratio = X_test.shape[0] / X.shape[0]
print(f"[INFO] Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}  (Train%={actual_train_ratio*100:.2f}  Test%={actual_test_ratio*100:.2f})\n")

# Safety checks
assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
assert abs(actual_train_ratio  0.70) < 1e6
assert abs(actual_test_ratio  0.30) < 1e6


# 5. FEATURE EXTRACTION + LINEAR SVM CLASSIFIERS

# We'll build four different feature types for each image:
#   A. Raw Pixels        → Flatten the 64x64 image into a 4096D vector.
#   B. LBP (Local Binary Patterns) → For each pixel compare neighbors to center.
#   C. HOG (Histogram of Oriented Gradients) → Capture edge orientations.
#   D. BoW (Bag of Visual Words) → Cluster patches and use histogram of visual word usage.
# For classification we use a Linear SVM (Support Vector Machine) from scikitlearn.
# We scale (standardize) the features so each dimension roughly has mean 0 and std 1.

try:
    from sklearn.svm import SVC
except Exception as e:
    raise ImportError("scikitlearn is required for SVM parts. Install with: pip install scikitlearn") from e

# >>> Utility: standardize features (zero mean, unit variance) so SVM training is stable.
def standardize(train, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1  # Avoid dividebyzero
    return (train  mean) / std, (test  mean) / std

# 
# A. RAW PIXELS
# 
X_train_raw = X_train.reshape(len(X_train), 1).astype(np.float32) / 255.0  # Scale to [0,1]
X_test_raw = X_test.reshape(len(X_test), 1).astype(np.float32) / 255.0
X_train_raw_s, X_test_raw_s = standardize(X_train_raw, X_test_raw)
start = time.time(); svm_raw = SVC(kernel='linear').fit(X_train_raw_s, y_train); train_time_raw = time.time()  start
start = time.time(); pred_raw = svm_raw.predict(X_test_raw_s); predict_time_raw = time.time()  start
acc_raw = (pred_raw == y_test).mean()
print(f"[RAW ] Accuracy={acc_raw*100:.2f}% train_time={train_time_raw:.2f}s predict_time={predict_time_raw:.2f}s")

# 
# B. LBP (Local Binary Patterns)
# 
# IDEA: For each pixel (excluding border), look at 8 neighbors. If neighbor >= center, write 1 else 0.
# Combine the 8 bits into a number 0..255. Then build a histogram counting occurrences of each pattern.

def lbp_image(img):
    h, w = img.shape
    out = np.zeros((h  2, w  2), dtype=np.uint8)  # Output smaller by 1pixel border
    for i in range(1, h  1):
        for j in range(1, w  1):
            c = img[i, j]  # center pixel
            code = 0
            # We check neighbors clockwise starting topleft. Each comparison adds a bit.
            code |= (img[i  1, j  1] >= c) << 7
            code |= (img[i  1, j    ] >= c) << 6
            code |= (img[i  1, j + 1] >= c) << 5
            code |= (img[i    , j + 1] >= c) << 4
            code |= (img[i + 1, j + 1] >= c) << 3
            code |= (img[i + 1, j    ] >= c) << 2
            code |= (img[i + 1, j  1] >= c) << 1
            code |= (img[i    , j  1] >= c) << 0
            out[i  1, j  1] = code
    return out

def lbp_histogram(img):
    lbp = lbp_image(img)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e9)  # Normalize so sum=1
    return hist

X_train_lbp = np.array([lbp_histogram(im) for im in X_train])
X_test_lbp = np.array([lbp_histogram(im) for im in X_test])
X_train_lbp_s, X_test_lbp_s = standardize(X_train_lbp, X_test_lbp)
start = time.time(); svm_lbp = SVC(kernel='linear').fit(X_train_lbp_s, y_train); train_time_lbp = time.time()  start
start = time.time(); pred_lbp = svm_lbp.predict(X_test_lbp_s); predict_time_lbp = time.time()  start
acc_lbp = (pred_lbp == y_test).mean()
print(f"[LBP ] Accuracy={acc_lbp*100:.2f}% train_time={train_time_lbp:.2f}s predict_time={predict_time_lbp:.2f}s")

# 
# C. HOG (Histogram of Oriented Gradients)
# 
# IDEA: Compute gradient (edge direction and strength) for each pixel. Divide image into cells
# (e.g., 8x8). For each cell, make a histogram counting how strong edges are for each of several
# orientation bins (e.g., 9 bins covering 0..180 degrees). Optionally normalize groups of cells
# (blocks) to reduce illumination effects.

def hog_features(img, cell=8, block=2, bins=9):
    img = img.astype(np.float32)
    gy, gx = np.gradient(img)          # Estimate vertical and horizontal gradients
    mag = np.sqrt(gx * gx + gy * gy)    # Magnitude (edge strength)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180  # Angle in [0,180)
    h, w = img.shape
    nc_y = h // cell                   # Number of cells along height
    nc_x = w // cell                   # Number of cells along width
    hist = np.zeros((nc_y, nc_x, bins), dtype=np.float32)
    bin_width = 180 / bins             # Degrees covered by each bin
    # Build percell orientation histograms
    for cy in range(nc_y):
        for cx in range(nc_x):
            cell_mag = mag[cy * cell:(cy + 1) * cell, cx * cell:(cx + 1) * cell]
            cell_ang = ang[cy * cell:(cy + 1) * cell, cx * cell:(cx + 1) * cell]
            for m, a in zip(cell_mag.ravel(), cell_ang.ravel()):
                bin_index = int(a // bin_width) % bins
                hist[cy, cx, bin_index] += m
    # Block normalization (group cells in overlapping 2x2 blocks)
    blocks_y = nc_y  block + 1
    blocks_x = nc_x  block + 1
    features = []
    for by in range(blocks_y):
        for bx in range(blocks_x):
            vec = hist[by:by + block, bx:bx + block, :].ravel()
            norm = np.linalg.norm(vec) + 1e6
            features.append(vec / norm)
    return np.concatenate(features)

X_train_hog = np.array([hog_features(im) for im in X_train])
X_test_hog = np.array([hog_features(im) for im in X_test])
X_train_hog_s, X_test_hog_s = standardize(X_train_hog, X_test_hog)
start = time.time(); svm_hog = SVC(kernel='linear').fit(X_train_hog_s, y_train); train_time_hog = time.time()  start
start = time.time(); pred_hog = svm_hog.predict(X_test_hog_s); predict_time_hog = time.time()  start
acc_hog = (pred_hog == y_test).mean()
print(f"[HOG ] Accuracy={acc_hog*100:.2f}% train_time={train_time_hog:.2f}s predict_time={predict_time_hog:.2f}s")

# 
# D. BAG OF VISUAL WORDS (BoW)
# 
# IDEA: Extract many small patches from images. Cluster them (kmeans) to create a "visual vocabulary".
# Each patch is assigned to the nearest centroid (visual word). For an image, we count how frequently
# each word appears (histogram). That histogram becomes the feature vector.

# Extract patches with given patch size and stride (step)
def extract_patches(img, patch=8, step=8):
    patches = []
    h, w = img.shape
    for i in range(0, h  patch + 1, step):
        for j in range(0, w  patch + 1, step):
            patches.append(img[i:i + patch, j:j + patch].ravel())
    return patches

all_patches = []
max_patches_per_image = 30  # Limit to control clustering time
for im in X_train:
    ps = extract_patches(im, patch=8, step=8)
    if len(ps) > max_patches_per_image:
        ps = ps[:max_patches_per_image]
    all_patches.extend(ps)

all_patches = np.array(all_patches)
num_visual_words = 60  # Number of clusters (size of vocabulary)
# Initialize centroids by random sampling of existing patches
rng_local = np.random.default_rng(RANDOM_SEED)
centroids = all_patches[rng_local.choice(len(all_patches), num_visual_words, replace=False)].astype(np.float32)

# Simple kmeans clustering (fixed iterations for simplicity)
for iteration in range(15):
    # Compute distances between each centroid and each patch (vectorized)
    dists = np.sqrt(((centroids[:, None]  all_patches[None, :]) ** 2).sum(axis=2))
    labels = np.argmin(dists, axis=0)  # Assign each patch to closest centroid
    new_centroids = []
    for ci in range(num_visual_words):
        pts = all_patches[labels == ci]
        if len(pts) > 0:
            new_centroids.append(pts.mean(axis=0))
        else:
            new_centroids.append(centroids[ci])  # Keep old centroid if no points assigned
    new_centroids = np.array(new_centroids)
    # If centroids barely change, we can stop early
    if np.allclose(new_centroids, centroids):
        break
    centroids = new_centroids

# Turn an image into a normalized histogram of visual word occurrences
def bow_histogram(im):
    ps = extract_patches(im, patch=8, step=8)
    if not ps:
        return np.zeros(num_visual_words)
    P = np.array(ps, dtype=np.float32)
    d = np.sqrt(((centroids[:, None]  P[None, :]) ** 2).sum(axis=2))
    lbl = np.argmin(d, axis=0)
    hist, _ = np.histogram(lbl, bins=num_visual_words, range=(0, num_visual_words))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e9)
    return hist

X_train_bow = np.array([bow_histogram(im) for im in X_train])
X_test_bow = np.array([bow_histogram(im) for im in X_test])
X_train_bow_s, X_test_bow_s = standardize(X_train_bow, X_test_bow)
start = time.time(); svm_bow = SVC(kernel='linear').fit(X_train_bow_s, y_train); train_time_bow = time.time()  start
start = time.time(); pred_bow = svm_bow.predict(X_test_bow_s); predict_time_bow = time.time()  start
acc_bow = (pred_bow == y_test).mean()
print(f"[BoW ] Accuracy={acc_bow*100:.2f}% train_time={train_time_bow:.2f}s predict_time={predict_time_bow:.2f}s\n")


# 6. SUMMARY TABLE

# We collect all results and print them sorted by accuracy so you can see which feature
# worked best in this run.

results = [
    ("Raw SVM", acc_raw, train_time_raw, predict_time_raw),
    ("LBP SVM", acc_lbp, train_time_lbp, predict_time_lbp),
    ("HOG SVM", acc_hog, train_time_hog, predict_time_hog),
    ("BoW SVM", acc_bow, train_time_bow, predict_time_bow)
]
# Sort by accuracy descending
results.sort(key=lambda x: x[1], reverse=True)

print("SUMMARY (sorted by accuracy):")
print(f"{'Method':<10} {'Acc%':>7} {'Train(s)':>10} {'Predict(s)':>11}")
print('' * 44)
for name, acc, tr, pr in results:
    print(f"{name:<10} {acc*100:7.2f} {tr:10.2f} {pr:11.2f}")

print("\nObservations:")
print(" Raw Pixels: Large dimensionality; linear SVM helps but may overfit if dataset small.")
print(" LBP: Robust to monotonic lighting changes; good for texture micropatterns.")
print(" HOG: Captures oriented edges and structure; often strong for shapelike textures.")
print(" BoW: Abstracts local patches into a distribution; loses spatial arrangement but can generalize.")

print("\n[DONE] All stages completed successfully.")

# End of file
