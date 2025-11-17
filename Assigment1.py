# import OpenCV just for reading images (not for processing)
import cv2 as cv  # just for imread, not for any processing, only reading the image from disk
import numpy as np  # numpy is like the backbone for arrays and math in python, we use it for all image math
import matplotlib.pyplot as plt  # matplotlib is for plotting images and histograms, like showing stuff visually
import os  # os is for file and folder stuff, like making directories
from PIL import Image  # PIL is used here just for saving images as pdf, not for processing

# show an image with matplotlib (and save if needed)
def show_image(title, img, cmap=None, save_folder=None, save_counter=None):
    plt.figure()  # this makes a new figure window for each image
    plt.title(title)  # set the title of the window to whatever we pass
    if cmap:
        plt.imshow(img, cmap=cmap)  # if cmap is given, show as grayscale (like for gray images)
    else:
        # convert BGR to RGB for matplotlib using numpy, not OpenCV (assignment rule!)
        if img.ndim == 3 and img.shape[2] == 3:  # check if it's a color image
            img_rgb = img[..., ::-1]  # this reverses the last axis, so BGR becomes RGB (matplotlib wants RGB)
            plt.imshow(img_rgb)  # show the color image
        else:
            plt.imshow(img)  # just in case it's not 3 channel, just show it as is
    plt.axis('off')  # don't show axis ticks or numbers, just the image
    if save_folder is not None and save_counter is not None:  # if we want to save the image
        filename = f"{save_counter:02d}_{title.replace(' ', '_').replace('(', '').replace(')', '')}.png"  # make a filename with counter and title
        plt.savefig(os.path.join(save_folder, filename), bbox_inches='tight', pad_inches=0.1)  # save the image to the folder
    plt.show()  # actually display the image

# path to the image
image_path = 'img.jpeg'  # this is the name of the image file, should be in the same folder
img = cv.imread(image_path)  # read the image using OpenCV, returns a numpy array (BGR format)
if img is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found.")  # if image is not found, stop and complain

# function to make grayscale (my own, not OpenCV)
def to_grayscale(src):
    h, w = src.shape[:2]  # get the height and width of the image
    gray = np.zeros((h, w), dtype=np.uint8)  # make an empty grayscale image (all zeros)
    for y in range(h):  # go over every row
        for x in range(w):  # go over every column
            b, g, r = src[y, x]  # get the blue, green, red values (OpenCV is BGR)
            gray[y, x] = int(0.2989 * r + 0.5870 * g + 0.1140 * b)  # use the formula for grayscale (weighted sum)
    return gray  # return the grayscale image

gray_img = to_grayscale(img)  # make grayscale version of the original image

# make output folders if not there
output_img_folder = os.path.join(os.getcwd(), 'output_images')  # make a path for output images folder
output_pdf_folder = os.path.join(os.getcwd(), 'output_pdf')  # make a path for output pdf folder
os.makedirs(output_img_folder, exist_ok=True)  # make the folder if it doesn't exist
os.makedirs(output_pdf_folder, exist_ok=True)  # same for pdf folder

img_save_counter = 1  # this is for naming files in order, so we know which is which

def plot_histogram(img, title="Histogram"):
    if img.ndim == 2:  # if it's a grayscale image
        plt.hist(img.ravel(), 256, [0, 256], color='black')  # plot the histogram in black
    elif img.ndim == 3 and img.shape[2] == 3:  # if it's a color image
        for idx, color in enumerate(('b', 'g', 'r')):  # loop over each channel
            plt.hist(img[:, :, idx].ravel(), 256, [0, 256], color=color, alpha=0.5, label=f'{color.upper()} channel')  # plot each channel
        plt.legend()  # show which color is which
    else:
        raise ValueError("Unsupported image shape for histogram.")  # if not 2D or 3D, error
    plt.title(title)  # set the title
    plt.xlabel('Pixel Intensity')  # x axis label
    plt.ylabel('Frequency')  # y axis label

# show and save histograms
plt.figure()  # make a new figure
plot_histogram(img, "Original (channels separated)")  # plot histogram for original color image
plt.tight_layout()  # make it look nice
plt.savefig(os.path.join(output_img_folder, f"{img_save_counter:02d}_Histogram_Original.png"), bbox_inches='tight', pad_inches=0.1)  # save
plt.close()  # close the figure
img_save_counter += 1  # increment counter
plt.figure()
plot_histogram(gray_img, "Grayscale")  # plot histogram for grayscale image
plt.tight_layout()
plt.savefig(os.path.join(output_img_folder, f"{img_save_counter:02d}_Histogram_Grayscale.png"), bbox_inches='tight', pad_inches=0.1)
plt.close()
img_save_counter += 1

# save hists for each channel
red = img[:, :, 2]  # red channel is at index 2 (OpenCV is BGR)
green = img[:, :, 1]  # green channel is at index 1
blue = img[:, :, 0]  # blue channel is at index 0


def add_gaussian_noise(src, mean=0, var=20):
    sigma = var ** 0.5  # standard deviation is sqrt of variance
    noise = np.random.normal(mean, sigma, src.shape)  # make random noise with normal distribution
    noisy = src.astype(np.float32) + noise  # add noise to the image (convert to float first)
    return np.clip(noisy, 0, 255).astype(np.uint8)  # clip to 0-255 and convert back to uint8

noisy_img = add_gaussian_noise(img)  # make a noisy version of the original image
show_image('Noisy Image', noisy_img, save_folder=output_img_folder, save_counter=img_save_counter)  # show and save noisy image
img_save_counter += 1
# Explicitly show R, G, B channel images (not just histograms)
red_img = np.zeros_like(img)  # make an empty image like the original
red_img[:, :, 2] = red        # set only the red channel, others stay zero
show_image("Red Channel Image", red_img, save_folder=output_img_folder, save_counter=img_save_counter)  # show and save
img_save_counter += 1

green_img = np.zeros_like(img)  # same for green
green_img[:, :, 1] = green
show_image("Green Channel Image", green_img, save_folder=output_img_folder, save_counter=img_save_counter)
img_save_counter += 1

blue_img = np.zeros_like(img)   # same for blue
blue_img[:, :, 0] = blue
show_image("Blue Channel Image", blue_img, save_folder=output_img_folder, save_counter=img_save_counter)
img_save_counter += 1

# noisy image
noisy_img = add_gaussian_noise(img)  # make a noisy version of the original image
show_image('Noisy Image', noisy_img, save_folder=output_img_folder, save_counter=img_save_counter)  # show and save noisy image
img_save_counter += 1

gray_noisy = to_grayscale(noisy_img)  # make grayscale version of noisy image
show_image('Grayscale Noisy Image', gray_noisy, cmap='gray', save_folder=output_img_folder, save_counter=img_save_counter)  # show and save
img_save_counter += 1

# noisy hists
plt.figure()
plot_histogram(noisy_img, "Noisy Image (channels separated)")  # plot histogram for noisy color image
plt.savefig(os.path.join(output_img_folder, f"{img_save_counter:02d}_Histogram_Noisy.png"), bbox_inches='tight', pad_inches=0.1)
plt.close()
img_save_counter += 1
plt.figure()
plot_histogram(gray_noisy, "Noisy Grayscale")  # plot histogram for noisy grayscale
plt.savefig(os.path.join(output_img_folder, f"{img_save_counter:02d}_Histogram_Noisy_Grayscale.png"), bbox_inches='tight', pad_inches=0.1)
plt.close()
img_save_counter += 1
for channel_img, channel_name in zip([noisy_img[:, :, 2], noisy_img[:, :, 1], noisy_img[:, :, 0]], ["Red_Noisy", "Green_Noisy", "Blue_Noisy"]):
    plt.figure()
    plot_histogram(channel_img, f"{channel_name} Channel (Noisy)")  # plot for each noisy channel
    plt.savefig(os.path.join(output_img_folder, f"{img_save_counter:02d}_Histogram_{channel_name}.png"), bbox_inches='tight', pad_inches=0.1)
    plt.close()
    img_save_counter += 1

def mean_filter(img, ksize): # mean filter for any kernel size
    pad = ksize // 2  # how much to pad on each side (half the kernel size)
    if img.ndim == 2: # if grayscale
        padded = np.pad(img, pad, mode='reflect') # pad the image so we can filter edges (reflect so border is handled)
        filtered = np.zeros_like(img) # make an empty image to store result
        for y in range(img.shape[0]): # go over every row
            for x in range(img.shape[1]): # go over every column
                region = padded[y:y+ksize, x:x+ksize] # get the region for the filter (this is the window)
                filtered[y, x] = np.mean(region) # take the mean of the region (average all values in window)
        return filtered.astype(np.uint8) # return as uint8 (image type)
    else: # if color image
        filtered = np.zeros_like(img) # make empty image
        for c in range(3): # for each channel (R, G, B)
            filtered[:, :, c] = mean_filter(img[:, :, c], ksize) # filter each channel separately
        return filtered

# Sobel kernels for 3x3, 5x5, 7x7 (these are hardcoded, you can look up their math)
SOBEL_KERNELS = {
    3: (np.array([[1,0,-1],[2,0,-2],[1,0,-1]]), np.array([[1,2,1],[0,0,0],[-1,-2,-1]])), # classic 3x3
    5: (np.array([[2,1,0,-1,-2],[3,2,0,-2,-3],[4,3,0,-3,-4],[3,2,0,-2,-3],[2,1,0,-1,-2]]),
        np.array([[2,3,4,3,2],[1,2,3,2,1],[0,0,0,0,0],[-1,-2,-3,-2,-1],[-2,-3,-4,-3,-2]])), # 5x5
    7: (np.array([[3,2,1,0,-1,-2,-3],[4,3,2,0,-2,-3,-4],[5,4,3,0,-3,-4,-5],[6,5,4,0,-4,-5,-6],[5,4,3,0,-3,-4,-5],[4,3,2,0,-2,-3,-4],[3,2,1,0,-1,-2,-3]]),
        np.array([[3,4,5,6,5,4,3],[2,3,4,5,4,3,2],[1,2,3,4,3,2,1],[0,0,0,0,0,0,0],[-1,-2,-3,-4,-3,-2,-1],[-2,-3,-4,-5,-4,-3,-2],[-3,-4,-5,-6,-5,-4,-3]])) # 7x7
}

def sobel_edge(img, ksize=3, direction='both'):
    Kx, Ky = SOBEL_KERNELS[ksize] # get kernels for this size (tuple)
    pad = ksize // 2 # how much to pad (half the kernel size)
    padded = np.pad(img, pad, mode='reflect') # pad the image (so we don't lose border info)
    gx = np.zeros_like(img, dtype=np.float32) # for x gradient (horizontal edges)
    gy = np.zeros_like(img, dtype=np.float32) # for y gradient (vertical edges)
    for y in range(img.shape[0]): # for every row
        for x in range(img.shape[1]): # for every col
            region = padded[y:y+ksize, x:x+ksize] # get region (window)
            gx[y, x] = np.sum(region * Kx) # convolve with x kernel (dot product)
            gy[y, x] = np.sum(region * Ky) # convolve with y kernel
    if direction == 'horizontal': # just x
        edge = np.abs(gx) # only horizontal edges
    elif direction == 'vertical': # just y
        edge = np.abs(gy) # only vertical edges
    else: # both
        edge = np.hypot(gx, gy) # magnitude (sqrt(gx^2 + gy^2))
    return np.clip(edge, 0, 255).astype(np.uint8) # keep in range (0-255)

# Laplacian kernels for 3x3, 5x5, 7x7 (again, hardcoded)
LAPLACIAN_KERNELS = {
    3: np.array([[0,1,0],[1,-4,1],[0,1,0]]), # 3x3
    5: np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]]), # 5x5
    7: np.array([[0,0,0,1,0,0,0],[0,0,1,2,1,0,0],[0,1,2,3,2,1,0],[1,2,3,-24,3,2,1],[0,1,2,3,2,1,0],[0,0,1,2,1,0,0],[0,0,0,1,0,0,0]]) # 7x7
}

def laplacian_edge(img, ksize=3):
    kernel = LAPLACIAN_KERNELS[ksize] # pick kernel (based on size)
    pad = ksize // 2 # how much to pad
    padded = np.pad(img, pad, mode='reflect') # pad the image
    edge = np.zeros_like(img, dtype=np.float32) # make empty
    for y in range(img.shape[0]): # for every row
        for x in range(img.shape[1]): # for every col
            region = padded[y:y+ksize, x:x+ksize] # get region
            edge[y, x] = np.sum(region * kernel) # convolve (sum of elementwise product)
    edge = np.abs(edge) # take abs (edges can be negative)
    return np.clip(edge, 0, 255).astype(np.uint8) # keep in range

def gaussian_kernel(size, sigma=1):
    ax = np.arange(-size//2 + 1., size//2 + 1.)  # axis (centered at 0)
    xx, yy = np.meshgrid(ax, ax)  # grid (2D)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))  # gaussian formula (bell curve)
    return kernel / np.sum(kernel)  # normalize (so sum is 1)

def apply_filter(img, kernel):
    pad = kernel.shape[0] // 2  # padding (half kernel size)
    padded = np.pad(img, pad, mode='reflect')  # pad (so we don't lose border)
    filtered = np.zeros_like(img, dtype=np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            region = padded[y:y+kernel.shape[0], x:x+kernel.shape[1]]  # get region (window)
            filtered[y, x] = np.sum(region * kernel)  # filter (sum of elementwise product)
    return filtered

def non_max_suppression(grad_mag, grad_dir):
    Z = np.zeros_like(grad_mag, dtype=np.float32)  # output (same shape)
    angle = grad_dir * 180. / np.pi  # to degrees (from radians)
    angle[angle < 0] += 180  # fix angles (make all positive)
    for i in range(1, grad_mag.shape[0]-1): # skip border
        for j in range(1, grad_mag.shape[1]-1):
            q = 255
            r = 255
            # figure out which neighbors to compare based on angle
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = grad_mag[i, j+1]
                r = grad_mag[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = grad_mag[i+1, j-1]
                r = grad_mag[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = grad_mag[i+1, j]
                r = grad_mag[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = grad_mag[i-1, j-1]
                r = grad_mag[i+1, j+1]
            # keep only if local max
            if (grad_mag[i,j] >= q) and (grad_mag[i,j] >= r):
                Z[i,j] = grad_mag[i,j]
            else:
                Z[i,j] = 0
    return Z

def hysteresis(img, low, high):
    M, N = img.shape  # size (rows, cols)
    res = np.zeros((M,N), dtype=np.uint8)  # output (all zeros)
    strong = 255  # strong edge (white)
    weak = 75  # weak edge (gray)
    strong_i, strong_j = np.where(img >= high)  # strongs (above high threshold)
    weak_i, weak_j = np.where((img <= high) & (img >= low))  # weaks (between low and high)
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    for i in range(1, M-1): # skip border
        for j in range(1, N-1):
            if res[i,j] == weak:
                # if any neighbor is strong, make this strong
                if ((res[i+1, j-1] == strong) or (res[i+1, j] == strong) or (res[i+1, j+1] == strong)
                    or (res[i, j-1] == strong) or (res[i, j+1] == strong)
                    or (res[i-1, j-1] == strong) or (res[i-1, j] == strong) or (res[i-1, j+1] == strong)):
                    res[i,j] = strong
                else:
                    res[i,j] = 0 # else, suppress
    return res

def canny_edge(img, low_thresh=None, high_thresh=None, gaussian_size=7, sigma=2):
    if img.ndim == 3:
        img = to_grayscale(img)  # make gray if not (so we work on 2D)
    img = img.astype(np.float32)  # float for math (avoid overflow)
    kernel = gaussian_kernel(gaussian_size, sigma)  # smooth (make kernel)
    smoothed = apply_filter(img, kernel)  # blur (apply kernel)
    Kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])  # sobel x (3x3, for canny)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])  # sobel y
    Gx = apply_filter(smoothed, Kx)  # grad x (horizontal)
    Gy = apply_filter(smoothed, Ky)  # grad y (vertical)
    grad_mag = np.hypot(Gx, Gy)  # mag (sqrt(Gx^2 + Gy^2))
    grad_mag = grad_mag / (grad_mag.max() + 1e-8) * 255  # normalize (avoid div by zero)
    grad_dir = np.arctan2(Gy, Gx)  # dir (angle)
    nms = non_max_suppression(grad_mag, grad_dir)  # thin edges (keep only local max)
    if low_thresh is None or high_thresh is None:
        high_thresh = np.percentile(nms, 90)  # auto threshold (top 10% as strong)
        low_thresh = high_thresh * 0.4 # weak is 40% of strong
    edge = hysteresis(nms, low_thresh, high_thresh)  # final edges (connect weak to strong)
    return np.clip(edge, 0, 255).astype(np.uint8)  # keep in range

# show original and gray
show_image('Original Image', img, save_folder=output_img_folder, save_counter=img_save_counter)  # show orig (color)
img_save_counter += 1
show_image('Grayscale Image', gray_img, cmap='gray', save_folder=output_img_folder, save_counter=img_save_counter)  # show gray
img_save_counter += 1

# try different filters and edge detectors
for k in [3, 5, 7]: # for all 3 sizes (assignment wants all)
    filtered_clean = mean_filter(gray_img, k)  # mean filter clean (no noise)
    filtered_noisy = mean_filter(gray_noisy, k)  # mean filter noisy (with noise)
    show_image(f'Mean Filter {k}x{k} (Clean)', filtered_clean, cmap='gray', save_folder=output_img_folder, save_counter=img_save_counter) # show mean filtered
    img_save_counter += 1
    show_image(f'Mean Filter {k}x{k} (Noisy)', filtered_noisy, cmap='gray', save_folder=output_img_folder, save_counter=img_save_counter) # show mean filtered noisy
    img_save_counter += 1
    sobel_clean = sobel_edge(filtered_clean, ksize=k)  # sobel clean (variable size)
    sobel_noisy = sobel_edge(filtered_noisy, ksize=k)  # sobel noisy
    show_image(f'Sobel Edge {k}x{k} (Clean)', sobel_clean, cmap='gray', save_folder=output_img_folder, save_counter=img_save_counter) # show sobel
    img_save_counter += 1
    show_image(f'Sobel Edge {k}x{k} (Noisy)', sobel_noisy, cmap='gray', save_folder=output_img_folder, save_counter=img_save_counter) # show sobel noisy
    img_save_counter += 1
    sobel_h_clean = sobel_edge(filtered_clean, ksize=k, direction='horizontal')  # sobel h (horizontal only)
    sobel_v_clean = sobel_edge(filtered_clean, ksize=k, direction='vertical')  # sobel v (vertical only)
    show_image(f'Sobel Horizontal {k}x{k} (Clean)', sobel_h_clean, cmap='gray', save_folder=output_img_folder, save_counter=img_save_counter) # show sobel h
    img_save_counter += 1
    show_image(f'Sobel Vertical {k}x{k} (Clean)', sobel_v_clean, cmap='gray', save_folder=output_img_folder, save_counter=img_save_counter) # show sobel v
    img_save_counter += 1
    lap_clean = laplacian_edge(filtered_clean, ksize=k)  # laplacian clean (variable size)
    lap_noisy = laplacian_edge(filtered_noisy, ksize=k)  # laplacian noisy
    show_image(f'Laplacian Edge {k}x{k} (Clean)', lap_clean, cmap='gray', save_folder=output_img_folder, save_counter=img_save_counter) # show laplacian
    img_save_counter += 1
    show_image(f'Laplacian Edge {k}x{k} (Noisy)', lap_noisy, cmap='gray', save_folder=output_img_folder, save_counter=img_save_counter) # show laplacian noisy
    img_save_counter += 1

# canny edge (my own)
canny_clean = canny_edge(gray_img)  # canny clean (no noise)
canny_noisy = canny_edge(gray_noisy)  # canny noisy
show_image('Canny Edge (Clean)', canny_clean, cmap='gray', save_folder=output_img_folder, save_counter=img_save_counter)  # show canny clean
img_save_counter += 1 # keeping tabs on image saves
show_image('Canny Edge (Noisy)', canny_noisy, cmap='gray', save_folder=output_img_folder, save_counter=img_save_counter)  # show canny noisy
img_save_counter += 1

# save all images as pdf (so you can see all results in one file)
from glob import glob  # for finding files (glob is like a file search)
image_files = sorted(glob(os.path.join(output_img_folder, '*.png')))  # get all pngs (sort so order is nice)
images = [Image.open(f).convert('RGB') for f in image_files]  # open them (PIL to open and convert to RGB)
if images:
    pdf_path = os.path.join(output_pdf_folder, 'output_images.pdf')  # pdf path (where to save)
    images[0].save(pdf_path, save_all=True, append_images=images[1:])  # save pdf (all images in one pdf)



