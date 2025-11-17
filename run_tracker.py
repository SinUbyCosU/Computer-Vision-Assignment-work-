
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Optical flow functions
def gaussian_kernel(size=5, sigma=1):
    k = np.arange(-(size//2), size//2+1)
    g = np.exp(-0.5*(k**2)/(sigma**2))
    g=g/g.sum()
    return np.outer(g,g)

def pyr_down(img):
    kernel = gaussian_kernel(5, 1.0)
    blurred = convolve2d(img, kernel, mode='same', boundary='symm')
    return blurred[::2, ::2]

def build_image_pyramid(img, levels=3):
    pyr= [img.astype(np.float32)]
    for i in range(1, levels):
        pyr.append(pyr_down(pyr[i-1]))
    return pyr[::-1]

def compute_gradients(img1, img2):
    Ix=0.5*(np.roll(img1, -1, axis=1) - np.roll(img1, 1, axis=1))
    Iy=0.5*(np.roll(img1, -1, axis=0) - np.roll(img1, 1, axis=0))
    It= img2- img1
    return Ix, Iy, It

def bilinear_interpolate(img, x, y):
    x0= np.floor(x).astype(int)
    x1=x0+1
    y0=np.floor(y).astype(int)
    y1=y0+1

    x0_clipped = np.clip(x0, 0, img.shape[1]-1)
    x1_clipped = np.clip(x1, 0, img.shape[1]-1)
    y0_clipped = np.clip(y0, 0, img.shape[0]-1)
    y1_clipped = np.clip(y1, 0, img.shape[0]-1)

    Ia = img[y0_clipped, x0_clipped]
    Ib = img[y1_clipped, x0_clipped]
    Ic = img[y0_clipped, x1_clipped]
    Id = img[y1_clipped, x1_clipped]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return Ia*wa + Ib*wb + Ic*wc + Id*wd

def lk_single_level(img1, img2, pts, win_size=9, eig_thresh=1e-4):
    h, w= img1.shape
    half= win_size // 2
    Ix, Iy, It_full = compute_gradients(img1, img2)
    flows= np.zeros((pts.shape[0],2), dtype=np.float32)
    status = np.zeros((pts.shape[0],), dtype=np.uint8)

    for i, p in enumerate(pts):
        x,y=p
        if x- half< 0 or x+ half >=w or y - half < 0 or y + half >= h:
            status[i] = 0
            continue

        xs= np.arange(x-half, x+half+1)
        ys= np.arange(y-half, y+half+1)
        X, Y= np.meshgrid(xs,ys)
        Xf= X.flatten().astype(np.float32)
        Yf= Y.flatten().astype(np.float32)

        Ix_w = bilinear_interpolate(Ix, Xf, Yf)
        Iy_w = bilinear_interpolate(Iy, Xf, Yf)
        It_w = bilinear_interpolate(It_full, Xf, Yf)

        A = np.vstack([Ix_w, Iy_w]).T
        ATA = A.T @ A
        eigvals = np.linalg.eigvals(ATA)
        if np.min(eigvals) < eig_thresh:
            status[i] = 0
            continue
        b = -It_w
        nu = np.linalg.lstsq(A, b, rcond=None)[0]
        flows[i] = nu[:2]
        status[i] = 1

    return flows, status

def pyramidal_lk(img1, img2, pts, levels=3, win_size=9):
    pyr1 = build_image_pyramid(img1, levels)
    pyr2 = build_image_pyramid(img2, levels)

    scale = 1.0 / (2 ** (levels-1))
    pts_level = pts * scale
    flow = np.zeros_like(pts_level)

    for lvl in range(levels):
        I1 = pyr1[lvl]
        I2 = pyr2[lvl]
        flows, status = lk_single_level(I1, I2, pts_level + flow, win_size=win_size)
        flow = flow + flows
        if lvl < levels-1:
            flow = flow * 2.0
            pts_level = pts_level * 2.0

    return flow, status

def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return img.astype(np.float32)

def detect_features(img, max_corners=2000, quality=0.0001, min_dist=2):
    # Use Shi-Tomasi (goodFeaturesToTrack) with smaller minDist and lower quality to get denser corners
    corners = cv2.goodFeaturesToTrack(img.astype(np.uint8),
                                      maxCorners=max_corners,
                                      qualityLevel=quality,
                                      minDistance=min_dist,
                                      blockSize=3,
                                      useHarrisDetector=False,
                                      k=0.04)
    if corners is None:
        return np.zeros((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)

def run(video_path, roi=None, out_dir='output', levels=3, win_size=9, fps_override=None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video: ' + video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = fps_override if fps_override is not None else 30.0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f'Loaded {len(frames)} frames at {fps} FPS')
    
    if len(frames) < 2:
        raise RuntimeError('Video must have at least 2 frames')

    h, w = frames[0].shape[:2]
    if roi is None:
        # Try to load ROI from roi_coordinates.txt if it exists
        roi_file = 'roi_coordinates.txt'
        if os.path.exists(roi_file):
            print(f'Loading ROI from {roi_file}')
            coords = []
            with open(roi_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        x, y = map(int, line.split(','))
                        coords.append([x, y])
            
            if len(coords) >= 3:
                trapezoid_pts = np.array(coords, dtype=np.int32)
                # Use bounding box for feature detection
                xs = [pt[0] for pt in coords]
                ys = [pt[1] for pt in coords]
                roi = (min(xs), min(ys), max(xs), max(ys))
                print(f'Using custom polygon ROI with {len(coords)} points')
            else:
                print(f'Invalid ROI file, using automatic detection')
                trapezoid_pts = None
        else:
            # Use intensity gradients to find road edges
            first_gray = to_gray(frames[0])
            
            # Define y-levels to sample
            roi_top_y = int(h*0.25)
            roi_bottom_y = int(h*0.95)
            
            # Find left and right edges at top (using horizontal gradient)
            top_row = first_gray[roi_top_y, :]
            top_grad = np.abs(np.gradient(top_row))
            # Smooth gradient
            kernel_size = 15
            top_grad_smooth = np.convolve(top_grad, np.ones(kernel_size)/kernel_size, mode='same')
            # Find edges in left and right halves
            mid = w // 2
            left_peaks = np.where(top_grad_smooth[:mid] > np.percentile(top_grad_smooth[:mid], 75))[0]
            right_peaks = np.where(top_grad_smooth[mid:] > np.percentile(top_grad_smooth[mid:], 75))[0] + mid
            roi_top_x0 = left_peaks[-1] if len(left_peaks) > 0 else int(w*0.3)
            roi_top_x1 = right_peaks[0] if len(right_peaks) > 0 else int(w*0.7)
            
            # Find left and right edges at bottom
            bottom_row = first_gray[roi_bottom_y, :]
            bottom_grad = np.abs(np.gradient(bottom_row))
            bottom_grad_smooth = np.convolve(bottom_grad, np.ones(kernel_size)/kernel_size, mode='same')
            left_peaks = np.where(bottom_grad_smooth[:mid] > np.percentile(bottom_grad_smooth[:mid], 75))[0]
            right_peaks = np.where(bottom_grad_smooth[mid:] > np.percentile(bottom_grad_smooth[mid:], 75))[0] + mid
            roi_bottom_x0 = left_peaks[-1] if len(left_peaks) > 0 else int(w*0.1)
            roi_bottom_x1 = right_peaks[0] if len(right_peaks) > 0 else int(w*0.9)
            
            # Store as trapezoid points
            trapezoid_pts = np.array([
                [roi_top_x0, roi_top_y],
                [roi_top_x1, roi_top_y],
                [roi_bottom_x1, roi_bottom_y],
                [roi_bottom_x0, roi_bottom_y]
            ], dtype=np.int32)
            # For rectangular detection region, use bounding box
            roi = (min(roi_bottom_x0, roi_top_x0), roi_top_y, 
                   max(roi_bottom_x1, roi_top_x1), roi_bottom_y)
    else:
        trapezoid_pts = None
    
    x0, y0, x1, y1 = roi

    gray = [to_gray(f) for f in frames]
    first_roi = gray[0][y0:y1, x0:x1]
    pts = detect_features(first_roi)
    if pts.shape[0] == 0:
        print('No features found in ROI')
        return [], [], None
    pts[:, 0] += x0
    pts[:, 1] += y0

    print(f'Tracking {pts.shape[0]} feature points...')

    pts_current = pts.copy()
    speeds = []
    overlays = []
    overlay_frames_idx = [0, len(frames)//2, max(0, len(frames)-2)]

    for t in range(len(frames)-1):
        I1, I2 = gray[t], gray[t+1]
        flow, status = pyramidal_lk(I1, I2, pts_current, levels=levels, win_size=win_size)
        mag = np.linalg.norm(flow, axis=1)
        good = status.astype(bool)
        speeds.append(float(np.median(mag[good]) * fps) if np.any(good) else 0.0)
        pts_current = pts_current + flow

        if t in overlay_frames_idx:
            vis = frames[t].copy()
            # Draw arrows scaled for visibility and a small circle at the start point
            arrow_scale = 4.0
            for i, p in enumerate(pts_current):
                if not good[i]:
                    continue
                x, y = int(p[0]), int(p[1])
                fx, fy = flow[i]
                # scale float flow for visibility
                dx, dy = int(np.round(fx * arrow_scale)), int(np.round(fy * arrow_scale))
                end_pt = (x + dx, y + dy)
                # draw line and arrowhead
                cv2.arrowedLine(vis, (x, y), end_pt, (0,255,0), 1, tipLength=0.2)
                # draw small circle at feature point
                cv2.circle(vis, (x, y), 2, (0,255,0), -1)
            # Draw trapezoid ROI if defined, else rectangle
            if trapezoid_pts is not None:
                cv2.polylines(vis, [trapezoid_pts], isClosed=True, color=(255,0,0), thickness=2)
            else:
                cv2.rectangle(vis, (x0, y0), (x1, y1), (255,0,0), 2)
            out_path = os.path.join(out_dir, f'overlay_{t:03d}.png')
            cv2.imwrite(out_path, vis)
            overlays.append(out_path)

    times = np.arange(len(speeds)) / fps
    plt.figure(figsize=(8,3))
    plt.plot(times, speeds)
    plt.xlabel('time (s)')
    plt.ylabel('speed proxy (pixels/sec)')
    plt.title('Median per-frame flow magnitude * FPS')
    plot_path = os.path.join(out_dir, 'speed_proxy.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f'Wrote {len(overlays)} overlay images')
    print(f'Wrote plot: {plot_path}')
    return speeds, overlays, plot_path

if __name__ == '__main__':
    VIDEO_PATH = 'CV.mp4'
    if not os.path.exists(VIDEO_PATH):
        print(f'ERROR: {VIDEO_PATH}')
        print('Place your video in this folder and name it CV.mp4')
        sys.exit(1)
    
    print('Starting vehicle speed tracker...')
    run(VIDEO_PATH, roi=None, out_dir='output')
    print('Done! Check the output folder for results.')
