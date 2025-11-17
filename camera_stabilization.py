import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Reuse optical flow functions
def gaussian_kernel(size=5, sigma=1):
    k = np.arange(-(size//2), size//2+1)
    g = np.exp(-0.5*(k**2)/(sigma**2))
    g = g / g.sum()
    return np.outer(g, g)

def pyr_down(img):
    kernel = gaussian_kernel(5, 1.0)
    blurred = convolve2d(img, kernel, mode='same', boundary='symm')
    return blurred[::2, ::2]

def build_image_pyramid(img, levels=3):
    pyr = [img.astype(np.float32)]
    for i in range(1, levels):
        pyr.append(pyr_down(pyr[i-1]))
    return pyr[::-1]

def compute_gradients(img1, img2):
    Ix = 0.5 * (np.roll(img1, -1, axis=1) - np.roll(img1, 1, axis=1))
    Iy = 0.5 * (np.roll(img1, -1, axis=0) - np.roll(img1, 1, axis=0))
    It = img2 - img1
    return Ix, Iy, It

def bilinear_interpolate(img, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

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
    h, w = img1.shape
    half = win_size // 2
    Ix, Iy, It_full = compute_gradients(img1, img2)
    flows = np.zeros((pts.shape[0], 2), dtype=np.float32)
    status = np.zeros((pts.shape[0],), dtype=np.uint8)

    for i, p in enumerate(pts):
        x, y = p
        if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
            status[i] = 0
            continue

        xs = np.arange(x-half, x+half+1)
        ys = np.arange(y-half, y+half+1)
        X, Y = np.meshgrid(xs, ys)
        Xf = X.flatten().astype(np.float32)
        Yf = Y.flatten().astype(np.float32)

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

def moving_average(data, window_size):
    if window_size < 1:
        return data
    window_size = min(window_size, len(data))
    smoothed = np.zeros_like(data)
    half = window_size // 2
    
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half + 1)
        smoothed[i] = np.mean(data[start:end])
    
    return smoothed

def stabilize_video(video_path, out_dir='stabilization_output', grid_spacing=20, 
                    smooth_window=15, levels=3, win_size=9):
   
    os.makedirs(out_dir, exist_ok=True)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video: ' + video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f'Loaded {len(frames)} frames at {fps} FPS')
    
    h, w = frames[0].shape[:2]
    
    # Create sparse grid of points for flow estimation
    xs = np.arange(grid_spacing, w - grid_spacing, grid_spacing)
    ys = np.arange(grid_spacing, h - grid_spacing, grid_spacing)
    X, Y = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)

    
    # Compute per-frame translation
    translations = []  # (dx, dy) for each frame
    
    for i in range(len(frames) - 1):
        gray1 = to_gray(frames[i])
        gray2 = to_gray(frames[i + 1])
        
        # Compute flow at grid points
        flow, status = pyramidal_lk(gray1, gray2, grid_pts, levels=levels, win_size=win_size)
        
        # Use median of good flow vectors as camera translation
        good_flow = flow[status.astype(bool)]
        if len(good_flow) > 0:
            dx = np.median(good_flow[:, 0])
            dy = np.median(good_flow[:, 1])
        else:
            dx, dy = 0.0, 0.0
        
        translations.append([dx, dy])
        
        if (i + 1) % 50 == 0:
            print(f'  Processed {i+1}/{len(frames)-1} frames')
    
    translations = np.array(translations)
    
    # Accumulate to get camera path
    camera_path = np.cumsum(translations, axis=0)
    camera_path = np.vstack([[0, 0], camera_path])  # Add initial position
    
    print(f'Smoothing camera path with window size {smooth_window}...')
    # Smooth the camera path
    smooth_path_x = moving_average(camera_path[:, 0], smooth_window)
    smooth_path_y = moving_average(camera_path[:, 1], smooth_window)
    smooth_path = np.column_stack([smooth_path_x, smooth_path_y])
    
    # Compute stabilization transforms (difference between actual and smooth path)
    stabilization_transforms = camera_path - smooth_path
    
    # Apply stabilization
    print('Stabilizing frames...')
    stabilized_frames = []
    
    for i, frame in enumerate(frames):
        dx, dy = stabilization_transforms[i]
        
        # Create translation matrix
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        
        # Warp frame
        stabilized = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        stabilized_frames.append(stabilized)
    
    # Save example frame with flow visualization (first frame)
    print('visualizations...')
    example_frame = frames[0].copy()
    gray1 = to_gray(frames[0])
    gray2 = to_gray(frames[1])
    flow, status = pyramidal_lk(gray1, gray2, grid_pts, levels=levels, win_size=win_size)
    
    arrow_scale = 3.0
    for i, pt in enumerate(grid_pts):
        if status[i]:
            x, y = int(pt[0]), int(pt[1])
            dx, dy = flow[i]
            end_x = int(x + dx * arrow_scale)
            end_y = int(y + dy * arrow_scale)
            cv2.arrowedLine(example_frame, (x, y), (end_x, end_y), (0, 255, 0), 1, tipLength=0.3)
            cv2.circle(example_frame, (x, y), 3, (0, 0, 255), -1)
    
    cv2.imwrite(os.path.join(out_dir, 'sparse_flow_example.png'), example_frame)
    
    # Plot raw vs smoothed translation
    times = np.arange(len(camera_path)) / fps
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(times, camera_path[:, 0], 'b-', alpha=0.5, label='Raw X')
    ax1.plot(times, smooth_path[:, 0], 'r-', linewidth=2, label='Smoothed X')
    ax1.set_ylabel('X Translation (pixels)')
    ax1.set_title('Camera Path: Horizontal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, camera_path[:, 1], 'b-', alpha=0.5, label='Raw Y')
    ax2.plot(times, smooth_path[:, 1], 'r-', linewidth=2, label='Smoothed Y')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Y Translation (px)')
    ax2.set_title('Camera Path: Vertical')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'translation_plot.png'), dpi=150)
    plt.close()
    
    # Save videos
    print('Saving videos...')
    
    # Use MP4 format with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Original video
    out_original = os.path.join(out_dir, 'original.mp4')
    writer = cv2.VideoWriter(out_original, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    
    # Stabilized video
    out_stabilized = os.path.join(out_dir, 'stabilized.mp4')
    writer = cv2.VideoWriter(out_stabilized, fourcc, fps, (w, h))
    for frame in stabilized_frames:
        writer.write(frame)
    writer.release()
    
    # Side-by-side comparison video
    out_comparison = os.path.join(out_dir, 'comparison_sidebyside.mp4')
    writer = cv2.VideoWriter(out_comparison, fourcc, fps, (w * 2, h))
    for orig, stab in zip(frames, stabilized_frames):
        # Add labels
        orig_labeled = orig.copy()
        stab_labeled = stab.copy()
        cv2.putText(orig_labeled, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2)
        cv2.putText(stab_labeled, 'Stabilized', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        combined = np.hstack([orig_labeled, stab_labeled])
        writer.write(combined)
    writer.release()
    
    print(f'Saved original video: {out_original}')
    print(f'Saved stabilized video: {out_stabilized}')
    print(f'Saved comparison video: {out_comparison}')
    
    # Create GIF snippet
    try:
        import imageio
        # Take a snippet with motion (frames 50-80)
        start_idx = min(50, len(frames) - 30)
        end_idx = start_idx + 30
        
        gif_frames = []
        for i in range(start_idx, end_idx):
            orig_labeled = frames[i].copy()
            stab_labeled = stabilized_frames[i].copy()
            cv2.putText(orig_labeled, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
            cv2.putText(stab_labeled, 'Stabilized', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            combined = np.hstack([orig_labeled, stab_labeled])
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            gif_frames.append(combined_rgb)
        
        gif_path = os.path.join(out_dir, 'comparison.gif')
        imageio.mimsave(gif_path, gif_frames, fps=fps, loop=0)
        print(f'Saved comparison GIF: {gif_path}')
    except ImportError:
        print('imageio not available, skipping GIF creation')
    
    print(f'\nStabilization complete! results {out_dir}/')
    print(f'Camera motion range: X=[{camera_path[:, 0].min():.1f}, {camera_path[:, 0].max():.1f}], '
          f'Y=[{camera_path[:, 1].min():.1f}, {camera_path[:, 1].max():.1f}]')
    
    return stabilized_frames

if __name__ == '__main__':
    VIDEO_PATH = 'CV2.mp4'
    if not os.path.exists(VIDEO_PATH):
        print(f'ERROR:  {VIDEO_PATH}')
        sys.exit(1)

    print('Starting translation stabilization...')
    # Use larger smoothing window for stronger stabilization
    stabilize_video(VIDEO_PATH, grid_spacing=40, smooth_window=25, levels=2, win_size=7)
    print('Done!')
