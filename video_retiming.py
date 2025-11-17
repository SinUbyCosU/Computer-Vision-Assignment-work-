import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Reuse optical flow functions from run_tracker.py
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

def compute_dense_flow(img1, img2, grid_spacing=5, levels=3, win_size=9):
    
    gray1 = to_gray(img1)
    gray2 = to_gray(img2)
    
    h, w = gray1.shape
    
    # Create dense grid of points
    xs = np.arange(0, w, grid_spacing)
    ys = np.arange(0, h, grid_spacing)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
    
    # Compute flow at grid points
    flow_pts, status = pyramidal_lk(gray1, gray2, pts, levels=levels, win_size=win_size)
    
    # Create dense flow field by interpolation
    flow_u = np.zeros((h, w), dtype=np.float32)
    flow_v = np.zeros((h, w), dtype=np.float32)
    
    # Fill in computed flow values
    for i, (x, y) in enumerate(pts):
        if status[i]:
            xi, yi = int(x), int(y)
            if 0 <= xi < w and 0 <= yi < h:
                flow_u[yi, xi] = flow_pts[i, 0]
                flow_v[yi, xi] = flow_pts[i, 1]
    
    # Interpolate to fill gaps (simple linear interpolation)
    # For simplicity, we'll use cv2.resize which does bilinear interpolation
    if grid_spacing > 1:
        h_grid, w_grid = len(ys), len(xs)
        flow_u_grid = flow_u[::grid_spacing, ::grid_spacing]
        flow_v_grid = flow_v[::grid_spacing, ::grid_spacing]
        flow_u = cv2.resize(flow_u_grid, (w, h), interpolation=cv2.INTER_LINEAR)
        flow_v = cv2.resize(flow_v_grid, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return flow_u, flow_v

def warp_image(img, flow_u, flow_v, scale=1.0):
  
    h, w = img.shape[:2]
    
    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Apply flow
    x_new = x_coords + scale * flow_u
    y_new = y_coords + scale * flow_v
    
    # Warp using cv2.remap (uses bilinear interpolation)
    warped = cv2.remap(img, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return warped

def synthesize_intermediate_frame(frame1, frame2, grid_spacing=5, levels=3, win_size=9):
  
    # Compute forward flow (frame1 -> frame2)
    flow_fwd_u, flow_fwd_v = compute_dense_flow(frame1, frame2, grid_spacing, levels, win_size)
    
    # Compute backward flow (frame2 -> frame1)
    flow_bwd_u, flow_bwd_v = compute_dense_flow(frame2, frame1, grid_spacing, levels, win_size)
    
    # Warp frame1 forward by 0.5 * flow
    warped1 = warp_image(frame1, flow_fwd_u, flow_fwd_v, scale=0.5)
    
    # Warp frame2 backward by 0.5 * flow
    warped2 = warp_image(frame2, flow_bwd_u, flow_bwd_v, scale=0.5)
    
    # Blend the two warped frames
    intermediate = (warped1.astype(np.float32) + warped2.astype(np.float32)) / 2.0
    intermediate = np.clip(intermediate, 0, 255).astype(np.uint8)
    
    return intermediate, (flow_fwd_u, flow_fwd_v), (flow_bwd_u, flow_bwd_v)

def run_retiming(video_path, start_frame=50, num_pairs=15, out_dir='retiming_output', 
                 grid_spacing=8, levels=3, win_size=9):
    
    os.makedirs(out_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video: ' + video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    
    # Load frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f'Loaded {len(frames)} frames at {fps} FPS')
    
    if start_frame + num_pairs + 1 > len(frames):
        print(f'Warning: Not enough frames. Using start_frame=0')
        start_frame = 0
        num_pairs = min(num_pairs, len(frames) - 1)
    
    # Process frame pairs
    original_sequence = []
    slow_motion_sequence = []
    
    print(f'Processing {num_pairs} frame pairs starting from frame {start_frame}...')
    
    for i in range(num_pairs):
        idx = start_frame + i
        frame1 = frames[idx]
        frame2 = frames[idx + 1]
        
        print(f'  Pair {i+1}/{num_pairs}: frames {idx} -> {idx+1}')
        
        # Synthesize intermediate frame
        intermediate, flow_fwd, flow_bwd = synthesize_intermediate_frame(
            frame1, frame2, grid_spacing, levels, win_size)
        
        # Add to sequences
        original_sequence.append(frame1)
        slow_motion_sequence.extend([frame1, intermediate])
        
        # Save visualization for first pair
        if i == 0:
            # Save original frames
            cv2.imwrite(os.path.join(out_dir, 'frame_t.png'), frame1)
            cv2.imwrite(os.path.join(out_dir, 'frame_t+1.png'), frame2)
            cv2.imwrite(os.path.join(out_dir, 'frame_intermediate.png'), intermediate)
            
            # Create difference heatmaps
            # Before warping: difference between frame1 and frame2
            diff_before = np.abs(frame2.astype(np.float32) - frame1.astype(np.float32))
            diff_before_gray = np.mean(diff_before, axis=2)
            
            # After warping: difference between intermediate and actual average
            actual_avg = (frame1.astype(np.float32) + frame2.astype(np.float32)) / 2.0
            diff_after = np.abs(intermediate.astype(np.float32) - actual_avg)
            diff_after_gray = np.mean(diff_after, axis=2)
            
            # Create heatmap visualizations
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            
            axes[0, 0].imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Frame t')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Frame t+1')
            axes[0, 1].axis('off')
            
            im1 = axes[1, 0].imshow(diff_before_gray, cmap='hot')
            axes[1, 0].set_title('Difference: (Frame(t+1) - Frame(t))')
            axes[1, 0].axis('off')
            plt.colorbar(im1, ax=axes[1, 0])
            
            im2 = axes[1, 1].imshow(diff_after_gray, cmap='hot')
            axes[1, 1].set_title('Difference: (Intermediate - Average)')
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1])
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'difference_heatmaps.png'), dpi=150)
            plt.close()
            
            # Visualize flow
            flow_fwd_u, flow_fwd_v = flow_fwd
            flow_mag = np.sqrt(flow_fwd_u**2 + flow_fwd_v**2)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(flow_mag, cmap='jet')
            plt.colorbar(label='Flow Magnitude (px)')
            plt.title('Forward Optical Flow Magnitude')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'flow_magnitude.png'), dpi=150)
            plt.close()
    
    # Add last frame
    slow_motion_sequence.append(frames[start_frame + num_pairs])
    
    # Save as video
    h, w = frames[0].shape[:2]
    
    # Save slow-motion video (2x slower) - use MP4 with x264
    out_video = os.path.join(out_dir, 'slow_motion_2x.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
    for frame in slow_motion_sequence:
        writer.write(frame)
    writer.release()
    print(f'Saved slow-motion video: {out_video}')
    
    # Create GIF of a short snippet (first 10 frames of slow motion)
    try:
        import imageio
        gif_frames = slow_motion_sequence[:min(20, len(slow_motion_sequence))]
        gif_frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in gif_frames]
        gif_path = os.path.join(out_dir, 'slow_motion_snippet.gif')
        imageio.mimsave(gif_path, gif_frames_rgb, fps=fps/2, loop=0)
        print(f'Saved GIF snippet: {gif_path}')
    except ImportError:
        print('imageio not available, skipping GIF creation')
    
    print(f'\nResults saved to {out_dir}/')
    print(f'Original : {len(original_sequence)} frames')
    print(f'Slow-motion : {len(slow_motion_sequence)} frames (2x slower)')
    
    return slow_motion_sequence

if __name__ == '__main__':
    VIDEO_PATH = 'CV.mp4'
    if not os.path.exists(VIDEO_PATH):
        print(f'ERROR{VIDEO_PATH}')
        sys.exit(1)
    
    print('Starting video retiming (2x slow motion)...')
    run_retiming(VIDEO_PATH, start_frame=50, num_pairs=15, 
                 grid_spacing=8, levels=3, win_size=9)
    print('Done!')
