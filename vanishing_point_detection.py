import os
import sys
import numpy as np
import cv2  # Only for I/O (reading/writing video and images)
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter as nd_gaussian_filter

def manual_canny(img, low_threshold=50, high_threshold=150):
    """
    Simplified Canny edge detection using NumPy/SciPy.
    Faster version with vectorized operations.
    """
    # 1. Gaussian smoothing
    blurred = gaussian_filter(img.astype(np.float32), sigma=1.4)
    
    # 2. Compute gradients using Sobel operators
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    grad_x = convolve(blurred, sobel_x)
    grad_y = convolve(blurred, sobel_y)
    
    # Gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Simplified non-maximum suppression (vectorized)
    # Just use gradient magnitude with thresholding
    edges = np.zeros_like(magnitude, dtype=np.uint8)
    edges[magnitude >= high_threshold] = 255
    edges[(magnitude >= low_threshold) & (magnitude < high_threshold)] = 128
    
    # Simple hysteresis: connect weak edges to strong ones
    from scipy.ndimage import binary_dilation
    strong_edges = (edges == 255)
    weak_edges = (edges == 128)
    
    # Dilate strong edges and keep weak edges connected to them
    dilated_strong = binary_dilation(strong_edges, iterations=1)
    connected_weak = weak_edges & dilated_strong
    
    final_edges = np.zeros_like(edges)
    final_edges[strong_edges | connected_weak] = 255
    
    return final_edges

def manual_hough_lines(edges, rho_res=2, theta_res=np.pi/90, threshold=40, 
                       min_line_length=50, max_line_gap=10):
    """
    Simplified Hough Transform - faster version with coarser resolution.
    """
    h, w = edges.shape
    
    # Get edge pixels (subsample for speed)
    edge_points = np.argwhere(edges > 0)
    
    if len(edge_points) == 0:
        return None
    
    # Subsample edge points for speed
    if len(edge_points) > 2000:
        indices = np.random.choice(len(edge_points), 2000, replace=False)
        edge_points = edge_points[indices]
    
    # Hough space parameters (coarser for speed)
    diag_len = int(np.sqrt(h**2 + w**2))
    rhos = np.arange(-diag_len, diag_len, rho_res)
    thetas = np.arange(0, np.pi, theta_res)
    
    # Accumulator
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint32)
    
    # Voting (vectorized where possible)
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    
    for y, x in edge_points:
        rho_values = x * cos_thetas + y * sin_thetas
        rho_indices = np.searchsorted(rhos, rho_values)
        rho_indices = np.clip(rho_indices, 0, len(rhos)-1)
        
        for t_idx, rho_idx in enumerate(rho_indices):
            accumulator[rho_idx, t_idx] += 1
    
    # Find peaks (top N strongest lines)
    flat_acc = accumulator.flatten()
    top_indices = np.argpartition(flat_acc, -20)[-20:]  # Top 20 lines
    top_indices = top_indices[flat_acc[top_indices] > threshold]
    
    lines = []
    for idx in top_indices:
        rho_idx, theta_idx = np.unravel_index(idx, accumulator.shape)
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        
        # Convert to line endpoints
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        x1 = np.clip(x1, 0, w-1)
        y1 = np.clip(y1, 0, h-1)
        x2 = np.clip(x2, 0, w-1)
        y2 = np.clip(y2, 0, h-1)
        
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length >= min_line_length:
            lines.append([[x1, y1, x2, y2]])
    
    return lines if len(lines) > 0 else None

def line_intersection(line1, line2):
    """
    Find intersection point of two lines.
    Line format: (x1, y1, x2, y2)
    Returns: (x, y) or None if parallel
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    
    if abs(denom) < 1e-6:  # Parallel lines
        return None
    
    px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
    py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
    
    return (px, py)

def filter_lines_by_angle(lines, min_angle=10, max_angle=80):
    """
    Filter lines to keep those likely converging to vanishing point.
    For road/hallway scenes, keep moderately angled lines.
    """
    if lines is None:
        return []
    
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle from horizontal in degrees
        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(abs(dy), abs(dx)) * 180 / np.pi
        
        # Calculate line length
        length = np.sqrt(dx**2 + dy**2)
        
        # Keep lines that:
        # 1. Are not too horizontal (angle > min_angle)
        # 2. Are not perfectly vertical (angle < max_angle)  
        # 3. Have reasonable length
        if min_angle < angle < max_angle and length > 40:
            filtered.append(line[0])
    
    return filtered

def compute_intersections_weighted(lines, img_shape,
                                   x_bounds=(0.2, 0.8), y_bounds=(0.0, 0.4)):
    """
    Compute all pairwise line intersections and a weight for each based on
    centrality and how high (towards the top) the intersection is.

    Returns:
        intersections: Nx2 array of (x, y)
        weights: Nx array of weights in [0, 1]
    """
    h, w = img_shape[:2]
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    intersections = []
    weights = []

    if lines is None or len(lines) < 2:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            vp = line_intersection(lines[i], lines[j])
            if vp is None:
                continue
            x, y = vp
            if (x_min * w) < x < (x_max * w) and (y_min * h) < y < (y_max * h):
                # Centrality weight (1 at center, 0 at edges) with stronger emphasis on center
                x_center_w = 1.0 - abs(x - w / 2) / (w / 2)
                x_center_w = max(0.0, x_center_w) ** 2.0
                # Upper weight (1 at top of ROI, 0 at bottom of ROI)
                y_upper_w = 1.0 - (y - (y_min * h)) / ((y_max - y_min) * h)
                wgt = max(0.0, x_center_w) * max(0.0, y_upper_w)
                intersections.append((x, y))
                weights.append(wgt)

    if len(intersections) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.array(intersections, dtype=np.float32), np.array(weights, dtype=np.float32)

class TemporalVPAccumulator:
    """
    Maintains a temporal heatmap of vanishing point likelihood over a bounded ROI.
    Each frame's intersections are splatted into a 2D accumulator with decay.
    """
    def __init__(self, frame_shape, x_bounds=(0.2, 0.8), y_bounds=(0.0, 0.45),
                 bin_size=4, decay=0.95, blur_sigma=0.8, center_bias=0.12):
        h, w = frame_shape[:2]
        self.w = w
        self.h = h
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.decay = decay
        self.bin_size = bin_size
        self.blur_sigma = blur_sigma

        x_min_px = int(x_bounds[0] * w)
        x_max_px = int(x_bounds[1] * w)
        y_min_px = int(y_bounds[0] * h)
        y_max_px = int(y_bounds[1] * h)
        self.x_min_px = x_min_px
        self.y_min_px = y_min_px
        self.grid_w = max(1, (x_max_px - x_min_px) // bin_size)
        self.grid_h = max(1, (y_max_px - y_min_px) // bin_size)

        self.acc = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # Precompute a horizontal center prior (Gaussian along x)
        cx = self.grid_w / 2.0
        sigma = max(1.0, self.grid_w * center_bias)
        x_coords = np.arange(self.grid_w, dtype=np.float32)
        prior_1d = np.exp(-0.5 * ((x_coords - cx) / sigma) ** 2)
        self.center_prior = np.tile(prior_1d[None, :], (self.grid_h, 1)).astype(np.float32)

    def _to_bin(self, x, y):
        bx = int((x - self.x_min_px) / self.bin_size)
        by = int((y - self.y_min_px) / self.bin_size)
        return bx, by

    def update(self, intersections, weights=None):
        # Decay previous heatmap
        self.acc *= self.decay

        if intersections is None or len(intersections) == 0:
            return

        if weights is None or len(weights) != len(intersections):
            weights = np.ones((len(intersections),), dtype=np.float32)

        # Splat each intersection into the accumulator
        for (x, y), wgt in zip(intersections, weights):
            bx, by = self._to_bin(x, y)
            if 0 <= bx < self.grid_w and 0 <= by < self.grid_h:
                self.acc[by, bx] += float(wgt)

        # Light blur to spread support locally
        if self.blur_sigma and self.blur_sigma > 0:
            self.acc = nd_gaussian_filter(self.acc, sigma=self.blur_sigma)

    def get_vp(self, refine_window=3):
        if np.all(self.acc <= 0):
            return None, 0.0
        # Apply center prior to gently pull VP toward center horizontally
        weighted_acc = self.acc * (self.center_prior + 1e-6)
        by, bx = np.unravel_index(np.argmax(weighted_acc), weighted_acc.shape)
        peak_val = float(weighted_acc[by, bx])

        # Optional local centroid refinement
        y0 = max(0, by - refine_window)
        y1 = min(self.grid_h, by + refine_window + 1)
        x0 = max(0, bx - refine_window)
        x1 = min(self.grid_w, bx + refine_window + 1)
        patch = weighted_acc[y0:y1, x0:x1]
        if patch.size > 0 and patch.sum() > 0:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            cy = (yy * patch).sum() / patch.sum()
            cx = (xx * patch).sum() / patch.sum()
        else:
            cy, cx = by, bx

        # Map back to pixel coordinates
        vp_x = self.x_min_px + (cx + 0.5) * self.bin_size
        vp_y = self.y_min_px + (cy + 0.5) * self.bin_size
        return (float(vp_x), float(vp_y)), peak_val

def estimate_vanishing_point_ransac(lines, img_shape, iterations=1000, threshold=15):
    """
    Estimate vanishing point with weighted clustering.
    For road/hallway scenes, VP should be in upper-center region.
    """
    if len(lines) < 2:
        return None
    
    h, w = img_shape[:2]
    
    # Collect all pairwise intersections with weights
    intersections = []
    weights = []
    
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            vp = line_intersection(lines[i], lines[j])
            if vp is not None:
                x, y = vp
                # VP should be in upper portion and roughly centered
                # Accept points in range: X [20%, 80%], Y [0, 40%]
                if 0.2*w < x < 0.8*w and 0 < y < 0.4*h:
                    intersections.append((x, y))
                    
                    # Weight by centrality and height
                    # Higher points (lower y) and more centered get more weight
                    x_center_weight = 1.0 - abs(x - w/2) / (w/2)
                    y_upper_weight = 1.0 - y / (0.4*h)  # Now using 40% upper region
                    weights.append(x_center_weight * y_upper_weight)
    
    if len(intersections) < 3:
        return None
    
    intersections = np.array(intersections)
    weights = np.array(weights)
    
    # Use weighted average instead of median
    total_weight = np.sum(weights)
    vp_x = np.sum(intersections[:, 0] * weights) / total_weight
    vp_y = np.sum(intersections[:, 1] * weights) / total_weight
    
    # Count inliers
    distances = np.sqrt((intersections[:, 0] - vp_x)**2 + 
                       (intersections[:, 1] - vp_y)**2)
    inliers = np.sum(distances < threshold * 5)
    
    # Need at least 3 inliers
    if inliers < 3:
        return None
    
    return (vp_x, vp_y, inliers)

def process_frame(frame, canny_low=30, canny_high=120, 
                  hough_threshold=30, min_line_length=35, max_line_gap=10,
                  ransac_iterations=500, ransac_threshold=15):
    """
    Process single frame to detect vanishing point.
    Uses manual Canny and Hough Transform implementations.
    
    Returns:
        vp: (x, y, inliers) or None
        lines: detected lines
        edges: edge image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Manual Canny edge detection
    edges = manual_canny(gray, low_threshold=canny_low, high_threshold=canny_high)
    
    # Manual Hough line detection
    lines = manual_hough_lines(edges, threshold=hough_threshold,
                               min_line_length=min_line_length,
                               max_line_gap=max_line_gap)
    
    # Filter lines by angle (keep diagonal lines for road/hallway scenes)
    filtered_lines = filter_lines_by_angle(lines, min_angle=15, max_angle=80)
    
    # Estimate vanishing point using RANSAC
    vp = None
    if len(filtered_lines) >= 2:
        vp = estimate_vanishing_point_ransac(filtered_lines, frame.shape,
                                              iterations=ransac_iterations,
                                              threshold=ransac_threshold)
    
    return vp, filtered_lines, edges

def draw_vanishing_point_overlay(frame, vp, lines, show_all_lines=True, label_prefix='VP'):
    """
    Draw detected lines and vanishing point on frame.
    """
    overlay = frame.copy()
    
    # Draw all detected lines
    if show_all_lines and lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), 
                    (0, 255, 0), 1, cv2.LINE_AA)
    
    # Draw vanishing point
    if vp is not None:
        # vp may be (x, y, support) or (x, y)
        if isinstance(vp, (list, tuple)) and len(vp) >= 2:
            vp_x, vp_y = vp[0], vp[1]
            support = vp[2] if len(vp) > 2 else None
        else:
            vp_x, vp_y, support = vp, None, None
        vp_x, vp_y = int(vp_x), int(vp_y)
        
        # Draw crosshair at vanishing point
        cv2.drawMarker(overlay, (vp_x, vp_y), (0, 0, 255), 
                      cv2.MARKER_CROSS, 30, 3)
        
        # Draw circle
        cv2.circle(overlay, (vp_x, vp_y), 15, (0, 0, 255), 2)
        
        # Draw lines from vanishing point to detected lines (radial pattern)
        if lines is not None and len(lines) > 0:
            for line in lines[:min(10, len(lines))]:  # Draw to first 10 lines
                x1, y1, x2, y2 = line
                midpoint = (int((x1+x2)/2), int((y1+y2)/2))
                cv2.line(overlay, (vp_x, vp_y), midpoint, 
                        (255, 0, 0), 1, cv2.LINE_AA)
        
        # Add text label
        if support is not None:
            text = f'{label_prefix}: ({vp_x}, {vp_y}) S: {support:.1f}'
        else:
            text = f'{label_prefix}: ({vp_x}, {vp_y})'
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(overlay, 'VP: Not Detected', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    return overlay

def run_vanishing_point_detection(
    video_path,
    out_dir='vanishing_output',
    save_every_n=10,
    x_bounds=(0.2, 0.8),
    y_bounds=(0.0, 0.5),
    canny_low=30,
    canny_high=120,
    hough_threshold=30,
    min_line_length=35,
    max_line_gap=10,
    ransac_iterations=500,
    ransac_threshold=15,
):
    """
    Process video and detect vanishing point in each frame.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video: ' + video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f'Loaded video: {total_frames} frames at {fps} FPS, {width}x{height}')
    
    # Storage for vanishing points (video-aggregated)
    vp_history = []
    frame_indices = []

    # Temporal accumulator over upper-center ROI
    vp_acc = TemporalVPAccumulator(
        (height, width),
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        bin_size=4,
        decay=0.96,
        blur_sigma=0.8,
    )
    
    # Process frames
    frame_idx = 0
    processed_frames = []
    
    # Use MP4 format with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = os.path.join(out_dir, 'vanishing_point_detection.mp4')
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
    
    print('Processing frames...')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect edges/lines and per-frame VP (for reference)
        per_vp, lines, edges = process_frame(
            frame,
            canny_low=canny_low,
            canny_high=canny_high,
            hough_threshold=hough_threshold,
            min_line_length=min_line_length,
            max_line_gap=max_line_gap,
            ransac_iterations=ransac_iterations,
            ransac_threshold=ransac_threshold,
        )

        # Update temporal accumulator with all intersections from this frame
        filtered_lines = lines if lines is not None else []
        inters, wts = compute_intersections_weighted(
            filtered_lines, frame.shape, x_bounds=x_bounds, y_bounds=y_bounds
        )
        vp_acc.update(inters, wts)

        # Get current video-aggregated VP
        video_vp, support = vp_acc.get_vp(refine_window=3)
        if video_vp is not None:
            vp = (video_vp[0], video_vp[1], support)
        else:
            vp = None
        
        # Draw overlay
        overlay = draw_vanishing_point_overlay(frame, vp, lines, label_prefix='Video VP')
        # Optionally draw per-frame VP (dim) if available
        if per_vp is not None:
            pvx, pvy, _ = per_vp
            cv2.drawMarker(overlay, (int(pvx), int(pvy)), (0, 128, 255),
                           cv2.MARKER_TILTED_CROSS, 20, 2)
        
        # Save to video
        writer.write(overlay)
        
        # Store vanishing point
        if vp is not None:
            vp_x, vp_y = vp[0], vp[1]
            vp_history.append((float(vp_x), float(vp_y)))
            frame_indices.append(frame_idx)
        
        # Save sample frames
        if frame_idx % save_every_n == 0 or frame_idx < 5:
            sample_path = os.path.join(out_dir, f'frame_{frame_idx:04d}.png')
            cv2.imwrite(sample_path, overlay)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f'  Processed {frame_idx}/{total_frames} frames')
    
    cap.release()
    writer.release()
    
    print(f'Saved output video: {out_video}')
    print(f'Detected vanishing point in {len(vp_history)}/{total_frames} frames '
          f'({100*len(vp_history)/total_frames:.1f}%)')
    
    # Plot vanishing point trajectory
    if len(vp_history) > 0:
        vp_array = np.array(vp_history)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
        
        # X coordinate over time
        ax1.plot(frame_indices, vp_array[:, 0], 'b-', alpha=0.5, label='Raw X')
        if len(vp_array) > 10:
            # Moving average
            window = min(11, len(vp_array))
            smooth_x = np.convolve(vp_array[:, 0], np.ones(window)/window, mode='valid')
            ax1.plot(frame_indices[window//2:len(smooth_x)+window//2], smooth_x, 
                    'r-', linewidth=2, label='Smoothed X')
        ax1.set_ylabel('X Coordinate (pixels)')
        ax1.set_title('Vanishing Point X-Coordinate Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Y coordinate over time
        ax2.plot(frame_indices, vp_array[:, 1], 'b-', alpha=0.5, label='Raw Y')
        if len(vp_array) > 10:
            smooth_y = np.convolve(vp_array[:, 1], np.ones(window)/window, mode='valid')
            ax2.plot(frame_indices[window//2:len(smooth_y)+window//2], smooth_y,
                    'r-', linewidth=2, label='Smoothed Y')
        ax2.set_ylabel('Y Coordinate (pixels)')
        ax2.set_xlabel('Frame Number')
        ax2.set_title('Vanishing Point Y-Coordinate Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 2D trajectory
        ax3.plot(vp_array[:, 0], vp_array[:, 1], 'b-', alpha=0.3)
        ax3.scatter(vp_array[:, 0], vp_array[:, 1], c=frame_indices, 
                   cmap='viridis', s=10, alpha=0.6)
        ax3.set_xlabel('X Coordinate (pixels)')
        ax3.set_ylabel('Y Coordinate (pixels)')
        ax3.set_title('Vanishing Point Trajectory (2D Path)')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        cbar = plt.colorbar(ax3.scatter(vp_array[:, 0], vp_array[:, 1], 
                                       c=frame_indices, cmap='viridis', s=10))
        cbar.set_label('Frame Number')
        
        plt.tight_layout()
        plot_path = os.path.join(out_dir, 'vanishing_point_trajectory.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f'Saved trajectory plot: {plot_path}')
        
        # Statistics
        print(f'\nVanishing Point Statistics:')
        print(f'  X: mean={np.mean(vp_array[:, 0]):.1f}, '
              f'std={np.std(vp_array[:, 0]):.1f}, '
              f'range=[{np.min(vp_array[:, 0]):.1f}, {np.max(vp_array[:, 0]):.1f}]')
        print(f'  Y: mean={np.mean(vp_array[:, 1]):.1f}, '
              f'std={np.std(vp_array[:, 1]):.1f}, '
              f'range=[{np.min(vp_array[:, 1]):.1f}, {np.max(vp_array[:, 1]):.1f}]')
    
    # Create GIF snippet
    try:
        import imageio
        gif_frames = []
        cap = cv2.VideoCapture(out_video)
        frame_count = 0
        while frame_count < min(100, total_frames):  # First 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            gif_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
        cap.release()
        
        if len(gif_frames) > 0:
            gif_path = os.path.join(out_dir, 'vanishing_point_demo.gif')
            imageio.mimsave(gif_path, gif_frames, fps=fps/2, loop=0)
            print(f'Saved demo GIF: {gif_path}')
    except ImportError:
        print('imageio not available, skipping GIF creation')
    
    print(f'\nResults saved to {out_dir}/')
    return vp_history

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Video-based Vanishing Point Detection (manual edges/lines)')
    parser.add_argument('--video', '-v', type=str, default='vanishing.mp4', help='Path to input video file')
    parser.add_argument('--out', '-o', type=str, default='vanishing_output', help='Output directory')
    parser.add_argument('--x-bounds', type=float, nargs=2, default=(0.2, 0.8), help='Normalized X bounds for VP ROI (e.g., 0.2 0.8)')
    parser.add_argument('--y-bounds', type=float, nargs=2, default=(0.0, 0.5), help='Normalized Y bounds for VP ROI (e.g., 0.0 0.5)')
    parser.add_argument('--canny', type=int, nargs=2, default=(30, 120), help='Canny low/high thresholds')
    parser.add_argument('--hough-threshold', type=int, default=30, help='Hough vote threshold')
    parser.add_argument('--min-line-length', type=int, default=35, help='Minimum line length')
    parser.add_argument('--max-line-gap', type=int, default=10, help='Maximum line gap')
    parser.add_argument('--ransac-iters', type=int, default=500, help='RANSAC iterations (unused in temporal)')
    parser.add_argument('--ransac-thresh', type=int, default=15, help='RANSAC inlier threshold (pixels)')
    parser.add_argument('--save-every-n', type=int, default=15, help='Save every Nth frame as PNG')

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f'ERROR: Video file not found: {args.video}')
        sys.exit(1)

    print('Starting vanishing point detection...')
    run_vanishing_point_detection(
        video_path=args.video,
        out_dir=args.out,
        save_every_n=args.save_every_n,
        x_bounds=tuple(args.x_bounds),
        y_bounds=tuple(args.y_bounds),
        canny_low=args.canny[0],
        canny_high=args.canny[1],
        hough_threshold=args.hough_threshold,
        min_line_length=args.min_line_length,
        max_line_gap=args.max_line_gap,
        ransac_iterations=args.ransac_iters,
        ransac_threshold=args.ransac_thresh,
    )
    print('Done!')
