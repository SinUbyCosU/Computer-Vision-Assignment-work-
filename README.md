ehicle speed-proxy via sparse pyramidal Lucas-Kanade (NumPy)

Files added:
- `optical_flow.py`: from-scratch pyramidal LK implementation (NumPy). Uses SciPy convolution for downsampling kernel.
- `vehicle_speed_proxy.py`: runner script that loads a video, selects an ROI, detects features, tracks them, writes overlay images and a speed plot.

Dependencies
- Python 3.8+
- numpy, scipy, opencv-python, matplotlib

Install (PowerShell):
```powershell
python -m pip install numpy scipy opencv-python matplotlib
```

Quick run
1. Put your short phone video in the folder and rename it to `input.mp4`, or edit `VIDEO_PATH` in `vehicle_speed_proxy.py`.
2. (Optional) edit ROI coordinates in `vehicle_speed_proxy.py` or set to None to use default.
3. Run in PowerShell:
```powershell
python vehicle_speed_proxy.py
```

Outputs
- `output/overlay_*.png`: example frames with sparse flow arrows overlaid.
- `output/speed_proxy.png`: plot of median per-frame magnitude * FPS (pixels/sec).

Notes and assumptions
- This implementation computes gradients using simple finite differences and builds a small pyramid using Gaussian downsampling (requires SciPy).
- Feature detection uses OpenCV's Shi-Tomasi (allowed for I/O/prep).
- If the input video FPS is not available, the script assumes 30 FPS.
- This is a minimal, instructional implementation and not optimized for large videos.
