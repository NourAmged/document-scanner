# Document Scanner

This project is a simple Document Scanner built with Python and OpenCV. It allows you to scan documents from an image or webcam feed, automatically detects the document edges, applies a perspective transform, and outputs a clean, top-down scanned version of the document.

## Features
- Scan documents from an image file or webcam stream
- Automatic edge detection and perspective transformation
- Adjustable Canny edge detection thresholds via trackbars
- Adaptive thresholding for clean, high-contrast output
- Save scanned and thresholded images with a timestamp

## Requirements
- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy

## Installation
1. Clone this repository or download the source code.
2. Install the required packages:
   ```bash
   pip install opencv-python numpy
   ```

## Usage
1. Place the image you want to scan in the project directory and update the `path_image` variable in `main.py`.
2. To use a webcam feed, set `web_cam_feed = True` and update the video source URL if needed.
3. Run the scanner:
   ```bash
   python main.py
   ```
4. Use the trackbars to adjust edge detection thresholds for best results.
5. Press `s` to save the scanned and thresholded images. Press `q` to quit.


## Example

