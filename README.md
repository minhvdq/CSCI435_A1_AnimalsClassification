# Usage Guide for a1.py

## Setup

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

2. Make sure your image dataset is organized in subfolders under `Animals10/`, e.g.:
   ```
   Animals10/
     Bird/
     Cat/
     ...
   ```

## Running

### 1. Feature Visualization

- **SIFT or HOG visualization:**
  ```sh
  python a1.py -f SIFT path/to/image.jpg
  python a1.py -f HOG path/to/image.jpg
  ```

### 2. Color Panel Extraction

- **HSV or YCrCb color panels:**
  ```sh
  python a1.py -c HSV path/to/image.jpg
  python a1.py -c YCrCb path/to/image.jpg
  ```

### 3. Classification

- **Run SIFT+SVM classification:**
  ```sh
  python a1.py -r Animals10
  ```

### 4. Save Feature Images

- **Save SIFT/HOG feature images to output folder:**
  ```sh
  python a1.py -fs SIFT path/to/image.jpg output
  python a1.py -fs HOG path/to/image.jpg output
  ```

## Notes

- All arguments are required for their respective options.
- Output images will be saved in the specified folder.