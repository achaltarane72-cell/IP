# Image Processing Practical Portal — Python Flask

This project follows the uploaded Image Processing Lab post-lab report structure.

## Practical modules
1. RGB / Grayscale, Arithmetic & Bitwise Operations
2. 2-D Geometric Transformations
3. Histogram Equalization, Spatial Enhancement & Thresholding
4. Spatial Domain Filters
5. Image Inpainting
6. Lossless Image Compression
7. Morphological Operations
8. Object Detection using Correlation
9. Top-Hat Transformation
10. Colour Space Conversion
11. Edge Detection

> The uploaded report says “Include all 12 practicals”, but its portal screenshots/list show modules 01 through 11. This package follows the 11 modules actually listed.

## Run locally
```powershell
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open: http://127.0.0.1:5000

## Pure Python mode
Example:
```powershell
python run_practical.py 2 Rotation
```
Output files are saved under `static/outputs/`.

## Deploy on Render
Push this folder to GitHub, then create a Render Web Service using the included `render.yaml`.
Start command:
```text
gunicorn app:app
```

## GitHub note
GitHub stores your source code. The Flask website itself runs on a Python host such as Render; GitHub Pages cannot execute the Flask backend.
