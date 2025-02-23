# Image to text coverter

## Description
This project is a GUI-based application for handwritten digit recognition using a trained OCR model. It allows users to upload an image containing a handwritten digit, preprocesses the image, and predicts the digit using a pre-trained deep learning model.

## Features
- Load and preprocess handwritten digit images.
- Predict digits using a trained OCR model.
- User-friendly interface built with Tkinter.
- Image display and prediction result visualization.

## Requirements
To run this project, install the following dependencies:

```bash
pip install opencv-python numpy tensorflow pillow
```

## Usage
1. Run the script:
   ```bash
   python run_ocr.py
   ```
2. Click the "Select Image" button and choose an image containing a handwritten digit.
3. The application will process the image and display the predicted digit.

## How to Run
To run this file, follow these steps:

1. Ensure that you have installed all required dependencies.
2. Make sure `ocr_model.h5` is present in the working directory.
3. Open a terminal or command prompt in the project directory.
4. Run the following command:
   ```bash
   python run_ocr.py
   ```
5. The GUI window will open, allowing you to select an image and see the prediction result.

## Files
- `run_ocr.py`: Main script that loads the model, processes images, and runs the Tkinter GUI.
- `ocr_model.h5`: Pre-trained model for digit recognition (ensure it's in the same directory).

## Dependencies
- Python 3.x
- OpenCV
- TensorFlow/Keras
- NumPy
- PIL (Pillow)
- Tkinter

## Notes
Ensure that `ocr_model.h5` is available in the working directory before running the script.

## License
This project is open-source. Feel free to modify and use it as needed.

