import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Canvas
from PIL import Image, ImageTk

# Load trained model
model = load_model("ocr_model.h5")


def preprocessImage(imagePath):
    """Preprocess the image to match the MNIST input format."""
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

def predictDigit():
    """Open file to select an image, predict the digit, and update the UI."""
    filePath = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", ".png;.jpg;*.jpeg")])
    
    if filePath:
        img = preprocessImage(filePath)
        prediction = model.predict(img)
        digit = np.argmax(prediction)

        # Update UI with prediction
        resultLabel.config(text=f"Predicted Digit: {digit}", font=("Arial", 18, "bold"), fg="#0A66C2")

        # Display selected image
        imgPil = Image.open(filePath)
        imgPil = imgPil.resize((250, 250))
        imgTk = ImageTk.PhotoImage(imgPil)
        imageLabel.config(image=imgTk)
        imageLabel.image = imgTk
    else:
        resultLabel.config(text="No image selected.", fg="red")

# Create UI window
root = tk.Tk()
root.title("Handwritten Digit Recognition")
root.geometry("500x600")
root.configure(bg="#F3F2EF")

# Header Frame
titleFrame = Frame(root, bg="#0A66C2", height=80)
titleFrame.pack(fill="x")
titleLabel = Label(titleFrame, text="Handwritten Digit Recognition", font=("Arial", 22, "bold"), fg="white", bg="#0A66C2")
titleLabel.pack(pady=20)

# Main Frame
mainFrame = Frame(root, bg="#FFFFFF", bd=2, relief="groove")
mainFrame.pack(pady=20, padx=20, fill="both", expand=True)

# Select Image Button
selectButton = Button(mainFrame, text="Select Image", command=predictDigit, font=("Arial", 14, "bold"), bg="#0A66C2", fg="white", padx=20, pady=10, relief="flat")
selectButton.pack(pady=20)

# Image Display
imageLabel = Label(mainFrame, bg="#FFFFFF")
imageLabel.pack(pady=10)

# Prediction Result
resultLabel = Label(mainFrame, text="", font=("Arial", 18, "bold"), bg="#FFFFFF")
resultLabel.pack(pady=20)

# Run the UI
root.mainloop()