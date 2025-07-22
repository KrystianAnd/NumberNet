# NumberNet – Digit Recognition from Images

NumberNet is a PyTorch-based project that allows you to:
- train a digit recognition model using the MNIST and SVHN datasets,
- segment digits from a provided image (e.g., receipt, paper, etc.),
- predict digit values from the image using a trained model.

---

## 🧠 Network Architecture (CNN)

The `NumberNet` model is a 3-layer convolutional neural network:
- 3 convolutional layers (Conv2d) with ReLU and max pooling,
- Dropout for regularization,
- 2 fully connected (Linear) layers,
- Softmax at the end (implied by `CrossEntropyLoss`).

---

## 💡 How Digit Segmentation and Prediction Works

The NumberNet model is trained on grayscale images with a resolution of 28×28 pixels.  
Therefore, before making a prediction, the input image is automatically transformed into this format:

- If the image is in color (RGB), it is converted to grayscale.
- The image is then thresholded and segmented to find a single digit.
- The detected digit is cropped, resized proportionally, and centered on a 28×28 background.
- Finally, the image is normalized (mean 0.5, std 0.5) and passed to the model.

This means you can use photos of handwritten digits, e.g. on paper, and the model will adapt them to the required format automatically.

---

## 🗂️ Project Structure
```
Numbernet/
│
├── numbernet/
│ ├── dataset.py # Dataset loading and preprocessing (MNIST, SVHN)
│ └── model.py # CNN model definition (NumberNet)
├── scripts/
│ ├── predict.py # Digit segmentation and prediction
│ └── train.py # Model training on MNIST + SVHN
├── models/
│ └── numbernet.pt # Trained model checkpoint
├── data/ # Downloaded datasets
├── tests/
│ └── test_model.py
├── requirements.txt 
└── pyproject.toml 
```
---

## 🚀 How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model

```
python train.py
```

### 3. Predict digit from image

```
python predict.py path/to/your/image.png
```

### ⚠️ If you get an error using `python` command

Some systems default to Python 2 or have different Python paths.  
If running the script with `python` causes issues, try this instead:

```
PYTHONPATH=. python3 scripts/predict.py path/to/your/image.png
```

---

## 🧪 Training Data

The project uses two datasets:

- MNIST – classic grayscale digit images,
- SVHN – digit photos from house number signs (in color).

Both datasets are automatically downloaded and processed when running train.py.

---

## 📊 Results

The model is evaluated on the combined MNIST and SVHN test sets.
The best-performing model (based on test accuracy) is automatically saved.

---

## 🛠️ Requirements (requirements.txt)

```
torch==2.2.0
torchvision==0.17.0
numpy==1.26.0
matplotlib==3.8.0
pillow==10.0.0
ipython==8.20.0
ruff==0.4.3
pytest==8.4.0
opencv-python==4.9.0.80
tqdm==4.66.2
```

---

## 📌  Notes

- The model assumes there's one dominant digit in the input image.
- Segmentation works best on high-contrast images with black backgrounds.
- You can train the model using CPU or GPU (if available).

---

## 👤 Author

Krystian Andrzejak

---

## 📝 License

MIT License – free to use and modify.