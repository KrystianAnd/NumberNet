import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from numbernet.model import NumberNet


def load_model(model_path: str) -> torch.nn.Module:
    model = NumberNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def preprocess_pil_image(image: Image.Image) -> torch.Tensor:
    image = image.resize((28, 28))
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    return transform(image).unsqueeze(0)


def resize_and_pad(image, size=28, pad_value=0):
    h, w = image.shape[:2]
    scale = min((size - 4) / w, (size - 4) / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size), pad_value, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized

    return canvas


def segment_digits(image_path: str) -> list[torch.Tensor]:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"File not found: {image_path}")

    if np.mean(img) < 127:
        img = cv2.bitwise_not(img)

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found!")
        return []

    candidates = []
    img_center_x = thresh.shape[1] // 2

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if area < 50 or h < 10 or w < 2:
            continue
        if h / w < 1.2:
            continue
        if abs((x + w // 2) - img_center_x) > 0.4 * thresh.shape[1]:
            continue

        candidates.append((x, y, w, h, area))

    if not candidates:
        print("No suitable contour found.")
        return []

    x, y, w, h, _ = max(candidates, key=lambda c: c[4])

    digit_crop = thresh[y : y + h, x : x + w]
    digit_padded = resize_and_pad(digit_crop)

    digit_pil = Image.fromarray(digit_padded)
    digit_tensor = preprocess_pil_image(digit_pil)

    return [digit_tensor]


def predict_image(model: torch.nn.Module, image_tensor: torch.Tensor) -> int:
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    return prediction.item()


def main():
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter the image path: ")

    if not os.path.exists(image_path):
        print(f"File '{image_path}' does not exist!")
        sys.exit(1)

    model_path = "models/numbernet.pt"
    model = load_model(model_path)

    digit_tensors = segment_digits(image_path)

    if not digit_tensors:
        print("No digits found in the image.")
        return

    result = ""
    for digit_tensor in digit_tensors:
        prediction = predict_image(model, digit_tensor)
        result += str(prediction)

    print(f"\nDetected number from '{image_path}': {result}")


if __name__ == "__main__":
    main()
