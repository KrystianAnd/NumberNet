import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from numbernet.model import NumberNet


def load_model(model_path):
    model = NumberNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image


def predict_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    return prediction.item()


def main():
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter the path to the image file: ")

    if not os.path.exists(image_path):
        print(f"File '{image_path}' does not exist!")
        sys.exit(1)

    model_path = "models/numbernet_cnn.pt"
    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)
    prediction = predict_image(model, image_tensor)

    print(f"\n Image: {image_path}")
    print(f"\n Predicted digit: {prediction}")


if __name__ == "__main__":
    main()
