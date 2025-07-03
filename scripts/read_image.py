import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from numbernet.model import NumberNet


def load_model(model_path):
    model = NumberNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def predict(model, image_tensor):
    output = model(image_tensor)
    probabilities = F.softmax(output, dim=1)
    prediction = torch.argmax(probabilities, dim=1)
    return prediction.item()


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/read_image.py path_to_image")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = "models/numbernet_cnn.pt"

    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)
    prediction = predict(model, image_tensor)

    print(f"Predicted digit for '{image_path}': {prediction}")


if __name__ == "__main__":
    main()
