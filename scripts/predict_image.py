import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from numbernet.model import NumberNet


def load_model(model_path):
    model = NumberNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def get_test_loader():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    return test_loader


def predict_sample(model, images):
    outputs = model(images)
    probabilities = F.softmax(outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    return predictions


def main():
    model_path = "models/numbernet_cnn.pt"
    model = load_model(model_path)

    test_loader = get_test_loader()
    images, labels = next(iter(test_loader))

    predictions = predict_sample(model, images)

    for i in range(5):
        print(f"Predicted: {predictions[i].item()}, Actual: {labels[i].item()}")


if __name__ == "__main__":
    main()
