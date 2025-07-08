import torch

from numbernet.model import NumberNet


def test_model_output_shape():
    model = NumberNet()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 10), "Output shape should be (1, 10) for 10 classes"


def test_model_predicts_number():
    model = NumberNet()
    model.load_state_dict(torch.load("models/numbernet_cnn.pt", map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        output = model(dummy_input)
        predicted = torch.argmax(output, dim=1).item()
    assert 0 <= predicted <= 9, "Predicted class should be between 0 and 9"
