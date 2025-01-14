from src.mlops.model import dreamer
import torch
import warnings
import pytest

def test_model():
    model = dreamer()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)

warnings.filterwarnings("ignore")

# def test_error_on_wrong_shape():
#     model = dreamer()
#     with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
#         model(torch.randn(1,2,3))
#     with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
#         model(torch.randn(1,1,28,29))