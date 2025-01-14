from mlops.model import dreamer
from mlops.data import corrupt_mnist

import torch
import warnings
import pytest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.mark.parametrize("batch_size", [32, 64])
def test_training_devices(batch_size: int) -> None:
    """Test if all components are on the same device during training."""
    model = dreamer().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for img, target in train_dataloader:
        # Check if input images and targets are on the correct device
        assert img.device == DEVICE, f"Input images are on {img.device}, expected {DEVICE}"
        assert target.device == DEVICE, f"Targets are on {target.device}, expected {DEVICE}"

        # Check if model parameters are on the correct device
        for param in model.parameters():
            assert param.device == DEVICE, f"Model parameter is on {param.device}, expected {DEVICE}"

        # Perform a forward pass and check if outputs are on the correct device
        y_pred = model(img.to(DEVICE))
        assert y_pred.device == DEVICE, f"Model outputs are on {y_pred.device}, expected {DEVICE}"

        break