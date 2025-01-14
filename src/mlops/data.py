import torch
import typer


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images as (X - mean(X)) / std(X).

    Parameters:
    - images: Tensor of shape (N, 1, 28, 28)

    Returns:
    - Normalized images
    """
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data from data/raw and saves it to processed directory at data/processed.

    Parameters:
    - raw_dir: Directory containing raw data.
    - processed_dir: Directory to save processed data.

    Returns:
    - None
    """
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST.

    Parameters:
    - None

    Returns:
    - Tuple of train and test datasets
    """
    train_images = torch.load("data/processed/train_images.pt", weights_only=True)
    train_target = torch.load("data/processed/train_target.pt", weights_only=True)
    test_images = torch.load("data/processed/test_images.pt", weights_only=True)
    test_target = torch.load("data/processed/test_target.pt", weights_only=True)

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
