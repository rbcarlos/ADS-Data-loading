import ads3 as ads3
import torch
import torchvision.transforms as transforms

from pathlib import Path
from PIL import Image
from torch.utils import data as D

torch.manual_seed(0)
INPUT_SIZE = 224
root = Path("data")
file_train = root / "train.txt"
folder_images = root / "image"


class CarDataset(D.Dataset):
    def __init__(self, labels: list):
        self.filenames = []
        self.labels = labels

        """Read the dataset index file"""
        with open(file_train, newline="\n") as trainfile:
            for line in trainfile:
                self.filenames.append(folder_images / line.strip())

        """Initialise the data pipeline"""
        self.transform = transforms.Compose(
            [
                transforms.Resize(INPUT_SIZE),
                transforms.CenterCrop(INPUT_SIZE),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index: int):
        """Get a sample from the dataset"""
        image = Image.open(str(self.filenames[index]))
        labelStr = self.filenames[index].parts[-3]
        label = self.labels.index(labelStr)
        return self.transform(image), label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.filenames)


if __name__ == "__main__":
    """Initialise dataset"""
    labels = ads3.get_labels()
    dataset = CarDataset(labels=labels)

    """Split train and test"""
    train_len = int(0.7 * len(dataset))
    valid_len = len(dataset) - train_len
    train, valid = D.random_split(dataset, lengths=[train_len, valid_len])

    # When running image augmentation you should define seperate training and validation!

    print("train size: %d, valid size %d" % (len(train), len(valid)))

    loader_train = D.DataLoader(
        train,
        batch_size=80,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=2,
    )
    loader_valid = D.DataLoader(
        valid,
        batch_size=80,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=2,
    )

    ads3.run_experiment(
        loader_train, loader_valid
    )  # For profiling feel free to lower epoch count via epoch=X
