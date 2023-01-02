import ads3_pt_profile as ads3
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

    def __getitem__(self, index: int):
        """Get a sample from the dataset"""
        image = Image.open(str(self.filenames[index]))
        labelStr = self.filenames[index].parts[-3]
        label = self.labels.index(labelStr)
        # we don't perform any augmentations at this point
        return image, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.filenames)

class DatasetTrain(D.Dataset):
    def __init__(self, data):
        self.data = data

        """Initialise the data pipeline"""
        self.transform = transforms.Compose(
            [
                transforms.Resize(INPUT_SIZE),
                transforms.CenterCrop(INPUT_SIZE),
                # the difference between the two datasets
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index: int):
        """Get a sample from the dataset"""
        x, y = self.data[index]
        return self.transform(x), y

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.data)

class DatasetValid(D.Dataset):
    def __init__(self, data):
        self.data = data

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
        x, y = self.data[index]
        return self.transform(x), y

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.data)


if __name__ == "__main__":
    """Initialise dataset"""
    labels = ads3.get_labels()
    # dataset without transforms
    dataset = CarDataset(labels=labels)

    """Split train and test"""
    train_len = int(0.7 * len(dataset))
    valid_len = len(dataset) - train_len
    train, valid = D.random_split(dataset, lengths=[train_len, valid_len])

    # dataset that does separate transforms
    train = DatasetTrain(train)
    valid = DatasetValid(valid)

    # When running image augmentation you should define seperate training and validation!

    print("train size: %d, valid size %d" % (len(train), len(valid)))

    loader_train = D.DataLoader(
        train,
        batch_size=20,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=2,
    )
    loader_valid = D.DataLoader(
        valid,
        batch_size=20,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=2,
    )

    log_file = "out/6py.csv"
    trace_file = "out/trace6py.json"

    ads3.run_experiment(
        loader_train, loader_valid, log_file, trace_file
    )  # For profiling feel free to lower epoch count via epoch=X
