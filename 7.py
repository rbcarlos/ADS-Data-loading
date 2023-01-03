import ads3 as ads3
import torch
import torchvision.transforms as transforms

from pathlib import Path
from PIL import Image
from torch.utils import data as D

import argparse

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
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                # brightness because of pictures taken against the sun
                # hue because of the cars of different colors
                # saturation and contrast because of outside lighting (shadows, fog, etc.)
                transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=.5, hue=.3, saturation=0.3, contrast=0.3)], p=0.5),
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
        # further augmentation only for train set
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


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j", "--threads", type=int,
        default=1
        )
    parser.add_argument(
        "--logfile", default="out/9py.csv"
    )
    parser.add_argument(
        "--tracefile", default="out/trace9py.json"
    )
    parser.add_argument(
        "-p", "--profile", action="store_true"
    )
    return parser

if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()

    if args.profile:
        import ads3_pt_profile as ads3
    else:
        import ads3 as ads3

    """Initialise dataset"""
    labels = ads3.get_labels()
    dataset = CarDataset(labels=labels)

    """Split train and test"""
    train_len = int(0.7 * len(dataset))
    valid_len = len(dataset) - train_len
    train, valid = D.random_split(dataset, lengths=[train_len, valid_len])

    train = DatasetTrain(train)
    valid = DatasetValid(valid)

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

    log_file = args.logfile
    trace_file = args.tracefile

    ads3.run_experiment(
        loader_train, loader_valid, log_file, trace_file
    )  # For profiling feel free to lower epoch count via epoch=X
